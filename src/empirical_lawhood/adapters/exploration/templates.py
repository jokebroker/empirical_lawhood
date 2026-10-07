"""Fourteen registered exploratory templates and their safe instantiation service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import ClassVar

from empirical_lawhood.kernel.admission import AdmissionSet, ReachabilityResult
from empirical_lawhood.kernel.atlases import ResponseAtlas
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.identification import StructuralConvergenceResult
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.models import ModelRelation, ViewModelSetSpec
from empirical_lawhood.kernel.provenance import EvidenceSnapshot, ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.discovery import (
    AnalysisObligations,
    AnalysisTemplate,
    ExplorationDiagnosticProjection,
    ExplorationEvidenceView,
    ExplorationIntent,
    QuantityReportingObligation,
    TemplateInstantiation,
    obligations_extension,
)
from empirical_lawhood.planning.exploration import (
    AnalysisProposal,
    AnalysisSpec,
    AnomalyKind,
    AnomalySignal,
    SearchAxis,
)


@dataclass(frozen=True, slots=True)
class TemplateLibrary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/exploration/template-library'

    library_id: str
    templates: tuple[AnalysisTemplate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.library_id, field_name="library_id")
        require_sorted_unique_ids(self.templates, attribute="template_id", field_name="templates")
        if len(self.templates) != 14:
            raise ValueError("Exploration template library must contain the complete fourteen families")

    def matching(self, kind: AnomalyKind) -> tuple[AnalysisTemplate, ...]:
        return tuple(template for template in self.templates if kind in template.anomaly_kinds)


@dataclass(frozen=True, slots=True)
class TemplateContext(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/exploration/template-context'

    context_id: str
    snapshot: ObjectIdentity
    evidence_view: ObjectIdentity
    search_axes: tuple[SearchAxis, ...]
    quantity_reporting: tuple[QuantityReportingObligation, ...]
    analysis_budget: ResourceBudget
    executed_template_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.context_id, field_name="context_id")
        require_sorted_unique_ids(self.search_axes, attribute="axis_id", field_name="search_axes")
        if not self.search_axes:
            raise ValueError("template context requires at least one registered search axis")
        require_sorted_unique_ids(
            self.quantity_reporting,
            attribute="quantity_id",
            field_name="quantity_reporting",
        )
        if not self.quantity_reporting:
            raise ValueError("template context requires native-unit quantity reporting")
        require_sorted_unique_strings(
            self.executed_template_ids,
            field_name="executed_template_ids",
        )


def _template(
    template_id: str,
    question: str,
    estimand: str,
    anomaly_kinds: tuple[AnomalyKind, ...],
    *,
    roles: tuple[str, ...],
    upstream: tuple[str, ...],
    axes: tuple[str, ...],
    pipeline: str,
    transforms: tuple[str, ...],
    estimators: tuple[str, ...],
    decompositions: tuple[str, ...],
    nulls: tuple[str, ...] = ("matched-null",),
    controls: tuple[str, ...] = ("source-backed-positive-control",),
    falsifiers: tuple[str, ...] = ("wrong-action", "wrong-time"),
) -> AnalysisTemplate:
    return AnalysisTemplate(
        template_id=template_id,
        template_version="1.0.0",
        question=question,
        estimand=estimand,
        intent=ExplorationIntent.RETROSPECTIVE_DISCOVERY,
        anomaly_kinds=tuple(sorted(anomaly_kinds)),
        required_role_ids=tuple(sorted(roles)),
        required_upstream_schema_ids=tuple(sorted(upstream)),
        required_search_axis_ids=tuple(sorted(axes)),
        registered_pipeline_key=pipeline,
        registered_pipeline_version="1.0.0",
        transform_keys=tuple(sorted(transforms)),
        estimator_keys=tuple(sorted(estimators)),
        decomposition_keys=tuple(sorted(decompositions)),
        matched_null_keys=tuple(sorted(nulls)),
        positive_control_keys=tuple(sorted(controls)),
        falsifier_keys=tuple(sorted(falsifiers)),
        terminal_missing_role_code="unevaluable-exploration",
        terminal_missing_upstream_code="prerequisite-not-met",
    )


def default_template_library() -> TemplateLibrary:
    """Generalize recurring post-hoc questions without hard-coding coordinates."""

    view = ExplorationEvidenceView.SCHEMA
    templates = (
        _template(
            "action-tangent-constraint-normal",
            "Where do supported action tangents align with receiver-constraint normals?",
            "Native-unit angle and intersection margin by supported local cell.",
            (AnomalyKind.ADMISSION_FRAGILITY, AnomalyKind.TRANSPORT_FAILURE),
            roles=("action", "constraint", "receiver"),
            upstream=(AdmissionSet.SCHEMA, ResponseLaw.SCHEMA),
            axes=("action-chart", "denominator-cell"),
            pipeline="exploration.response-geometry",
            transforms=("within-fold-native-scaling",),
            estimators=("local-angle",),
            decompositions=("constraint-normal", "response-tangent"),
        ),
        _template(
            "active-coordinate-quotient-equivalence",
            "Which action coordinates are active, null or empirically equivalent locally?",
            "Supported local rank, null quotient and action-equivalence classes.",
            (AnomalyKind.COORDINATE_FAILURE, AnomalyKind.COUNTERFEIT_RANK),
            roles=("action", "receiver", "support"),
            upstream=(ResponseLaw.SCHEMA,),
            axes=("action-chart", "denominator-cell"),
            pipeline="exploration.response-geometry",
            transforms=("within-fold-native-scaling",),
            estimators=("stable-local-rank",),
            decompositions=("active-null-quotient",),
        ),
        _template(
            "support-chart-denominator-fragmentation",
            "Where do support, chart overlap or denominator fragmentation limit inference?",
            "Coverage and overlap by denominator, chart, gauge and horizon cell.",
            (AnomalyKind.SUPPORT_GAP,),
            roles=("action", "denominator", "independent-unit", "receiver", "support"),
            upstream=(view,),
            axes=("denominator-cell", "horizon"),
            pipeline="exploration.support-structure",
            transforms=("training-only-standardization",),
            estimators=("grouped-support-coverage",),
            decompositions=("chart-overlap", "denominator-fragmentation"),
        ),
        _template(
            "admission-margin-threshold-fragility",
            "Which admission gates or thresholds make a decision fragile?",
            "Per-gate native-unit margin and decision sensitivity surface.",
            (AnomalyKind.ADMISSION_FRAGILITY,),
            roles=("constraint", "receiver", "uncertainty"),
            upstream=(AdmissionSet.SCHEMA,),
            axes=("admission-gate", "denominator-cell"),
            pipeline="exploration.admission-structure",
            transforms=("one-sided-bound-preserving",),
            estimators=("threshold-sensitivity",),
            decompositions=("per-gate-margin",),
        ),
        _template(
            "counterfeit-rank-representation-preprocessing",
            "Does apparent rank survive representation and preprocessing controls?",
            "Stable rank and principal-angle variation across frozen representations.",
            (AnomalyKind.COORDINATE_FAILURE, AnomalyKind.COUNTERFEIT_RANK),
            roles=("action", "receiver", "support"),
            upstream=(view,),
            axes=("representation", "transform"),
            pipeline="exploration.response-geometry",
            transforms=("fold-local-preprocessing", "native-coordinate-control"),
            estimators=("stable-local-rank",),
            decompositions=("representation", "transform"),
        ),
        _template(
            "residual-memory-closure",
            "What retained history is required to close the local response state?",
            "Held-out residual dependence and same-state/different-history contrast.",
            (AnomalyKind.RESIDUAL_MEMORY,),
            roles=("history", "receiver", "time"),
            upstream=(view,),
            axes=("history-ladder", "horizon"),
            pipeline="exploration.response-dynamics",
            transforms=("causal-lag-construction",),
            estimators=("residual-memory-test",),
            decompositions=("history-ladder",),
            falsifiers=("reversed-history", "wrong-history"),
        ),
        _template(
            "replicate-denominator-gauge-horizon-chart-transport",
            (
                "Which structures recur or fail across replicate, denominator, gauge, horizon "
                "or chart?"
            ),
            "Directional recurrence and rejected-transport envelope.",
            (AnomalyKind.COORDINATE_FAILURE, AnomalyKind.TRANSPORT_FAILURE),
            roles=("denominator", "independent-unit", "receiver", "time"),
            upstream=(ModelRelation.SCHEMA,),
            axes=("denominator-cell", "gauge", "horizon"),
            pipeline="exploration.transport-structure",
            transforms=("direction-preserving-alignment",),
            estimators=("held-out-directional-transport",),
            decompositions=("replicate", "transport-axis"),
        ),
        _template(
            "distributional-receiver-opposing-components",
            "Do distributional receiver components move coherently or oppose one another?",
            "Native component displacement and distributional divergence by receiver gauge.",
            (AnomalyKind.CONTRADICTION, AnomalyKind.UNEXPECTED_NULL),
            roles=("receiver", "support"),
            upstream=(view,),
            axes=("gauge", "receiver-component"),
            pipeline="exploration.response-geometry",
            transforms=("mass-preserving-normalization",),
            estimators=("distributional-displacement",),
            decompositions=("opposing-components",),
        ),
        _template(
            "plant-observer-selector-reachability-loss",
            "Where is response lost between plant, observer, selector and reachable action?",
            "Stage-local rank, support and reachability loss.",
            (AnomalyKind.TRANSPORT_FAILURE,),
            roles=("action", "observer", "receiver", "selector"),
            upstream=(ReachabilityResult.SCHEMA, ResponseAtlas.SCHEMA),
            axes=("interface", "stage"),
            pipeline="exploration.reachability-structure",
            transforms=("interface-aligned-projection",),
            estimators=("stage-loss",),
            decompositions=("observer", "plant", "reachability", "selector"),
        ),
        _template(
            "wrong-action-donor-specificity-topology",
            "Which donors or wrong actions retain receiver advantage and where?",
            "Wrong-action retention topology by action, donor and denominator cell.",
            (AnomalyKind.CONTRADICTION,),
            roles=("action", "receiver", "support"),
            upstream=(view,),
            axes=("action-chart", "donor", "denominator-cell"),
            pipeline="exploration.support-structure",
            transforms=("support-matched-action-permutation",),
            estimators=("action-specificity",),
            decompositions=("donor-topology",),
            falsifiers=("wrong-action", "wrong-donor", "wrong-time"),
        ),
        _template(
            "relaxation-propagator-semigroup-arrow",
            "Does the observed response obey a stable relaxation, propagator or time arrow?",
            "Finite-horizon propagator consistency and reverse-time asymmetry.",
            (AnomalyKind.RESIDUAL_MEMORY,),
            roles=("history", "receiver", "time"),
            upstream=(view,),
            axes=("horizon", "time-direction"),
            pipeline="exploration.response-dynamics",
            transforms=("causal-window", "reverse-time-control"),
            estimators=("finite-horizon-propagator",),
            decompositions=("relaxation-mode", "semigroup-defect"),
            falsifiers=("reverse-time", "wrong-time"),
        ),
        _template(
            "feedback-screening-delivery-actuator-shadow",
            (
                "Is apparent invariance caused by feedback screening, delivery loss or "
                "actuator shadow?"
            ),
            "Requested/applied/realized channel contrast under source-backed clocks.",
            (AnomalyKind.TRANSPORT_FAILURE, AnomalyKind.UNEXPECTED_NULL),
            roles=("action", "applied-action", "receiver", "time"),
            upstream=(view,),
            axes=("action-channel", "horizon"),
            pipeline="exploration.transport-structure",
            transforms=("requested-applied-clock-alignment",),
            estimators=("channel-specificity",),
            decompositions=("actuator-shadow", "delivery", "feedback-screening"),
        ),
        _template(
            "numerical-view-model-set-discrepancy",
            "Does structure persist across numerical views and the frozen plausible model set?",
            "Per-view structural difference and model-set discrepancy envelope.",
            (AnomalyKind.CONTRADICTION, AnomalyKind.COUNTERFEIT_RANK),
            roles=("numerical-view", "receiver", "support"),
            upstream=(ViewModelSetSpec.SCHEMA, StructuralConvergenceResult.SCHEMA),
            axes=("model", "numerical-view"),
            pipeline="exploration.model-set-structure",
            transforms=("native-unit-view-alignment",),
            estimators=("structural-discrepancy",),
            decompositions=("model", "numerical-view"),
        ),
        _template(
            "independent-unit-information-design-sensitivity",
            (
                "Is the negative result information-limited under the physical independent-unit "
                "design?"
            ),
            "Detectable-effect and design-sensitivity envelope at the physical-unit level.",
            (AnomalyKind.INFORMATION_LIMIT,),
            roles=("independent-unit", "receiver", "uncertainty"),
            upstream=(view,),
            axes=("design-sensitivity", "receiver"),
            pipeline="exploration.information-limit",
            transforms=("physical-unit-aggregation",),
            estimators=("design-sensitivity-envelope",),
            decompositions=("information-bound",),
        ),
    )
    return TemplateLibrary(
        library_id="exploration-analysis-templates",
        templates=tuple(sorted(templates, key=lambda template: template.template_id)),
    )


def _family_member_ids(axes: tuple[SearchAxis, ...]) -> tuple[str, ...]:
    members = (
        ".".join(
            f"{axis.axis_id}--{candidate}" for axis, candidate in zip(axes, values, strict=True)
        )
        for values in product(*(axis.candidate_ids for axis in axes))
    )
    return tuple(sorted(members))


class AnalysisProposalFactory:
    """Instantiate only registered schemas and primitives; never generated code."""

    factory_key = "exploration.registered-template-factory"
    factory_version = "1.0.0"

    @staticmethod
    def _bounded_instance_id(
        prefix: str,
        snapshot: EvidenceSnapshot,
        signal: AnomalySignal,
        template: AnalysisTemplate,
    ) -> str:
        payload = ":".join((snapshot.fingerprint(), signal.signal_id, template.template_id)).encode(
            "utf-8"
        )
        suffix = hashlib.sha256(payload).hexdigest()[:16]
        return f"{prefix}.exploration.{template.template_id}.{suffix}"

    def instantiate(
        self,
        snapshot: EvidenceSnapshot,
        view: ExplorationEvidenceView,
        signal: AnomalySignal,
        template: AnalysisTemplate,
        context: TemplateContext,
    ) -> TemplateInstantiation:
        self._validate_inputs(snapshot, view, signal, template, context)
        available_axes = {axis.axis_id: axis for axis in context.search_axes}
        selected_axes = tuple(
            available_axes[axis_id]
            for axis_id in template.required_search_axis_ids
            if axis_id in available_axes
        )
        if not selected_axes:
            selected_axes = (context.search_axes[0],)
        missing_roles = tuple(
            sorted(set(template.required_role_ids) - set(view.available_role_ids))
        )
        intrinsic_schemas = {
            ExplorationDiagnosticProjection.SCHEMA,
            ExplorationEvidenceView.SCHEMA,
        }
        available_upstream = set(view.available_upstream_schema_ids) | intrinsic_schemas
        missing_upstream = tuple(
            sorted(set(template.required_upstream_schema_ids) - available_upstream)
        )
        missing_axes = tuple(sorted(set(template.required_search_axis_ids) - set(available_axes)))
        reasons = []
        if missing_roles:
            reasons.append(template.terminal_missing_role_code)
        if missing_upstream or missing_axes:
            reasons.append(template.terminal_missing_upstream_code)
        reason_codes = tuple(sorted(set(reasons)))
        analysis_id = self._bounded_instance_id("analysis", snapshot, signal, template)
        obligations = AnalysisObligations(
            obligations_id=f"obligations.{analysis_id}",
            analysis_id=analysis_id,
            intent=template.intent,
            competing_explanation_ids=(
                "declared-response-structure",
                "measurement-or-observation-artifact",
                "selection-scaling-or-aggregation-artifact",
            ),
            transform_keys=template.transform_keys,
            estimator_keys=template.estimator_keys,
            decomposition_keys=template.decomposition_keys,
            grouping_rule_keys=("physical-independent-unit",),
            folding_rule_keys=("grouped-outcome-visible-resampling",),
            causal_clock_ids=view.grouping_clock_ids,
            application_clock_ids=view.grouping_clock_ids,
            matched_null_keys=template.matched_null_keys,
            positive_control_keys=template.positive_control_keys,
            falsifier_keys=template.falsifier_keys,
            uncertainty_rule_key="physical-unit-resampling",
            influence_rule_key="leave-one-physical-unit-out",
            evaluability_rule_key="required-role-and-support-complete",
            multiplicity_rule_key="complete-family-reporting",
            sensitivity_keys=("aggregation", "native-scaling", "selection"),
            reporting_keys=("all-attempts", "native-units", "normalized-secondary"),
            registered_family_member_ids=_family_member_ids(selected_axes),
            quantity_reporting=context.quantity_reporting,
            non_promotion_statement=(
                'Outcome-visible exploration can nominate a fresh test but cannot promote measurement-through-controller-use.'
            ),
            outcome_access=view.outcome_access,
            parent_visibility_ceiling=view.visibility_ceiling,
            visibility_ceiling=view.visibility_ceiling,
        )
        analysis = AnalysisSpec(
            analysis_id=analysis_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_fingerprint=snapshot.fingerprint(),
            system_id=view.system_id,
            relation=view.relation,
            question=template.question,
            estimand=template.estimand,
            independent_unit_id=view.independent_unit_id,
            projection_ids=view.available_projection_ids,
            receiver_quantity_ids=view.relation.receiver_quantity_ids,
            denominator_quantity_ids=view.relation.denominator_quantity_ids,
            action_quantity_ids=view.relation.action_quantity_ids,
            horizon_ids=(view.relation.horizon.horizon_id,),
            grouping_clock_ids=view.grouping_clock_ids,
            registered_pipeline_key=template.registered_pipeline_key,
            registered_pipeline_version=template.registered_pipeline_version,
            null_capability_keys=template.matched_null_keys,
            falsifier_capability_keys=template.falsifier_keys,
            search_axes=selected_axes,
            uncertainty_method_key="physical-unit-resampling",
            budget=context.analysis_budget,
            stopping_rule="Execute and retain the complete registered family exactly once.",
            outcome_access=view.outcome_access,
            parent_visibility_ceiling=view.visibility_ceiling,
            visibility_ceiling=view.visibility_ceiling,
            extensions=(obligations_extension(obligations),),
        )
        prerequisites = tuple(sorted({*missing_roles, *missing_upstream, *missing_axes}))
        proposal = AnalysisProposal(
            proposal_id=self._bounded_instance_id("proposal", snapshot, signal, template),
            analysis=analysis,
            anomaly_signal_ids=(signal.signal_id,),
            intent=(
                "Apply a registered bounded diagnostic family and retain null, failure and "
                "unevaluable outcomes."
            ),
            prerequisite_ids=prerequisites,
            selection_rationale=(
                "Template matched the typed anomaly; portfolio selection remains multi-objective."
            ),
        )
        return TemplateInstantiation(
            instantiation_id=self._bounded_instance_id("instantiation", snapshot, signal, template),
            template=ObjectIdentity.from_record(template.template_id, template),
            proposal=proposal,
            obligations=obligations,
            ready=not reason_codes,
            reason_codes=reason_codes,
        )

    @staticmethod
    def _validate_inputs(
        snapshot: EvidenceSnapshot,
        view: ExplorationEvidenceView,
        signal: AnomalySignal,
        template: AnalysisTemplate,
        context: TemplateContext,
    ) -> None:
        if context.snapshot != ObjectIdentity.from_record(snapshot.snapshot_id, snapshot):
            raise ValueError("template context binds another EvidenceSnapshot")
        if context.evidence_view != ObjectIdentity.from_record(view.view_id, view):
            raise ValueError("template context binds another evidence view")
        if (
            signal.snapshot_id != snapshot.snapshot_id
            or signal.relation_id != view.relation.relation_id
        ):
            raise ValueError("analysis signal belongs to another snapshot or relation")
        if signal.kind not in template.anomaly_kinds:
            raise ValueError("analysis template does not support the anomaly kind")


def instantiate_matching_templates(
    snapshot: EvidenceSnapshot,
    view: ExplorationEvidenceView,
    signals: tuple[AnomalySignal, ...],
    context: TemplateContext,
    library: TemplateLibrary | None = None,
) -> tuple[TemplateInstantiation, ...]:
    selected_library = library or default_template_library()
    factory = AnalysisProposalFactory()
    values = (
        factory.instantiate(snapshot, view, signal, template, context)
        for signal in signals
        for template in selected_library.matching(signal.kind)
    )
    return tuple(sorted(values, key=lambda value: value.instantiation_id))
