"""Frozen development/evaluation seam for shared response-algebra methods.

This module contains no substrate logic.  It binds the common transparent
estimator to a development-only projection, freezes all selectable scientific
structure, and permits the evaluator to grade a sealed evaluation batch
without changing support, state, numerical view, thresholds or class
hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import ClassVar, Protocol, TypeVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.response_algebra import (
    ResponseAlgebraIdentificationResult,
    SignatureAxis,
    StructuralClassDisposition,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.response_algebra import ResponseAlgebraProtocolSpec
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .contracts import DataSplit
from .response_algebra import (
    ResponseAlgebraMethodConfig,
    ResponseAlgebraMethodInput,
    ResponseAlgebraMethodResult,
    ReceiverEquivalenceCriterion,
    identify_response_algebra,
)


RESPONSE_ALGEBRA_OBSERVATION_KEY = "response-algebra.identity-observation"
RESPONSE_ALGEBRA_NUMERICAL_KEY = "response-algebra.numerical-qualification"
RESPONSE_ALGEBRA_IDENTIFIER_KEY = "response-algebra.direct-affine-finite"
RESPONSE_ALGEBRA_EVALUATOR_KEY = "response-algebra.transparent-evaluator"
RESPONSE_ALGEBRA_REPORTER_KEY = "response-algebra.class-card-reporter"
RESPONSE_ALGEBRA_PROTOCOL_VERSION = "1.0.0"


class _SplitRecord(Protocol):
    @property
    def split(self) -> DataSplit: ...


_SplitRecordT = TypeVar("_SplitRecordT", bound=_SplitRecord)


def _calibration_records(values: tuple[_SplitRecordT, ...]) -> tuple[_SplitRecordT, ...]:
    return tuple(value for value in values if value.split is DataSplit.CALIBRATION)


@dataclass(frozen=True, slots=True)
class ResponseAlgebraNumericalQualification(CanonicalRecord):
    """Development-visible numerical floor and view qualification."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-numerical-qualification'

    qualification_id: str
    primary_numerical_view_id: str
    comparison_numerical_view_ids: tuple[str, ...]
    receiver_criteria: tuple[ReceiverEquivalenceCriterion, ...]
    independent_unit_ids: tuple[str, ...]
    qualified: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_stable_id(
            self.primary_numerical_view_id,
            field_name="primary_numerical_view_id",
        )
        require_sorted_unique_strings(
            self.comparison_numerical_view_ids,
            field_name="comparison_numerical_view_ids",
        )
        require_sorted_unique_ids(
            self.receiver_criteria,
            attribute="criterion_id",
            field_name="receiver_criteria",
        )
        require_sorted_unique_strings(
            self.independent_unit_ids,
            field_name="independent_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.qualified == bool(self.reason_codes):
            raise ValueError("numerical qualification status and reasons disagree")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraMethodSelection(CanonicalRecord):
    """One statically registered option in the shared, non-substrate menu."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-method-selection'

    selection_id: str
    method_key: str
    method_version: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.selection_id, field_name="selection_id")
        validate_stable_id(self.method_key, field_name="method_key")
        validate_semantic_version(self.method_version)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraDevelopmentFreeze(CanonicalRecord):
    """Immutable scientific choices selected without evaluation outcomes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-development-freeze'

    freeze_id: str
    protocol_id: str
    protocol_sha256: str
    development_projection_sha256: str
    method_config: ResponseAlgebraMethodConfig
    method_config_sha256: str
    method_menu: tuple[ResponseAlgebraMethodSelection, ...]
    selected_method_id: str
    development_unit_ids: tuple[str, ...]
    evaluation_unit_ids: tuple[str, ...]
    selected_state_view_ids: tuple[str, ...]
    selected_numerical_view_id: str
    primary_hypothesis_id: str
    class_hypothesis_sha256s: tuple[str, ...]
    evaluator_key: str
    support_rank_state_threshold_refit_forbidden: bool
    evaluation_outcomes_observed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("freeze_id", self.freeze_id),
            ("protocol_id", self.protocol_id),
            ("selected_method_id", self.selected_method_id),
            ("selected_numerical_view_id", self.selected_numerical_view_id),
            ("primary_hypothesis_id", self.primary_hypothesis_id),
            ("evaluator_key", self.evaluator_key),
        ):
            validate_stable_id(value, field_name=name)
        for name, value in (
            ("protocol_sha256", self.protocol_sha256),
            ("development_projection_sha256", self.development_projection_sha256),
            ("method_config_sha256", self.method_config_sha256),
        ):
            validate_sha256(value, field_name=name)
        if self.method_config.fingerprint() != self.method_config_sha256:
            raise ValueError("method config differs from its frozen fingerprint")
        require_sorted_unique_ids(
            self.method_menu,
            attribute="selection_id",
            field_name="method_menu",
        )
        if not self.method_menu:
            raise ValueError("response-algebra method menu must not be empty")
        if self.selected_method_id not in {value.selection_id for value in self.method_menu}:
            raise ValueError("selected response-algebra method is outside the frozen menu")
        require_sorted_unique_strings(
            self.development_unit_ids,
            field_name="development_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.evaluation_unit_ids,
            field_name="evaluation_unit_ids",
            allow_empty=False,
        )
        if set(self.development_unit_ids) & set(self.evaluation_unit_ids):
            raise ValueError("development and evaluation units overlap in the freeze")
        require_sorted_unique_strings(
            self.selected_state_view_ids,
            field_name="selected_state_view_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.class_hypothesis_sha256s,
            field_name="class_hypothesis_sha256s",
            allow_empty=False,
        )
        for digest in self.class_hypothesis_sha256s:
            validate_sha256(digest, field_name="class_hypothesis_sha256s")
        if not self.support_rank_state_threshold_refit_forbidden:
            raise ValueError("evaluation must forbid structural and threshold refitting")
        if self.evaluation_outcomes_observed:
            raise ValueError("a development freeze cannot observe evaluation outcomes")
        expected_views = tuple(
            sorted(
                value
                for value in (
                    self.method_config.primary_state_view_id,
                    self.method_config.augmented_state_view_id,
                    self.method_config.full_state_view_id,
                )
                if value is not None
            )
        )
        if self.selected_state_view_ids != expected_views:
            raise ValueError("frozen state views differ from the method config")
        if self.selected_numerical_view_id != self.method_config.primary_numerical_view_id:
            raise ValueError("frozen numerical view differs from the method config")


@dataclass(frozen=True, slots=True)
class FrozenClassAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/frozen-class-assessment'

    assessment_id: str
    hypothesis_id: str
    class_definition_id: str
    disposition: StructuralClassDisposition
    matched_axes: tuple[SignatureAxis, ...]
    opposed_axes: tuple[SignatureAxis, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("hypothesis_id", self.hypothesis_id),
            ("class_definition_id", self.class_definition_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.matched_axes, field_name="matched_axes")
        require_sorted_unique_strings(self.opposed_axes, field_name="opposed_axes")
        if set(self.matched_axes) & set(self.opposed_axes):
            raise ValueError("a frozen class axis cannot both match and oppose")
        if self.disposition is StructuralClassDisposition.MATCHED and (
            not self.matched_axes or self.opposed_axes
        ):
            raise ValueError("matched frozen class requires matched axes only")
        if self.disposition is StructuralClassDisposition.OPPOSED and not self.opposed_axes:
            raise ValueError("opposed frozen class requires an opposed axis")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraEvaluationOutput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-evaluation-output'

    evaluation_id: str
    freeze_id: str
    method_result: ResponseAlgebraMethodResult
    class_assessments: tuple[FrozenClassAssessment, ...]
    support_rank_state_threshold_refit_performed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        require_sorted_unique_ids(
            self.class_assessments,
            attribute="assessment_id",
            field_name="class_assessments",
        )
        if not self.class_assessments:
            raise ValueError("evaluation output requires frozen class assessments")
        if self.support_rank_state_threshold_refit_performed:
            raise ValueError("response-algebra evaluation cannot refit frozen structure")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraClassCard(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-algebra-class-card'

    card_id: str
    evaluation_id: str
    freeze_id: str
    signature_id: str
    primary_hypothesis_id: str
    primary_disposition: StructuralClassDisposition
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    caveat_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("card_id", self.card_id),
            ("evaluation_id", self.evaluation_id),
            ("freeze_id", self.freeze_id),
            ("signature_id", self.signature_id),
            ("primary_hypothesis_id", self.primary_hypothesis_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.caveat_codes, field_name="caveat_codes")
        if self.maximum_evidence_ceiling in {
            EvidenceCeiling.ADMISSION,
            EvidenceCeiling.CONTROLLER_USE,
        }:
            raise ValueError("response-algebra class card cannot exceed local law")


def _split_unit_ids(method_input: ResponseAlgebraMethodInput, split: DataSplit) -> tuple[str, ...]:
    unit_ids: set[str] = set()
    record_groups = (
        method_input.delivered_words,
        method_input.delivered_generators,
        method_input.word_responses,
        method_input.affine_transitions,
        method_input.discrete_transitions,
        method_input.generator_transitions,
    )
    for records in record_groups:
        unit_ids.update(value.independent_unit_id for value in records if value.split is split)
    return tuple(sorted(unit_ids))


def development_projection_sha256(method_input: ResponseAlgebraMethodInput) -> str:
    """Hash only development-visible records, excluding held-out outcomes."""

    projection = {
        "case_token": method_input.case_token,
        "state_views": method_input.state_views,
        "delivered_words": _calibration_records(method_input.delivered_words),
        "delivered_generators": _calibration_records(method_input.delivered_generators),
        "word_responses": _calibration_records(method_input.word_responses),
        "affine_transitions": _calibration_records(method_input.affine_transitions),
        "discrete_transitions": _calibration_records(method_input.discrete_transitions),
        "generator_transitions": _calibration_records(method_input.generator_transitions),
        "two_sided_actions_delivered": method_input.two_sided_actions_delivered,
        "local_state_closed": method_input.local_state_closed,
        "hybrid_switching_observed": method_input.hybrid_switching_observed,
    }
    return sha256(canonical_json_bytes(projection)).hexdigest()


def calibration_only_response_algebra_input(
    method_input: ResponseAlgebraMethodInput,
) -> ResponseAlgebraMethodInput:
    """Construct the exact development-visible projection without held-out rows."""

    return ResponseAlgebraMethodInput(
        case_token=method_input.case_token,
        state_views=method_input.state_views,
        delivered_words=_calibration_records(method_input.delivered_words),
        delivered_generators=_calibration_records(method_input.delivered_generators),
        word_responses=_calibration_records(method_input.word_responses),
        affine_transitions=_calibration_records(method_input.affine_transitions),
        discrete_transitions=_calibration_records(method_input.discrete_transitions),
        generator_transitions=_calibration_records(method_input.generator_transitions),
        two_sided_actions_delivered=method_input.two_sided_actions_delivered,
        local_state_closed=method_input.local_state_closed,
        hybrid_switching_observed=method_input.hybrid_switching_observed,
        evaluator_reveal_attestation_id=method_input.evaluator_reveal_attestation_id,
    )


def freeze_response_algebra_development(
    *,
    freeze_id: str,
    protocol: ResponseAlgebraProtocolSpec,
    method_input: ResponseAlgebraMethodInput,
    config: ResponseAlgebraMethodConfig,
    implementation_sha256: str,
) -> ResponseAlgebraDevelopmentFreeze:
    """Freeze the one shared method and every selectable structural choice."""

    if protocol.method_key != config.method_key or protocol.method_version != config.method_version:
        raise ValueError("protocol and response-algebra method config differ")
    if protocol.evaluator_key != RESPONSE_ALGEBRA_EVALUATOR_KEY:
        raise ValueError("protocol selects another response-algebra evaluator")
    if _split_unit_ids(method_input, DataSplit.HELD_OUT):
        raise ValueError("development freeze input contains held-out records")
    development_ids = _split_unit_ids(method_input, DataSplit.CALIBRATION)
    evaluation_ids = protocol.word_design.evaluation_unit_ids
    if not development_ids or not evaluation_ids:
        raise ValueError("freeze requires disjoint development and sealed evaluation units")
    if set(development_ids) & set(evaluation_ids):
        raise ValueError("method input reuses a development unit in evaluation")
    if development_ids != protocol.word_design.development_unit_ids:
        raise ValueError("method development units differ from the protocol")
    selected_views = tuple(
        sorted(
            value
            for value in (
                config.primary_state_view_id,
                config.augmented_state_view_id,
                config.full_state_view_id,
            )
            if value is not None
        )
    )
    primary = next(value for value in protocol.class_hypotheses if value.primary)
    menu = (
        ResponseAlgebraMethodSelection(
            selection_id="shared-direct-affine-finite",
            method_key=config.method_key,
            method_version=config.method_version,
            implementation_sha256=implementation_sha256,
        ),
    )
    return ResponseAlgebraDevelopmentFreeze(
        freeze_id=freeze_id,
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol.fingerprint(),
        development_projection_sha256=development_projection_sha256(method_input),
        method_config=config,
        method_config_sha256=config.fingerprint(),
        method_menu=menu,
        selected_method_id=menu[0].selection_id,
        development_unit_ids=development_ids,
        evaluation_unit_ids=evaluation_ids,
        selected_state_view_ids=selected_views,
        selected_numerical_view_id=config.primary_numerical_view_id,
        primary_hypothesis_id=primary.hypothesis_id,
        class_hypothesis_sha256s=tuple(
            sorted(value.fingerprint() for value in protocol.class_hypotheses)
        ),
        evaluator_key=protocol.evaluator_key,
        support_rank_state_threshold_refit_forbidden=True,
        evaluation_outcomes_observed=False,
    )


def evaluate_frozen_response_algebra(
    *,
    protocol: ResponseAlgebraProtocolSpec,
    method_input: ResponseAlgebraMethodInput,
    freeze: ResponseAlgebraDevelopmentFreeze,
) -> ResponseAlgebraEvaluationOutput:
    """Reveal and grade held-out responses under an exact development freeze."""

    if protocol.fingerprint() != freeze.protocol_sha256:
        raise ValueError("evaluation protocol differs from the development freeze")
    hypotheses = tuple(sorted(value.fingerprint() for value in protocol.class_hypotheses))
    if hypotheses != freeze.class_hypothesis_sha256s:
        raise ValueError("evaluation class hypotheses differ from the freeze")
    primary = next(value for value in protocol.class_hypotheses if value.primary)
    if primary.hypothesis_id != freeze.primary_hypothesis_id:
        raise ValueError("evaluation primary hypothesis differs from the freeze")
    if development_projection_sha256(method_input) != freeze.development_projection_sha256:
        raise ValueError("evaluation changed development-visible evidence")
    if _split_unit_ids(method_input, DataSplit.CALIBRATION) != freeze.development_unit_ids:
        raise ValueError("evaluation changed development independent units")
    if _split_unit_ids(method_input, DataSplit.HELD_OUT) != freeze.evaluation_unit_ids:
        raise ValueError("evaluation independent units differ from the freeze")
    selected = next(
        value for value in freeze.method_menu if value.selection_id == freeze.selected_method_id
    )
    if (
        selected.method_key != freeze.method_config.method_key
        or selected.method_version != freeze.method_config.method_version
    ):
        raise ValueError("frozen method selection and config differ")
    result = identify_response_algebra(method_input, freeze.method_config)
    observed = result.signature.axis_values()
    assessments: list[FrozenClassAssessment] = []
    for hypothesis in protocol.class_hypotheses:
        definition = hypothesis.class_definition
        matched = tuple(
            sorted(
                criterion.axis
                for criterion in definition.criteria
                if observed[criterion.axis] in criterion.allowed_values
            )
        )
        opposed = tuple(
            sorted(
                criterion.axis
                for criterion in definition.criteria
                if observed[criterion.axis] not in criterion.allowed_values
                and observed[criterion.axis] != "UNEVALUABLE"
            )
        )
        assessments.append(
            FrozenClassAssessment(
                assessment_id=f"class-assessment.{method_input.case_token}.{hypothesis.hypothesis_id}",
                hypothesis_id=hypothesis.hypothesis_id,
                class_definition_id=definition.class_id,
                disposition=definition.assess(result.signature),
                matched_axes=matched,
                opposed_axes=opposed,
            )
        )
    return ResponseAlgebraEvaluationOutput(
        evaluation_id=f"response-algebra-evaluation.{method_input.case_token}",
        freeze_id=freeze.freeze_id,
        method_result=result,
        class_assessments=tuple(sorted(assessments, key=lambda value: value.assessment_id)),
        support_rank_state_threshold_refit_performed=False,
    )


def _schema_sha256(schema: str) -> str:
    return sha256(schema.encode("utf-8")).hexdigest()


def response_algebra_protocol_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    """Register the exact shared observation/method/evaluation/reporting family."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    read_write = (
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    development = tuple(sorted((CapabilityPermission.READ_DEVELOPMENT, *read_write)))
    evaluator_permissions = tuple(
        sorted(
            (
                CapabilityPermission.READ_DEVELOPMENT,
                *read_write,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            )
        )
    )
    report_permissions = tuple(sorted((CapabilityPermission.READ_OUTCOME_VISIBLE, *read_write)))
    resources = ResourceBudget(
        cpu_cores=4,
        memory_bytes=8 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=3_600,
        source_scan_bytes=4 * 1024**3,
        output_bytes=512 * 1024**2,
    )
    specs = (
        (
            RESPONSE_ALGEBRA_EVALUATOR_KEY,
            CapabilityKind.EVALUATOR,
            ResponseAlgebraDevelopmentFreeze.SCHEMA,
            tuple(
                sorted(
                    (
                        ResponseAlgebraDevelopmentFreeze.SCHEMA,
                        ResponseAlgebraMethodInput.SCHEMA,
                        ResponseAlgebraMethodResult.SCHEMA,
                        ResponseAlgebraProtocolSpec.SCHEMA,
                    )
                )
            ),
            tuple(
                sorted(
                    (
                        ResponseAlgebraEvaluationOutput.SCHEMA,
                        ResponseAlgebraMethodResult.SCHEMA,
                    )
                )
            ),
            evaluator_permissions,
            OutcomeAccess.EVALUATOR_REVEAL,
            True,
            False,
        ),
        (
            RESPONSE_ALGEBRA_OBSERVATION_KEY,
            CapabilityKind.OBSERVATION_OPERATOR,
            ResponseAlgebraProtocolSpec.SCHEMA,
            tuple(
                sorted(
                    (
                        'empirical-lawhood/kernel/letter-action-word',
                        'empirical-lawhood/methods/affine-transition',
                        'empirical-lawhood/methods/word-response',
                    )
                )
            ),
            (ResponseAlgebraMethodInput.SCHEMA,),
            read_write,
            OutcomeAccess.EVALUATION_SEALED,
            True,
            False,
        ),
        (
            RESPONSE_ALGEBRA_REPORTER_KEY,
            CapabilityKind.REPORTER,
            ResponseAlgebraDevelopmentFreeze.SCHEMA,
            tuple(
                sorted(
                    (
                        ResponseAlgebraEvaluationOutput.SCHEMA,
                        ResponseAlgebraMethodResult.SCHEMA,
                    )
                )
            ),
            (ResponseAlgebraClassCard.SCHEMA,),
            report_permissions,
            OutcomeAccess.EVALUATION_REVEALED,
            True,
            False,
        ),
        (
            RESPONSE_ALGEBRA_IDENTIFIER_KEY,
            CapabilityKind.RESPONSE_ALGEBRA_IDENTIFIER,
            ResponseAlgebraMethodConfig.SCHEMA,
            tuple(
                sorted(
                    (
                        ResponseAlgebraMethodInput.SCHEMA,
                        ResponseAlgebraNumericalQualification.SCHEMA,
                    )
                )
            ),
            tuple(
                sorted(
                    (
                        ResponseAlgebraDevelopmentFreeze.SCHEMA,
                        ResponseAlgebraIdentificationResult.SCHEMA,
                        ResponseAlgebraMethodResult.SCHEMA,
                    )
                )
            ),
            development,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            False,
            True,
        ),
        (
            RESPONSE_ALGEBRA_NUMERICAL_KEY,
            CapabilityKind.NUMERICAL_QUALIFIER,
            ResponseAlgebraMethodConfig.SCHEMA,
            (ResponseAlgebraMethodInput.SCHEMA,),
            (ResponseAlgebraNumericalQualification.SCHEMA,),
            development,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            True,
            False,
        ),
    )
    manifests = tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version=RESPONSE_ALGEBRA_PROTOCOL_VERSION,
            kind=kind,
            config_schema=config_schema,
            config_schema_sha256=_schema_sha256(config_schema),
            input_schema_ids=input_schemas,
            output_schema_ids=output_schemas,
            permissions=permissions,
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=outcome,
            resource_ceiling=resources,
            deterministic=deterministic,
            seed_required=seed_required,
            language_id="python",
            runtime_id="cpython-3.11-response-algebra-protocol",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=(
                "development-evaluation-independent-units",
                "evaluator-only-class-assignment",
                "exact-action-ledger-and-clock",
                "no-child-specific-estimator-copy",
                "no-llm-rl-or-live-actuation",
                "support-rank-state-thresholds-frozen",
            ),
            implementation_sha256=implementation_sha256,
        )
        for (
            key,
            kind,
            config_schema,
            input_schemas,
            output_schemas,
            permissions,
            outcome,
            deterministic,
            seed_required,
        ) in specs
    )
    return CapabilityRegistry(
        registry_id="response-algebra-shared-protocol",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def build_response_algebra_protocol_template(
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
    resource_budget_by_step_id: dict[str, ResourceBudget],
) -> ProtocolTemplate:
    """Build the one generic freeze/reveal DAG without a substrate branch."""

    expected_step_ids = {"evaluate", "identify", "numerical", "observe", "report"}
    if set(config_by_step_id) != expected_step_ids:
        raise ValueError("response-algebra protocol config step set differs")
    if set(resource_budget_by_step_id) != expected_step_ids:
        raise ValueError("response-algebra protocol resource step set differs")
    definitions = (
        (
            "evaluate",
            ScientificStage.EVALUATE,
            RESPONSE_ALGEBRA_EVALUATOR_KEY,
            ("identify", "observe"),
            tuple(
                sorted(
                    (
                        ResponseAlgebraEvaluationOutput.SCHEMA,
                        ResponseAlgebraMethodResult.SCHEMA,
                    )
                )
            ),
            OutcomeAccess.EVALUATOR_REVEAL,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            BarrierKind.REVEAL,
        ),
        (
            "identify",
            ScientificStage.FREEZE,
            RESPONSE_ALGEBRA_IDENTIFIER_KEY,
            ("numerical", "observe"),
            (ResponseAlgebraDevelopmentFreeze.SCHEMA,),
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            BarrierKind.FREEZE,
        ),
        (
            "numerical",
            ScientificStage.QUALIFY,
            RESPONSE_ALGEBRA_NUMERICAL_KEY,
            ("observe",),
            (ResponseAlgebraNumericalQualification.SCHEMA,),
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            BarrierKind.NONE,
        ),
        (
            "observe",
            ScientificStage.TRANSFORM,
            RESPONSE_ALGEBRA_OBSERVATION_KEY,
            (),
            (ResponseAlgebraMethodInput.SCHEMA,),
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            BarrierKind.NONE,
        ),
        (
            "report",
            ScientificStage.REPORT,
            RESPONSE_ALGEBRA_REPORTER_KEY,
            ("evaluate",),
            (ResponseAlgebraClassCard.SCHEMA,),
            OutcomeAccess.EVALUATION_REVEALED,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            BarrierKind.NONE,
        ),
    )
    steps: list[ProtocolStepTemplate] = []
    for (
        step_id,
        stage,
        capability_key,
        dependencies,
        output_schemas,
        outcome_access,
        visibility,
        barrier,
    ) in definitions:
        manifest = registry.resolve(capability_key, RESPONSE_ALGEBRA_PROTOCOL_VERSION)
        config = config_by_step_id[step_id]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError("response-algebra step config schema differs from its manifest")
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=stage,
                capability_key=capability_key,
                capability_version=RESPONSE_ALGEBRA_PROTOCOL_VERSION,
                config=config,
                dependency_step_ids=dependencies,
                outputs=tuple(
                    OutputTemplate(
                        output_id=f"{step_id}.{index}",
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/json",
                        filename_suffix=".json",
                    )
                    for index, schema in enumerate(output_schemas)
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=outcome_access,
                visibility_ceiling=visibility,
                resource_budget=resource_budget_by_step_id[step_id],
                resource_lock_ids=(f"response-algebra-{step_id}",),
                barrier=barrier,
                maximum_attempts=2,
                obligation_ids=(f"response-algebra-{step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id="response-algebra-shared-protocol-template",
        template_version=RESPONSE_ALGEBRA_PROTOCOL_VERSION,
        steps=tuple(steps),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


class _StaticResponseAlgebraRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        records_by_schema: dict[str, CanonicalRecord],
    ) -> None:
        self.manifest = manifest
        self._records_by_schema = records_by_schema
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        outputs = tuple(
            TaskOutputPayload(
                output_id=port.output_id,
                payload=self._records_by_schema[port.payload_schema].canonical_bytes(),
            )
            for port in context.output_ports
        )
        return RunnerResult(
            outputs=outputs,
            checks=(ReceiptCheck("response-algebra-static-runtime-contract", True, ()),),
        )


class ResponseAlgebraCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Static synthetic provider used to prove generic protocol plumbing."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        records_by_schema: dict[str, CanonicalRecord],
    ) -> None:
        expected = response_algebra_protocol_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("response-algebra provider registry differs")
        for schema, record in records_by_schema.items():
            if record.SCHEMA != schema:
                raise ValueError("response-algebra static record schema differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = len(registry.capabilities)
        self.records_by_schema = dict(records_by_schema)
        self._runners = tuple(
            _StaticResponseAlgebraRunner(manifest, self.records_by_schema)
            for manifest in registry.capabilities
        )

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("response-algebra provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("response-algebra plan registry/source differs")
        if any(task.external_inputs for task in plan.tasks):
            raise ValueError("synthetic response-algebra conformance has external inputs")
        return ()

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("response-algebra semantic registry differs")
        record_types = {
            ResponseAlgebraClassCard.SCHEMA: ResponseAlgebraClassCard,
            ResponseAlgebraDevelopmentFreeze.SCHEMA: ResponseAlgebraDevelopmentFreeze,
            ResponseAlgebraEvaluationOutput.SCHEMA: ResponseAlgebraEvaluationOutput,
            ResponseAlgebraIdentificationResult.SCHEMA: ResponseAlgebraIdentificationResult,
            ResponseAlgebraMethodInput.SCHEMA: ResponseAlgebraMethodInput,
            ResponseAlgebraMethodResult.SCHEMA: ResponseAlgebraMethodResult,
            ResponseAlgebraNumericalQualification.SCHEMA: (ResponseAlgebraNumericalQualification),
        }
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                manifest,
                payload_schema=schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(field.name for field in fields(record_types[schema]))),
            )
            for manifest in registry.capabilities
            for schema in manifest.output_schema_ids
        )


__all__ = [
    "FrozenClassAssessment",
    "RESPONSE_ALGEBRA_EVALUATOR_KEY",
    "RESPONSE_ALGEBRA_IDENTIFIER_KEY",
    "RESPONSE_ALGEBRA_NUMERICAL_KEY",
    "RESPONSE_ALGEBRA_OBSERVATION_KEY",
    "RESPONSE_ALGEBRA_PROTOCOL_VERSION",
    "RESPONSE_ALGEBRA_REPORTER_KEY",
    "ResponseAlgebraCampaignRuntimeProvider",
    "ResponseAlgebraClassCard",
    "ResponseAlgebraDevelopmentFreeze",
    "ResponseAlgebraEvaluationOutput",
    "ResponseAlgebraMethodSelection",
    "ResponseAlgebraNumericalQualification",
    "build_response_algebra_protocol_template",
    "calibration_only_response_algebra_input",
    "development_projection_sha256",
    "evaluate_frozen_response_algebra",
    "freeze_response_algebra_development",
    "response_algebra_protocol_registry",
]
