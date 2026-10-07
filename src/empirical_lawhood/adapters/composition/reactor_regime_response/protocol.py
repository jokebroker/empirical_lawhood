"""Frozen B, C and D public task topology before phase issue.

Cross-issue B packages are exact external inputs to C.  They are deliberately
not represented as dependencies on tasks in another run.
"""

from __future__ import annotations

from empirical_lawhood.adapters.methods.reactor_regime_response.calibration_pipeline import RegimeCalibrationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.config import PREFIX, ROOTS, ReactorRegimeResponseDesign
from empirical_lawhood.adapters.methods.reactor_regime_response.extension_bundle import CAPABILITY as METHOD_CAPABILITY
from empirical_lawhood.adapters.methods.reactor_regime_response.law_terminal import RegimeJointLawResult
from empirical_lawhood.adapters.methods.reactor_regime_response.model_records import RegimeFitPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.discovery_records import RegimeDiscoveryTraining, RegimeDiscoveryDevelopment
from empirical_lawhood.adapters.methods.reactor_regime_response.discovery_diagnostics import RegimePhaseDiagnostics, RegimeDiscoveryConfirmation
from empirical_lawhood.adapters.methods.reactor_regime_response.preassay_readout import RegimePreassayReadout, RegimeOpportunityReadout
from empirical_lawhood.adapters.methods.reactor_regime_response.nomination_records import RegimeNominationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.qualification_pipeline import RegimeQualificationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_closeout import RegimeDRootSealResult
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_reveal import RegimeDRootRevealResult
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_cohort import ReactorRegimeResponseCohortProspective
from empirical_lawhood.adapters.methods.reactor_regime_response.prospective_decision import CausalValidityRegimeAssignment
from empirical_lawhood.adapters.simulators.reactor_regime_response.prospective_records import RegimeDNativeRoot
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.simulators.reactor_regime_response.config import ReactorRegimeNativeConfig
from empirical_lawhood.adapters.simulators.reactor_regime_response.extension_bundle import CAPABILITY as SOURCE_CAPABILITY
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.composition.experiment_authoring import study_template_from_protocol as _programme_template
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, ContentIdentityPolicy, StudyTemplate
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.capabilities import CapabilityPermission
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificInputRole,
    ScientificStage,
)
from dataclasses import replace
from typing import Callable, cast

from .resources import phase_cpu_reservations


def phase_protocol(
    phase: str,
    design: ReactorRegimeResponseDesign,
    native: ReactorRegimeNativeConfig,
) -> ProtocolTemplate:
    """Return the exact B or C local task census, with causal seal barriers."""
    if native.design != design:
        raise ValueError("reactor phase or fixed native design differs")
    if phase == "D":
        return _d_protocol(design, native)
    if phase not in ("B", "C"):
        raise ValueError("reactor phase differs")
    roles = ("fit", "nomination") if phase == "B" else ("calibration", "qualification")
    roots = tuple(root for root, role, _, _ in ROOTS if role in roles)
    cpu_reservations = dict(phase_cpu_reservations(phase))
    source_config = _config_ref(native, ObjectIdentity.from_record(native.config_id, native), SOURCE_CAPABILITY)
    method_config = _config_ref(design, ObjectIdentity.from_record(design.config_id, design), METHOD_CAPABILITY)

    def source_step(root: str, role: str, kind: str) -> ProtocolStepTemplate:
        task = f"regime.{kind}.{root}"
        deps = (
            () if kind == "prepare" else
            (f"regime.prepare.{root}", f"regime.seal.{root}", "regime.calibration")
            if role == "qualification" else
            (f"regime.prepare.{root}", f"regime.seal.{root}")
        )
        outputs = (
            (("causal", RegimeCausalPreparation.SCHEMA), ("private", RegimePrivatePreparation.SCHEMA))
            if kind == "prepare"
            else (("assay", RegimeAssayPanel.SCHEMA),)
        )
        source_scan_bytes = 128 * 1024**2 if kind == "prepare" else 256 * 1024**2
        return ProtocolStepTemplate(
            task,
            ScientificStage.PREPARE if kind == "prepare" else ScientificStage.ACQUIRE,
            SOURCE_CAPABILITY.capability_key,
            SOURCE_CAPABILITY.capability_version,
            source_config,
            tuple(sorted(deps)),
            _outputs(outputs),
            SOURCE_CAPABILITY.permissions,
            OutcomeAccess.EVALUATION_SEALED,
            VisibilityCeiling.PROSPECTIVE,
            ResourceBudget(1, 8 * 1024**3, 0, cpu_reservations[task], source_scan_bytes, 128 * 1024**2),
            (),
            BarrierKind.FREEZE if kind == "assay" else BarrierKind.NONE,
            1,
            (f"reactor-regime.{phase.lower()}.{role}.{kind}.{root}",),
        )

    def method_step(
        task: str,
        deps: tuple[str, ...],
        schema: str,
        *,
        sealed: bool = False,
        terminal: bool = False,
        extra_outputs: tuple[tuple[str, str], ...] = (),
    ) -> ProtocolStepTemplate:
        # Cohort reducers bind up to 64 full preparation/assay records.
        # Compressed constant software fixtures underestimate their bytes.
        # This is a scan ceiling, not a larger worker memory allocation.
        source_scan_bytes = (
            8 * 1024**3 if task in (
                "regime.fit", "regime.nomination", "regime.calibration",
                "regime.qualification", "regime.law-qualification",
            ) else 128 * 1024**2 if task.startswith("regime.seal.")
            else 256 * 1024**2
        )
        return ProtocolStepTemplate(
            task,
            ScientificStage.EVALUATE if sealed else ScientificStage.FREEZE,
            METHOD_CAPABILITY.capability_key,
            METHOD_CAPABILITY.capability_version,
            method_config,
            tuple(sorted(set(deps))),
            _outputs((("science", schema), *extra_outputs)),
            METHOD_CAPABILITY.permissions if sealed else tuple(
                permission for permission in METHOD_CAPABILITY.permissions
                if permission is not CapabilityPermission.REVEAL_OUTCOMES
            ),
            OutcomeAccess.EVALUATOR_REVEAL if sealed else OutcomeAccess.EVALUATION_SEALED,
            VisibilityCeiling.OUTCOME_VISIBLE if terminal else VisibilityCeiling.PROSPECTIVE,
            ResourceBudget(1, 8 * 1024**3, 0, cpu_reservations[task], source_scan_bytes, 128 * 1024**2),
            (),
            BarrierKind.REVEAL if sealed else BarrierKind.FREEZE,
            1,
            (f"{PREFIX}.phase-{phase.lower()}.single-terminal",)
            if terminal else (f"reactor-regime.{phase.lower()}.{task}",),
        )

    steps: list[ProtocolStepTemplate] = []
    for root in roots:
        role = next(role for name, role, _, _ in ROOTS if name == root)
        steps.append(source_step(root, role, "prepare"))
        seal_deps = [f"regime.prepare.{root}"]
        if role != "fit" and phase == "B":
            seal_deps.append("regime.fit")
        if role == "qualification":
            seal_deps.append("regime.calibration")
        steps.append(method_step(f"regime.seal.{root}", tuple(seal_deps), RegimePredictionSeal.SCHEMA,
                                 extra_outputs=(("preassay", RegimePreassayReadout.SCHEMA),)))
        steps.append(source_step(root, role, "assay"))

    if phase == "B":
        fit_roots = tuple(root for root, role, _, _ in ROOTS if role == "fit")
        nom_roots = tuple(root for root, role, _, _ in ROOTS if role == "nomination")
        steps.append(method_step(
            "regime.fit",
            tuple(f"regime.{kind}.{root}" for root in fit_roots for kind in ("prepare", "seal", "assay")),
            RegimeFitPackage.SCHEMA,
            extra_outputs=(("discovery-training", RegimeDiscoveryTraining.SCHEMA),),
        ))
        steps.append(method_step(
            "regime.nomination",
            ("regime.fit", *(f"regime.{kind}.{root}" for root in nom_roots for kind in ("prepare", "seal", "assay"))),
            RegimeNominationPackage.SCHEMA,
            extra_outputs=(("discovery-development", RegimeDiscoveryDevelopment.SCHEMA),
                           ("diagnostics", RegimePhaseDiagnostics.SCHEMA),
                           ("opportunities", RegimeOpportunityReadout.SCHEMA)),
        ))
        steps.append(method_step(
            "regime.adjudication",
            ("regime.fit", "regime.nomination"),
            ScientificAdjudicationRecord.SCHEMA,
            sealed=True,
            terminal=True,
        ))
    else:
        cal_roots = tuple(root for root, role, _, _ in ROOTS if role == "calibration")
        qual_roots = tuple(root for root, role, _, _ in ROOTS if role == "qualification")
        steps.append(method_step(
            "regime.calibration",
            tuple(f"regime.{kind}.{root}" for root in cal_roots for kind in ("prepare", "seal", "assay")),
            RegimeCalibrationPackage.SCHEMA,
            extra_outputs=(("diagnostics", RegimePhaseDiagnostics.SCHEMA),
                           ("opportunities", RegimeOpportunityReadout.SCHEMA)),
        ))
        steps.append(method_step(
            "regime.qualification",
            ("regime.calibration", *(f"regime.{kind}.{root}" for root in qual_roots for kind in ("prepare", "seal", "assay"))),
            RegimeQualificationPackage.SCHEMA,
            extra_outputs=(("diagnostics", RegimePhaseDiagnostics.SCHEMA),
                           ("regime-confirmation", RegimeDiscoveryConfirmation.SCHEMA),
                           ("opportunities", RegimeOpportunityReadout.SCHEMA)),
        ))
        steps.append(method_step(
            "regime.law-qualification",
            ("regime.calibration", "regime.qualification", *(f"regime.assay.{root}" for root in qual_roots)),
            RegimeJointLawResult.SCHEMA,
        ))
        steps.append(method_step(
            "regime.adjudication",
            ("regime.law-qualification", "regime.qualification"),
            ScientificAdjudicationRecord.SCHEMA,
            sealed=True,
            terminal=True,
        ))
    return ProtocolTemplate(
        f"reactor-regime-phase-{phase.lower()}.protocol",
        "1.0.0",
        tuple(sorted(steps, key=lambda step: step.step_id)),
        False,
        False,
        True,
    )


def _d_protocol(
    design: ReactorRegimeResponseDesign,
    native: ReactorRegimeNativeConfig,
) -> ProtocolTemplate:
    roots = tuple(root for root, role, _, _ in ROOTS if role == "prospective")
    cpu = dict(phase_cpu_reservations("D"))
    source_config = _config_ref(native, ObjectIdentity.from_record(native.config_id, native), SOURCE_CAPABILITY)
    method_config = _config_ref(design, ObjectIdentity.from_record(design.config_id, design), METHOD_CAPABILITY)
    steps: list[ProtocolStepTemplate] = []

    def add(
        task: str, deps: tuple[str, ...], outputs: tuple[tuple[str, str], ...],
        *, source: bool = False, reveal: bool = False, terminal: bool = False,
    ) -> None:
        capability = SOURCE_CAPABILITY if source else METHOD_CAPABILITY
        stage = (
            ScientificStage.PREPARE if task.startswith("regime.prepare.") else
            ScientificStage.ACQUIRE if source else
            ScientificStage.EVALUATE if reveal else ScientificStage.FREEZE
        )
        permissions = capability.permissions if source or reveal else tuple(
            permission for permission in capability.permissions
            if permission is not CapabilityPermission.REVEAL_OUTCOMES
        )
        steps.append(ProtocolStepTemplate(
            task, stage, capability.capability_key, capability.capability_version,
            source_config if source else method_config,
            tuple(sorted(set(deps))), _outputs(outputs), permissions,
            OutcomeAccess.EVALUATOR_REVEAL if reveal else OutcomeAccess.EVALUATION_SEALED,
            VisibilityCeiling.OUTCOME_VISIBLE if terminal else VisibilityCeiling.PROSPECTIVE,
            ResourceBudget(1, 8 * 1024**3, 0, cpu[task],
                           8 * 1024**3 if task in ("regime.d-plan", "regime.d-cohort")
                           else 1024**3 if task.startswith(("regime.d-seal.", "regime.d-reveal."))
                           else 256 * 1024**2 if source else 128 * 1024**2,
                           256 * 1024**2 if task.startswith("regime.action.") else 128 * 1024**2),
            (),
            BarrierKind.REVEAL if reveal else BarrierKind.FREEZE
            if not task.startswith("regime.prepare.") else BarrierKind.NONE,
            1,
            (f"{PREFIX}.phase-d.single-terminal",) if terminal else
            (f"reactor-regime.d.{task}",),
        ))

    for root in roots:
        prepare = f"regime.prepare.{root}"
        seal = f"regime.seal.{root}"
        assignment = f"regime.assignment.{root}"
        action = f"regime.action.{root}"
        d_seal = f"regime.d-seal.{root}"
        d_reveal = f"regime.d-reveal.{root}"
        add(prepare, (), (("causal", RegimeCausalPreparation.SCHEMA),
                          ("private", RegimePrivatePreparation.SCHEMA)), source=True)
        add(seal, (prepare,), (("science", RegimePredictionSeal.SCHEMA),))
        add(assignment, (prepare, seal), (("science", CausalValidityRegimeAssignment.SCHEMA),))
        add(action, (prepare, assignment, "regime.d-plan"),
            (("native", RegimeDNativeRoot.SCHEMA),), source=True)
        add(d_seal, (action, "regime.d-plan", assignment, prepare),
            (("science", RegimeDRootSealResult.SCHEMA),))
        add(d_reveal, (d_seal, action, assignment, prepare, "regime.d-plan"),
            (("science", RegimeDRootRevealResult.SCHEMA),), reveal=True)
    add("regime.d-plan", tuple(
        task for root in roots
        for task in (f"regime.prepare.{root}", f"regime.assignment.{root}")
    ), (("science", ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA),))
    add("regime.d-cohort", (
        "regime.d-plan", *(f"regime.d-reveal.{root}" for root in roots)
    ), (("science", ReactorRegimeResponseCohortProspective.SCHEMA),), reveal=True)
    add("regime.d-adjudication", ("regime.d-cohort", "regime.d-plan"),
        (("science", ScientificAdjudicationRecord.SCHEMA),),
        reveal=True, terminal=True)
    return ProtocolTemplate(
        "reactor-regime-phase-d.protocol", "1.0.0",
        tuple(sorted(steps, key=lambda step: step.step_id)),
        False, False, True,
    )


def phase_template(
    phase: str,
    design: ReactorRegimeResponseDesign,
    native: ReactorRegimeNativeConfig,
    source: EmpiricalStudySource,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
    *,
    prior_fit: ArtifactIdentity | None = None,
    prior_nomination: ArtifactIdentity | None = None,
    prior_discovery: ArtifactIdentity | None = None,
    prior_calibration: ArtifactIdentity | None = None,
    prior_qualification: ArtifactIdentity | None = None,
    prior_law: ArtifactIdentity | None = None,
) -> StudyTemplate:
    """Bind exact source and prior-phase records to the ordinary graph compiler."""
    if phase == "D":
        priors = (
            prior_fit, prior_nomination, prior_calibration,
            prior_qualification, prior_law,
        )
        if any(value is None for value in priors):
            raise ValueError("D requires all five exact B/C package identities")
        assert all(value is not None for value in priors)
        return _d_template(
            design, native, source, experiment, registry,
            cast(tuple[ArtifactIdentity, ArtifactIdentity, ArtifactIdentity,
                       ArtifactIdentity, ArtifactIdentity], priors),
        )
    if (phase == "C") != (prior_fit is not None and prior_nomination is not None and prior_discovery is not None):
        raise ValueError("C requires both exact B package identities; B requires neither")
    protocol = phase_protocol(phase, design, native)
    prefix = f"{PREFIX}.phase-{phase.lower()}"
    source_id = f"{prefix}.source-bundle"
    first_source_task = next(
        step.step_id for step in protocol.steps if step.step_id.startswith("regime.prepare.")
    )
    template = _programme_template(
        protocol,
        source,
        source_id,
        experiment,
        registry,
        prefix=prefix,
        source_task_id=first_source_task,
        terminal_task_id="regime.adjudication",
        primary_output_ids={
            ScientificStage.PREPARE: "causal",
            ScientificStage.ACQUIRE: "assay",
            ScientificStage.FREEZE: "science",
            ScientificStage.EVALUATE: "science",
        },
    )
    source_edge = next(edge for edge in template.graph.edges if edge.external_input_id == source_id)
    edges = [
        edge for edge in template.graph.edges
        if not (
            edge.producer_output_id == "private"
            and (edge.consumer_node_id.startswith("regime.seal.") or edge.consumer_node_id == "regime.fit")
        )
        and not (edge.producer_output_id == "discovery-training"
                 and edge.consumer_node_id != "regime.nomination")
        and not (edge.producer_output_id == "discovery-development"
                 and edge.consumer_node_id != "regime.adjudication")
        and not (edge.producer_output_id == "diagnostics"
                 and not (edge.producer_node_id == "regime.calibration"
                          and edge.consumer_node_id == "regime.qualification"))
        and not (edge.producer_output_id == "regime-confirmation"
                 and edge.consumer_node_id != "regime.adjudication")
        and edge.producer_output_id != "opportunities"
        and not (edge.producer_output_id == "preassay"
                 and edge.consumer_node_id not in ("regime.nomination", "regime.calibration", "regime.qualification"))
    ]
    edges.extend(
        replace(
            source_edge,
            edge_id=f"{prefix}.source-edge.{task.step_id}",
            consumer_node_id=task.step_id,
        )
        for task in protocol.steps
        if task.capability_key == SOURCE_CAPABILITY.capability_key
        and task.step_id != first_source_task
    )
    # A source preparation may have run before calibration, but the sealed
    # qualification assay is unreachable until calibration was persisted.
    external = list(template.graph.external_inputs)
    if phase == "C":
        assert prior_fit is not None and prior_nomination is not None
        for artifact, consumer_ids in (
            (prior_discovery, ("regime.qualification",)),
            (prior_fit, tuple(
                step.step_id for step in protocol.steps
                if step.step_id == "regime.calibration"
                or step.step_id == "regime.qualification"
                or step.step_id == "regime.law-qualification"
                or step.step_id.startswith("regime.seal.")
            )),
            (prior_nomination, tuple(
                step.step_id for step in protocol.steps
                if step.step_id == "regime.calibration"
                or step.step_id == "regime.qualification"
                or step.step_id == "regime.law-qualification"
                or step.step_id.startswith("regime.seal.")
            )),
        ):
            assert artifact is not None
            external.append(CandidateGraphExternalInput(
                artifact.artifact_id,
                ScientificInputRole.MODEL,
                artifact.artifact_id,
                ContentIdentityPolicy.EXACT_SHA256,
                artifact.sha256,
                artifact.payload_schema,
                artifact.media_type,
                artifact.size_bytes,
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
            ))
            for task_id in consumer_ids:
                edges.append(CandidateGraphEdge(
                    f"edge.{artifact.artifact_id}.{task_id}",
                    None,
                    None,
                    artifact.artifact_id,
                    task_id,
                    f"prior-{artifact.artifact_id}",
                    ScientificInputRole.MODEL,
                    artifact.artifact_id,
                    artifact.payload_schema,
                    artifact.media_type,
                    artifact.size_bytes,
                    OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.FREEZE,
                ))
    graph = replace(
        template.graph,
        external_inputs=tuple(sorted(external, key=lambda value: value.input_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )
    coverage = replace(
        template.coverage,
        bindings=tuple(
            replace(
                binding,
                contributor_edge_ids=tuple(
                    edge.edge_id for edge in graph.edges
                    if edge.consumer_node_id == binding.proof_owner_node_id
                ),
            )
            for binding in template.coverage.bindings
        ),
    )
    return replace(template, graph=graph, coverage=coverage)


def _d_template(
    design: ReactorRegimeResponseDesign,
    native: ReactorRegimeNativeConfig,
    source: EmpiricalStudySource,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
    priors: tuple[ArtifactIdentity, ArtifactIdentity, ArtifactIdentity,
                  ArtifactIdentity, ArtifactIdentity],
) -> StudyTemplate:
    protocol = _d_protocol(design, native)
    prefix = f"{PREFIX}.phase-d"
    source_id = f"{prefix}.source-bundle"
    first = next(step.step_id for step in protocol.steps if step.step_id.startswith("regime.prepare."))
    template = _programme_template(
        protocol, source, source_id, experiment, registry,
        prefix=prefix, source_task_id=first,
        terminal_task_id="regime.d-adjudication",
        primary_output_ids={
            ScientificStage.PREPARE: "causal",
            ScientificStage.ACQUIRE: "native",
            ScientificStage.FREEZE: "science",
            ScientificStage.EVALUATE: "science",
        },
    )
    source_edge = next(edge for edge in template.graph.edges if edge.external_input_id == source_id)

    def keep(edge: CandidateGraphEdge) -> bool:
        child = edge.consumer_node_id
        schema = edge.payload_schema
        if edge.producer_node_id is None:
            return True
        if child.startswith("regime.seal."):
            return schema == RegimeCausalPreparation.SCHEMA
        if child.startswith("regime.assignment."):
            return schema in (
                RegimeCausalPreparation.SCHEMA,
                RegimePredictionSeal.SCHEMA,
            )
        if child == "regime.d-plan":
            return schema in (
                RegimeCausalPreparation.SCHEMA,
                CausalValidityRegimeAssignment.SCHEMA,
            )
        return True

    edges = [edge for edge in template.graph.edges if keep(edge)]
    edges.extend(
        replace(source_edge,
                edge_id=f"{prefix}.source-edge.{step.step_id}",
                consumer_node_id=step.step_id)
        for step in protocol.steps
        if step.capability_key == SOURCE_CAPABILITY.capability_key
        and step.step_id != first
    )
    external = list(template.graph.external_inputs)
    fit, nomination, calibration, qualification, law = priors
    consumers: tuple[tuple[ArtifactIdentity, Callable[[str], bool]], ...] = (
        (fit, lambda task: task.startswith(("regime.seal.", "regime.assignment."))),
        (nomination, lambda task: task.startswith(("regime.prepare.", "regime.seal.", "regime.assignment.", "regime.action.")) or task == "regime.d-adjudication"),
        (calibration, lambda task: task.startswith(("regime.assignment.", "regime.action."))),
        (qualification, lambda task: task.startswith(("regime.prepare.", "regime.seal.", "regime.assignment.", "regime.action.")) or task == "regime.d-adjudication"),
        (law, lambda task: task.startswith(("regime.prepare.", "regime.seal.", "regime.assignment.", "regime.action.")) or task in ("regime.d-plan", "regime.d-adjudication")),
    )
    for artifact, accepts in consumers:
        prior_access = OutcomeAccess.EVALUATION_SEALED
        external.append(CandidateGraphExternalInput(
            artifact.artifact_id, ScientificInputRole.MODEL,
            artifact.artifact_id, ContentIdentityPolicy.EXACT_SHA256,
            artifact.sha256, artifact.payload_schema, artifact.media_type,
            artifact.size_bytes, prior_access,
            VisibilityCeiling.PROSPECTIVE,
        ))
        for step in protocol.steps:
            if accepts(step.step_id):
                edges.append(CandidateGraphEdge(
                    f"edge.{artifact.artifact_id}.{step.step_id}",
                    None, None, artifact.artifact_id, step.step_id,
                    f"prior-{artifact.artifact_id}", ScientificInputRole.MODEL,
                    artifact.artifact_id, artifact.payload_schema,
                    artifact.media_type, artifact.size_bytes,
                    prior_access,
                    VisibilityCeiling.PROSPECTIVE, BarrierKind.FREEZE,
                ))
    graph = replace(
        template.graph,
        external_inputs=tuple(sorted(external, key=lambda item: item.input_id)),
        edges=tuple(sorted(edges, key=lambda item: item.edge_id)),
    )
    coverage = replace(
        template.coverage,
        bindings=tuple(replace(
            binding,
            contributor_edge_ids=tuple(
                edge.edge_id for edge in graph.edges
                if edge.consumer_node_id == binding.proof_owner_node_id
            ),
        ) for binding in template.coverage.bindings),
    )
    return replace(template, graph=graph, coverage=coverage)
