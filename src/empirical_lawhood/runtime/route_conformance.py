"""Canonical topology and terminal receipt for public route conformance in the repository."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure

from .capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
)


class PublicRouteConformanceStage(StrEnum):
    OBSERVATION = "OBSERVATION"
    CANDIDATE_FAMILY = "CANDIDATE_FAMILY"
    COMPONENT_QUALIFICATION = "COMPONENT_QUALIFICATION"
    LAW_QUALIFICATION = "LAW_QUALIFICATION"
    ATLAS = "ATLAS"
    ADMISSION_RECEIPTS = "ADMISSION_RECEIPTS"
    CONTROLLER_PROGRAMME = "CONTROLLER_PROGRAMME"
    CONTROLLER_COMPILE = "CONTROLLER_COMPILE"
    CONTROLLER_TICK = "CONTROLLER_TICK"
    SEALED_REFERENCE = "SEALED_REFERENCE"
    AUTHORIZED_REVEAL = "AUTHORIZED_REVEAL"
    CONTROLLER_UNIT_EVALUATION = "CONTROLLER_UNIT_EVALUATION"
    CONTROLLER_COHORT_ADJUDICATION = "CONTROLLER_COHORT_ADJUDICATION"


PUBLIC_ROUTE_STAGE_ORDER = tuple(PublicRouteConformanceStage)

STANDARD_PUBLIC_ROUTE_OUTPUT_SCHEMAS = {
    PublicRouteConformanceStage.OBSERVATION: ('empirical-lawhood/methods/identification-dataset',),
    PublicRouteConformanceStage.CANDIDATE_FAMILY: ('empirical-lawhood/methods/component-uncertainty-family-assessment',),
    PublicRouteConformanceStage.COMPONENT_QUALIFICATION: (
        'empirical-lawhood/methods/component-qualification-assessment',
    ),
    PublicRouteConformanceStage.LAW_QUALIFICATION: ('empirical-lawhood/kernel/law-qualification-result',),
    PublicRouteConformanceStage.ATLAS: ('empirical-lawhood/kernel/response-atlas',),
    PublicRouteConformanceStage.ADMISSION_RECEIPTS: ('empirical-lawhood/planning/controlled-map-admission-receipt-corpus',),
    PublicRouteConformanceStage.CONTROLLER_PROGRAMME: (
        'empirical-lawhood/planning/admission-controller-study',
    ),
    PublicRouteConformanceStage.CONTROLLER_COMPILE: (
        'empirical-lawhood/runtime/compiled-admission-controller-study',
    ),
    PublicRouteConformanceStage.CONTROLLER_TICK: ('empirical-lawhood/runtime/admission-controller-tick-receipt',),
    PublicRouteConformanceStage.SEALED_REFERENCE: (
        'empirical-lawhood/runtime/sealed-repeated-delivery-controller-bundle',
    ),
    PublicRouteConformanceStage.AUTHORIZED_REVEAL: (
        'empirical-lawhood/runtime/revealed-repeated-delivery-controller-bundle',
    ),
    PublicRouteConformanceStage.CONTROLLER_UNIT_EVALUATION: ('empirical-lawhood/runtime/repeated-delivery-controller-unit-evaluation',),
    PublicRouteConformanceStage.CONTROLLER_COHORT_ADJUDICATION: ('empirical-lawhood/runtime/repeated-delivery-controller-cohort-adjudication',),
}

REQUIRED_PUBLIC_ROUTE_CONFORMANCE_CASE_IDS = (
    "archive-observation-candidate-stop",
    "atlas-admission-controller-continuity",
    "controlled-io-refusal-and-support",
    "finite-action-word-order",
    "member-refinement-shape",
    "nested-controller-independent-unit",
    "partial-support-shape",
    "point-contract-parity",
    "provider-artifact-envelope-recovery",
    "public-execution-envelope-issue-compile",
    "pybamm-pathwise-shape",
)


_STAGE_PROOF_OWNERS = {
    PublicRouteConformanceStage.OBSERVATION: (
        "empirical_lawhood.adapters.methods.evidence_projection",
        'project_response_method_identification_dataset',
    ),
    PublicRouteConformanceStage.CANDIDATE_FAMILY: (
        "empirical_lawhood.adapters.methods.law_assessment",
        "CandidateFamilyAssembler",
    ),
    PublicRouteConformanceStage.COMPONENT_QUALIFICATION: (
        "empirical_lawhood.adapters.methods.law_assessment",
        "QualificationProfileEvaluatorRegistry",
    ),
    PublicRouteConformanceStage.LAW_QUALIFICATION: (
        "empirical_lawhood.adapters.methods.law_assessment",
        "ResponseLawQualificationService",
    ),
    PublicRouteConformanceStage.ATLAS: (
        "empirical_lawhood.adapters.geometry.services",
        "AtlasAssembler",
    ),
    PublicRouteConformanceStage.ADMISSION_RECEIPTS: (
        'empirical_lawhood.adapters.geometry.admission_receipts',
        'AdmissionReceiptCorpusAssembler',
    ),
    PublicRouteConformanceStage.CONTROLLER_PROGRAMME: (
        'empirical_lawhood.planning.controller_study',
        'AdmissionControllerStudy',
    ),
    PublicRouteConformanceStage.CONTROLLER_COMPILE: (
        "empirical_lawhood.runtime.controller_compiler",
        "ControllerCompiler",
    ),
    PublicRouteConformanceStage.CONTROLLER_TICK: (
        "empirical_lawhood.runtime.controller_runtime",
        "ControllerRuntime",
    ),
    PublicRouteConformanceStage.SEALED_REFERENCE: (
        "empirical_lawhood.runtime.controller_evaluation_nested",
        'SealedRepeatedDeliveryControllerBundle',
    ),
    PublicRouteConformanceStage.AUTHORIZED_REVEAL: (
        'empirical_lawhood.adapters.control.study_bridge',
        "ControllerCampaignBridge.reveal_nested",
    ),
    PublicRouteConformanceStage.CONTROLLER_UNIT_EVALUATION: (
        'empirical_lawhood.adapters.control.study_bridge',
        "ControllerCampaignBridge.evaluate_nested_unit",
    ),
    PublicRouteConformanceStage.CONTROLLER_COHORT_ADJUDICATION: (
        'empirical_lawhood.adapters.control.study_bridge',
        "ControllerCampaignBridge.adjudicate_nested_cohort",
    ),
}

_STAGE_CAPABILITY_KINDS = {
    PublicRouteConformanceStage.OBSERVATION: CapabilityKind.OBSERVATION_OPERATOR,
    PublicRouteConformanceStage.CANDIDATE_FAMILY: CapabilityKind.LAW_IDENTIFIER,
    PublicRouteConformanceStage.COMPONENT_QUALIFICATION: CapabilityKind.NUMERICAL_QUALIFIER,
    PublicRouteConformanceStage.LAW_QUALIFICATION: CapabilityKind.LAW_IDENTIFIER,
    PublicRouteConformanceStage.ATLAS: CapabilityKind.ATLAS_ASSEMBLER,
    PublicRouteConformanceStage.ADMISSION_RECEIPTS: CapabilityKind.ADMISSION_EVALUATOR,
    PublicRouteConformanceStage.CONTROLLER_PROGRAMME: CapabilityKind.CONTROLLER_SYNTHESIZER,
    PublicRouteConformanceStage.CONTROLLER_COMPILE: CapabilityKind.CONTROLLER_SYNTHESIZER,
    PublicRouteConformanceStage.CONTROLLER_TICK: CapabilityKind.ONLINE_GATE_EVALUATOR,
    PublicRouteConformanceStage.SEALED_REFERENCE: CapabilityKind.EVALUATOR,
    PublicRouteConformanceStage.AUTHORIZED_REVEAL: CapabilityKind.EVALUATOR,
    PublicRouteConformanceStage.CONTROLLER_UNIT_EVALUATION: CapabilityKind.EVALUATOR,
    PublicRouteConformanceStage.CONTROLLER_COHORT_ADJUDICATION: CapabilityKind.EVALUATOR,
}

_STAGE_EVIDENCE_CEILINGS = {
    PublicRouteConformanceStage.OBSERVATION: EvidenceCeiling.MEASUREMENT,
    PublicRouteConformanceStage.CANDIDATE_FAMILY: EvidenceCeiling.ORDER_RELATION,
    PublicRouteConformanceStage.COMPONENT_QUALIFICATION: EvidenceCeiling.RESPONSE,
    PublicRouteConformanceStage.LAW_QUALIFICATION: EvidenceCeiling.LOCAL_LAW,
    PublicRouteConformanceStage.ATLAS: EvidenceCeiling.LOCAL_LAW,
    PublicRouteConformanceStage.ADMISSION_RECEIPTS: EvidenceCeiling.ADMISSION,
    PublicRouteConformanceStage.CONTROLLER_PROGRAMME: EvidenceCeiling.ADMISSION,
    PublicRouteConformanceStage.CONTROLLER_COMPILE: EvidenceCeiling.ADMISSION,
    PublicRouteConformanceStage.CONTROLLER_TICK: EvidenceCeiling.ADMISSION,
    PublicRouteConformanceStage.SEALED_REFERENCE: EvidenceCeiling.CONTROLLER_USE,
    PublicRouteConformanceStage.AUTHORIZED_REVEAL: EvidenceCeiling.CONTROLLER_USE,
    PublicRouteConformanceStage.CONTROLLER_UNIT_EVALUATION: EvidenceCeiling.CONTROLLER_USE,
    PublicRouteConformanceStage.CONTROLLER_COHORT_ADJUDICATION: EvidenceCeiling.CONTROLLER_USE,
}


@dataclass(frozen=True, slots=True)
class PublicRouteProofOwnerBinding(CanonicalRecord):
    """Exact logical owner resolved inside one clean implementation source closure."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-proof-owner-binding'

    owner_id: str
    stage: PublicRouteConformanceStage
    owner_module: str
    owner_symbol: str
    implementation_source: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.owner_id, field_name="owner_id")
        validate_nonempty(self.owner_module, field_name="owner_module")
        validate_nonempty(self.owner_symbol, field_name="owner_symbol")
        if self.implementation_source.object_schema != ImplementationSourceClosure.SCHEMA:
            raise ValueError("public-route proof owner lacks an exact source closure")


@dataclass(frozen=True, slots=True)
class PublicRouteStageConfiguration(CanonicalRecord):
    """Static stage contract used to fingerprint the public capability config."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-stage-configuration'

    config_id: str
    stage: PublicRouteConformanceStage
    input_schema_ids: tuple[str, ...]
    output_schema_ids: tuple[str, ...]
    external_artifact_plane_required: bool
    maximum_outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        for field_name, values in (
            ("input_schema_ids", self.input_schema_ids),
            ("output_schema_ids", self.output_schema_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name)
            for value in values:
                validate_schema(value)
        if not self.output_schema_ids or not self.external_artifact_plane_required:
            raise ValueError("public-route stage config lacks output/custody semantics")


@dataclass(frozen=True, slots=True)
class PublicRouteSemanticOutputContract(CanonicalRecord):
    """Machine-readable meaning and claim ceiling for one public stage output."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-semantic-output-contract'

    contract_id: str
    stage: PublicRouteConformanceStage
    proof_owner: ObjectIdentity
    capability_manifest: ObjectIdentity
    stage_config: ObjectIdentity
    output_schema_ids: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    maximum_outcome_access: OutcomeAccess
    prohibited_inferences: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        require_sorted_unique_strings(
            self.output_schema_ids,
            field_name="output_schema_ids",
            allow_empty=False,
        )
        for value in self.output_schema_ids:
            validate_schema(value)
        require_sorted_unique_strings(
            self.prohibited_inferences,
            field_name="prohibited_inferences",
            allow_empty=False,
        )
        if self.proof_owner.object_schema != PublicRouteProofOwnerBinding.SCHEMA:
            raise ValueError("public-route semantics bind another proof-owner schema")
        if self.capability_manifest.object_schema != CapabilityManifest.SCHEMA:
            raise ValueError("public-route semantics bind another capability schema")
        if self.stage_config.object_schema != PublicRouteStageConfiguration.SCHEMA:
            raise ValueError("public-route semantics bind another config schema")


@dataclass(frozen=True, slots=True)
class PublicRouteStageBinding(CanonicalRecord):
    """One statically owned graph stage and its complete output schema set."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-stage-binding'

    binding_id: str
    stage: PublicRouteConformanceStage
    predecessor_stages: tuple[PublicRouteConformanceStage, ...]
    proof_owner: ObjectIdentity
    capability_manifest: ObjectIdentity
    output_schema_ids: tuple[str, ...]
    semantic_output_contracts: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        require_sorted_unique_strings(
            tuple(value.value for value in self.predecessor_stages),
            field_name="predecessor_stages",
        )
        require_sorted_unique_strings(
            self.output_schema_ids,
            field_name="output_schema_ids",
            allow_empty=False,
        )
        for value in self.output_schema_ids:
            validate_schema(value)
        require_sorted_unique_ids(
            self.semantic_output_contracts,
            attribute="object_id",
            field_name="semantic_output_contracts",
        )
        if not self.semantic_output_contracts:
            raise ValueError("public-route stage lacks a semantic output contract")


@dataclass(frozen=True, slots=True)
class RunEnvelopeConformanceGraph(CanonicalRecord):
    """Static exact-owner graph; it is a conformance topology, not experiment evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-envelope-conformance-graph'

    graph_id: str
    stages: tuple[PublicRouteStageBinding, ...]
    issued_package_schema: str
    run_plan_schema: str
    execution_plan_schema: str
    execution_envelope_schema: str
    recovery_index_schema: str
    external_artifact_plane_required: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.graph_id, field_name="graph_id")
        require_sorted_unique_ids(
            self.stages,
            attribute="binding_id",
            field_name="stages",
        )
        for value in (
            self.issued_package_schema,
            self.run_plan_schema,
            self.execution_plan_schema,
            self.execution_envelope_schema,
            self.recovery_index_schema,
        ):
            validate_schema(value)
        by_stage = {value.stage: value for value in self.stages}
        if set(by_stage) != set(PUBLIC_ROUTE_STAGE_ORDER):
            raise ValueError("public-route conformance stage roster is incomplete")
        ordinal = {value: index for index, value in enumerate(PUBLIC_ROUTE_STAGE_ORDER)}
        for stage, binding in by_stage.items():
            if any(ordinal[parent] >= ordinal[stage] for parent in binding.predecessor_stages):
                raise ValueError("public-route predecessor is not an earlier stage")
            expected = (
                () if ordinal[stage] == 0 else (PUBLIC_ROUTE_STAGE_ORDER[ordinal[stage] - 1],)
            )
            if binding.predecessor_stages != expected:
                raise ValueError("public-route graph is not the exact public stage chain")
        if (
            self.issued_package_schema != 'empirical-lawhood/api/envelope-experiment-package'
            or self.run_plan_schema != 'empirical-lawhood/runtime/envelope-run-plan'
            or self.execution_plan_schema != 'empirical-lawhood/runtime/envelope-execution-plan'
            or self.execution_envelope_schema != 'empirical-lawhood/runtime/run-execution-envelope'
            or self.recovery_index_schema != 'empirical-lawhood/runtime/envelope-run-recovery-index'
        ):
            raise ValueError("public-route conformance uses another campaign topology")
        if (
            not self.external_artifact_plane_required
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("public-route graph overclaims its conformance boundary")

    def binding(self, stage: PublicRouteConformanceStage) -> PublicRouteStageBinding:
        return next(value for value in self.stages if value.stage is stage)


@dataclass(frozen=True, slots=True)
class PublicRouteStageOutput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-stage-output'

    output_id: str
    stage: PublicRouteConformanceStage
    record: ObjectIdentity
    parents: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.output_id, field_name="output_id")
        require_sorted_unique_ids(
            self.parents,
            attribute="object_id",
            field_name="parents",
        )


@dataclass(frozen=True, slots=True)
class RunEnvelopeOperationalConformanceReceipt(CanonicalRecord):
    """Executed provider/artifact/envelope leg joined to the scientific chain."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-envelope-operational-conformance-receipt'

    receipt_id: str
    provider_registry_sha256: str
    execution_plan: ObjectIdentity
    execution_envelope: ObjectIdentity
    recovery_index: ObjectIdentity
    task_receipts: tuple[ObjectIdentity, ...]
    published_artifacts: tuple[ObjectIdentity, ...]
    interruption_case_ids: tuple[str, ...]
    source_envelope_bytes_sha256: str
    recovered_envelope_bytes_sha256: str
    scientific_recomputation_count: int
    scientific_retuning_count: int
    terminal: bool
    created_empirical_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_sha256(
            self.provider_registry_sha256,
            field_name="provider_registry_sha256",
        )
        if (
            self.execution_plan.object_schema != 'empirical-lawhood/runtime/envelope-execution-plan'
            or self.execution_envelope.object_schema != 'empirical-lawhood/runtime/run-execution-envelope'
            or self.recovery_index.object_schema != 'empirical-lawhood/runtime/envelope-run-recovery-index'
        ):
            raise ValueError("operational conformance uses another campaign topology")
        require_sorted_unique_ids(
            self.task_receipts,
            attribute="object_id",
            field_name="task_receipts",
        )
        require_sorted_unique_ids(
            self.published_artifacts,
            attribute="object_id",
            field_name="published_artifacts",
        )
        require_sorted_unique_strings(
            self.interruption_case_ids,
            field_name="interruption_case_ids",
            allow_empty=False,
        )
        if not self.task_receipts or not self.published_artifacts:
            raise ValueError("operational conformance lacks receipt-first publication")
        validate_sha256(
            self.source_envelope_bytes_sha256,
            field_name="source_envelope_bytes_sha256",
        )
        validate_sha256(
            self.recovered_envelope_bytes_sha256,
            field_name="recovered_envelope_bytes_sha256",
        )
        if self.source_envelope_bytes_sha256 != self.recovered_envelope_bytes_sha256:
            raise ValueError("operational recovery changed envelope bytes")
        if self.scientific_recomputation_count or self.scientific_retuning_count:
            raise ValueError("operational recovery recomputed or retuned science")
        if not self.terminal or self.created_empirical_evidence:
            raise ValueError("operational conformance overclaims its terminal boundary")


@dataclass(frozen=True, slots=True)
class RunEnvelopeRouteConformanceReceipt(CanonicalRecord):
    """Exact integration receipt over records independently produced by sole owners."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-envelope-route-conformance-receipt'

    receipt_id: str
    graph: RunEnvelopeConformanceGraph
    stage_outputs: tuple[PublicRouteStageOutput, ...]
    issued_package: ObjectIdentity
    run_plan: ObjectIdentity
    execution_plan: ObjectIdentity
    execution_envelope: ObjectIdentity
    recovery_index: ObjectIdentity
    operational_conformance: RunEnvelopeOperationalConformanceReceipt
    public_composition: ObjectIdentity
    external_artifact_plane_id: str
    terminal: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.external_artifact_plane_id,
            field_name="external_artifact_plane_id",
        )
        require_sorted_unique_ids(
            self.stage_outputs,
            attribute="output_id",
            field_name="stage_outputs",
        )
        outputs = {value.stage: value for value in self.stage_outputs}
        if len(outputs) != len(self.stage_outputs) or set(outputs) != set(PUBLIC_ROUTE_STAGE_ORDER):
            raise ValueError("public-route output roster is not exact")
        for stage, output in outputs.items():
            if output.record.object_schema not in self.graph.binding(stage).output_schema_ids:
                raise ValueError("public-route output schema differs from its static contract")
            ordinal = PUBLIC_ROUTE_STAGE_ORDER.index(stage)
            expected_parents = (
                () if ordinal == 0 else (outputs[PUBLIC_ROUTE_STAGE_ORDER[ordinal - 1]].record,)
            )
            if output.parents != expected_parents:
                raise ValueError("public-route output lineage differs from its exact chain")
        expected_schemas = (
            (self.issued_package, self.graph.issued_package_schema),
            (self.run_plan, self.graph.run_plan_schema),
            (self.execution_plan, self.graph.execution_plan_schema),
            (self.execution_envelope, self.graph.execution_envelope_schema),
            (self.recovery_index, self.graph.recovery_index_schema),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("public-route campaign identity differs from its graph")
        if self.public_composition.object_schema != RunEnvelopeRouteComposition.SCHEMA:
            raise ValueError("public-route receipt binds another public composition")
        if self.operational_conformance.provider_registry_sha256 == "0" * 64:
            raise ValueError("public-route operational provider is not bound")
        if (
            not self.terminal
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("public-route receipt is not terminal architecture conformance")


@dataclass(frozen=True, slots=True)
class RunEnvelopeConformanceCaseReceipt(CanonicalRecord):
    """One exact truth-known case; labels alone never count as conformance."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-envelope-conformance-case-receipt'

    case_id: str
    covered_stages: tuple[PublicRouteConformanceStage, ...]
    evidence_records: tuple[ObjectIdentity, ...]
    terminal_disposition: str
    terminal: bool
    created_empirical_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        require_sorted_unique_strings(
            tuple(value.value for value in self.covered_stages),
            field_name="covered_stages",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.evidence_records,
            attribute="object_id",
            field_name="evidence_records",
        )
        validate_nonempty(self.terminal_disposition, field_name="terminal_disposition")
        if not self.evidence_records or not self.terminal or self.created_empirical_evidence:
            raise ValueError("public-route case lacks terminal truth-known evidence")


@dataclass(frozen=True, slots=True)
class RunEnvelopeConformanceSuiteReceipt(CanonicalRecord):
    """Exact case roster joined to the reproducible public and operational receipt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-envelope-conformance-suite-receipt'

    suite_id: str
    primary_receipt: ObjectIdentity
    public_composition: ObjectIdentity
    operational_conformance: ObjectIdentity
    provider_registry_sha256: str
    cases: tuple[RunEnvelopeConformanceCaseReceipt, ...]
    recovery_injection_case_ids: tuple[str, ...]
    terminal: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.suite_id, field_name="suite_id")
        if self.primary_receipt.object_schema != RunEnvelopeRouteConformanceReceipt.SCHEMA:
            raise ValueError("public-route suite binds another primary receipt")
        if self.public_composition.object_schema != RunEnvelopeRouteComposition.SCHEMA:
            raise ValueError("public-route suite binds another public composition")
        if (
            self.operational_conformance.object_schema
            != RunEnvelopeOperationalConformanceReceipt.SCHEMA
        ):
            raise ValueError("public-route suite binds another operational receipt")
        validate_sha256(self.provider_registry_sha256, field_name="provider_registry_sha256")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if tuple(value.case_id for value in self.cases) != (
            REQUIRED_PUBLIC_ROUTE_CONFORMANCE_CASE_IDS
        ):
            raise ValueError("public-route conformance case roster is incomplete")
        require_sorted_unique_strings(
            self.recovery_injection_case_ids,
            field_name="recovery_injection_case_ids",
            allow_empty=False,
        )
        if (
            not self.terminal
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("public-route suite overclaims its conformance boundary")


@dataclass(frozen=True, slots=True)
class RunEnvelopeRouteComposition(CanonicalRecord):
    """Complete content-addressed public route and operational provider binding."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-envelope-route-composition'

    composition_id: str
    graph: RunEnvelopeConformanceGraph
    implementation_source: ImplementationSourceClosure
    proof_owners: tuple[PublicRouteProofOwnerBinding, ...]
    stage_configs: tuple[PublicRouteStageConfiguration, ...]
    capabilities: tuple[CapabilityManifest, ...]
    semantic_output_contracts: tuple[PublicRouteSemanticOutputContract, ...]
    provider_registry_sha256: str
    external_artifact_plane_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.composition_id, field_name="composition_id")
        validate_sha256(
            self.provider_registry_sha256,
            field_name="provider_registry_sha256",
        )
        if self.provider_registry_sha256 == "0" * 64:
            raise ValueError("public composition lacks its operational provider registry")
        require_sorted_unique_ids(
            self.proof_owners,
            attribute="owner_id",
            field_name="proof_owners",
        )
        require_sorted_unique_ids(
            self.stage_configs,
            attribute="config_id",
            field_name="stage_configs",
        )
        require_sorted_unique_ids(
            self.capabilities,
            attribute="registry_id",
            field_name="capabilities",
        )
        require_sorted_unique_ids(
            self.semantic_output_contracts,
            attribute="contract_id",
            field_name="semantic_output_contracts",
        )
        if not self.external_artifact_plane_required:
            raise ValueError("public composition must retain the external artifact plane")
        by_stage = {value.stage: value for value in self.graph.stages}
        owners = {value.stage: value for value in self.proof_owners}
        configs = {value.stage: value for value in self.stage_configs}
        semantics = {value.stage: value for value in self.semantic_output_contracts}
        capabilities = {
            PublicRouteConformanceStage(
                value.capability_key.removeprefix("response-law.").upper()
            ): value
            for value in self.capabilities
        }
        expected_stages = set(PUBLIC_ROUTE_STAGE_ORDER)
        if any(
            set(values) != expected_stages
            for values in (by_stage, owners, configs, capabilities, semantics)
        ):
            raise ValueError("public composition stage roster is incomplete")
        source_identity = ObjectIdentity.from_record(
            self.implementation_source.source_closure_id,
            self.implementation_source,
        )
        for stage in PUBLIC_ROUTE_STAGE_ORDER:
            binding = by_stage[stage]
            owner = owners[stage]
            config = configs[stage]
            capability = capabilities[stage]
            semantic = semantics[stage]
            owner_identity = ObjectIdentity.from_record(owner.owner_id, owner)
            capability_identity = ObjectIdentity.from_record(
                capability.capability_key,
                capability,
            )
            config_identity = ObjectIdentity.from_record(config.config_id, config)
            semantic_identity = ObjectIdentity.from_record(semantic.contract_id, semantic)
            if owner.implementation_source != source_identity:
                raise ValueError("public proof owner differs from the composition source")
            if capability.implementation_sha256 != self.implementation_source.source_tree_sha256:
                raise ValueError("public capability differs from the clean source tree")
            if (
                binding.proof_owner != owner_identity
                or binding.capability_manifest != capability_identity
                or binding.semantic_output_contracts != (semantic_identity,)
                or binding.output_schema_ids != config.output_schema_ids
                or semantic.proof_owner != owner_identity
                or semantic.capability_manifest != capability_identity
                or semantic.stage_config != config_identity
                or semantic.output_schema_ids != config.output_schema_ids
                or semantic.maximum_evidence_ceiling is not capability.maximum_evidence_ceiling
                or semantic.maximum_outcome_access is not capability.maximum_outcome_access
            ):
                raise ValueError("public composition graph/capability semantics drifted")


def _stage_outcome_access(stage: PublicRouteConformanceStage) -> OutcomeAccess:
    if stage in {
        PublicRouteConformanceStage.OBSERVATION,
        PublicRouteConformanceStage.CANDIDATE_FAMILY,
        PublicRouteConformanceStage.COMPONENT_QUALIFICATION,
        PublicRouteConformanceStage.LAW_QUALIFICATION,
        PublicRouteConformanceStage.AUTHORIZED_REVEAL,
        PublicRouteConformanceStage.CONTROLLER_UNIT_EVALUATION,
        PublicRouteConformanceStage.CONTROLLER_COHORT_ADJUDICATION,
    }:
        return OutcomeAccess.EVALUATOR_REVEAL
    if stage is PublicRouteConformanceStage.SEALED_REFERENCE:
        return OutcomeAccess.EVALUATION_SEALED
    return OutcomeAccess.OUTCOME_BLIND


def _stage_permissions(
    stage: PublicRouteConformanceStage,
) -> tuple[CapabilityPermission, ...]:
    values = {
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    }
    if stage is PublicRouteConformanceStage.AUTHORIZED_REVEAL:
        values.update(
            {
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            }
        )
    elif stage in {
        PublicRouteConformanceStage.CONTROLLER_UNIT_EVALUATION,
        PublicRouteConformanceStage.CONTROLLER_COHORT_ADJUDICATION,
    }:
        values.add(CapabilityPermission.READ_SEALED_OUTCOMES)
    return tuple(sorted(values, key=str))


def standard_public_route_composition(
    *,
    implementation_source: ImplementationSourceClosure,
    provider_registry_sha256: str,
) -> RunEnvelopeRouteComposition:
    """Build the exact public stage composition from a clean source closure."""

    if not implementation_source.clean_worktree:
        raise ValueError("public-route composition requires a clean source closure")
    source_identity = ObjectIdentity.from_record(
        implementation_source.source_closure_id,
        implementation_source,
    )
    owners: list[PublicRouteProofOwnerBinding] = []
    configs: list[PublicRouteStageConfiguration] = []
    capabilities: list[CapabilityManifest] = []
    semantics: list[PublicRouteSemanticOutputContract] = []
    bindings: list[PublicRouteStageBinding] = []
    for index, stage in enumerate(PUBLIC_ROUTE_STAGE_ORDER, start=1):
        slug = stage.value.lower()
        input_schemas = (
            ()
            if index == 1
            else STANDARD_PUBLIC_ROUTE_OUTPUT_SCHEMAS[PUBLIC_ROUTE_STAGE_ORDER[index - 2]]
        )
        output_schemas = STANDARD_PUBLIC_ROUTE_OUTPUT_SCHEMAS[stage]
        owner_module, owner_symbol = _STAGE_PROOF_OWNERS[stage]
        owner = PublicRouteProofOwnerBinding(
            owner_id=f"proof-owner.{slug}",
            stage=stage,
            owner_module=owner_module,
            owner_symbol=owner_symbol,
            implementation_source=source_identity,
        )
        config = PublicRouteStageConfiguration(
            config_id=f"public-route-config.{slug}",
            stage=stage,
            input_schema_ids=tuple(sorted(input_schemas)),
            output_schema_ids=output_schemas,
            external_artifact_plane_required=True,
            maximum_outcome_access=_stage_outcome_access(stage),
        )
        capability = CapabilityManifest(
            capability_key=f"response-law.{slug}",
            capability_version="1.0.0",
            kind=_STAGE_CAPABILITY_KINDS[stage],
            config_schema=PublicRouteStageConfiguration.SCHEMA,
            config_schema_sha256=hashlib.sha256(
                PublicRouteStageConfiguration.SCHEMA.encode("utf-8")
            ).hexdigest(),
            input_schema_ids=tuple(sorted(input_schemas)),
            output_schema_ids=output_schemas,
            permissions=_stage_permissions(stage),
            maximum_evidence_ceiling=_STAGE_EVIDENCE_CEILINGS[stage],
            maximum_outcome_access=_stage_outcome_access(stage),
            resource_ceiling=ResourceBudget(
                cpu_cores=4,
                memory_bytes=8 * 1024**3,
                gpu_devices=0,
                wall_time_seconds=3_600,
                source_scan_bytes=100 * 1024**3,
                output_bytes=1024**3,
            ),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11-response-law-route",
            requires_clean_commit=True,
            requires_active_mount=False,
            requires_network=False,
            conformance_check_ids=(f"response-law-public-route-{slug}",),
            implementation_sha256=implementation_source.source_tree_sha256,
        )
        owner_identity = ObjectIdentity.from_record(owner.owner_id, owner)
        capability_identity = ObjectIdentity.from_record(
            capability.capability_key,
            capability,
        )
        config_identity = ObjectIdentity.from_record(config.config_id, config)
        semantic = PublicRouteSemanticOutputContract(
            contract_id=f"semantic-contract.{slug}",
            stage=stage,
            proof_owner=owner_identity,
            capability_manifest=capability_identity,
            stage_config=config_identity,
            output_schema_ids=output_schemas,
            maximum_evidence_ceiling=capability.maximum_evidence_ceiling,
            maximum_outcome_access=capability.maximum_outcome_access,
            prohibited_inferences=(
                "architecture-conformance-is-not-empirical-evidence",
                "stage-output-cannot-author-a-later-rung",
            ),
        )
        bindings.append(
            PublicRouteStageBinding(
                binding_id=f"stage-binding.{index:02d}-{slug}",
                stage=stage,
                predecessor_stages=(() if index == 1 else (PUBLIC_ROUTE_STAGE_ORDER[index - 2],)),
                proof_owner=owner_identity,
                capability_manifest=capability_identity,
                output_schema_ids=output_schemas,
                semantic_output_contracts=(
                    ObjectIdentity.from_record(semantic.contract_id, semantic),
                ),
            )
        )
        owners.append(owner)
        configs.append(config)
        capabilities.append(capability)
        semantics.append(semantic)
    graph = RunEnvelopeConformanceGraph(
        graph_id="graph.public-execution-envelope-route-conformance",
        stages=tuple(bindings),
        issued_package_schema='empirical-lawhood/api/envelope-experiment-package',
        run_plan_schema='empirical-lawhood/runtime/envelope-run-plan',
        execution_plan_schema='empirical-lawhood/runtime/envelope-execution-plan',
        execution_envelope_schema='empirical-lawhood/runtime/run-execution-envelope',
        recovery_index_schema='empirical-lawhood/runtime/envelope-run-recovery-index',
        external_artifact_plane_required=True,
        created_empirical_evidence=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return RunEnvelopeRouteComposition(
        composition_id="composition.public-execution-envelope-route",
        graph=graph,
        implementation_source=implementation_source,
        proof_owners=tuple(sorted(owners, key=lambda value: value.owner_id)),
        stage_configs=tuple(sorted(configs, key=lambda value: value.config_id)),
        capabilities=tuple(sorted(capabilities, key=lambda value: value.registry_id)),
        semantic_output_contracts=tuple(sorted(semantics, key=lambda value: value.contract_id)),
        provider_registry_sha256=provider_registry_sha256,
        external_artifact_plane_required=True,
    )


def standard_public_route_conformance_graph(
    *,
    implementation_source: ImplementationSourceClosure,
    provider_registry_sha256: str,
) -> RunEnvelopeConformanceGraph:
    """Return the graph from the exact source/provider-bound public composition."""

    return standard_public_route_composition(
        implementation_source=implementation_source,
        provider_registry_sha256=provider_registry_sha256,
    ).graph


__all__ = [
    "PUBLIC_ROUTE_STAGE_ORDER",
    "REQUIRED_PUBLIC_ROUTE_CONFORMANCE_CASE_IDS",
    "STANDARD_PUBLIC_ROUTE_OUTPUT_SCHEMAS",
    'RunEnvelopeConformanceGraph',
    'RunEnvelopeRouteComposition',
    'RunEnvelopeConformanceCaseReceipt',
    'RunEnvelopeOperationalConformanceReceipt',
    "PublicRouteProofOwnerBinding",
    'RunEnvelopeRouteConformanceReceipt',
    'RunEnvelopeConformanceSuiteReceipt',
    "PublicRouteConformanceStage",
    "PublicRouteSemanticOutputContract",
    "PublicRouteStageConfiguration",
    "PublicRouteStageBinding",
    "PublicRouteStageOutput",
    "standard_public_route_conformance_graph",
    "standard_public_route_composition",
]
