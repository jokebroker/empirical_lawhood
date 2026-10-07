"""Public route conformance records for resource envelopes.

This module and the execution envelope route retain the same thirteen
scientific stage contracts. This module binds resource and JIT identities,
scientific graph parity, and recovery identities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
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
from empirical_lawhood.planning.identification_evidence import IdentificationEvidenceManifest

from .capabilities import CapabilityManifest, CapabilityPermission
from .execution_envelope import validate_optional_jit_identities, JitGraphSignatureManifest, PredevelopmentJitSignatureCensus
from .adjudication import ScientificAdjudicationRecord
from .route_conformance import PUBLIC_ROUTE_STAGE_ORDER, STANDARD_PUBLIC_ROUTE_OUTPUT_SCHEMAS, PublicRouteConformanceStage, PublicRouteProofOwnerBinding, PublicRouteSemanticOutputContract, PublicRouteStageBinding, PublicRouteStageConfiguration, PublicRouteStageOutput, standard_public_route_composition


# Runtime cannot import the API layer.  The literal is checked against the API
# record in focused tests and is part of this versioned public-route contract.
STANDARD_ISSUED_EXPERIMENT_PACKAGE_SCHEMA = 'empirical-lawhood/api/experiment-package'
RESOURCE_ROUTE_IDENTIFICATION_INPUT_SCHEMA = IdentificationEvidenceManifest.SCHEMA

REQUIRED_RESOURCE_ENVELOPE_CONFORMANCE_CASE_IDS = (
    "archive-observation-candidate-stop",
    "atlas-admission-controller-continuity",
    "controlled-io-refusal-and-support",
    "finite-action-word-order",
    "member-refinement-shape",
    "nested-controller-independent-unit",
    "partial-support-shape",
    "point-contract-parity",
    "provider-artifact-envelope-recovery",
    "public-authority-stop-before-reveal",
    "public-resource-envelope-issue-compile-jit",
    "public-supported-stage-chain",
    "public-zero-law-condition-false",
    "pybamm-pathwise-shape",
)


class PublicRouteCaseSemantics(StrEnum):
    """Closed interpretation of one resource envelope conformance case."""

    MATRIX_EVIDENCE = "MATRIX_EVIDENCE"
    SUPPORTED = "SUPPORTED"
    ZERO_LAW = "ZERO_LAW"
    AUTHORITY_STOP = "AUTHORITY_STOP"
    RECOVERY = "RECOVERY"


class PublicRouteConformanceScope(StrEnum):
    """Closed claim boundary for a repository-only route receipt."""

    ARCHITECTURE_TRUTH_KNOWN = "ARCHITECTURE_TRUTH_KNOWN"


class PublicRouteInvocationSurface(StrEnum):
    """How the independently owned scientific stages were invoked."""

    INDEPENDENT_OWNER_RECEIPT_JOIN = "INDEPENDENT_OWNER_RECEIPT_JOIN"
    EMPIRICAL_LAWHOOD_API = "EMPIRICAL_LAWHOOD_API"


REQUIRED_RESOURCE_ENVELOPE_SEMANTIC_CASES = (
    ("provider-artifact-envelope-recovery", PublicRouteCaseSemantics.RECOVERY),
    ("public-authority-stop-before-reveal", PublicRouteCaseSemantics.AUTHORITY_STOP),
    ("public-supported-stage-chain", PublicRouteCaseSemantics.SUPPORTED),
    ("public-zero-law-condition-false", PublicRouteCaseSemantics.ZERO_LAW),
)

RESOURCE_ENVELOPE_CASE_SEMANTICS_BY_ID = tuple(
    (
        case_id,
        dict(REQUIRED_RESOURCE_ENVELOPE_SEMANTIC_CASES).get(
            case_id,
            PublicRouteCaseSemantics.MATRIX_EVIDENCE,
        ),
    )
    for case_id in REQUIRED_RESOURCE_ENVELOPE_CONFORMANCE_CASE_IDS
)


def _sorted_stages(
    values: tuple[PublicRouteConformanceStage, ...],
) -> tuple[PublicRouteConformanceStage, ...]:
    return tuple(sorted(values, key=lambda value: value.value))


_ALL_STAGES = _sorted_stages(PUBLIC_ROUTE_STAGE_ORDER)
_ZERO_LAW_COVERED_STAGES = _sorted_stages(PUBLIC_ROUTE_STAGE_ORDER[:5])
_ZERO_LAW_CONDITION_FALSE_STAGES = _sorted_stages(PUBLIC_ROUTE_STAGE_ORDER[5:])
_AUTHORITY_STOP_COVERED_STAGES = _sorted_stages(PUBLIC_ROUTE_STAGE_ORDER[:10])
_AUTHORITY_STOP_CONDITION_FALSE_STAGES = _sorted_stages(PUBLIC_ROUTE_STAGE_ORDER[10:])

_ZERO_LAW_REQUIRED_EVIDENCE_SCHEMAS = {
    'empirical-lawhood/methods/identification-dataset',
    'empirical-lawhood/methods/component-uncertainty-family-assessment',
    'empirical-lawhood/methods/component-qualification-assessment',
    'empirical-lawhood/kernel/law-qualification-result',
    'empirical-lawhood/methods/law-qualification-batch',
    'empirical-lawhood/planning/atlas-assembly-obstruction',
}
_AUTHORITY_STOP_REQUIRED_EVIDENCE_SCHEMAS = {
    'empirical-lawhood/runtime/sealed-repeated-delivery-controller-bundle',
}


@dataclass(frozen=True, slots=True)
class PublicRouteConformanceGraph(CanonicalRecord):
    """Exact current public topology; conformance is not empirical evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-conformance-graph'

    graph_id: str
    stages: tuple[PublicRouteStageBinding, ...]
    issued_package_schema: str
    run_plan_schema: str
    execution_plan_schema: str
    execution_resource_envelope_spec_schema: str
    predevelopment_jit_signature_census_schema: str
    jit_graph_signature_manifest_schema: str
    recovery_index_schema: str
    scientific_graph_parity_receipt_schema: str
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
        schema_values = (
            self.issued_package_schema,
            self.run_plan_schema,
            self.execution_plan_schema,
            self.execution_resource_envelope_spec_schema,
            self.predevelopment_jit_signature_census_schema,
            self.jit_graph_signature_manifest_schema,
            self.recovery_index_schema,
            self.scientific_graph_parity_receipt_schema,
        )
        for value in schema_values:
            validate_schema(value)
        expected_schemas = (
            STANDARD_ISSUED_EXPERIMENT_PACKAGE_SCHEMA,
            'empirical-lawhood/runtime/run-plan',
            'empirical-lawhood/runtime/execution-plan',
            'empirical-lawhood/runtime/execution-resource-envelope-spec',
            PredevelopmentJitSignatureCensus.SCHEMA,
            JitGraphSignatureManifest.SCHEMA,
            'empirical-lawhood/runtime/run-recovery-index',
            'empirical-lawhood/runtime/resource-graph-preservation-receipt',
        )
        if schema_values != expected_schemas:
            raise ValueError("resource envelope public-route uses another campaign topology")
        by_stage = {value.stage: value for value in self.stages}
        if len(by_stage) != len(self.stages) or set(by_stage) != set(PUBLIC_ROUTE_STAGE_ORDER):
            raise ValueError("resource envelope public-route conformance stage roster is incomplete")
        ordinal = {value: index for index, value in enumerate(PUBLIC_ROUTE_STAGE_ORDER)}
        for stage, binding in by_stage.items():
            expected_predecessors = (
                () if ordinal[stage] == 0 else (PUBLIC_ROUTE_STAGE_ORDER[ordinal[stage] - 1],)
            )
            if binding.predecessor_stages != expected_predecessors:
                raise ValueError("resource envelope public-route graph is not the exact public stage chain")
            if binding.output_schema_ids != STANDARD_PUBLIC_ROUTE_OUTPUT_SCHEMAS[stage]:
                raise ValueError("resource envelope public-route stage output contract drifted")
        if (
            not self.external_artifact_plane_required
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("resource envelope public-route graph overclaims its conformance boundary")

    def binding(self, stage: PublicRouteConformanceStage) -> PublicRouteStageBinding:
        return next(value for value in self.stages if value.stage is stage)


@dataclass(frozen=True, slots=True)
class PublicRouteComposition(CanonicalRecord):
    """Source/provider-bound current public composition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-composition'

    composition_id: str
    graph: PublicRouteConformanceGraph
    implementation_source: ImplementationSourceClosure
    proof_owners: tuple[PublicRouteProofOwnerBinding, ...]
    stage_configs: tuple[PublicRouteStageConfiguration, ...]
    capabilities: tuple[CapabilityManifest, ...]
    semantic_output_contracts: tuple[PublicRouteSemanticOutputContract, ...]
    provider_registry_sha256: str
    external_artifact_plane_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.composition_id, field_name="composition_id")
        validate_sha256(self.provider_registry_sha256, field_name="provider_registry_sha256")
        if self.provider_registry_sha256 == "0" * 64:
            raise ValueError("resource envelope public composition lacks its operational provider registry")
        if not self.implementation_source.clean_worktree:
            raise ValueError("resource envelope public composition requires a clean source closure")
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
            raise ValueError("resource envelope public composition must retain the external artifact plane")

        bindings = {value.stage: value for value in self.graph.stages}
        owners = {value.stage: value for value in self.proof_owners}
        configs = {value.stage: value for value in self.stage_configs}
        semantics = {value.stage: value for value in self.semantic_output_contracts}
        try:
            capabilities = {
                PublicRouteConformanceStage(
                    value.capability_key.removeprefix("response-law.").upper()
                ): value
                for value in self.capabilities
            }
        except ValueError as error:
            raise ValueError("resource envelope public composition has an unknown stage capability") from error
        expected_stages = set(PUBLIC_ROUTE_STAGE_ORDER)
        if any(
            len(values) != len(PUBLIC_ROUTE_STAGE_ORDER) or set(values) != expected_stages
            for values in (bindings, owners, configs, capabilities, semantics)
        ):
            raise ValueError("resource envelope public composition stage roster is incomplete")

        source_identity = ObjectIdentity.from_record(
            self.implementation_source.source_closure_id,
            self.implementation_source,
        )
        for stage in PUBLIC_ROUTE_STAGE_ORDER:
            binding = bindings[stage]
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
                raise ValueError("resource envelope public proof owner differs from its source closure")
            if capability.implementation_sha256 != self.implementation_source.source_tree_sha256:
                raise ValueError("resource envelope public capability differs from the clean source tree")
            if (
                binding.proof_owner != owner_identity
                or binding.capability_manifest != capability_identity
                or binding.semantic_output_contracts != (semantic_identity,)
                or binding.output_schema_ids != config.output_schema_ids
                or config.input_schema_ids != capability.input_schema_ids
                or semantic.proof_owner != owner_identity
                or semantic.capability_manifest != capability_identity
                or semantic.stage_config != config_identity
                or semantic.output_schema_ids != config.output_schema_ids
                or semantic.maximum_evidence_ceiling is not capability.maximum_evidence_ceiling
                or semantic.maximum_outcome_access is not capability.maximum_outcome_access
            ):
                raise ValueError("resource envelope public composition graph/capability semantics drifted")


@dataclass(frozen=True, slots=True)
class PublicRouteOperationalConformanceReceipt(CanonicalRecord):
    """Provider execution and publication receipt with exact recovery replay."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-operational-conformance-receipt'

    receipt_id: str
    provider_registry_sha256: str
    issued_package: ObjectIdentity
    run_plan: ObjectIdentity
    execution_plan: ObjectIdentity
    execution_resource_envelope_spec: ObjectIdentity
    predevelopment_jit_signature_census: ObjectIdentity | None
    jit_graph_signature_manifest: ObjectIdentity | None
    recovery_index: ObjectIdentity
    scientific_graph_parity_receipt: ObjectIdentity
    task_receipts: tuple[ObjectIdentity, ...]
    published_artifacts: tuple[ObjectIdentity, ...]
    recovery_injection_case_ids: tuple[str, ...]
    source_recovery_state_bytes_sha256: str
    recovered_recovery_state_bytes_sha256: str
    scientific_recomputation_count: int
    scientific_retuning_count: int
    terminal: bool
    created_empirical_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_sha256(self.provider_registry_sha256, field_name="provider_registry_sha256")
        if self.provider_registry_sha256 == "0" * 64:
            raise ValueError("resource envelope operational conformance lacks its provider registry")
        validate_optional_jit_identities(
            self.predevelopment_jit_signature_census, self.jit_graph_signature_manifest
        )
        observed_schemas = (
            self.issued_package.object_schema,
            self.run_plan.object_schema,
            self.execution_plan.object_schema,
            self.execution_resource_envelope_spec.object_schema,
            PredevelopmentJitSignatureCensus.SCHEMA,
            JitGraphSignatureManifest.SCHEMA,
            self.recovery_index.object_schema,
            self.scientific_graph_parity_receipt.object_schema,
        )
        expected_schemas = (
            STANDARD_ISSUED_EXPERIMENT_PACKAGE_SCHEMA,
            'empirical-lawhood/runtime/run-plan',
            'empirical-lawhood/runtime/execution-plan',
            'empirical-lawhood/runtime/execution-resource-envelope-spec',
            PredevelopmentJitSignatureCensus.SCHEMA,
            JitGraphSignatureManifest.SCHEMA,
            'empirical-lawhood/runtime/run-recovery-index',
            'empirical-lawhood/runtime/resource-graph-preservation-receipt',
        )
        if observed_schemas != expected_schemas:
            raise ValueError("resource envelope operational conformance uses another campaign topology")
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
            self.recovery_injection_case_ids,
            field_name="recovery_injection_case_ids",
            allow_empty=False,
        )
        if not self.task_receipts or not self.published_artifacts:
            raise ValueError("resource envelope operational conformance lacks receipt-first publication")
        validate_sha256(
            self.source_recovery_state_bytes_sha256,
            field_name="source_recovery_state_bytes_sha256",
        )
        validate_sha256(
            self.recovered_recovery_state_bytes_sha256,
            field_name="recovered_recovery_state_bytes_sha256",
        )
        if self.source_recovery_state_bytes_sha256 != self.recovered_recovery_state_bytes_sha256:
            raise ValueError("resource envelope operational recovery changed recovery-state bytes")
        if self.scientific_recomputation_count or self.scientific_retuning_count:
            raise ValueError("resource envelope operational recovery recomputed or retuned science")
        if not self.terminal or self.created_empirical_evidence:
            raise ValueError("resource envelope operational conformance overclaims its terminal boundary")


@dataclass(frozen=True, slots=True)
class PublicRouteConformanceReceipt(CanonicalRecord):
    """Current exact integration receipt over independently produced records."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-conformance-receipt'

    receipt_id: str
    graph: PublicRouteConformanceGraph
    stage_outputs: tuple[PublicRouteStageOutput, ...]
    issued_package: ObjectIdentity
    run_plan: ObjectIdentity
    execution_plan: ObjectIdentity
    execution_resource_envelope_spec: ObjectIdentity
    predevelopment_jit_signature_census: ObjectIdentity | None
    jit_graph_signature_manifest: ObjectIdentity | None
    recovery_index: ObjectIdentity
    scientific_graph_parity_receipt: ObjectIdentity
    operational_conformance: PublicRouteOperationalConformanceReceipt
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
            raise ValueError("resource envelope public-route output roster is not exact")
        for stage, output in outputs.items():
            if output.record.object_schema not in self.graph.binding(stage).output_schema_ids:
                raise ValueError("resource envelope public-route output schema differs from its contract")
            ordinal = PUBLIC_ROUTE_STAGE_ORDER.index(stage)
            expected_parents = (
                () if ordinal == 0 else (outputs[PUBLIC_ROUTE_STAGE_ORDER[ordinal - 1]].record,)
            )
            if output.parents != expected_parents:
                raise ValueError("resource envelope public-route output lineage differs from its exact chain")

        campaign_identities = (
            self.issued_package,
            self.run_plan,
            self.execution_plan,
            self.execution_resource_envelope_spec,
            self.predevelopment_jit_signature_census,
            self.jit_graph_signature_manifest,
            self.recovery_index,
            self.scientific_graph_parity_receipt,
        )
        graph_schemas = (
            self.graph.issued_package_schema,
            self.graph.run_plan_schema,
            self.graph.execution_plan_schema,
            self.graph.execution_resource_envelope_spec_schema,
            self.graph.predevelopment_jit_signature_census_schema,
            self.graph.jit_graph_signature_manifest_schema,
            self.graph.recovery_index_schema,
            self.graph.scientific_graph_parity_receipt_schema,
        )
        validate_optional_jit_identities(
            self.predevelopment_jit_signature_census, self.jit_graph_signature_manifest
        )
        if any(
            (value is None and index not in {4, 5})
            or (value is not None and value.object_schema != schema)
            for index, (value, schema) in enumerate(
                zip(campaign_identities, graph_schemas, strict=True)
            )
        ):
            raise ValueError("resource envelope public-route campaign identity differs from its graph")
        operational_identities = (
            self.operational_conformance.issued_package,
            self.operational_conformance.run_plan,
            self.operational_conformance.execution_plan,
            self.operational_conformance.execution_resource_envelope_spec,
            self.operational_conformance.predevelopment_jit_signature_census,
            self.operational_conformance.jit_graph_signature_manifest,
            self.operational_conformance.recovery_index,
            self.operational_conformance.scientific_graph_parity_receipt,
        )
        if operational_identities != campaign_identities:
            raise ValueError("resource envelope public-route operational identities are discontinuous")
        if self.public_composition.object_schema != PublicRouteComposition.SCHEMA:
            raise ValueError("resource envelope public-route receipt binds another public composition")
        if self.operational_conformance.provider_registry_sha256 == "0" * 64:
            raise ValueError("resource envelope public-route operational provider is not bound")
        if (
            not self.terminal
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("resource envelope public-route receipt is not terminal conformance")


@dataclass(frozen=True, slots=True)
class PublicRouteConformanceCaseReceipt(CanonicalRecord):
    """One conformance case with evidence and closed terminal semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-conformance-case-receipt'

    case_id: str
    semantics: PublicRouteCaseSemantics
    covered_stages: tuple[PublicRouteConformanceStage, ...]
    condition_false_stages: tuple[PublicRouteConformanceStage, ...]
    authority_stop_stage: PublicRouteConformanceStage | None
    evidence_records: tuple[ObjectIdentity, ...]
    terminal_disposition: str
    operational_success: bool
    recovered_without_recomputation: bool
    scientific_recomputation_count: int
    scientific_retuning_count: int
    terminal: bool
    created_empirical_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        require_sorted_unique_strings(
            tuple(value.value for value in self.covered_stages),
            field_name="covered_stages",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            tuple(value.value for value in self.condition_false_stages),
            field_name="condition_false_stages",
        )
        if set(self.covered_stages).intersection(self.condition_false_stages):
            raise ValueError("resource envelope public-route case both covers and skips one stage")
        require_sorted_unique_ids(
            self.evidence_records,
            attribute="object_id",
            field_name="evidence_records",
        )
        validate_nonempty(self.terminal_disposition, field_name="terminal_disposition")
        if not self.evidence_records or not self.operational_success:
            raise ValueError("resource envelope public-route case lacks successful canonical evidence")
        if self.scientific_recomputation_count or self.scientific_retuning_count:
            raise ValueError("resource envelope public-route case recomputed or retuned science")
        if not self.terminal or self.created_empirical_evidence:
            raise ValueError("resource envelope public-route case lacks terminal truth-known evidence")

        evidence_schemas = {value.object_schema for value in self.evidence_records}
        if self.semantics is PublicRouteCaseSemantics.SUPPORTED:
            required_schemas = {
                schema
                for schemas in STANDARD_PUBLIC_ROUTE_OUTPUT_SCHEMAS.values()
                for schema in schemas
            }
            if (
                self.covered_stages != _ALL_STAGES
                or self.condition_false_stages
                or self.authority_stop_stage is not None
                or self.recovered_without_recomputation
                or not required_schemas.issubset(evidence_schemas)
            ):
                raise ValueError("supported resource envelope public-route case is incomplete")
        elif self.semantics is PublicRouteCaseSemantics.ZERO_LAW:
            if (
                self.covered_stages != _ZERO_LAW_COVERED_STAGES
                or self.condition_false_stages != _ZERO_LAW_CONDITION_FALSE_STAGES
                or self.authority_stop_stage is not None
                or self.recovered_without_recomputation
                or not _ZERO_LAW_REQUIRED_EVIDENCE_SCHEMAS.issubset(evidence_schemas)
            ):
                raise ValueError("zero-law resource envelope public-route case is inconsistent")
        elif self.semantics is PublicRouteCaseSemantics.AUTHORITY_STOP:
            if (
                self.covered_stages != _AUTHORITY_STOP_COVERED_STAGES
                or self.condition_false_stages != _AUTHORITY_STOP_CONDITION_FALSE_STAGES
                or self.authority_stop_stage is not PublicRouteConformanceStage.AUTHORIZED_REVEAL
                or self.recovered_without_recomputation
                or not _AUTHORITY_STOP_REQUIRED_EVIDENCE_SCHEMAS.issubset(evidence_schemas)
            ):
                raise ValueError("authority-stop resource envelope public-route case is inconsistent")
        elif self.semantics is PublicRouteCaseSemantics.RECOVERY:
            if (
                self.condition_false_stages
                or self.authority_stop_stage is not None
                or not self.recovered_without_recomputation
                or PublicRouteOperationalConformanceReceipt.SCHEMA not in evidence_schemas
            ):
                raise ValueError("recovery resource envelope public-route case is inconsistent")
        elif (
            self.condition_false_stages
            or self.authority_stop_stage is not None
            or self.recovered_without_recomputation
        ):
            raise ValueError("matrix resource envelope public-route case claims terminal route semantics")


@dataclass(frozen=True, slots=True)
class PublicRouteConformanceSuiteReceipt(CanonicalRecord):
    """Closed case roster joined to one public route and operational receipt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-conformance-suite-receipt'

    suite_id: str
    primary_receipt: ObjectIdentity
    public_composition: ObjectIdentity
    operational_conformance: ObjectIdentity
    provider_registry_sha256: str
    cases: tuple[PublicRouteConformanceCaseReceipt, ...]
    recovery_injection_case_ids: tuple[str, ...]
    terminal: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.suite_id, field_name="suite_id")
        if self.primary_receipt.object_schema != PublicRouteConformanceReceipt.SCHEMA:
            raise ValueError("resource envelope public-route suite binds another primary receipt")
        if self.public_composition.object_schema != PublicRouteComposition.SCHEMA:
            raise ValueError("resource envelope public-route suite binds another public composition")
        if (
            self.operational_conformance.object_schema
            != PublicRouteOperationalConformanceReceipt.SCHEMA
        ):
            raise ValueError("resource envelope public-route suite binds another operational receipt")
        validate_sha256(self.provider_registry_sha256, field_name="provider_registry_sha256")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if tuple(value.case_id for value in self.cases) != (
            REQUIRED_RESOURCE_ENVELOPE_CONFORMANCE_CASE_IDS
        ):
            raise ValueError("resource envelope public-route conformance case roster is incomplete")
        by_id = {value.case_id: value for value in self.cases}
        observed_semantics = tuple(
            (case_id, by_id[case_id].semantics)
            for case_id in REQUIRED_RESOURCE_ENVELOPE_CONFORMANCE_CASE_IDS
        )
        if observed_semantics != RESOURCE_ENVELOPE_CASE_SEMANTICS_BY_ID:
            raise ValueError("resource envelope public-route semantic case roster is incomplete")
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
            raise ValueError("resource envelope public-route suite overclaims its conformance boundary")


RESOURCE_ENVELOPE_EXECUTABLE_GATE_GAP_REASONS = (
    "CURRENT_PROVIDER_EXECUTION_EVIDENCE_ABSENT",
    "CURRENT_PROVIDER_RECOVERY_EVIDENCE_ABSENT",
    "PUBLIC_FACADE_SCIENTIFIC_STAGE_API_ABSENT",
    "SCIENTIFIC_OWNER_PROVIDER_REGISTRY_DIFFERS",
)


@dataclass(frozen=True, slots=True)
class PublicRouteExecutableGateGap(CanonicalRecord):
    """Record the blockers for executable public route conformance.

    This record keeps the resource envelope identities separate from execution
    and recovery evidence from a different provider registry. It does not
    establish an executable public route.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/public-route-executable-gate-gap'

    gap_id: str
    graph: PublicRouteConformanceGraph
    supported_stage_outputs: tuple[PublicRouteStageOutput, ...]
    zero_law_evidence: tuple[ObjectIdentity, ...]
    issued_package: ObjectIdentity
    run_plan: ObjectIdentity
    execution_plan: ObjectIdentity
    execution_resource_envelope_spec: ObjectIdentity
    predevelopment_jit_signature_census: ObjectIdentity | None
    jit_graph_signature_manifest: ObjectIdentity | None
    recovery_index: ObjectIdentity
    scientific_graph_parity_receipt: ObjectIdentity
    scientific_owner_provider_registry_sha256: str
    current_provider_registry_sha256: str
    public_composition: ObjectIdentity
    reason_codes: tuple[str, ...]
    conformance_scope: PublicRouteConformanceScope
    invocation_surface: PublicRouteInvocationSurface
    public_api_covered_stages: tuple[PublicRouteConformanceStage, ...]
    gate_ready: bool
    terminal: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.gap_id, field_name="gap_id")
        require_sorted_unique_ids(
            self.supported_stage_outputs,
            attribute="output_id",
            field_name="supported_stage_outputs",
        )
        outputs = {value.stage: value for value in self.supported_stage_outputs}
        if len(outputs) != len(self.supported_stage_outputs) or set(outputs) != set(
            PUBLIC_ROUTE_STAGE_ORDER
        ):
            raise ValueError("resource envelope public-route gate gap lacks the exact supported owner chain")
        for stage, output in outputs.items():
            if output.record.object_schema not in self.graph.binding(stage).output_schema_ids:
                raise ValueError("resource envelope public-route gate-gap output differs from its contract")
            ordinal = PUBLIC_ROUTE_STAGE_ORDER.index(stage)
            expected_parents = (
                () if ordinal == 0 else (outputs[PUBLIC_ROUTE_STAGE_ORDER[ordinal - 1]].record,)
            )
            if output.parents != expected_parents:
                raise ValueError("resource envelope public-route gate-gap lineage differs from its exact chain")
        require_sorted_unique_ids(
            self.zero_law_evidence,
            attribute="object_id",
            field_name="zero_law_evidence",
        )
        if not _ZERO_LAW_REQUIRED_EVIDENCE_SCHEMAS.issubset(
            {value.object_schema for value in self.zero_law_evidence}
        ):
            raise ValueError("resource envelope public-route gate gap lacks a complete zero-law owner chain")
        campaign_identities = (
            self.issued_package,
            self.run_plan,
            self.execution_plan,
            self.execution_resource_envelope_spec,
            self.predevelopment_jit_signature_census,
            self.jit_graph_signature_manifest,
            self.recovery_index,
            self.scientific_graph_parity_receipt,
        )
        graph_schemas = (
            self.graph.issued_package_schema,
            self.graph.run_plan_schema,
            self.graph.execution_plan_schema,
            self.graph.execution_resource_envelope_spec_schema,
            self.graph.predevelopment_jit_signature_census_schema,
            self.graph.jit_graph_signature_manifest_schema,
            self.graph.recovery_index_schema,
            self.graph.scientific_graph_parity_receipt_schema,
        )
        validate_optional_jit_identities(
            self.predevelopment_jit_signature_census, self.jit_graph_signature_manifest
        )
        if any(
            (value is None and index not in {4, 5})
            or (value is not None and value.object_schema != schema)
            for index, (value, schema) in enumerate(
                zip(campaign_identities, graph_schemas, strict=True)
            )
        ):
            raise ValueError("resource envelope public-route gate gap uses another campaign topology")
        validate_sha256(
            self.scientific_owner_provider_registry_sha256,
            field_name="scientific_owner_provider_registry_sha256",
        )
        validate_sha256(
            self.current_provider_registry_sha256,
            field_name="current_provider_registry_sha256",
        )
        if self.scientific_owner_provider_registry_sha256 == (
            self.current_provider_registry_sha256
        ):
            raise ValueError("resource envelope public-route provider mismatch gap is not present")
        if self.public_composition.object_schema != PublicRouteComposition.SCHEMA:
            raise ValueError("resource envelope public-route gate gap binds another composition")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.reason_codes != RESOURCE_ENVELOPE_EXECUTABLE_GATE_GAP_REASONS:
            raise ValueError("resource envelope public-route gate gap reason ledger is incomplete")
        require_sorted_unique_strings(
            tuple(value.value for value in self.public_api_covered_stages),
            field_name="public_api_covered_stages",
        )
        if (
            self.conformance_scope is not PublicRouteConformanceScope.ARCHITECTURE_TRUTH_KNOWN
            or self.invocation_surface
            is not PublicRouteInvocationSurface.INDEPENDENT_OWNER_RECEIPT_JOIN
            or self.public_api_covered_stages
            or self.gate_ready
            or not self.terminal
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("resource envelope public-route gate gap overclaims executable conformance")


def standard_resource_envelope_route_composition(
    *,
    implementation_source: ImplementationSourceClosure,
    provider_registry_sha256: str,
) -> PublicRouteComposition:
    """Build the resource envelope route from the execution envelope composition."""

    baseline = standard_public_route_composition(
        implementation_source=implementation_source,
        provider_registry_sha256=provider_registry_sha256,
    )
    configs = {value.stage: value for value in baseline.stage_configs}
    capabilities = {
        PublicRouteConformanceStage(
            value.capability_key.removeprefix("response-law.").upper()
        ): value
        for value in baseline.capabilities
    }
    semantics = {value.stage: value for value in baseline.semantic_output_contracts}
    configs[PublicRouteConformanceStage.OBSERVATION] = replace(
        configs[PublicRouteConformanceStage.OBSERVATION],
        input_schema_ids=(RESOURCE_ROUTE_IDENTIFICATION_INPUT_SCHEMA,),
    )
    capabilities[PublicRouteConformanceStage.OBSERVATION] = replace(
        capabilities[PublicRouteConformanceStage.OBSERVATION],
        input_schema_ids=(RESOURCE_ROUTE_IDENTIFICATION_INPUT_SCHEMA,),
    )
    for stage in (
        PublicRouteConformanceStage.CONTROLLER_UNIT_EVALUATION,
        PublicRouteConformanceStage.CONTROLLER_COHORT_ADJUDICATION,
    ):
        capability = capabilities[stage]
        capabilities[stage] = replace(
            capability,
            permissions=tuple(
                sorted(
                    {
                        *capability.permissions,
                        CapabilityPermission.READ_OUTCOME_VISIBLE,
                    }
                )
            ),
        )
    controller_cohort = capabilities[PublicRouteConformanceStage.CONTROLLER_COHORT_ADJUDICATION]
    capabilities[PublicRouteConformanceStage.CONTROLLER_COHORT_ADJUDICATION] = replace(
        controller_cohort,
        output_schema_ids=tuple(
            sorted({*controller_cohort.output_schema_ids, ScientificAdjudicationRecord.SCHEMA})
        ),
    )
    for stage in (
        PublicRouteConformanceStage.CONTROLLER_PROGRAMME,
        PublicRouteConformanceStage.CONTROLLER_COMPILE,
    ):
        capability = capabilities[stage]
        capabilities[stage] = replace(
            capability,
            conformance_check_ids=tuple(
                sorted(
                    {
                        *capability.conformance_check_ids,
                        "corrected-current-controller-contract",
                    }
                )
            ),
        )
    stages = []
    for binding in baseline.graph.stages:
        stage = binding.stage
        capability = capabilities[stage]
        config = configs[stage]
        semantic = replace(
            semantics[stage],
            capability_manifest=ObjectIdentity.from_record(
                capability.capability_key,
                capability,
            ),
            stage_config=ObjectIdentity.from_record(config.config_id, config),
        )
        semantics[stage] = semantic
        stages.append(
            replace(
                binding,
                capability_manifest=ObjectIdentity.from_record(
                    capability.capability_key,
                    capability,
                ),
                semantic_output_contracts=(
                    ObjectIdentity.from_record(semantic.contract_id, semantic),
                ),
            )
        )
    graph = PublicRouteConformanceGraph(
        graph_id="graph.public-resource-envelope-route-conformance",
        stages=tuple(stages),
        issued_package_schema=STANDARD_ISSUED_EXPERIMENT_PACKAGE_SCHEMA,
        run_plan_schema='empirical-lawhood/runtime/run-plan',
        execution_plan_schema='empirical-lawhood/runtime/execution-plan',
        execution_resource_envelope_spec_schema='empirical-lawhood/runtime/execution-resource-envelope-spec',
        predevelopment_jit_signature_census_schema=(PredevelopmentJitSignatureCensus.SCHEMA),
        jit_graph_signature_manifest_schema=JitGraphSignatureManifest.SCHEMA,
        recovery_index_schema='empirical-lawhood/runtime/run-recovery-index',
        scientific_graph_parity_receipt_schema='empirical-lawhood/runtime/resource-graph-preservation-receipt',
        external_artifact_plane_required=True,
        created_empirical_evidence=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return PublicRouteComposition(
        composition_id="composition.public-resource-envelope-route",
        graph=graph,
        implementation_source=baseline.implementation_source,
        proof_owners=baseline.proof_owners,
        stage_configs=tuple(sorted(configs.values(), key=lambda value: value.config_id)),
        capabilities=tuple(sorted(capabilities.values(), key=lambda value: value.registry_id)),
        semantic_output_contracts=tuple(
            sorted(semantics.values(), key=lambda value: value.contract_id)
        ),
        provider_registry_sha256=baseline.provider_registry_sha256,
        external_artifact_plane_required=True,
    )


def standard_resource_envelope_route_conformance_graph(
    *,
    implementation_source: ImplementationSourceClosure,
    provider_registry_sha256: str,
) -> PublicRouteConformanceGraph:
    """Return the source/provider-bound current conformance graph."""

    return standard_resource_envelope_route_composition(
        implementation_source=implementation_source,
        provider_registry_sha256=provider_registry_sha256,
    ).graph


__all__ = [
    "RESOURCE_ROUTE_IDENTIFICATION_INPUT_SCHEMA",
    "RESOURCE_ENVELOPE_CASE_SEMANTICS_BY_ID",
    "RESOURCE_ENVELOPE_EXECUTABLE_GATE_GAP_REASONS",
    "REQUIRED_RESOURCE_ENVELOPE_CONFORMANCE_CASE_IDS",
    "REQUIRED_RESOURCE_ENVELOPE_SEMANTIC_CASES",
    "STANDARD_ISSUED_EXPERIMENT_PACKAGE_SCHEMA",
    'PublicRouteCaseSemantics',
    'PublicRouteConformanceScope',
    'PublicRouteInvocationSurface',
    'PublicRouteComposition',
    'PublicRouteConformanceCaseReceipt',
    'PublicRouteConformanceGraph',
    'PublicRouteConformanceReceipt',
    'PublicRouteConformanceSuiteReceipt',
    'PublicRouteExecutableGateGap',
    'PublicRouteOperationalConformanceReceipt',
    'standard_resource_envelope_route_composition',
    'standard_resource_envelope_route_conformance_graph',
]
