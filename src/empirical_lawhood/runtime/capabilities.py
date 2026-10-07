"""Static, versioned capability manifests and registry conformance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


class CapabilityKind(StrEnum):
    SOURCE = "SOURCE"
    SIMULATOR = "SIMULATOR"
    TRANSFORM = "TRANSFORM"
    ANALYSIS = "ANALYSIS"
    ANOMALY_DETECTOR = "ANOMALY_DETECTOR"
    PORTFOLIO_PLANNER = "PORTFOLIO_PLANNER"
    HYPOTHESIS_SYNTHESIZER = "HYPOTHESIS_SYNTHESIZER"
    PROSPECTIVE_NOMINATOR = "PROSPECTIVE_NOMINATOR"
    EXPERIMENT_DESIGNER = "EXPERIMENT_DESIGNER"
    HYPOTHESIS_ADJUDICATOR = "HYPOTHESIS_ADJUDICATOR"
    LAW_IDENTIFIER = "LAW_IDENTIFIER"
    RESPONSE_ALGEBRA_IDENTIFIER = "RESPONSE_ALGEBRA_IDENTIFIER"
    FALSIFIER = "FALSIFIER"
    OBSERVATION_OPERATOR = "OBSERVATION_OPERATOR"
    NUMERICAL_QUALIFIER = "NUMERICAL_QUALIFIER"
    DISCREPANCY_ESTIMATOR = "DISCREPANCY_ESTIMATOR"
    TRANSPORT_TESTER = "TRANSPORT_TESTER"
    ATLAS_ASSEMBLER = "ATLAS_ASSEMBLER"
    ADMISSION_EVALUATOR = "ADMISSION_EVALUATOR"
    REACHABILITY_EVALUATOR = "REACHABILITY_EVALUATOR"
    CONTROLLER_SYNTHESIZER = "CONTROLLER_SYNTHESIZER"
    CONTROLLER_OBSERVER = "CONTROLLER_OBSERVER"
    ONLINE_GATE_EVALUATOR = "ONLINE_GATE_EVALUATOR"
    NATIVE_DELIVERY = "NATIVE_DELIVERY"
    EVALUATOR = "EVALUATOR"
    REPORTER = "REPORTER"
    APPROVAL_GATE = "APPROVAL_GATE"
    ACTUATOR = "ACTUATOR"


class CapabilityPermission(StrEnum):
    READ_EXTERNAL_ARTIFACTS = "READ_EXTERNAL_ARTIFACTS"
    WRITE_EXTERNAL_ARTIFACTS = "WRITE_EXTERNAL_ARTIFACTS"
    READ_DEVELOPMENT = "READ_DEVELOPMENT"
    READ_OUTCOME_VISIBLE = "READ_OUTCOME_VISIBLE"
    READ_SEALED_OUTCOMES = "READ_SEALED_OUTCOMES"
    READ_FROZEN_MODELS = "READ_FROZEN_MODELS"
    REVEAL_OUTCOMES = "REVEAL_OUTCOMES"
    WRITE_CATALOG = "WRITE_CATALOG"
    APPROVE_NONACTUATING = "APPROVE_NONACTUATING"
    COMMAND_ACTUATOR = "COMMAND_ACTUATOR"


_OUTCOME_ACCESS_RANK = {
    OutcomeAccess.OUTCOME_BLIND: 0,
    OutcomeAccess.EVALUATION_SEALED: 0,
    OutcomeAccess.DEVELOPMENT_VISIBLE: 1,
    OutcomeAccess.EVALUATOR_REVEAL: 2,
    OutcomeAccess.EVALUATION_REVEALED: 2,
    OutcomeAccess.PRIVILEGED_TRUTH: 3,
}

_EVIDENCE_CEILING_RANK = {ceiling: index for index, ceiling in enumerate(EvidenceCeiling)}


def input_access_allowed(
    requirement: CapabilityRequirement,
    access: OutcomeAccess,
    *,
    reveal_barrier_authorized: bool = False,
    frozen_model_authorized: bool = False,
) -> bool:
    """One input permission rule shared by preflight and actual worker reads.

    Preflight may prove the conditional reveal route. Only the execution owner
    can establish that its declared barrier has actual reveal authority.
    """
    permissions = set(requirement.required_permissions)
    if CapabilityPermission.READ_EXTERNAL_ARTIFACTS not in permissions:
        return False
    if access is OutcomeAccess.DEVELOPMENT_VISIBLE:
        return bool(
            permissions.intersection(
                {
                    CapabilityPermission.READ_DEVELOPMENT,
                    CapabilityPermission.READ_OUTCOME_VISIBLE,
                    CapabilityPermission.REVEAL_OUTCOMES,
                }
            )
        )
    if access is OutcomeAccess.EVALUATION_REVEALED:
        return bool(
            permissions.intersection(
                {CapabilityPermission.READ_OUTCOME_VISIBLE, CapabilityPermission.REVEAL_OUTCOMES}
            )
        )
    if access is OutcomeAccess.PRIVILEGED_TRUTH:
        return (
            requirement.requested_outcome_access
            in {OutcomeAccess.EVALUATOR_REVEAL, OutcomeAccess.PRIVILEGED_TRUTH}
            and CapabilityPermission.REVEAL_OUTCOMES in permissions
        )
    if access is OutcomeAccess.EVALUATION_SEALED and reveal_barrier_authorized:
        if (
            CapabilityPermission.READ_OUTCOME_VISIBLE in permissions
            and requirement.requested_outcome_access
            in {OutcomeAccess.EVALUATOR_REVEAL, OutcomeAccess.EVALUATION_REVEALED}
        ):
            return True
    if _OUTCOME_ACCESS_RANK[access] > _OUTCOME_ACCESS_RANK[requirement.requested_outcome_access]:
        return False
    if access is OutcomeAccess.EVALUATION_SEALED:
        return CapabilityPermission.READ_SEALED_OUTCOMES in permissions
    if access is OutcomeAccess.EVALUATOR_REVEAL:
        return CapabilityPermission.REVEAL_OUTCOMES in permissions or (
            frozen_model_authorized
            and CapabilityPermission.READ_FROZEN_MODELS in permissions
            and requirement.requested_outcome_access is OutcomeAccess.EVALUATOR_REVEAL
        )
    return True


@dataclass(frozen=True, slots=True)
class CapabilityConfigRef(CanonicalRecord):
    """Content-addressed canonical configuration; never an import path or command."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capability-config-ref'

    config_id: str
    config_schema: str
    config_schema_sha256: str
    content_sha256: str
    artifact_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_schema(self.config_schema)
        validate_sha256(
            self.config_schema_sha256,
            field_name="config_schema_sha256",
        )
        validate_sha256(self.content_sha256, field_name="content_sha256")
        validate_stable_id(self.artifact_id, field_name="artifact_id")


@dataclass(frozen=True, slots=True)
class ImplementationSourceClosure(CanonicalRecord):
    """Exact tracked source bytes underlying one registered implementation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/implementation-source-closure'

    closure_id: str
    implementation_commit: str
    source_files: tuple[ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.closure_id, field_name="closure_id")
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        if not isinstance(self.source_files, tuple) or not self.source_files:
            raise ValueError("implementation source closure requires tracked source files")
        if len(self.source_files) > 256:
            raise ValueError("implementation source closure exceeds its file-count limit")
        if any(not isinstance(value, ArtifactIdentity) for value in self.source_files):
            raise ValueError("implementation source closure contains another record type")
        require_sorted_unique_ids(
            self.source_files,
            attribute="artifact_id",
            field_name="source_files",
        )


@dataclass(frozen=True, slots=True)
class CapabilityManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capability-manifest'

    capability_key: str
    capability_version: str
    kind: CapabilityKind
    config_schema: str
    config_schema_sha256: str
    input_schema_ids: tuple[str, ...]
    output_schema_ids: tuple[str, ...]
    permissions: tuple[CapabilityPermission, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    maximum_outcome_access: OutcomeAccess
    resource_ceiling: ResourceBudget
    deterministic: bool
    seed_required: bool
    language_id: str
    runtime_id: str
    requires_clean_commit: bool
    requires_active_mount: bool
    requires_network: bool
    conformance_check_ids: tuple[str, ...]
    implementation_sha256: str

    @property
    def registry_id(self) -> str:
        return f"{self.capability_key}@{self.capability_version}"

    def __post_init__(self) -> None:
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_schema(self.config_schema)
        validate_sha256(
            self.config_schema_sha256,
            field_name="config_schema_sha256",
        )
        for name, values in (
            ("input_schema_ids", self.input_schema_ids),
            ("output_schema_ids", self.output_schema_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
            for schema in values:
                validate_schema(schema)
        require_sorted_unique_strings(self.permissions, field_name="permissions")
        for name, value in (
            ("language_id", self.language_id),
            ("runtime_id", self.runtime_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.conformance_check_ids,
            field_name="conformance_check_ids",
            allow_empty=False,
        )
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        self._validate_execution_contract()
        self._validate_authority_permissions()
        self._validate_response_algebra_boundary()

    def _validate_execution_contract(self) -> None:
        if self.deterministic and self.seed_required:
            raise ValueError("a deterministic capability cannot require a random seed")

    def _validate_authority_permissions(self) -> None:
        if (
            CapabilityPermission.COMMAND_ACTUATOR in self.permissions
            and self.kind is not CapabilityKind.ACTUATOR
        ):
            raise ValueError("only an actuator capability may command an actuator")
        if (
            CapabilityPermission.REVEAL_OUTCOMES in self.permissions
            and self.kind is not CapabilityKind.EVALUATOR
        ):
            raise ValueError("only an evaluator capability may reveal outcomes")
        approval_permission = CapabilityPermission.APPROVE_NONACTUATING
        if self.kind is CapabilityKind.APPROVAL_GATE:
            if self.permissions != (approval_permission,):
                raise ValueError("approval gate receives only nonactuating approval authority")
            if self.maximum_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
                raise ValueError("approval gate must remain outcome-blind")
        elif approval_permission in self.permissions:
            raise ValueError("only the approval gate may receive approval authority")

    def _validate_response_algebra_boundary(self) -> None:
        if self.kind is not CapabilityKind.RESPONSE_ALGEBRA_IDENTIFIER:
            return
        if self.maximum_evidence_ceiling in {
            EvidenceCeiling.ADMISSION,
            EvidenceCeiling.CONTROLLER_USE,
        }:
            raise ValueError("response-algebra identifier is capped at local law")
        if self.maximum_outcome_access in {
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }:
            raise ValueError("response-algebra identifier cannot reveal evaluation outcomes")
        forbidden = {
            CapabilityPermission.APPROVE_NONACTUATING,
            CapabilityPermission.COMMAND_ACTUATOR,
            CapabilityPermission.READ_OUTCOME_VISIBLE,
            CapabilityPermission.REVEAL_OUTCOMES,
            CapabilityPermission.READ_FROZEN_MODELS,
            CapabilityPermission.WRITE_CATALOG,
        }
        if forbidden & set(self.permissions):
            raise ValueError("response-algebra identifier has a forbidden permission")
        result_schema = 'empirical-lawhood/kernel/response-algebra-identification-result'
        if result_schema not in self.output_schema_ids:
            raise ValueError("response-algebra identifier must emit the typed result schema")


@dataclass(frozen=True, slots=True)
class CapabilityRequirement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capability-requirement'

    capability_key: str
    capability_version: str
    kind: CapabilityKind
    config: CapabilityConfigRef
    required_input_schema_ids: tuple[str, ...]
    required_output_schema_ids: tuple[str, ...]
    required_permissions: tuple[CapabilityPermission, ...]
    requested_evidence_ceiling: EvidenceCeiling
    requested_outcome_access: OutcomeAccess
    requested_resources: ResourceBudget

    def __post_init__(self) -> None:
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        for name, values in (
            ("required_input_schema_ids", self.required_input_schema_ids),
            ("required_output_schema_ids", self.required_output_schema_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
            for schema in values:
                validate_schema(schema)
        require_sorted_unique_strings(
            self.required_permissions,
            field_name="required_permissions",
        )


class CapabilityConformanceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CapabilityRegistry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capability-registry'

    registry_id: str
    capabilities: tuple[CapabilityManifest, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        require_sorted_unique_ids(
            self.capabilities,
            attribute="registry_id",
            field_name="capabilities",
        )
        if not self.capabilities:
            raise ValueError("capability registry must not be empty")

    def resolve(self, capability_key: str, capability_version: str) -> CapabilityManifest:
        validate_stable_id(capability_key, field_name="capability_key")
        validate_semantic_version(capability_version)
        for capability in self.capabilities:
            if (
                capability.capability_key == capability_key
                and capability.capability_version == capability_version
            ):
                return capability
        raise KeyError(f"unregistered capability: {capability_key}@{capability_version}")

    def require(self, requirement: CapabilityRequirement) -> CapabilityManifest:
        try:
            capability = self.resolve(
                requirement.capability_key,
                requirement.capability_version,
            )
        except KeyError as error:
            raise CapabilityConformanceError(str(error)) from error
        reasons = _conformance_reasons(capability, requirement)
        if reasons:
            raise CapabilityConformanceError(",".join(sorted(reasons)))
        return capability


def _conformance_reasons(
    capability: CapabilityManifest,
    requirement: CapabilityRequirement,
) -> list[str]:
    reasons: list[str] = []
    if capability.kind is not requirement.kind:
        reasons.append("CAPABILITY_KIND_MISMATCH")
    if (
        capability.config_schema != requirement.config.config_schema
        or capability.config_schema_sha256 != requirement.config.config_schema_sha256
    ):
        reasons.append("CONFIG_SCHEMA_NOT_SUPPORTED")
    if not set(requirement.required_input_schema_ids).issubset(capability.input_schema_ids):
        reasons.append("INPUT_SCHEMA_NOT_SUPPORTED")
    if not set(requirement.required_output_schema_ids).issubset(capability.output_schema_ids):
        reasons.append("OUTPUT_SCHEMA_NOT_SUPPORTED")
    if not set(requirement.required_permissions).issubset(capability.permissions):
        reasons.append("CAPABILITY_PERMISSION_MISSING")
    if (
        _EVIDENCE_CEILING_RANK[requirement.requested_evidence_ceiling]
        > _EVIDENCE_CEILING_RANK[capability.maximum_evidence_ceiling]
    ):
        reasons.append("EVIDENCE_CEILING_EXCEEDED")
    if (
        _OUTCOME_ACCESS_RANK[requirement.requested_outcome_access]
        > _OUTCOME_ACCESS_RANK[capability.maximum_outcome_access]
    ):
        reasons.append("OUTCOME_ACCESS_EXCEEDED")
    if not capability.resource_ceiling.contains(requirement.requested_resources):
        reasons.append("CAPABILITY_RESOURCE_EXCEEDED")
    return reasons
