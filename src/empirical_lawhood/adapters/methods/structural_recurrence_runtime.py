"Registered runtime surface for the inherited margin structural recurrence forecast methods.\n\nThe source manifest fingerprints the executing target package's method files.\nIt makes no assertion about an earlier closeout or its receipts.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters.methods import structural_recurrence as core
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceQualificationBranch, StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetDesignFreeze, StructuralRecurrenceTargetStage
from empirical_lawhood.adapters.methods.action_fiber_structural_recurrence import ActionFiberStructuralRecurrenceTargetResult
from empirical_lawhood.adapters.methods.structural_bootstrap_inputs import StructuralBootstrapInputCensus
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import MarginStructuralRecurrenceForecastAdjudication, MarginStructuralRecurrenceForecastConformance, MarginStructuralRecurrenceForecastMethodFreeze, MarginStructuralRecurrenceForecastAdmissionHandoff, MarginStructuralRecurrenceForecastPredictionIssue, MarginStructuralRecurrenceForecastTargetMatch, adjudicate, build_prediction_issue, match_target
from empirical_lawhood.adapters.methods.margin_structural_recurrence_conformance import execute_conformance
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityRegistration,
)


MARGIN_FORECAST_CAPABILITY_VERSION = "1.0.0"
MAX_MARGIN_FORECAST_RUNTIME_BYTES = 16 * 1024 * 1024


class MarginStructuralRecurrenceForecastRuntimeOperation(StrEnum):
    CONFORMANCE = "CONFORMANCE"
    PREDICT = "PREDICT"
    MATCH = "MATCH"
    ADJUDICATE = "ADJUDICATE"
    TARGET_COMPATIBILITY_AUDIT = "TARGET_COMPATIBILITY_AUDIT"


class MarginStructuralRecurrenceForecastPortKind(StrEnum):
    CURRENT_SCIENTIFIC_FUNCTION = "CURRENT_SCIENTIFIC_FUNCTION"
    ADDITIVE_BOUNDARY_AUDIT = "ADDITIVE_BOUNDARY_AUDIT"


_CAPABILITY_KEY_BY_OPERATION = {
    MarginStructuralRecurrenceForecastRuntimeOperation.CONFORMANCE: 'method.margin-structural-recurrence-forecast.conformance',
    MarginStructuralRecurrenceForecastRuntimeOperation.PREDICT: 'method.margin-structural-recurrence-forecast.predict',
    MarginStructuralRecurrenceForecastRuntimeOperation.MATCH: 'method.margin-structural-recurrence-forecast.match',
    MarginStructuralRecurrenceForecastRuntimeOperation.ADJUDICATE: 'method.margin-structural-recurrence-forecast.adjudicate',
    MarginStructuralRecurrenceForecastRuntimeOperation.TARGET_COMPATIBILITY_AUDIT: (
        'method.margin-structural-recurrence-forecast.target-compatibility-audit'
    ),
}


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastFrozenSourceFile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-frozen-source-file'

    source_id: str
    repository_path: str
    sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.source_id, field_name="source_id")
        validate_nonempty(self.repository_path, field_name="repository_path")
        validate_sha256(self.sha256, field_name="sha256")
        if (
            self.repository_path.startswith(("/", "."))
            or ".." in self.repository_path.split("/")
            or not self.repository_path.startswith("src/empirical_lawhood/adapters/methods/")
        ):
            raise ValueError('frozen structural recurrence source path escapes the method package')


_METHOD_FILES = (
    "structural_recurrence.py",
    "structural_recurrence_targets.py",
    "action_fiber_structural_recurrence.py",
    "action_fiber_structural_recurrence_conformance.py",
    "margin_structural_recurrence_forecast.py",
    "margin_structural_recurrence_conformance.py",
    "structural_bootstrap_inputs.py",
)


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastMethodSourceManifest(CanonicalRecord):
    'Current target source bytes of the structural forecast method.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-method-source-manifest'

    manifest_id: str
    source_files: tuple[MarginStructuralRecurrenceForecastFrozenSourceFile, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        require_sorted_unique_ids(
            self.source_files, attribute="source_id", field_name="source_files"
        )
        expected = {
            f"src/empirical_lawhood/adapters/methods/{name}" for name in _METHOD_FILES
        }
        if {row.repository_path for row in self.source_files} != expected:
            raise ValueError('structural recurrence method source roster differs from the shipped package')


def method_source_manifest() -> MarginStructuralRecurrenceForecastMethodSourceManifest:
    'Bind the seven method and numerical-input files as current executing target bytes.'

    method_dir = Path(__file__).resolve().parent
    values = []
    for filename in _METHOD_FILES:
        path = method_dir / filename
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 2 * 1024 * 1024:
            raise ValueError('structural recurrence method source is missing or outside its byte bound')
        values.append(
            MarginStructuralRecurrenceForecastFrozenSourceFile(
                source_id=f"margin-structural-recurrence-forecast-source.{filename.removesuffix('.py').replace('_', '-')}",
                repository_path=f"src/empirical_lawhood/adapters/methods/{filename}",
                sha256=sha256(path.read_bytes()).hexdigest(),
            )
        )
    return MarginStructuralRecurrenceForecastMethodSourceManifest(
        manifest_id='independent-substrate-grounding.margin-structural-recurrence-forecast.method-source-manifest',
        source_files=tuple(sorted(values, key=lambda row: row.source_id)),
    )


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastSemanticPortBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-semantic-port-binding'

    binding_id: str
    operation: MarginStructuralRecurrenceForecastRuntimeOperation
    port_kind: MarginStructuralRecurrenceForecastPortKind
    capability_key: str
    current_callable_symbol: str
    request_schema: str
    output_schema: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_nonempty(self.current_callable_symbol, field_name='current_callable_symbol')
        if self.capability_key != _CAPABILITY_KEY_BY_OPERATION[self.operation]:
            raise ValueError("semantic port binding selects the wrong capability")
        if self.port_kind is MarginStructuralRecurrenceForecastPortKind.ADDITIVE_BOUNDARY_AUDIT:
            if self.operation is not MarginStructuralRecurrenceForecastRuntimeOperation.TARGET_COMPATIBILITY_AUDIT:
                raise ValueError("only target compatibility is an additive port binding")
        elif self.operation is MarginStructuralRecurrenceForecastRuntimeOperation.TARGET_COMPATIBILITY_AUDIT:
            raise ValueError("target compatibility must remain visibly additive")


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastSemanticPortInventory(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-semantic-port-inventory'

    inventory_id: str
    method_source_manifest: ObjectIdentity
    bindings: tuple[MarginStructuralRecurrenceForecastSemanticPortBinding, ...]
    scientific_operators_changed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.inventory_id, field_name="inventory_id")
        require_sorted_unique_ids(
            self.bindings,
            attribute="binding_id",
            field_name="bindings",
        )
        if {value.operation for value in self.bindings} != set(MarginStructuralRecurrenceForecastRuntimeOperation):
            raise ValueError("semantic port inventory must cover the exact operation roster")
        if self.scientific_operators_changed:
            raise ValueError('independent substrate grounding cannot change frozen margin structural recurrence forecast semantics')


def semantic_port_inventory() -> MarginStructuralRecurrenceForecastSemanticPortInventory:
    manifest = method_source_manifest()
    values = (
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.CONFORMANCE,
            MarginStructuralRecurrenceForecastPortKind.CURRENT_SCIENTIFIC_FUNCTION,
            'empirical_lawhood.adapters.methods.margin_structural_recurrence_conformance.execute_conformance',
            MarginStructuralRecurrenceForecastConformanceRequest.SCHEMA,
            MarginStructuralRecurrenceForecastConformance.SCHEMA,
        ),
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.PREDICT,
            MarginStructuralRecurrenceForecastPortKind.CURRENT_SCIENTIFIC_FUNCTION,
            'empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast.build_prediction_issue',
            MarginStructuralRecurrenceForecastPredictionRequest.SCHEMA,
            MarginStructuralRecurrenceForecastPredictionIssue.SCHEMA,
        ),
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.MATCH,
            MarginStructuralRecurrenceForecastPortKind.CURRENT_SCIENTIFIC_FUNCTION,
            'empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast.match_target',
            MarginStructuralRecurrenceForecastMatchRequest.SCHEMA,
            MarginStructuralRecurrenceForecastTargetMatch.SCHEMA,
        ),
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.ADJUDICATE,
            MarginStructuralRecurrenceForecastPortKind.CURRENT_SCIENTIFIC_FUNCTION,
            'empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast.adjudicate',
            MarginStructuralRecurrenceForecastAdjudicationRequest.SCHEMA,
            MarginStructuralRecurrenceForecastAdjudication.SCHEMA,
        ),
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.TARGET_COMPATIBILITY_AUDIT,
            MarginStructuralRecurrenceForecastPortKind.ADDITIVE_BOUNDARY_AUDIT,
            'empirical_lawhood.adapters.methods.structural_recurrence_runtime.audit_target_compatibility',
            MarginStructuralRecurrenceForecastCompatibilityAuditRequest.SCHEMA,
            MarginStructuralRecurrenceForecastCompatibilityAudit.SCHEMA,
        ),
    )
    bindings = tuple(
        MarginStructuralRecurrenceForecastSemanticPortBinding(
            binding_id=f"'independent-substrate-grounding.margin-structural-recurrence-forecast.port.'{operation.value.lower().replace('_', '-')}",
            operation=operation,
            port_kind=kind,
            capability_key=_CAPABILITY_KEY_BY_OPERATION[operation],
            current_callable_symbol=symbol,
            request_schema=request_schema,
            output_schema=output_schema,
        )
        for operation, kind, symbol, request_schema, output_schema in values
    )
    return MarginStructuralRecurrenceForecastSemanticPortInventory(
        inventory_id='independent-substrate-grounding.margin-structural-recurrence-forecast.semantic-port-inventory',
        method_source_manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        bindings=tuple(sorted(bindings, key=lambda value: value.binding_id)),
        scientific_operators_changed=False,
    )


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastMethodPortFreeze(CanonicalRecord):
    """Immutable M06 exit record for the semantic-preserving runtime port."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-method-port-freeze'

    freeze_id: str
    method_source_manifest: ObjectIdentity
    semantic_inventory: ObjectIdentity
    registry: ObjectIdentity
    capability_registration_sha256s: tuple[str, ...]
    base_fixture_pass_count: int
    base_control_pass_count: int
    margin_fixture_pass_count: int
    margin_control_pass_count: int
    direct_registered_bytes_equal: bool
    refusal_suite_passed: bool
    receipt_recovery_replay_passed: bool
    scientific_operators_changed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        require_sorted_unique_strings(
            self.capability_registration_sha256s,
            field_name="capability_registration_sha256s",
            allow_empty=False,
        )
        if len(self.capability_registration_sha256s) != len(MarginStructuralRecurrenceForecastRuntimeOperation):
            raise ValueError("method-port freeze must bind the exact capability roster")
        if (
            self.base_fixture_pass_count,
            self.base_control_pass_count,
            self.margin_fixture_pass_count,
            self.margin_control_pass_count,
        ) != (24, 13, 12, 10):
            raise ValueError('method-port freeze conformance counts differ from margin structural recurrence forecast')
        if not all(
            (
                self.direct_registered_bytes_equal,
                self.refusal_suite_passed,
                self.receipt_recovery_replay_passed,
            )
        ):
            raise ValueError("method-port freeze requires every M03--M05 check")
        if self.scientific_operators_changed:
            raise ValueError('method-port freeze cannot ratify changed margin structural recurrence forecast semantics')


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastRuntimeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-runtime-config'

    config_id: str
    operation: MarginStructuralRecurrenceForecastRuntimeOperation
    capability_key: str
    capability_version: str
    method_source_manifest_sha256: str
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(
            self.method_source_manifest_sha256,
            field_name='method_source_manifest_sha256',
        )
        if self.capability_key != _CAPABILITY_KEY_BY_OPERATION[self.operation]:
            raise ValueError('structural recurrence runtime operation and capability key differ')
        if self.capability_version != MARGIN_FORECAST_CAPABILITY_VERSION:
            raise ValueError('structural recurrence runtime capability public revision differs')
        if self.method_source_manifest_sha256 != method_source_manifest().fingerprint():
            raise ValueError('structural recurrence runtime config binds the wrong current method source manifest')
        if not 0 < self.maximum_input_bytes <= MAX_MARGIN_FORECAST_RUNTIME_BYTES:
            raise ValueError('structural recurrence runtime input bound is outside the registered limit')


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastConformanceRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-conformance-request'

    request_id: str
    conformance_id: str
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze
    execution_authority: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.conformance_id, field_name="conformance_id")


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastPredictionRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-prediction-request'

    request_id: str
    issue_id: str
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze
    conformance: MarginStructuralRecurrenceForecastConformance
    design: StructuralRecurrenceTargetDesignFreeze
    development: StructuralRecurrenceStageEvidence
    evaluation_count: int
    prospective_validation_count: int
    scientific_bootstrap_inputs: StructuralBootstrapInputCensus

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.issue_id, field_name="issue_id")
        if self.development.stage is not StructuralRecurrenceTargetStage.DEVELOPMENT:
            raise ValueError('structural recurrence prediction request requires development evidence')
        if self.evaluation_count < 1 or self.prospective_validation_count < 1:
            raise ValueError('structural recurrence prediction request requires future panel sizes')


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastMatchRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-match-request'

    request_id: str
    prediction: MarginStructuralRecurrenceForecastPredictionIssue
    admission_handoff: MarginStructuralRecurrenceForecastAdmissionHandoff
    result: ActionFiberStructuralRecurrenceTargetResult

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastAdjudicationRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-adjudication-request'

    request_id: str
    conformance: MarginStructuralRecurrenceForecastConformance
    matches: tuple[MarginStructuralRecurrenceForecastTargetMatch, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        require_sorted_unique_ids(self.matches, attribute="match_id", field_name="matches")


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastCompatibilityAuditRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-compatibility-audit-request'

    request_id: str
    audit_id: str
    design: StructuralRecurrenceTargetDesignFreeze
    requested_level: core.TargetLevel

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.audit_id, field_name="audit_id")


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastCompatibilityAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-compatibility-audit'

    audit_id: str
    target_design: ObjectIdentity
    requested_level: core.TargetLevel
    source_branch: StructuralRecurrenceQualificationBranch
    compatibility_map_valid: bool
    complete_role_binding: bool
    split_rosters_disjoint: bool
    compatible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = all(
            (
                self.source_branch is StructuralRecurrenceQualificationBranch.ENTER_LAW_QUALIFICATION,
                self.compatibility_map_valid,
                self.complete_role_binding,
                self.split_rosters_disjoint,
                not self.reason_codes,
            )
        )
        if self.compatible != expected:
            raise ValueError('structural recurrence compatibility disposition is not audit-derived')


def audit_target_compatibility(
    request: MarginStructuralRecurrenceForecastCompatibilityAuditRequest,
) -> MarginStructuralRecurrenceForecastCompatibilityAudit:
    design = request.design
    reasons: set[str] = set()
    if design.qualification.branch is not StructuralRecurrenceQualificationBranch.ENTER_LAW_QUALIFICATION:
        reasons.add("SOURCE_QUALIFICATION_NOT_ENTERED")
    if not design.compatibility.map_valid or design.compatibility.required_new_role_ids:
        reasons.add("NATIVE_ROLE_MAP_INCOMPATIBLE")
    complete_roles = all(value.available for value in design.compatibility.bindings)
    if not complete_roles:
        reasons.add("REQUIRED_NATIVE_ROLE_ABSENT")
    level_rank = {value: index for index, value in enumerate(core.TargetLevel)}
    if level_rank[request.requested_level] > level_rank[design.maximum_level]:
        reasons.add("REQUESTED_LEVEL_EXCEEDS_TARGET_CEILING")
    rosters = tuple(unit_id for roster in design.rosters for unit_id in roster.unit_ids)
    seeds = tuple(seed for roster in design.rosters for seed in roster.seeds)
    split_disjoint = len(rosters) == len(set(rosters)) and len(seeds) == len(set(seeds))
    if not split_disjoint:
        reasons.add("SPLIT_ROSTER_OVERLAP")
    if design.selected_source_implementation.object_fingerprint != (
        design.qualification.selected_source_fingerprint
    ):
        reasons.add("SOURCE_IDENTITY_MISMATCH")
    return MarginStructuralRecurrenceForecastCompatibilityAudit(
        audit_id=request.audit_id,
        target_design=ObjectIdentity.from_record(design.design_id, design),
        requested_level=request.requested_level,
        source_branch=design.qualification.branch,
        compatibility_map_valid=(
            design.compatibility.map_valid and not design.compatibility.required_new_role_ids
        ),
        complete_role_binding=complete_roles,
        split_rosters_disjoint=split_disjoint,
        compatible=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def runtime_config(operation: MarginStructuralRecurrenceForecastRuntimeOperation) -> MarginStructuralRecurrenceForecastRuntimeConfig:
    return MarginStructuralRecurrenceForecastRuntimeConfig(
        config_id=f"'independent-substrate-grounding.margin-structural-recurrence-forecast.'{operation.value.lower().replace('_', '-')}.config",
        operation=operation,
        capability_key=_CAPABILITY_KEY_BY_OPERATION[operation],
        capability_version=MARGIN_FORECAST_CAPABILITY_VERSION,
        method_source_manifest_sha256=method_source_manifest().fingerprint(),
        maximum_input_bytes=MAX_MARGIN_FORECAST_RUNTIME_BYTES,
    )


def _decode_config(payload: bytes, operation: MarginStructuralRecurrenceForecastRuntimeOperation) -> MarginStructuralRecurrenceForecastRuntimeConfig:
    value = decode_canonical_bytes(
        payload,
        MarginStructuralRecurrenceForecastRuntimeConfig,
        maximum_bytes=64 * 1024,
    )
    if value.operation is not operation:
        raise ValueError('structural recurrence runtime config selects another operation')
    return value


def registered_conformance(config_payload: bytes, request_payload: bytes) -> bytes:
    config = _decode_config(config_payload, MarginStructuralRecurrenceForecastRuntimeOperation.CONFORMANCE)
    request = decode_canonical_bytes(
        request_payload,
        MarginStructuralRecurrenceForecastConformanceRequest,
        maximum_bytes=config.maximum_input_bytes,
    )
    return execute_conformance(
        conformance_id=request.conformance_id,
        method_freeze=request.method_freeze,
        execution_authority=request.execution_authority,
    ).canonical_bytes()


def registered_predict(config_payload: bytes, request_payload: bytes) -> bytes:
    config = _decode_config(config_payload, MarginStructuralRecurrenceForecastRuntimeOperation.PREDICT)
    request = decode_canonical_bytes(
        request_payload,
        MarginStructuralRecurrenceForecastPredictionRequest,
        maximum_bytes=config.maximum_input_bytes,
    )
    return build_prediction_issue(
        issue_id=request.issue_id,
        method_freeze=request.method_freeze,
        conformance=request.conformance,
        design=request.design,
        development=request.development,
        evaluation_count=request.evaluation_count,
        prospective_validation_count=request.prospective_validation_count,
        scientific_bootstrap_inputs=request.scientific_bootstrap_inputs,
    ).canonical_bytes()


def registered_match(config_payload: bytes, request_payload: bytes) -> bytes:
    config = _decode_config(config_payload, MarginStructuralRecurrenceForecastRuntimeOperation.MATCH)
    request = decode_canonical_bytes(
        request_payload,
        MarginStructuralRecurrenceForecastMatchRequest,
        maximum_bytes=config.maximum_input_bytes,
    )
    return match_target(
        prediction=request.prediction,
        admission_handoff=request.admission_handoff,
        result=request.result,
    ).canonical_bytes()


def registered_adjudicate(config_payload: bytes, request_payload: bytes) -> bytes:
    config = _decode_config(config_payload, MarginStructuralRecurrenceForecastRuntimeOperation.ADJUDICATE)
    request = decode_canonical_bytes(
        request_payload,
        MarginStructuralRecurrenceForecastAdjudicationRequest,
        maximum_bytes=config.maximum_input_bytes,
    )
    return adjudicate(
        conformance=request.conformance,
        matches=request.matches,
    ).canonical_bytes()


def registered_compatibility_audit(config_payload: bytes, request_payload: bytes) -> bytes:
    config = _decode_config(
        config_payload,
        MarginStructuralRecurrenceForecastRuntimeOperation.TARGET_COMPATIBILITY_AUDIT,
    )
    request = decode_canonical_bytes(
        request_payload,
        MarginStructuralRecurrenceForecastCompatibilityAuditRequest,
        maximum_bytes=config.maximum_input_bytes,
    )
    return audit_target_compatibility(request).canonical_bytes()


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=512 * 1024**2,
        gpu_devices=0,
        wall_time_seconds=300,
        source_scan_bytes=MAX_MARGIN_FORECAST_RUNTIME_BYTES,
        output_bytes=MAX_MARGIN_FORECAST_RUNTIME_BYTES,
    )


def structural_recurrence_runtime_manifests(*, implementation_sha256: str) -> tuple[CapabilityManifest, ...]:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    specifications = (
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.CONFORMANCE,
            CapabilityKind.ANALYSIS,
            (MarginStructuralRecurrenceForecastConformanceRequest.SCHEMA,),
            (MarginStructuralRecurrenceForecastConformance.SCHEMA,),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.PRIVILEGED_TRUTH,
            ('margin-structural-recurrence-forecast-base-24-fixtures-13-controls', 'margin-structural-recurrence-forecast-margin-12-fixtures-10-controls'),
        ),
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.PREDICT,
            CapabilityKind.PROSPECTIVE_NOMINATOR,
            (MarginStructuralRecurrenceForecastPredictionRequest.SCHEMA,),
            (MarginStructuralRecurrenceForecastPredictionIssue.SCHEMA,),
            EvidenceCeiling.ADMISSION,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            ('margin-structural-recurrence-forecast-development-only', 'margin-structural-recurrence-forecast-no-cross-target-pooling'),
        ),
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.MATCH,
            CapabilityKind.HYPOTHESIS_ADJUDICATOR,
            (MarginStructuralRecurrenceForecastMatchRequest.SCHEMA,),
            (MarginStructuralRecurrenceForecastTargetMatch.SCHEMA,),
            EvidenceCeiling.CONTROLLER_USE,
            OutcomeAccess.EVALUATION_REVEALED,
            ('margin-structural-recurrence-forecast-exact-action-identity', 'margin-structural-recurrence-forecast-safety-noncompensating'),
        ),
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.ADJUDICATE,
            CapabilityKind.HYPOTHESIS_ADJUDICATOR,
            (MarginStructuralRecurrenceForecastAdjudicationRequest.SCHEMA,),
            (MarginStructuralRecurrenceForecastAdjudication.SCHEMA,),
            EvidenceCeiling.CONTROLLER_USE,
            OutcomeAccess.EVALUATION_REVEALED,
            ('margin-structural-recurrence-forecast-bounded-claim-ceiling', 'margin-structural-recurrence-forecast-no-independent-generality'),
        ),
        (
            MarginStructuralRecurrenceForecastRuntimeOperation.TARGET_COMPATIBILITY_AUDIT,
            CapabilityKind.ANALYSIS,
            (MarginStructuralRecurrenceForecastCompatibilityAuditRequest.SCHEMA,),
            (MarginStructuralRecurrenceForecastCompatibilityAudit.SCHEMA,),
            EvidenceCeiling.MEASUREMENT,
            OutcomeAccess.OUTCOME_BLIND,
            ('independent-substrate-grounding-native-role-completeness', 'independent-substrate-grounding-split-roster-disjointness'),
        ),
    )
    values = []
    for operation, kind, inputs, outputs, ceiling, access, checks in specifications:
        values.append(
            CapabilityManifest(
                capability_key=_CAPABILITY_KEY_BY_OPERATION[operation],
                capability_version=MARGIN_FORECAST_CAPABILITY_VERSION,
                kind=kind,
                config_schema=MarginStructuralRecurrenceForecastRuntimeConfig.SCHEMA,
                config_schema_sha256=sha256(MarginStructuralRecurrenceForecastRuntimeConfig.SCHEMA.encode("ascii")).hexdigest(),
                input_schema_ids=inputs,
                output_schema_ids=outputs,
                permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
                maximum_evidence_ceiling=ceiling,
                maximum_outcome_access=access,
                resource_ceiling=_budget(),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id='cpython-3.11-margin-structural-recurrence-forecast-semantic-port',
                requires_clean_commit=True,
                requires_active_mount=False,
                requires_network=False,
                conformance_check_ids=tuple(sorted(checks)),
                implementation_sha256=implementation_sha256,
            )
        )
    return tuple(sorted(values, key=lambda value: value.registry_id))


def structural_recurrence_runtime_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id='independent-substrate-grounding-margin-structural-recurrence-forecast-runtime',
        capabilities=structural_recurrence_runtime_manifests(
            implementation_sha256=implementation_sha256,
        ),
    )


def structural_recurrence_candidate_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key='independent-substrate-grounding.margin-structural-recurrence-forecast-runtime-provider',
            provider_version=MARGIN_FORECAST_CAPABILITY_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=64 * 1024,
        )
        for manifest in structural_recurrence_runtime_manifests(
            implementation_sha256=implementation_sha256,
        )
    )


__all__ = [
    'MAX_MARGIN_FORECAST_RUNTIME_BYTES',
    'MARGIN_FORECAST_CAPABILITY_VERSION',
    'MarginStructuralRecurrenceForecastAdjudicationRequest',
    'MarginStructuralRecurrenceForecastCompatibilityAudit',
    'MarginStructuralRecurrenceForecastCompatibilityAuditRequest',
    'MarginStructuralRecurrenceForecastConformanceRequest',
    'MarginStructuralRecurrenceForecastFrozenSourceFile',
    'MarginStructuralRecurrenceForecastMatchRequest',
    'MarginStructuralRecurrenceForecastMethodPortFreeze',
    'MarginStructuralRecurrenceForecastMethodSourceManifest',
    'MarginStructuralRecurrenceForecastPredictionRequest',
    'MarginStructuralRecurrenceForecastPortKind',
    'MarginStructuralRecurrenceForecastRuntimeConfig',
    'MarginStructuralRecurrenceForecastRuntimeOperation',
    'MarginStructuralRecurrenceForecastSemanticPortBinding',
    'MarginStructuralRecurrenceForecastSemanticPortInventory',
    "audit_target_compatibility",
    'method_source_manifest',
    'structural_recurrence_candidate_registrations',
    'structural_recurrence_runtime_manifests',
    'structural_recurrence_runtime_registry',
    "registered_adjudicate",
    "registered_compatibility_audit",
    "registered_conformance",
    "registered_match",
    "registered_predict",
    "runtime_config",
    "semantic_port_inventory",
]
