"""target-authoring qualification boundary no-science design-basis protocol for independent substrate grounding.

The protocol exercises ordinary compilation, graph parity, receipts, recovery
and semantic validation for all four external interface shapes.  Its records
contain only frozen method/design identities and boolean conformance facts.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateInterfaceShape, IndependentSubstrateConformanceReport, IndependentSubstrateScientificDesignBasis, IndependentSubstrateTargetKind
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)


INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION = "1.0.0"
INDEPENDENT_SUBSTRATE_VERIFY_KEY = "method.independent-substrate-grounding.verify-design-basis"
INDEPENDENT_SUBSTRATE_FREEZE_KEY = "method.independent-substrate-grounding.freeze-target-authoring"


class IndependentSubstrateDesignBasisOperation(StrEnum):
    VERIFY_INTERFACE_SHAPE = "VERIFY_INTERFACE_SHAPE"
    FREEZE_TARGET_AUTHORING = "FREEZE_TARGET_AUTHORING"


@dataclass(frozen=True, slots=True)
class IndependentSubstrateDesignBasisRuntimeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-design-basis-runtime-config'

    config_id: str
    operation: IndependentSubstrateDesignBasisOperation
    capability_key: str
    capability_version: str
    design_basis_sha256: str
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.design_basis_sha256, field_name="design_basis_sha256")
        expected = {
            IndependentSubstrateDesignBasisOperation.VERIFY_INTERFACE_SHAPE: INDEPENDENT_SUBSTRATE_VERIFY_KEY,
            IndependentSubstrateDesignBasisOperation.FREEZE_TARGET_AUTHORING: INDEPENDENT_SUBSTRATE_FREEZE_KEY,
        }[self.operation]
        if self.capability_key != expected:
            raise ValueError("independent substrate grounding design-basis operation and capability differ")
        if self.capability_version != INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION:
            raise ValueError("independent substrate grounding design-basis capability version differs")
        if not 0 < self.maximum_input_bytes <= 1024 * 1024:
            raise ValueError("independent substrate grounding design-basis input bound differs")


def decode_design_basis_config(payload: bytes) -> IndependentSubstrateDesignBasisRuntimeConfig:
    return decode_canonical_bytes(
        payload,
        IndependentSubstrateDesignBasisRuntimeConfig,
        maximum_bytes=64 * 1024,
    )


@dataclass(frozen=True, slots=True)
class IndependentSubstrateDesignBasisVerification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-design-basis-verification'

    verification_id: str
    target_slot: IndependentSubstrateTargetKind
    interface_shape: IndependentSubstrateInterfaceShape
    design_basis: ObjectIdentity
    ordinary_dag_shape_supported: bool
    receipt_recovery_supported: bool
    cpaim_surface_present: bool
    nature_surface_present: bool
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.verification_id, field_name="verification_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = all(
            (
                self.ordinary_dag_shape_supported,
                self.receipt_recovery_supported,
                not self.cpaim_surface_present,
                not self.nature_surface_present,
                not self.reason_codes,
            )
        )
        if self.passed != expected:
            raise ValueError("design-basis verification status is not fact-derived")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateTargetAuthoringManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-target-authoring-manifest'

    manifest_id: str
    design_basis: ObjectIdentity
    method_port_freeze: ObjectIdentity
    applicability_overlay: ObjectIdentity
    interface_conformance: ObjectIdentity
    interface_conformance_case_count: int
    interface_conformance_all_passed: bool
    verifications: tuple[ObjectIdentity, ...]
    allowed_target_slots: tuple[IndependentSubstrateTargetKind, ...]
    source_qualification_unlocked: bool
    target_issue_unlocked: bool
    cpaim_in_scope: bool
    nature_work_in_scope: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        require_sorted_unique_ids(
            self.verifications,
            attribute="object_id",
            field_name="verifications",
        )
        if len(self.verifications) != 4:
            raise ValueError("target-authoring manifest requires all four shape verifications")
        if (
            self.interface_conformance.object_schema != IndependentSubstrateConformanceReport.SCHEMA
            or self.interface_conformance_case_count != 14
            or not self.interface_conformance_all_passed
        ):
            raise ValueError("target-authoring manifest requires the passed truth-known interface conformance design report")
        if self.allowed_target_slots != tuple(IndependentSubstrateTargetKind):
            raise ValueError("target-authoring slot roster differs")
        if not self.source_qualification_unlocked or self.target_issue_unlocked:
            raise ValueError("target-authoring qualification boundary unlocks qualification, never target issue")
        if self.cpaim_in_scope or self.nature_work_in_scope:
            raise ValueError("target-authoring qualification boundary cannot reactivate deferred scope")


def design_basis_runtime_config(
    operation: IndependentSubstrateDesignBasisOperation,
    *,
    design_basis_sha256: str,
) -> IndependentSubstrateDesignBasisRuntimeConfig:
    key = {
        IndependentSubstrateDesignBasisOperation.VERIFY_INTERFACE_SHAPE: INDEPENDENT_SUBSTRATE_VERIFY_KEY,
        IndependentSubstrateDesignBasisOperation.FREEZE_TARGET_AUTHORING: INDEPENDENT_SUBSTRATE_FREEZE_KEY,
    }[operation]
    return IndependentSubstrateDesignBasisRuntimeConfig(
        config_id=f"independent-substrate-grounding.design-basis.{operation.value.lower().replace('_', '-')}.config",
        operation=operation,
        capability_key=key,
        capability_version=INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION,
        design_basis_sha256=design_basis_sha256,
        maximum_input_bytes=1024 * 1024,
    )


def freeze_target_authoring_manifest(
    *,
    manifest_id: str,
    design_basis: ObjectIdentity,
    method_port_freeze: ObjectIdentity,
    applicability_overlay: ObjectIdentity,
    interface_conformance: IndependentSubstrateConformanceReport,
    verifications: tuple[ObjectIdentity, ...],
) -> IndependentSubstrateTargetAuthoringManifest:
    """Bind a passed truth-known interface conformance report without duplicating its canonical payload."""

    if not interface_conformance.all_passed or interface_conformance.design_basis != design_basis:
        raise ValueError("truth-known interface conformance conformance does not close the exact design basis")
    return IndependentSubstrateTargetAuthoringManifest(
        manifest_id=manifest_id,
        design_basis=design_basis,
        method_port_freeze=method_port_freeze,
        applicability_overlay=applicability_overlay,
        interface_conformance=ObjectIdentity.from_record(
            interface_conformance.report_id,
            interface_conformance,
        ),
        interface_conformance_case_count=len(interface_conformance.case_results),
        interface_conformance_all_passed=interface_conformance.all_passed,
        verifications=verifications,
        allowed_target_slots=tuple(IndependentSubstrateTargetKind),
        source_qualification_unlocked=True,
        target_issue_unlocked=False,
        cpaim_in_scope=False,
        nature_work_in_scope=False,
    )


def _resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=128 * 1024**2,
        gpu_devices=0,
        wall_time_seconds=5,
        source_scan_bytes=1024 * 1024,
        output_bytes=64 * 1024,
    )


def independent_substrate_design_basis_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    manifests = (
        CapabilityManifest(
            capability_key=INDEPENDENT_SUBSTRATE_FREEZE_KEY,
            capability_version=INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION,
            kind=CapabilityKind.ANALYSIS,
            config_schema=IndependentSubstrateDesignBasisRuntimeConfig.SCHEMA,
            config_schema_sha256=sha256(
                IndependentSubstrateDesignBasisRuntimeConfig.SCHEMA.encode("ascii")
            ).hexdigest(),
            input_schema_ids=(IndependentSubstrateDesignBasisVerification.SCHEMA,),
            output_schema_ids=(IndependentSubstrateTargetAuthoringManifest.SCHEMA,),
            permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
            maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
            maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
            resource_ceiling=_resources(),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11-independent-substrate-grounding-design-basis",
            requires_clean_commit=True,
            requires_active_mount=False,
            requires_network=False,
            conformance_check_ids=(
                "cpaim-and-nature-excluded",
                "source-qualification-only-unlock",
            ),
            implementation_sha256=implementation_sha256,
        ),
        CapabilityManifest(
            capability_key=INDEPENDENT_SUBSTRATE_VERIFY_KEY,
            capability_version=INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION,
            kind=CapabilityKind.ANALYSIS,
            config_schema=IndependentSubstrateDesignBasisRuntimeConfig.SCHEMA,
            config_schema_sha256=sha256(
                IndependentSubstrateDesignBasisRuntimeConfig.SCHEMA.encode("ascii")
            ).hexdigest(),
            input_schema_ids=(IndependentSubstrateScientificDesignBasis.SCHEMA,),
            output_schema_ids=(IndependentSubstrateDesignBasisVerification.SCHEMA,),
            permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
            maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
            maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
            resource_ceiling=_resources(),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11-independent-substrate-grounding-design-basis",
            requires_clean_commit=True,
            requires_active_mount=False,
            requires_network=False,
            conformance_check_ids=(
                "four-external-interface-shapes",
                "no-scientific-target-payload",
            ),
            implementation_sha256=implementation_sha256,
        ),
    )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-design-basis-runtime",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def independent_substrate_design_basis_candidate_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key="independent-substrate-grounding.design-basis-runtime-provider",
            provider_version=INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=64 * 1024,
        )
        for manifest in independent_substrate_design_basis_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


_SHAPES = (
    (
        "verify-boptest-action-ledger",
        IndependentSubstrateTargetKind.BOPTEST_BRIDGE,
        IndependentSubstrateInterfaceShape.ACTION_LEDGER,
    ),
    ("verify-freegsnke-continuous", IndependentSubstrateTargetKind.FREEGSNKE, IndependentSubstrateInterfaceShape.VARIABLE_LEVEL),
    (
        "verify-grid2op-irregular-graph",
        IndependentSubstrateTargetKind.GRID2OP,
        IndependentSubstrateInterfaceShape.IRREGULAR_GRAPH,
    ),
    (
        "verify-nrel-continuous-archive",
        IndependentSubstrateTargetKind.NREL_INVERTER,
        IndependentSubstrateInterfaceShape.CONTINUOUS_PHYSICAL_ARCHIVE,
    ),
)


def build_independent_substrate_design_basis_protocol(
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    step_ids = {value[0] for value in _SHAPES} | {"freeze-target-authoring"}
    if set(config_by_step_id) != step_ids:
        raise ValueError("independent substrate grounding design-basis config roster differs")
    steps = []
    for step_id, _slot, _shape in _SHAPES:
        manifest = registry.resolve(INDEPENDENT_SUBSTRATE_VERIFY_KEY, INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION)
        config = config_by_step_id[step_id]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError("independent substrate grounding design-basis config differs from capability schema")
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=ScientificStage.QUALIFY,
                capability_key=INDEPENDENT_SUBSTRATE_VERIFY_KEY,
                capability_version=INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION,
                config=config,
                dependency_step_ids=(),
                outputs=(
                    OutputTemplate(
                        output_id=f"{step_id}.record",
                        payload_schema=IndependentSubstrateDesignBasisVerification.SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    ),
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                resource_budget=ResourceBudget(
                    cpu_cores=1,
                    memory_bytes=1_000_000,
                    gpu_devices=0,
                    wall_time_seconds=2,
                    source_scan_bytes=1000,
                    output_bytes=1000,
                ),
                resource_lock_ids=(f"independent-substrate-grounding-{step_id}",),
                barrier=BarrierKind.NONE,
                maximum_attempts=2,
                obligation_ids=(f"independent-substrate-grounding-{step_id}-contract",),
            )
        )
    freeze_manifest = registry.resolve(INDEPENDENT_SUBSTRATE_FREEZE_KEY, INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION)
    freeze_config = config_by_step_id["freeze-target-authoring"]
    if (
        freeze_config.config_schema != freeze_manifest.config_schema
        or freeze_config.config_schema_sha256 != freeze_manifest.config_schema_sha256
    ):
        raise ValueError("independent substrate grounding design-basis config differs from capability schema")
    steps.append(
        ProtocolStepTemplate(
            step_id="freeze-target-authoring",
            stage=ScientificStage.FREEZE,
            capability_key=INDEPENDENT_SUBSTRATE_FREEZE_KEY,
            capability_version=INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION,
            config=freeze_config,
            dependency_step_ids=tuple(value[0] for value in _SHAPES),
            outputs=(
                OutputTemplate(
                    output_id="freeze-target-authoring.record",
                    payload_schema=IndependentSubstrateTargetAuthoringManifest.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    filename_suffix=".json",
                ),
            ),
            required_permissions=freeze_manifest.permissions,
            requested_outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=ResourceBudget(
                cpu_cores=1,
                memory_bytes=1_000_000,
                gpu_devices=0,
                wall_time_seconds=2,
                source_scan_bytes=6000,
                output_bytes=4000,
            ),
            resource_lock_ids=("independent-substrate-grounding-freeze-target-authoring",),
            barrier=BarrierKind.FREEZE,
            maximum_attempts=2,
            obligation_ids=("independent-substrate-grounding-freeze-target-authoring-contract",),
        )
    )
    return ProtocolTemplate(
        template_id="independent-substrate-grounding-design-basis-no-science-protocol",
        template_version=INDEPENDENT_SUBSTRATE_PROTOCOL_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


class _IndependentSubstrateDesignBasisRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        records_by_task_id: dict[str, CanonicalRecord],
    ) -> None:
        self.manifest = manifest
        self.records_by_task_id = records_by_task_id
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        if not context.input_ports:
            raise ValueError("independent substrate grounding design-basis runner requires frozen inputs")
        record = self.records_by_task_id[context.task_id]
        if (
            len(context.output_ports) != 1
            or context.output_ports[0].payload_schema != record.SCHEMA
        ):
            raise ValueError("independent substrate grounding design-basis output differs from its task")
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=record.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck("independent-substrate-grounding-design-basis-runtime-contract", True, ()),),
        )


class IndependentSubstrateDesignBasisRuntimeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        records_by_task_id: dict[str, CanonicalRecord],
    ) -> None:
        expected = independent_substrate_design_basis_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("independent substrate grounding design-basis provider registry differs")
        expected_tasks = {value[0] for value in _SHAPES} | {"freeze-target-authoring"}
        if set(records_by_task_id) != expected_tasks:
            raise ValueError("independent substrate grounding design-basis provider task roster differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        by_key = {
            INDEPENDENT_SUBSTRATE_VERIFY_KEY: {
                task_id: record
                for task_id, record in records_by_task_id.items()
                if task_id != "freeze-target-authoring"
            },
            INDEPENDENT_SUBSTRATE_FREEZE_KEY: {
                "freeze-target-authoring": records_by_task_id["freeze-target-authoring"]
            },
        }
        self._runners = tuple(
            _IndependentSubstrateDesignBasisRunner(manifest, by_key[manifest.capability_key])
            for manifest in registry.capabilities
        )
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("independent substrate grounding design-basis provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("independent substrate grounding design-basis provider plan/source differs")
        return ()

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("independent substrate grounding design-basis semantic registry differs")
        record_types = {
            IndependentSubstrateDesignBasisVerification.SCHEMA: IndependentSubstrateDesignBasisVerification,
            IndependentSubstrateTargetAuthoringManifest.SCHEMA: IndependentSubstrateTargetAuthoringManifest,
        }
        values = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                record_type = record_types[schema]
                values.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(sorted(field.name for field in fields(record_type))),
                    )
                )
        return tuple(sorted(values, key=lambda value: (value.capability_key, value.payload_schema)))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("independent substrate grounding design-basis adjudication registry differs")
        return None


__all__ = [
    'IndependentSubstrateDesignBasisOperation',
    'IndependentSubstrateDesignBasisRuntimeConfig',
    'IndependentSubstrateDesignBasisRuntimeProvider',
    'IndependentSubstrateDesignBasisVerification',
    'IndependentSubstrateTargetAuthoringManifest',
    'build_independent_substrate_design_basis_protocol',
    "decode_design_basis_config",
    "design_basis_runtime_config",
    "freeze_target_authoring_manifest",
    'independent_substrate_design_basis_candidate_registrations',
    'independent_substrate_design_basis_registry',
]
