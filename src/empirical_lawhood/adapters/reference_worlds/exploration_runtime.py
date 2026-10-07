"""Static runtime provider for the committed outcome-visible reference wave."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256

from empirical_lawhood.adapters.exploration.execution import (
    AutomaticSkeptic,
    HypothesisSynthesizer,
    RegisteredAnalysisExecutor,
)
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.discovery import SkepticReport
from empirical_lawhood.planning.exploration import (
    AnalysisAttempt,
    AnalysisSpec,
    ExplorationPlan,
    ExploratoryFinding,
    HypothesisSet,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.exploration import (
    ExplorationWaveInput,
    ExplorationWaveResult,
    validate_exploration_wave_result,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationOutputContract


REFERENCE_DUAL_LOOP_SOURCE = b"reference outcome-visible summary"
REFERENCE_EXPLORATION_SOURCE = b"reference executable exploration input\n"
REFERENCE_EXPLORATION_SOURCE_SCHEMA = (
    'empirical-lawhood/reference-worlds/exploration-source'
)


def _reference_exploration_budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=128_000_000,
        gpu_devices=0,
        wall_time_seconds=60,
        source_scan_bytes=1_000_000,
        output_bytes=1_000_000,
    )


def input_derived_exploration_execution_registry(
    *,
    registry_id: str,
    source_schema_ids: tuple[str, ...],
    resource_ceiling: ResourceBudget | None = None,
) -> CapabilityRegistry:
    """Build the static registry shared by maintained input-derived waves."""

    if (
        tuple(sorted(set(source_schema_ids))) != source_schema_ids
        or not source_schema_ids
    ):
        raise ValueError("input-derived exploration schemas must be sorted and unique")
    budget = resource_ceiling or _reference_exploration_budget()

    permissions = tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            )
        )
    )

    def manifest(
        *,
        capability_key: str,
        capability_version: str,
        kind: CapabilityKind,
        config_schema: str,
        input_schema_ids: tuple[str, ...],
        output_schema_ids: tuple[str, ...],
        conformance_check_ids: tuple[str, ...],
        implementation_bytes: bytes,
    ) -> CapabilityManifest:
        return CapabilityManifest(
            capability_key=capability_key,
            capability_version=capability_version,
            kind=kind,
            config_schema=config_schema,
            config_schema_sha256=sha256(config_schema.encode("utf-8")).hexdigest(),
            input_schema_ids=input_schema_ids,
            output_schema_ids=output_schema_ids,
            permissions=permissions,
            maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            resource_ceiling=budget,
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=conformance_check_ids,
            implementation_sha256=sha256(implementation_bytes).hexdigest(),
        )

    values = (
        manifest(
            capability_key="exploration.pipeline",
            capability_version="1.0.0",
            kind=CapabilityKind.ANALYSIS,
            config_schema=AnalysisSpec.SCHEMA,
            input_schema_ids=tuple(
                sorted(
                    (
                        ExplorationWaveInput.SCHEMA,
                        *source_schema_ids,
                    )
                )
            ),
            output_schema_ids=(ExploratoryFinding.SCHEMA,),
            conformance_check_ids=("input-derived-exploration-analysis",),
            implementation_bytes=b'input-derived-exploration-analysis',
        ),
        manifest(
            capability_key="exploration.skeptic",
            capability_version="1.0.0",
            kind=CapabilityKind.ANALYSIS,
            config_schema=ExplorationPlan.SCHEMA,
            input_schema_ids=tuple(
                sorted((ExplorationWaveInput.SCHEMA, ExploratoryFinding.SCHEMA))
            ),
            output_schema_ids=(SkepticReport.SCHEMA,),
            conformance_check_ids=("input-derived-complete-skeptic-family",),
            implementation_bytes=b'input-derived-exploration-skeptic',
        ),
        manifest(
            capability_key="exploration.synthesis",
            capability_version="1.0.0",
            kind=CapabilityKind.HYPOTHESIS_SYNTHESIZER,
            config_schema=ExplorationPlan.SCHEMA,
            input_schema_ids=tuple(
                sorted(
                    (
                        ExplorationWaveInput.SCHEMA,
                        ExploratoryFinding.SCHEMA,
                        SkepticReport.SCHEMA,
                    )
                )
            ),
            output_schema_ids=tuple(
                sorted((ExplorationWaveResult.SCHEMA, HypothesisSet.SCHEMA))
            ),
            conformance_check_ids=("input-derived-receipt-bound-synthesis",),
            implementation_bytes=b'input-derived-exploration-synthesis',
        ),
    )
    return CapabilityRegistry(
        registry_id=registry_id,
        capabilities=values,
    )


def reference_exploration_execution_registry() -> CapabilityRegistry:
    """Return the static three-capability registry for executable reference waves."""

    return input_derived_exploration_execution_registry(
        registry_id="reference-exploration-execution-registry",
        source_schema_ids=(REFERENCE_EXPLORATION_SOURCE_SCHEMA,),
    )


REFERENCE_EXPLORATION_EXECUTION_REGISTRY_SHA256 = (
    reference_exploration_execution_registry().fingerprint()
)


class ReferenceExplorationRunner:
    """Emit only the prebound, non-promotable reference records."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        findings: tuple[ExploratoryFinding, ...],
        hypotheses: HypothesisSet,
    ) -> None:
        self.manifest = manifest
        self.findings = findings
        self.hypotheses = hypotheses

    def execute(self, context: TaskContext) -> RunnerResult:
        if self.manifest.capability_key == "exploration.synthesis":
            record: CanonicalRecord = self.hypotheses
        else:
            matches = tuple(
                finding
                for finding in self.findings
                if any(
                    attempt.analysis_id in context.task_id
                    for attempt in finding.attempts
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    "reference exploration task has no unique bound finding"
                )
            record = matches[0]
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id, payload=record.canonical_bytes()
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("outcome-visible-lineage-retained", True, ()),
                ReceiptCheck("scientific-promotion-authority-absent", True, ()),
            ),
        )


class ReferenceExplorationRuntimeProvider(CampaignRuntimeProvider):
    """Read-only truth-known exploration provider with no approval capability."""

    capability_count = 2

    def __init__(self, *, registry: CapabilityRegistry) -> None:
        self.registry_sha256 = registry.fingerprint()
        self._validate_registry(registry)

    def _validate_registry(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("reference exploration registry identity differs")
        if tuple(item.capability_key for item in registry.capabilities) != (
            "exploration.pipeline",
            "exploration.synthesis",
        ):
            raise ValueError("reference exploration capability set differs")

    @staticmethod
    def _records(
        source_records: tuple[CanonicalRecord, ...],
    ) -> tuple[tuple[ExploratoryFinding, ...], HypothesisSet]:
        findings = tuple(
            sorted(
                (
                    item
                    for item in source_records
                    if isinstance(item, ExploratoryFinding)
                ),
                key=lambda item: item.finding_id,
            )
        )
        hypotheses = tuple(
            item for item in source_records if isinstance(item, HypothesisSet)
        )
        if not findings or len(hypotheses) != 1:
            raise ValueError("reference exploration source records are incomplete")
        return findings, hypotheses[0]

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._validate_registry(registry)
        findings, hypotheses = self._records(source_records)
        return tuple(
            ReferenceExplorationRunner(manifest, findings, hypotheses)
            for manifest in registry.capabilities
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError("reference exploration plan identity differs")
        configs = {
            item.fingerprint(): item
            for item in source_records
            if isinstance(item, (AnalysisSpec, ExplorationPlan))
        }
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        values: list[ExternalInputPayload] = []
        for artifact_id in sorted(specs):
            spec = specs[artifact_id]
            if spec.expected_content_sha256 is None:
                raise ValueError("reference exploration input lacks a content identity")
            config = configs.get(spec.expected_content_sha256)
            if config is None:
                if artifact_id != "reference-outcome-visible-summary":
                    raise ValueError("reference exploration input is not registered")
                if spec.expected_payload_schema is None:
                    raise ValueError(
                        "reference exploration source lacks a payload schema"
                    )
                payload = REFERENCE_DUAL_LOOP_SOURCE
                payload_schema = spec.expected_payload_schema
                profile = ArtifactProfile.TEXT_PARAMETERS
                media_type = "text/plain"
                logical_sha256 = None
            else:
                payload = config.canonical_bytes()
                payload_schema = config.SCHEMA
                profile = ArtifactProfile.CANONICAL_JSON
                media_type = "application/json"
                logical_sha256 = config.fingerprint()
            parent_record = spec if config is None else config
            parent_id = spec.input_id if config is None else artifact_id
            parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(parent_id, parent_record),
                visibility_ceiling=(
                    spec.expected_visibility_ceiling
                    or VisibilityCeiling.OUTCOME_VISIBLE
                ),
                outcome_access=(
                    spec.expected_outcome_access or OutcomeAccess.EVALUATION_REVEALED
                ),
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=payload_schema,
                    profile=profile,
                    media_type=media_type,
                    payload=payload,
                    visibility_ceiling=(
                        spec.expected_visibility_ceiling
                        or VisibilityCeiling.OUTCOME_VISIBLE
                    ),
                    outcome_access=(
                        spec.expected_outcome_access
                        or OutcomeAccess.EVALUATION_REVEALED
                    ),
                    parent_visibility_ceilings=(parent.visibility_ceiling,),
                    lineage_parents=(parent,),
                    logical_content_sha256=logical_sha256,
                )
            )
        return tuple(values)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._validate_registry(registry)
        record_types = {
            ExploratoryFinding.SCHEMA: ExploratoryFinding,
            HypothesisSet.SCHEMA: HypothesisSet,
        }
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                manifest,
                payload_schema=payload_schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(field.name for field in fields(record_types[payload_schema]))
                ),
            )
            for manifest in registry.capabilities
            for payload_schema in manifest.output_schema_ids
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        self._validate_registry(registry)
        return None


@dataclass(frozen=True, slots=True)
class RegisteredExplorationInput:
    """One exact non-record input statically owned by a runtime provider."""

    logical_artifact_id: str
    payload_schema: str
    profile: ArtifactProfile
    media_type: str
    payload: bytes


def _record_payloads(
    source_records: tuple[CanonicalRecord, ...],
    registered_inputs: tuple[RegisteredExplorationInput, ...],
) -> dict[str, bytes]:
    values: dict[str, bytes] = {
        value.logical_artifact_id: value.payload for value in registered_inputs
    }
    if len(values) != len(registered_inputs):
        raise ValueError("registered exploration input identity is duplicated")
    for record in source_records:
        if isinstance(record, ExplorationWaveInput):
            artifact_id = record.wave_input_id
        elif isinstance(record, AnalysisSpec):
            artifact_id = f"analysis-spec.{record.analysis_id}"
        elif isinstance(record, ExplorationPlan):
            artifact_id = f"exploration-plan.{record.plan_id}"
        else:
            raise ValueError(
                "executable exploration source records contain an output or unknown type"
            )
        if artifact_id in values:
            raise ValueError(
                "executable exploration source record identity is duplicated"
            )
        values[artifact_id] = record.canonical_bytes()
    return values


class ReferenceExplorationExecutionRunner:
    """Derive one task output from verified input and dependency ports."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        source_payloads: dict[str, bytes],
    ) -> None:
        self.manifest = manifest
        self.source_payloads = source_payloads

    def _read_inputs(
        self,
        context: TaskContext,
    ) -> dict[str, tuple[bytes, str, str]]:
        payloads: dict[str, tuple[bytes, str, str]] = {}
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError("exploration worker input size differs")
            if port.artifact_id in context.external_input_artifact_ids:
                try:
                    expected = self.source_payloads[port.artifact_id]
                except KeyError as error:
                    raise ValueError(
                        "exploration task received an unknown source input"
                    ) from error
                if payload != expected:
                    raise ValueError(
                        "exploration worker input differs from its frozen bytes"
                    )
            payloads[port.artifact_id] = (
                payload,
                port.materialization_id,
                port.payload_schema,
            )
        if set(payloads) != {port.artifact_id for port in context.input_ports}:
            raise ValueError("exploration worker inputs do not resolve exactly once")
        return payloads

    @staticmethod
    def _wave_input(
        context: TaskContext,
        payloads: dict[str, tuple[bytes, str, str]],
    ) -> ExplorationWaveInput:
        wave_ports = tuple(
            port
            for port in context.input_ports
            if port.payload_schema == ExplorationWaveInput.SCHEMA
            and port.artifact_id in context.external_input_artifact_ids
        )
        if len(wave_ports) != 1:
            raise ValueError("exploration task lacks one frozen wave input")
        port = wave_ports[0]
        payload = payloads[port.artifact_id][0]
        return decode_canonical_bytes(
            payload,
            ExplorationWaveInput,
            maximum_bytes=port.size_bytes,
        )

    @staticmethod
    def _receipt_bindings(
        context: TaskContext,
        expected_task_ids: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        bindings = {value.task_id: value for value in context.dependency_receipts}
        if tuple(sorted(bindings)) != tuple(sorted(expected_task_ids)):
            raise ValueError("exploration task lacks complete dependency receipts")
        for task_id, binding in bindings.items():
            if not binding.receipt_id.startswith(
                f"receipt.{context.run_id}.{task_id}.attempt-"
            ):
                raise ValueError("exploration dependency receipt identity differs")
        return {
            task_id: binding.output_materialization_ids
            for task_id, binding in bindings.items()
        }

    @staticmethod
    def _receipt_payload(
        *,
        task_id: str,
        artifact_id: str,
        payload_schema: str,
        bindings: dict[str, tuple[str, ...]],
        payloads: dict[str, tuple[bytes, str, str]],
    ) -> bytes:
        try:
            payload, materialization_id, observed_schema = payloads[artifact_id]
            materialization_ids = bindings[task_id]
        except KeyError as error:
            raise ValueError(
                "receipt-named exploration dependency is absent"
            ) from error
        if (
            materialization_id not in materialization_ids
            or observed_schema != payload_schema
        ):
            raise ValueError("exploration dependency differs from its receipt binding")
        return payload

    @staticmethod
    def _findings(
        wave_input: ExplorationWaveInput,
        bindings: dict[str, tuple[str, ...]],
        payloads: dict[str, tuple[bytes, str, str]],
    ) -> tuple[ExploratoryFinding, ...]:
        findings: list[ExploratoryFinding] = []
        for instantiation in wave_input.instantiations:
            proposal = instantiation.proposal
            analysis_id = proposal.analysis.analysis_id
            task_id = f"explore.{analysis_id}"
            artifact_id = f"finding.{wave_input.plan.plan_id}.{analysis_id}"
            payload = ReferenceExplorationExecutionRunner._receipt_payload(
                task_id=task_id,
                artifact_id=artifact_id,
                payload_schema=ExploratoryFinding.SCHEMA,
                bindings=bindings,
                payloads=payloads,
            )
            finding = decode_canonical_bytes(
                payload,
                ExploratoryFinding,
                maximum_bytes=len(payload),
            )
            if (
                finding.plan_id != wave_input.plan.plan_id
                or finding.proposal_id != proposal.proposal_id
                or {attempt.analysis_id for attempt in finding.attempts}
                != {analysis_id}
            ):
                raise ValueError("receipt-bound finding changes its frozen analysis")
            findings.append(finding)
        return tuple(sorted(findings, key=lambda value: value.finding_id))

    @staticmethod
    def _attempts(
        findings: tuple[ExploratoryFinding, ...],
    ) -> tuple[AnalysisAttempt, ...]:
        return tuple(
            sorted(
                (attempt for finding in findings for attempt in finding.attempts),
                key=lambda value: value.attempt_id,
            )
        )

    def _analysis_output(
        self,
        context: TaskContext,
        wave_input: ExplorationWaveInput,
    ) -> ExploratoryFinding:
        matches = tuple(
            value
            for value in wave_input.instantiations
            if context.config.content_sha256 == value.proposal.analysis.fingerprint()
        )
        if len(matches) != 1:
            raise ValueError("analysis task does not bind one selected instantiation")
        instantiation = matches[0]
        analysis_id = instantiation.proposal.analysis.analysis_id
        relevant = tuple(
            value
            for value in wave_input.observations
            if value.analysis_id == analysis_id
        )
        _, finding = RegisteredAnalysisExecutor().execute_proposal(
            wave_input.snapshot,
            wave_input.plan,
            instantiation,
            relevant,
        )
        return finding

    def execute(self, context: TaskContext) -> RunnerResult:
        payloads = self._read_inputs(context)
        wave_input = self._wave_input(context, payloads)
        outputs: dict[str, CanonicalRecord]
        if self.manifest.capability_key == "exploration.pipeline":
            if context.dependency_receipts:
                raise ValueError("analysis task cannot consume dependency receipts")
            record = self._analysis_output(context, wave_input)
            outputs = {context.output_ports[0].output_id: record}
        else:
            analysis_task_ids = tuple(
                sorted(
                    f"explore.{value.proposal.analysis.analysis_id}"
                    for value in wave_input.instantiations
                )
            )
            if self.manifest.capability_key == "exploration.skeptic":
                bindings = self._receipt_bindings(context, analysis_task_ids)
                findings = self._findings(wave_input, bindings, payloads)
                skeptic = AutomaticSkeptic().evaluate(
                    wave_input.plan,
                    wave_input.instantiations,
                    wave_input.observations,
                    findings,
                )
                outputs = {"skeptic-report": skeptic}
            elif self.manifest.capability_key == "exploration.synthesis":
                dependency_task_ids = tuple(
                    sorted((*analysis_task_ids, "exploration-skeptic"))
                )
                bindings = self._receipt_bindings(context, dependency_task_ids)
                findings = self._findings(wave_input, bindings, payloads)
                skeptic_payload = self._receipt_payload(
                    task_id="exploration-skeptic",
                    artifact_id=f"skeptic-report.{wave_input.plan.plan_id}",
                    payload_schema=SkepticReport.SCHEMA,
                    bindings=bindings,
                    payloads=payloads,
                )
                skeptic = decode_canonical_bytes(
                    skeptic_payload,
                    SkepticReport,
                    maximum_bytes=len(skeptic_payload),
                )
                synthesis = HypothesisSynthesizer().synthesize(
                    wave_input.plan,
                    findings,
                    skeptic,
                )
                wave_result = ExplorationWaveResult(
                    result_id=f"wave-result.{wave_input.plan.plan_id}",
                    plan=ObjectIdentity.from_record(
                        wave_input.plan.plan_id,
                        wave_input.plan,
                    ),
                    attempts=self._attempts(findings),
                    findings=findings,
                    skeptic_report=skeptic,
                    hypothesis_synthesis=synthesis,
                )
                validate_exploration_wave_result(wave_result, wave_input)
                outputs = {
                    "hypothesis-set": synthesis.hypothesis_set,
                    "wave-result": wave_result,
                }
            else:
                raise ValueError("unknown executable exploration capability")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=outputs[port.output_id].canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("expected-output-oracle-absent", True, ()),
                ReceiptCheck("receipt-bound-dependencies-complete", True, ()),
                ReceiptCheck("receipt-bound-dependencies-decoded", True, ()),
                ReceiptCheck("verified-wave-input-consumed", True, ()),
            ),
        )


class ReferenceExplorationExecutionRuntimeProvider(CampaignRuntimeProvider):
    """Input-derived, non-promotable runtime provider for maintained waves."""

    registry_sha256 = REFERENCE_EXPLORATION_EXECUTION_REGISTRY_SHA256
    capability_count = 3

    def __init__(
        self,
        *,
        registry: CapabilityRegistry | None = None,
        registered_inputs: tuple[RegisteredExplorationInput, ...] | None = None,
    ) -> None:
        self.registry = registry or reference_exploration_execution_registry()
        self.registry_sha256 = self.registry.fingerprint()
        self.registered_inputs = registered_inputs or (
            RegisteredExplorationInput(
                logical_artifact_id="reference-exploration-source",
                payload_schema=REFERENCE_EXPLORATION_SOURCE_SCHEMA,
                profile=ArtifactProfile.TEXT_PARAMETERS,
                media_type="text/plain",
                payload=REFERENCE_EXPLORATION_SOURCE,
            ),
        )
        identifiers = tuple(
            value.logical_artifact_id for value in self.registered_inputs
        )
        if tuple(sorted(set(identifiers))) != identifiers:
            raise ValueError("registered exploration inputs must be sorted and unique")

    def _validate_registry(self, registry: CapabilityRegistry) -> None:
        if registry != self.registry or registry.fingerprint() != self.registry_sha256:
            raise ValueError(
                "input-derived exploration execution registry identity differs"
            )
        if tuple(value.capability_key for value in registry.capabilities) != (
            "exploration.pipeline",
            "exploration.skeptic",
            "exploration.synthesis",
        ):
            raise ValueError(
                "input-derived exploration execution capability set differs"
            )

    @staticmethod
    def _wave_input(
        source_records: tuple[CanonicalRecord, ...],
    ) -> ExplorationWaveInput:
        values = tuple(
            value for value in source_records if isinstance(value, ExplorationWaveInput)
        )
        if len(values) != 1:
            raise ValueError(
                "input-derived exploration requires exactly one wave input"
            )
        if any(
            isinstance(
                value, (ExploratoryFinding, HypothesisSet, ExplorationWaveResult)
            )
            for value in source_records
        ):
            raise ValueError(
                "input-derived exploration runtime inputs contain expected outputs"
            )
        return values[0]

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._validate_registry(registry)
        self._wave_input(source_records)
        payloads = _record_payloads(source_records, self.registered_inputs)
        return tuple(
            ReferenceExplorationExecutionRunner(manifest, payloads)
            for manifest in registry.capabilities
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError(
                "input-derived exploration execution plan identity differs"
            )
        wave_input = self._wave_input(source_records)
        payloads = _record_payloads(source_records, self.registered_inputs)
        registered_inputs = {
            value.logical_artifact_id: value for value in self.registered_inputs
        }
        records_by_fingerprint = {
            value.fingerprint(): value for value in source_records
        }
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        values: list[ExternalInputPayload] = []
        for artifact_id in sorted(specs):
            spec = specs[artifact_id]
            try:
                payload = payloads[artifact_id]
            except KeyError as error:
                raise ValueError(
                    "reference exploration input is not registered"
                ) from error
            if spec.expected_content_sha256 is None:
                raise ValueError("reference exploration input lacks a content identity")
            if sha256(payload).hexdigest() != spec.expected_content_sha256:
                raise ValueError("reference exploration input content identity differs")
            record = records_by_fingerprint.get(spec.expected_content_sha256)
            if record is None:
                registered = registered_inputs.get(artifact_id)
                if (
                    registered is None
                    or spec.expected_payload_schema != registered.payload_schema
                ):
                    raise ValueError(
                        "input-derived exploration raw input is not recognized"
                    )
                payload_schema = registered.payload_schema
                profile = registered.profile
                media_type = registered.media_type
                logical_sha256 = None
                parent_record: CanonicalRecord = spec
                parent_id = spec.input_id
            else:
                payload_schema = record.SCHEMA
                profile = ArtifactProfile.CANONICAL_JSON
                media_type = "application/json"
                logical_sha256 = record.fingerprint()
                parent_record = record
                parent_id = artifact_id
            primary_parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(parent_id, parent_record),
                visibility_ceiling=(
                    spec.expected_visibility_ceiling
                    or VisibilityCeiling.OUTCOME_VISIBLE
                ),
                outcome_access=(
                    spec.expected_outcome_access or OutcomeAccess.EVALUATION_REVEALED
                ),
            )
            parents = [primary_parent]
            if spec.identity_scope_sha256 == wave_input.plan.fingerprint():
                parents.append(
                    ArtifactLineageParent(
                        identity=ObjectIdentity.from_record(
                            wave_input.plan.plan_id,
                            wave_input.plan,
                        ),
                        visibility_ceiling=wave_input.plan.visibility_ceiling,
                        outcome_access=wave_input.plan.outcome_access,
                    )
                )
            elif spec.identity_scope_sha256 == wave_input.snapshot.fingerprint():
                parents.append(
                    ArtifactLineageParent(
                        identity=ObjectIdentity.from_record(
                            wave_input.snapshot.snapshot_id,
                            wave_input.snapshot,
                        ),
                        visibility_ceiling=wave_input.snapshot.visibility_ceiling,
                        outcome_access=wave_input.snapshot.outcome_access,
                    )
                )
            lineage_parents = tuple(
                sorted(
                    parents,
                    key=lambda value: (
                        value.identity.object_id,
                        value.identity.object_schema,
                        value.identity.object_version,
                        value.identity.object_fingerprint,
                    ),
                )
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=payload_schema,
                    profile=profile,
                    media_type=media_type,
                    payload=payload,
                    visibility_ceiling=primary_parent.visibility_ceiling,
                    outcome_access=primary_parent.outcome_access,
                    parent_visibility_ceilings=tuple(
                        parent.visibility_ceiling for parent in lineage_parents
                    ),
                    lineage_parents=lineage_parents,
                    logical_content_sha256=logical_sha256,
                )
            )
        return tuple(values)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._validate_registry(registry)
        record_types = {
            ExploratoryFinding.SCHEMA: ExploratoryFinding,
            SkepticReport.SCHEMA: SkepticReport,
            HypothesisSet.SCHEMA: HypothesisSet,
            ExplorationWaveResult.SCHEMA: ExplorationWaveResult,
        }
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                manifest,
                payload_schema=payload_schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(field.name for field in fields(record_types[payload_schema]))
                ),
            )
            for manifest in registry.capabilities
            for payload_schema in manifest.output_schema_ids
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        self._validate_registry(registry)
        return None
