"""Deterministic runtime provider for the committed minimal reference campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
    encode_scientific_adjudication,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)


REFERENCE_ADJUDICATION_SCOPE_ID = "reference-campaign-plumbing"


@dataclass(frozen=True, slots=True)
class ReferenceAdjudicationFixture(CanonicalRecord):
    """Truth-known sealed input used only to validate adjudication plumbing."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-adjudication-fixture'

    fixture_id: str
    evaluability: AdjudicationEvaluability
    scientific_status: ScientificStatus
    admission_status: AdmissionStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.fixture_id, field_name="fixture_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.evaluability is AdjudicationEvaluability.UNEVALUABLE and (
            self.scientific_status is not ScientificStatus.UNEVALUABLE
            or self.admission_status is not AdmissionStatus.UNEVALUABLE
        ):
            raise ValueError(
                "unevaluable reference fixture must retain unevaluable states"
            )
        if self.evaluability is AdjudicationEvaluability.EVALUABLE and (
            self.scientific_status
            in {ScientificStatus.NOT_TESTED, ScientificStatus.UNEVALUABLE}
            or self.admission_status
            in {AdmissionStatus.NOT_EVALUATED, AdmissionStatus.UNEVALUABLE}
        ):
            raise ValueError("evaluable reference fixture requires evaluated states")


REFERENCE_NEGATIVE_ADJUDICATION_FIXTURE = ReferenceAdjudicationFixture(
    fixture_id="reference-negative-adjudication",
    evaluability=AdjudicationEvaluability.EVALUABLE,
    scientific_status=ScientificStatus.NOT_SUPPORTED,
    admission_status=AdmissionStatus.EMPTY,
    reason_codes=(
        "ADMISSION_INTERSECTION_NOT_ESTABLISHED",
        "WRONG_ACTION_NOT_DISPLACED",
    ),
)


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("reference adjudication fixture contains duplicate fields")
        result[key] = value
    return result


def _fixture_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"reference fixture {field_name} must be a string")
    return value


def _decode_reference_fixture(payload: bytes) -> ReferenceAdjudicationFixture:
    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("reference adjudication fixture is not strict JSON") from error
    if (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8") != payload:
        raise ValueError("reference adjudication fixture is not canonical")
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "value",
        "version",
    }:
        raise ValueError("reference adjudication fixture envelope differs")
    if (
        document["schema"] != ReferenceAdjudicationFixture.SCHEMA
        or document["version"] != ReferenceAdjudicationFixture.VERSION
    ):
        raise ValueError("reference adjudication fixture schema differs")
    value = document["value"]
    expected = {field.name for field in fields(ReferenceAdjudicationFixture)}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("reference adjudication fixture fields differ")
    reason_codes = value["reason_codes"]
    if not isinstance(reason_codes, list) or not all(
        isinstance(reason, str) for reason in reason_codes
    ):
        raise ValueError("reference adjudication fixture reasons differ")
    return ReferenceAdjudicationFixture(
        fixture_id=_fixture_string(value["fixture_id"], field_name="fixture_id"),
        evaluability=AdjudicationEvaluability(
            _fixture_string(value["evaluability"], field_name="evaluability")
        ),
        scientific_status=ScientificStatus(
            _fixture_string(value["scientific_status"], field_name="scientific_status")
        ),
        admission_status=AdmissionStatus(
            _fixture_string(value["admission_status"], field_name="admission_status")
        ),
        reason_codes=tuple(reason_codes),
    )


class ReferenceCampaignRunner:
    """Path-free deterministic runner for one statically frozen manifest."""

    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        if self.manifest.capability_key == "reference.evaluate":
            return self._execute_evaluation(context)
        outputs = tuple(
            TaskOutputPayload(
                output_id=port.output_id,
                payload=canonical_json_bytes(
                    {
                        "capability_key": self.manifest.capability_key,
                        "input_artifact_ids": context.external_input_artifact_ids,
                        "input_materialization_ids": (
                            context.dependency_input_materialization_ids
                        ),
                        "output_id": port.output_id,
                        "schema": port.payload_schema,
                        "task_id": context.task_id,
                    }
                ),
            )
            for port in context.output_ports
        )
        return RunnerResult(
            outputs=outputs,
            checks=(ReceiptCheck("reference-runner-contract-passed", True, ()),),
        )

    def _execute_evaluation(self, context: TaskContext) -> RunnerResult:
        adjudication_context = context.scientific_adjudication_context
        if adjudication_context is None:
            raise ValueError("reference evaluator lacks its adjudication context")
        sealed_ports = tuple(
            port
            for port in context.input_ports
            if port.outcome_access is OutcomeAccess.EVALUATION_SEALED
        )
        if len(sealed_ports) != 1:
            raise ValueError("reference evaluator requires one sealed outcome input")
        fixture_payload = sealed_ports[0].read()
        if sealed_ports[0].bytes_read != sealed_ports[0].size_bytes:
            raise ValueError(
                "reference evaluator did not consume the complete sealed fixture"
            )
        fixture = _decode_reference_fixture(fixture_payload)
        logical_output_ids = tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
            )
        )
        if len(logical_output_ids) != len(context.output_ports):
            raise ValueError("reference evaluator output lacks logical identity")
        input_materialization_ids = tuple(sorted(context.input_materialization_ids))
        record = ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication_context.execution_plan,
            input_materialization_ids=input_materialization_ids,
            output_logical_artifact_ids=logical_output_ids,
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication_context.evidence_world_id,
            evidence_world_kind=adjudication_context.evidence_world_kind,
            relation=adjudication_context.relation,
            independent_unit_id=adjudication_context.independent_unit_id,
            information_cutoffs=adjudication_context.information_cutoffs,
            visibility_ceiling=adjudication_context.visibility_ceiling,
            outcome_access=adjudication_context.outcome_access,
            evaluability=fixture.evaluability,
            scientific_status=fixture.scientific_status,
            admission_status=fixture.admission_status,
            reason_codes=fixture.reason_codes,
            fixture_scope_id=adjudication_context.fixture_scope_id,
            plumbing_only=adjudication_context.plumbing_only,
        )
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=encode_scientific_adjudication(
                        record,
                        payload_schema=port.payload_schema,
                    ),
                )
                for port in context.output_ports
            ),
            checks=(ReceiptCheck("reference-adjudication-contract-passed", True, ()),),
        )


class ReferenceCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Registered truth-known source/runner bundle; no physical-world dispatch."""

    capability_count = 6

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        adjudication_fixture: ReferenceAdjudicationFixture = (
            REFERENCE_NEGATIVE_ADJUDICATION_FIXTURE
        ),
    ) -> None:
        self.registry_sha256 = registry.fingerprint()
        self._validate_registry(registry)
        self.adjudication_fixture = adjudication_fixture

    def _validate_registry(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("reference campaign registry identity differs")
        expected_keys = (
            "reference.develop-a",
            "reference.develop-b",
            "reference.evaluate",
            "reference.freeze",
            "reference.prepare",
            "reference.report",
        )
        if (
            tuple(manifest.capability_key for manifest in registry.capabilities)
            != expected_keys
        ):
            raise ValueError("reference campaign capability set differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._validate_registry(registry)
        return tuple(
            ReferenceCampaignRunner(manifest) for manifest in registry.capabilities
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError("reference plan registry identity differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        configs = {
            task.capability.config.artifact_id: task.capability.config
            for task in plan.tasks
        }
        values: list[ExternalInputPayload] = []
        for artifact_id in sorted(specs):
            spec = specs[artifact_id]
            config = configs.get(artifact_id)
            if config is None:
                payload = self.adjudication_fixture.canonical_bytes()
                payload_schema = ReferenceAdjudicationFixture.SCHEMA
                logical_sha256 = None
                outcome_access = OutcomeAccess.EVALUATION_SEALED
            else:
                key = artifact_id.removeprefix("config-artifact.")
                payload = f"config payload for {key}".encode("utf-8")
                payload_schema = config.config_schema
                logical_sha256 = config.content_sha256
                outcome_access = OutcomeAccess.OUTCOME_BLIND
            parent_id = spec.input_id if config is None else config.config_id
            parent_visibility = (
                spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
            )
            if config is None:
                if spec.identity_scope_sha256 is None:
                    raise ValueError(
                        "sealed reference input lacks an enclosing-scope identity"
                    )
                parent_identity = ObjectIdentity(
                    object_id=parent_id,
                    object_schema='empirical-lawhood/runtime/external-input-scope',
                    object_version="1.0.0",
                    object_fingerprint=spec.identity_scope_sha256,
                )
            else:
                parent_identity = ObjectIdentity.from_record(parent_id, config)
            parent = ArtifactLineageParent(
                identity=parent_identity,
                visibility_ceiling=parent_visibility,
                outcome_access=spec.expected_outcome_access or outcome_access,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=payload_schema,
                    profile=(
                        ArtifactProfile.CANONICAL_JSON
                        if config is None
                        else ArtifactProfile.TEXT_PARAMETERS
                    ),
                    media_type=("application/json" if config is None else "text/plain"),
                    payload=payload,
                    visibility_ceiling=(
                        spec.expected_visibility_ceiling
                        or VisibilityCeiling.PROSPECTIVE
                    ),
                    outcome_access=spec.expected_outcome_access or outcome_access,
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
        generic_keys = tuple(
            sorted(
                (
                    "capability_key",
                    "input_artifact_ids",
                    "input_materialization_ids",
                    "output_id",
                    "schema",
                    "task_id",
                )
            )
        )
        adjudication_value_keys = tuple(
            sorted(field.name for field in fields(ScientificAdjudicationRecord))
        )
        contracts: list[CapabilityOutputSemanticContract] = []
        for manifest in registry.capabilities:
            for payload_schema in manifest.output_schema_ids:
                evaluation = manifest.capability_key == "reference.evaluate"
                contracts.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=payload_schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=(
                            ("schema", "value", "version")
                            if evaluation
                            else generic_keys
                        ),
                        value_keys=adjudication_value_keys if evaluation else (),
                        record_version=(
                            ScientificAdjudicationRecord.VERSION if evaluation else None
                        ),
                        capability_key_field=None if evaluation else "capability_key",
                        task_id_field=None if evaluation else "task_id",
                        output_id_field=None if evaluation else "output_id",
                    )
                )
        return tuple(contracts)

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract:
        self._validate_registry(registry)
        manifest = registry.resolve("reference.evaluate", "1.0.0")
        return ScientificAdjudicationOutputContract(
            capability_key=manifest.capability_key,
            capability_version=manifest.capability_version,
            output_id="evaluate.evaluation",
            payload_schema='empirical-lawhood/testing/fixtures/reference-world-pipeline/evaluation-output',
            maximum_bytes=3_000,
            fixture_scope_id=REFERENCE_ADJUDICATION_SCOPE_ID,
            plumbing_only=True,
        )
