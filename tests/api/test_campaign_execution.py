# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from tests.approval_support import approval_service_for_path
from tests.execution_support import EnforcingTestTaskExecutor

from empirical_lawhood.adapters.reference_worlds.runtime import (
    ReferenceAdjudicationFixture,
    ReferenceCampaignRuntimeProvider,
)
from empirical_lawhood.api import (
    CampaignStatusRequest,
    CompileCampaignRequest,
    DocumentRequest,
    ResumeCampaignRequest,
    RunCampaignRequest,
    create_api,
    load_authoring,
)
from empirical_lawhood.api.execution import (
    CampaignExecutionService,
    CampaignConcurrencyError,
    CampaignInternalError,
    CampaignIdentityError,
    CampaignTaskError,
    CampaignValidationError,
)
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    FilesystemState,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.bounded_process import BoundedProcessResult
from empirical_lawhood.infrastructure.execution import (
    DirectTaskExecutor,
    ExecutionResourceCapacity,
    LiveLeaseError,
    LocalExecutionResourceAdmitter,
    LocalProcessExecutor,
    ResourceAdmissionError,
    SchedulerConsistencyError,
    TaskProcessError,
    operational_failure_diagnostic,
)
from empirical_lawhood.infrastructure.recovery import RunRecoveryError, decode_task_recovery_event
from empirical_lawhood.infrastructure.sql import create_catalog_engine, upgrade_catalog
from empirical_lawhood.infrastructure.task_receipts import (
    ExternalTaskReceiptStore,
    decode_artifact_manifest,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.artifacts import (
    ArtifactProfile,
    ExternalRootContract,
)
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    decode_scientific_adjudication,
)
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.execution import (
    ExecutionAssuranceProfile,
    OperationalFailureClass,
    RunnerResult,
    StreamedTaskOutput,
    TaskOutputPayload,
)
from empirical_lawhood.runtime.recovery import TaskRecoveryEvent
from empirical_lawhood.runtime.plans import ScientificStage
from empirical_lawhood.runtime.providers import (
    MAX_IN_MEMORY_EXTERNAL_INPUT_BYTES,
    CampaignRuntimeProviderRegistry,
    ExternalInputPayload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "tests/fixtures/reference-campaign.json"


def test_public_run_preview_accepts_campaign_package_above_former_64_mib_bound(
    tmp_path: Path,
) -> None:
    package = load_authoring(PACKAGE)
    payload = package.canonical_bytes()
    expanded = tmp_path / "expanded-campaign-package.json"
    expanded.write_bytes(payload + b" " * (64 * 1024**2 + 1 - len(payload)))
    api, external, catalog = _api(tmp_path)

    result = api.run_campaign(RunCampaignRequest(expanded, confirmed=False))

    assert result.status.value == "BLOCKED"
    assert result.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


class _TestFilesystemInspector:
    def __init__(self, root: Path, *, free_bytes: int = 100_000_000) -> None:
        self.root = root.resolve()
        self.free_bytes = free_bytes

    def inspect(self, _contract: ExternalRootContract) -> FilesystemState:
        return FilesystemState(
            canonical_root=str(self.root),
            active_mount=True,
            writable=True,
            free_bytes=self.free_bytes,
            path_is_symlink=False,
        )


class _InactiveFilesystemInspector(_TestFilesystemInspector):
    def inspect(self, _contract: ExternalRootContract) -> FilesystemState:
        return FilesystemState(
            canonical_root=str(self.root),
            active_mount=False,
            writable=False,
            free_bytes=100_000_000,
            path_is_symlink=False,
        )


def _api(
    tmp_path: Path,
    *,
    free_bytes: int = 100_000_000,
    provider: ReferenceCampaignRuntimeProvider | None = None,
    executor: Any = None,
    executor_enforces_resources: bool = True,
    executor_enforces_no_network: bool = False,
    enforce_git_identity: bool = False,
    execution_assurance_profile: ExecutionAssuranceProfile = (
        ExecutionAssuranceProfile.TRUSTED_LOCAL
    ),
) -> tuple[Any, Path, Path]:
    external = tmp_path / "external"
    external.mkdir()
    catalog = tmp_path / "experiment_catalog.sqlite3"
    contract = ExternalRootContract(
        storage_root_id="test-reference-root",
        logical_name="Injected reference campaign root",
        canonical_path=str(external.resolve()),
        required_mount_path=str(external.resolve()),
        mount_contract_schema='empirical-lawhood/testing/fixtures/reference-root',
        minimum_free_bytes=1,
    )

    def engine_factory():  # type: ignore[no-untyped-def]
        engine = create_catalog_engine(f"sqlite+pysqlite:///{catalog}")
        upgrade_catalog(engine)
        return engine

    service = CampaignExecutionService(
        repo_root=REPO_ROOT,
        artifact_plane=ExternalArtifactPlane(
            GuardedExternalRoot(
                contract,
                _TestFilesystemInspector(external, free_bytes=free_bytes),
            )
        ),
        engine_factory=engine_factory,
        catalog_path=catalog,
        providers=CampaignRuntimeProviderRegistry(
            (provider or ReferenceCampaignRuntimeProvider(registry=load_authoring(PACKAGE).registry),)
        ),
        approval_service=approval_service_for_path(PACKAGE),
        executor=(
            EnforcingTestTaskExecutor(
                executor,
                network_isolation=executor_enforces_no_network,
            )
            if executor_enforces_resources
            else executor or DirectTaskExecutor()
        ),
        enforce_git_identity=enforce_git_identity,
        execution_assurance_profile=execution_assurance_profile,
    )
    return create_api(repo_root=REPO_ROOT, execution_service=service), external, catalog


@pytest.mark.parametrize("enforce_git_identity", (False, True))
@pytest.mark.parametrize("executor_enforces_resources", (False, True))
def test_git_identity_and_executor_resource_enforcement_are_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enforce_git_identity: bool,
    executor_enforces_resources: bool,
) -> None:
    git_calls: list[tuple[str, ...]] = []

    def exact_clean_git(
        command: list[str],
        **_kwargs: object,
    ) -> BoundedProcessResult:
        arguments = tuple(command)
        git_calls.append(arguments)
        if arguments[-2:] == ("rev-parse", "HEAD"):
            return BoundedProcessResult(0, ("a" * 40 + "\n").encode(), b"")
        if arguments[-2:] == ("status", "--porcelain"):
            return BoundedProcessResult(0, b"", b"")
        raise AssertionError(f"unexpected Git command: {arguments}")

    monkeypatch.setattr("empirical_lawhood.api.execution.run_bounded_command", exact_clean_git)
    api, external, catalog = _api(
        tmp_path,
        executor_enforces_resources=executor_enforces_resources,
        enforce_git_identity=enforce_git_identity,
        execution_assurance_profile=ExecutionAssuranceProfile.STRICT_ISOLATED,
    )

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert bool(git_calls) is enforce_git_identity
    if executor_enforces_resources:
        assert result.succeeded
        assert result.payload is not None
        assert result.payload.operational_status == "SUCCEEDED"
    else:
        assert result.status.value == "BLOCKED"
        assert result.reason_codes == ("EXECUTION_RESOURCE_UNAVAILABLE",)
        assert not external.joinpath("runs").exists()
        assert not catalog.exists()


def test_flag_only_executor_enforcement_boolean_is_not_capability_evidence(
    tmp_path: Path,
) -> None:
    class _FlagOnlyExecutor:
        enforces_resources = True

        def execute(self, *args: object, **kwargs: object) -> object:
            return DirectTaskExecutor().execute(*args, **kwargs)  # type: ignore[arg-type]

    api, external, catalog = _api(
        tmp_path,
        executor=_FlagOnlyExecutor(),
        executor_enforces_resources=False,
        enforce_git_identity=False,
        execution_assurance_profile=ExecutionAssuranceProfile.STRICT_ISOLATED,
    )

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.status.value == "BLOCKED"
    assert result.reason_codes == ("EXECUTION_RESOURCE_UNAVAILABLE",)
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


def test_public_service_stops_truthful_local_executor_before_descendant_probe(
    tmp_path: Path,
) -> None:
    class _DetachedDescendantProbeBoundary(LocalProcessExecutor):
        """Model work that must never be dispatched without aggregate enforcement."""

        def __init__(self) -> None:
            super().__init__()
            self.invocations = 0

        def execute(
            self,
            runner: Any,
            context: Any,
            *,
            timeout_seconds: int,
        ) -> RunnerResult:
            self.invocations += 1
            raise AssertionError(
                "detached or surviving-grandchild probe reached the local executor"
            )

    executor = _DetachedDescendantProbeBoundary()
    api, external, catalog = _api(
        tmp_path,
        executor=executor,
        executor_enforces_resources=False,
        enforce_git_identity=False,
        execution_assurance_profile=ExecutionAssuranceProfile.STRICT_ISOLATED,
    )
    service = api._execution_service
    assert service is not None
    assert service.executor is executor
    assert not executor.enforcement_capability.enforces_resources
    assert not service.executor_enforces_resources

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.status.value == "BLOCKED"
    assert result.reason_codes == ("EXECUTION_RESOURCE_UNAVAILABLE",)
    assert result.errors[0].category.value == "AUTHORITY"
    assert executor.invocations == 0
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


@pytest.mark.parametrize("enforce_git_identity", (False, True))
def test_git_identity_does_not_supply_executor_network_isolation(
    tmp_path: Path,
    *,
    enforce_git_identity: bool,
) -> None:
    isolated_api, _external, _catalog = _api(
        tmp_path,
        executor_enforces_no_network=True,
        enforce_git_identity=enforce_git_identity,
    )
    service = isolated_api._execution_service
    assert service is not None
    assert service.executor_enforces_resources
    assert service.supports_exploration_no_network

    service.executor = DirectTaskExecutor()
    assert not service.executor_enforces_resources
    assert not service.supports_exploration_no_network


class _AdjudicationFieldMutatingExecutor:
    def __init__(self, field_name: str, value: object) -> None:
        self.field_name = field_name
        self.value = value
        self._delegate = DirectTaskExecutor()

    def execute(self, runner, context, *, timeout_seconds):  # type: ignore[no-untyped-def]
        result = self._delegate.execute(
            runner,
            context,
            timeout_seconds=timeout_seconds,
        )
        if context.task_id != "evaluate":
            return result
        assert len(result.outputs) == 1
        output = result.outputs[0]
        assert isinstance(output, TaskOutputPayload)
        document = json.loads(output.payload.decode("utf-8"))
        cast(dict[str, object], document["value"])[self.field_name] = self.value
        payload = (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        return RunnerResult(
            outputs=(replace(output, payload=payload),),
            checks=result.checks,
        )


class _BoundedTestOutputSource:
    def __init__(self, payload: bytes, *, fail_midstream: bool = False) -> None:
        self.payload = payload
        self.fail_midstream = fail_midstream
        self.closed = False

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        split = max(1, min(maximum_chunk_bytes, len(self.payload) // 2))
        yield self.payload[:split]
        if self.fail_midstream:
            raise RuntimeError("injected service stream failure")
        for offset in range(split, len(self.payload), maximum_chunk_bytes):
            yield self.payload[offset : offset + maximum_chunk_bytes]

    def close(self) -> None:
        self.closed = True


class _ObservedExternalInputSource:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.requested_chunk_bounds: list[int] = []
        self.maximum_yielded_chunk = 0
        self.closed = False

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        self.requested_chunk_bounds.append(maximum_chunk_bytes)
        chunk_bytes = max(1, min(7, maximum_chunk_bytes))
        for offset in range(0, len(self.payload), chunk_bytes):
            chunk = self.payload[offset : offset + chunk_bytes]
            self.maximum_yielded_chunk = max(self.maximum_yielded_chunk, len(chunk))
            yield chunk

    def close(self) -> None:
        self.closed = True


class _StreamingExternalInputProvider(ReferenceCampaignRuntimeProvider):
    def __init__(self, *, registry) -> None:
        super().__init__(registry=registry)
        self.observed_source: _ObservedExternalInputSource | None = None

    def external_inputs(self, plan, source_records=()):  # type: ignore[no-untyped-def]
        values = super().external_inputs(plan, source_records)
        first = values[0]
        payload = b"".join(first.chunks())
        first.close()
        source = _ObservedExternalInputSource(payload)
        self.observed_source = source
        return (replace(first, source=source), *values[1:])


class _StreamingPrepareExecutor:
    def __init__(self, *, wrong_schema: bool = False, fail_midstream: bool = False) -> None:
        self.wrong_schema = wrong_schema
        self.fail_midstream = fail_midstream
        self._delegate = DirectTaskExecutor()
        self.sources: list[_BoundedTestOutputSource] = []

    def execute(self, runner, context, *, timeout_seconds):  # type: ignore[no-untyped-def]
        result = self._delegate.execute(
            runner,
            context,
            timeout_seconds=timeout_seconds,
        )
        if context.task_id != "prepare":
            return result
        assert len(result.outputs) == 1
        output = result.outputs[0]
        assert isinstance(output, TaskOutputPayload)
        payload = output.payload
        if self.wrong_schema:
            document = json.loads(payload)
            document["schema"] = 'empirical-lawhood/testing/fixtures/wrong-service-stream'
            payload = (
                json.dumps(
                    document,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        source = _BoundedTestOutputSource(
            payload,
            fail_midstream=self.fail_midstream,
        )
        self.sources.append(source)
        return RunnerResult(
            outputs=(
                StreamedTaskOutput(
                    output_id=output.output_id,
                    size_bytes=len(payload),
                    physical_sha256=hashlib.sha256(payload).hexdigest(),
                    source=source,
                    logical_content_sha256=output.logical_content_sha256,
                ),
            ),
            checks=result.checks,
        )


class _WrongSchemaPrepareByteExecutor:
    def __init__(self) -> None:
        self._delegate = DirectTaskExecutor()

    def execute(self, runner, context, *, timeout_seconds):  # type: ignore[no-untyped-def]
        result = self._delegate.execute(
            runner,
            context,
            timeout_seconds=timeout_seconds,
        )
        if context.task_id != "prepare":
            return result
        output = result.outputs[0]
        assert isinstance(output, TaskOutputPayload)
        document = json.loads(output.payload)
        document["schema"] = 'empirical-lawhood/testing/fixtures/wrong-service-byte'
        payload = (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        return RunnerResult(
            outputs=(replace(output, payload=payload),),
            checks=result.checks,
        )


class _MissingOutputContractProvider(ReferenceCampaignRuntimeProvider):
    def output_semantic_contracts(  # type: ignore[no-untyped-def]
        self, registry, execution_plan=None
    ):
        return super().output_semantic_contracts(registry)[1:]


class _WrongSemanticDigestProvider(ReferenceCampaignRuntimeProvider):
    def output_semantic_contracts(  # type: ignore[no-untyped-def]
        self, registry, execution_plan=None
    ):
        contracts = super().output_semantic_contracts(registry)
        return (replace(contracts[0], capability_implementation_sha256="0" * 64), *contracts[1:])


def test_external_input_small_byte_convenience_is_explicitly_capped() -> None:
    payload = b"x" * MAX_IN_MEMORY_EXTERNAL_INPUT_BYTES
    bounded = ExternalInputPayload.from_bytes(
        logical_artifact_id="external-input.at-limit",
        payload_schema='empirical-lawhood/testing/fixtures/external-input',
        profile=ArtifactProfile.TEXT_PARAMETERS,
        media_type="text/plain",
        payload=payload,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(),
        lineage_parents=(),
    )
    chunks = tuple(bounded.chunks())
    assert b"".join(chunks) == payload
    assert max(map(len, chunks)) <= 1024 * 1024
    assert bounded.size_bytes == bounded.maximum_bytes == len(payload)
    assert bounded.source_sha256 == hashlib.sha256(payload).hexdigest()
    bounded.close()
    with pytest.raises(ValueError, match="small-byte convenience limit"):
        ExternalInputPayload.from_bytes(
            logical_artifact_id="external-input.over-limit",
            payload_schema='empirical-lawhood/testing/fixtures/external-input',
            profile=ArtifactProfile.TEXT_PARAMETERS,
            media_type="text/plain",
            payload=b"x" * (MAX_IN_MEMORY_EXTERNAL_INPUT_BYTES + 1),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            parent_visibility_ceilings=(),
            lineage_parents=(),
        )


def test_public_service_consumes_provider_input_as_a_bounded_stream(
    tmp_path: Path,
) -> None:
    provider = _StreamingExternalInputProvider(registry=load_authoring(PACKAGE).registry)
    api, external, _catalog = _api(tmp_path, provider=provider)

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.succeeded
    source = provider.observed_source
    assert source is not None
    assert source.requested_chunk_bounds
    assert max(source.requested_chunk_bounds) <= 1024 * 1024
    assert source.maximum_yielded_chunk <= 7
    assert source.closed
    assert external.joinpath(
        'runs/synthetic-reference-run-plan/inputs/config-artifact.reference.develop-a.bin'
    ).is_file()


def test_preview_and_execution_use_the_same_preprovider_contract_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _external, _catalog = _api(tmp_path)
    service = api._execution_service
    assert service is not None
    original = service.preexecution_contract_closure
    observed = []

    def recording_closure(**values):  # type: ignore[no-untyped-def]
        matrix = original(**values)
        observed.append(matrix)
        return matrix

    monkeypatch.setattr(service, "preexecution_contract_closure", recording_closure)

    preview = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=False))
    assert preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
    assert len(observed) == 1
    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.succeeded
    assert len(observed) == 2  # One fresh closure per operation; no trust is cached.
    assert observed[0] == observed[1]
    assert observed[0].passed
    assert len(observed[0].checks) == 6


def test_execution_identity_refuses_before_issued_payload_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _external, _catalog = _api(tmp_path)
    service = api._execution_service
    assert service is not None

    def refuse_identity(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise CampaignIdentityError("campaign implementation commit differs")

    def unexpected_closure(**kwargs):  # type: ignore[no-untyped-def]
        pytest.fail("identity drift must refuse before issued payload reads")

    monkeypatch.setattr(service, "_check_identity", refuse_identity)
    monkeypatch.setattr(service, "preexecution_contract_closure", unexpected_closure)
    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))
    assert result.reason_codes == ("EXECUTION_IDENTITY_REFUSED",)


@pytest.mark.parametrize("confirmed", (False, True))
def test_missing_publication_validator_refuses_before_run_writes(tmp_path, confirmed):
    from empirical_lawhood.infrastructure.artifacts import (
        ArtifactIdentityConflict,
    )

    class UnavailableValidator:
        def generic_validation(self, *, profile, payload_schema):
            raise ArtifactIdentityConflict("injected unavailable output validator")

    api, external, catalog = _api(tmp_path)
    api._execution_service.artifact_plane.validators = UnavailableValidator()
    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=confirmed))
    assert result.reason_codes == ("CAPABILITY_NOT_REGISTERED",)
    assert "lacks its registered validator" in result.errors[0].message
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


def _assert_prepare_output_absent(external: Path) -> None:
    output = external / 'runs/synthetic-reference-run-plan/outputs/prepare/prepared.json'
    assert not output.exists()
    assert not output.with_name(f"{output.name}.manifest.json").exists()
    assert not tuple(external.glob('runs/synthetic-reference-run-plan/receipts/prepare/*.json'))
    assert not tuple(external.rglob("*.stream.partial"))


def test_public_service_requires_complete_provider_output_contract_set(
    tmp_path: Path,
) -> None:
    api, external, catalog = _api(
        tmp_path,
        provider=_MissingOutputContractProvider(registry=load_authoring(PACKAGE).registry),
    )

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert not result.succeeded
    assert result.reason_codes == ("CAPABILITY_NOT_REGISTERED",)
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


def test_public_service_rejects_semantic_validator_implementation_substitution(
    tmp_path: Path,
) -> None:
    api, external, catalog = _api(
        tmp_path,
        provider=_WrongSemanticDigestProvider(registry=load_authoring(PACKAGE).registry),
    )

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert not result.succeeded
    assert result.reason_codes == ("CAPABILITY_NOT_REGISTERED",)
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


@pytest.mark.parametrize(
    ("executor",),
    (
        (_WrongSchemaPrepareByteExecutor(),),
        (_StreamingPrepareExecutor(wrong_schema=True),),
        (_StreamingPrepareExecutor(fail_midstream=True),),
    ),
)
def test_public_output_contract_or_midstream_failure_publishes_no_task_success(
    tmp_path: Path,
    *,
    executor: object,
) -> None:
    api, external, _catalog = _api(
        tmp_path,
        executor=executor,
    )

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))
    status = api.campaign_status(
        CampaignStatusRequest(
            'synthetic-reference-run-plan',
            include_attempt_history=True,
        )
    )

    assert not result.succeeded
    assert result.payload is not None
    assert result.payload.operational_status == "FAILED"
    _assert_prepare_output_absent(external)
    assert status.succeeded
    assert status.payload is not None
    prepare_attempts = tuple(
        attempt for attempt in status.payload.attempt_history if attempt.task_id == "prepare"
    )
    assert len(prepare_attempts) == 1
    assert {attempt.disposition for attempt in prepare_attempts} == {"FAILED"}
    expected_failure_class = (
        OperationalFailureClass.CONTRACT_REFUSAL
        if isinstance(executor, _WrongSchemaPrepareByteExecutor)
        else OperationalFailureClass.PROVIDER_RUNNER_DEFECT
    )
    assert {attempt.failure_class for attempt in prepare_attempts} == {expected_failure_class.value}
    assert {attempt.retryable for attempt in prepare_attempts} == {False}
    if isinstance(executor, _StreamingPrepareExecutor):
        assert len(executor.sources) == 1
        assert all(source.closed for source in executor.sources)


def test_production_composition_requires_explicit_operator_storage() -> None:
    with pytest.raises(ValueError, match="explicit OperatorStorageProfile"):
        create_api(repo_root=REPO_ROOT)


def test_doctor_and_write_preview_report_injected_storage_facts(tmp_path: Path) -> None:
    api, external, _catalog = _api(tmp_path)

    doctor = api.doctor()
    assert doctor.succeeded
    assert doctor.payload is not None
    assert doctor.payload.external_present
    assert doctor.payload.external_root == str(external.resolve())
    assert doctor.payload.mount_active
    assert doctor.payload.canonical_contained
    assert doctor.payload.symlink_safe
    assert doctor.payload.observed_free_bytes == 100_000_000
    assert doctor.payload.effective_write_floor_bytes == 1
    assert doctor.payload.storage_read_ready
    assert doctor.payload.storage_write_ready
    assert doctor.payload.catalog_state == "ABSENT"
    assert doctor.payload.catalog_reason_codes == ("LOCAL_CATALOG_ABSENT",)
    backends = {value.backend_id: value for value in doctor.payload.backends}
    assert backends["trusted-local-process"].state == "ENABLED"
    assert backends["exploration-no-network"].reason_codes == ("NO_NETWORK_ISOLATION_UNAVAILABLE",)
    actions = {value.action_id: value for value in doctor.payload.action_readiness}
    assert actions["local-validation"].executable_now
    assert actions["external-artifact-write"].infrastructure_ready
    assert not actions["external-artifact-write"].executable_now
    assert actions["external-artifact-write"].authority_required == ("external-write-confirmation",)
    assert not actions["catalog-read"].infrastructure_ready
    assert actions["catalog-initialize"].infrastructure_ready
    assert actions["exploration-execution"].infrastructure_ready
    assert not actions["exploration-execution"].executable_now
    assert actions["exploration-execution"].authority_required == ("outcome-reveal",)
    assert tuple(value.boundary_id for value in doctor.payload.authority_boundaries) == (
        "acquisition-source-terms",
        "durable-plan-authorization",
        "external-write-confirmation",
        "live-actuation",
        "local-catalog-mutation-confirmation",
        "outcome-reveal",
        "remote-resource-authority",
    )

    preview = api.run_campaign(RunCampaignRequest(PACKAGE))
    assert preview.payload is not None
    assert preview.payload.observed_free_bytes == 100_000_000
    assert preview.payload.effective_write_floor_bytes == 8000
    assert preview.payload.storage_write_ready
    assert preview.payload.storage_reason_codes == ()
    assert not external.joinpath("runs").exists()


def test_campaign_plan_output_allocation_is_an_enforced_write_floor(
    tmp_path: Path,
) -> None:
    api, external, catalog = _api(tmp_path, free_bytes=7_999)

    preview = api.run_campaign(RunCampaignRequest(PACKAGE))
    assert preview.payload is not None
    assert preview.payload.effective_write_floor_bytes == 8_000
    assert not preview.payload.storage_write_ready
    assert preview.payload.storage_reason_codes == ("EXTERNAL_FREE_SPACE_BELOW_FLOOR",)

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))
    assert not result.succeeded
    assert result.status.value == "BLOCKED"
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


def test_reference_campaign_runs_with_negative_science_and_successful_operation(
    tmp_path: Path,
) -> None:
    api, external, catalog = _api(tmp_path)

    preview = api.run_campaign(RunCampaignRequest(PACKAGE))
    assert not preview.succeeded
    assert preview.status.value == "BLOCKED"
    assert preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.succeeded
    assert result.payload is not None
    assert result.payload.operational_status == "SUCCEEDED"
    assert result.payload.adjudication_state == "ADJUDICATED"
    assert result.payload.adjudication_evaluability == "EVALUABLE"
    assert result.payload.scientific_status == "NOT_SUPPORTED"
    assert result.payload.admission_status == "EMPTY"
    assert result.payload.scientific_reason_codes == (
        "ADMISSION_INTERSECTION_NOT_ESTABLISHED",
        "WRONG_ACTION_NOT_DISPLACED",
    )
    assert result.payload.adjudication_id == ('adjudication.synthetic-reference-run-plan.evaluate')
    assert result.payload.adjudication_fingerprint is not None
    assert result.payload.adjudication_artifact_id == (
        'artifact.synthetic-reference-run-plan.evaluate.evaluation'
    )
    assert result.payload.adjudication_materialization_id is not None
    assert result.payload.adjudication_receipt_id == (
        'receipt.synthetic-reference-run-plan.evaluate.attempt-001'
    )
    assert result.payload.evidence_world_id == "w01-stable-linear"
    assert result.payload.fixture_scope_id == "reference-campaign-plumbing"
    assert result.payload.plumbing_only
    assert result.payload.completed_task_ids == (
        "develop-a",
        "develop-b",
        "evaluate",
        "freeze",
        "prepare",
        "report",
    )
    assert catalog.is_file()
    assert b"reference-negative-adjudication" not in catalog.read_bytes()
    assert external.joinpath('runs/synthetic-reference-run-plan/plans/campaign-package.json').is_file()
    for name in ("campaign-package", "run-plan", "execution-plan"):
        manifest = decode_artifact_manifest(
            external.joinpath(
                f"runs/synthetic-reference-run-plan/plans/{name}.json.manifest.json"
            ).read_bytes()
        )
        assert manifest.logical.visibility_ceiling is VisibilityCeiling.OUTCOME_VISIBLE
        assert manifest.logical.outcome_access is OutcomeAccess.EVALUATION_REVEALED
        assert manifest.logical.lineage_parents
    report_manifest = decode_artifact_manifest(
        external.joinpath(
            'runs/synthetic-reference-run-plan/outputs/report/report.json.manifest.json'
        ).read_bytes()
    )
    assert report_manifest.logical.outcome_access is OutcomeAccess.EVALUATION_REVEALED
    assert report_manifest.logical.visibility_ceiling is VisibilityCeiling.OUTCOME_VISIBLE
    assert {parent.outcome_access for parent in report_manifest.logical.lineage_parents} == {
        OutcomeAccess.OUTCOME_BLIND,
        OutcomeAccess.EVALUATION_REVEALED,
    }

    service = api._execution_service
    assert service is not None
    receipt = ExternalTaskReceiptStore(service.artifact_plane).read(
        'synthetic-reference-run-plan',
        "evaluate",
        'synthetic-reference-run-plan.evaluate.attempt-001',
    )
    assert receipt is not None
    adjudication = decode_scientific_adjudication(
        external.joinpath(
            'runs/synthetic-reference-run-plan/outputs/evaluate/evaluation.json'
        ).read_bytes(),
        payload_schema='empirical-lawhood/testing/fixtures/reference-world-pipeline/evaluation-output',
    )
    assert adjudication.input_materialization_ids == receipt.input_materialization_ids
    assert adjudication.output_logical_artifact_ids == tuple(
        value.logical_artifact_id for value in receipt.output_logical_artifacts
    )
    assert adjudication.required_receipt_ids == (
        'receipt.synthetic-reference-run-plan.freeze.attempt-001',
    )


def test_truth_known_positive_fixture_is_receipt_bound_and_scope_limited(
    tmp_path: Path,
) -> None:
    provider = ReferenceCampaignRuntimeProvider(
        registry=load_authoring(PACKAGE).registry,
        adjudication_fixture=ReferenceAdjudicationFixture(
            fixture_id='synthetic-reference-positive-adjudication',
            evaluability=AdjudicationEvaluability.EVALUABLE,
            scientific_status=ScientificStatus.SUPPORTED,
            admission_status=AdmissionStatus.ADMITTED,
            reason_codes=("REFERENCE_TRUTH_SUPPORTED",),
        )
    )
    api, _external, _catalog = _api(tmp_path, provider=provider)

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.succeeded
    assert result.payload is not None
    assert result.payload.adjudication_state == "ADJUDICATED"
    assert result.payload.scientific_status == "SUPPORTED"
    assert result.payload.admission_status == "ADMITTED"
    assert result.payload.scientific_reason_codes == ("REFERENCE_TRUTH_SUPPORTED",)
    assert result.payload.fixture_scope_id == "reference-campaign-plumbing"
    assert result.payload.plumbing_only
    assert result.payload.adjudication_artifact_id is not None
    assert result.payload.adjudication_receipt_id is not None


def test_receipt_dependent_revealed_admission_may_own_scientific_adjudication(
    tmp_path: Path,
) -> None:
    api, _external, _catalog = _api(tmp_path)
    package, _run_plan, execution_plan = api._compile_path(PACKAGE)
    service = api._execution_service
    assert service is not None
    provider = service.providers.resolve(execution_plan.registry_sha256)
    contract = provider.scientific_adjudication_contract(
        package.registry,
        execution_plan,
    )
    assert contract is not None
    task = next(
        value
        for value in execution_plan.tasks
        if value.capability.capability_key == contract.capability_key
        and any(output.output_id == contract.output_id for output in value.outputs)
    )
    admission_task = replace(
        task,
        stage=ScientificStage.ADMISSION,
        capability=replace(
            task.capability,
            kind=CapabilityKind.ADMISSION_EVALUATOR,
        ),
        outputs=tuple(
            (
                replace(
                    output,
                    outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                    visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                )
                if output.output_id == contract.output_id
                else output
            )
            for output in task.outputs
        ),
    )
    admission_plan = replace(
        execution_plan,
        tasks=tuple(
            admission_task if value.task_id == task.task_id else value
            for value in execution_plan.tasks
        ),
    )

    binding = service._bind_scientific_adjudication(
        package,
        admission_plan,
        contract,
    )

    assert binding is not None
    assert binding.task == admission_task


def test_receipt_dependent_historical_development_report_may_own_adjudication(
    tmp_path: Path,
) -> None:
    api, _external, _catalog = _api(tmp_path)
    package, _run_plan, execution_plan = api._compile_path(PACKAGE)
    service = api._execution_service
    assert service is not None
    provider = service.providers.resolve(execution_plan.registry_sha256)
    contract = provider.scientific_adjudication_contract(
        package.registry,
        execution_plan,
    )
    assert contract is not None
    task = next(
        value
        for value in execution_plan.tasks
        if value.capability.capability_key == contract.capability_key
        and any(output.output_id == contract.output_id for output in value.outputs)
    )
    historical_report = replace(
        task,
        stage=ScientificStage.REPORT,
        capability=replace(task.capability, kind=CapabilityKind.REPORTER),
        outputs=tuple(
            replace(
                output,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            )
            if output.output_id == contract.output_id
            else output
            for output in task.outputs
        ),
    )
    historical_plan = replace(
        execution_plan,
        tasks=tuple(
            historical_report if value.task_id == task.task_id else value
            for value in execution_plan.tasks
        ),
    )

    binding = service._bind_scientific_adjudication(
        package,
        historical_plan,
        contract,
    )

    assert binding is not None
    assert binding.task == historical_report


def test_missing_durable_adjudication_receipt_emits_no_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ExternalTaskReceiptStore.read

    def missing_evaluation(self, run_id, task_id, attempt_id):  # type: ignore[no-untyped-def]
        if task_id == "evaluate":
            return None
        return original(self, run_id, task_id, attempt_id)

    monkeypatch.setattr(ExternalTaskReceiptStore, "read", missing_evaluation)
    api, _external, _catalog = _api(tmp_path)

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.succeeded
    assert result.payload is not None
    assert result.payload.operational_status == "SUCCEEDED"
    assert result.payload.adjudication_state == "NOT_ADJUDICATED"
    assert result.payload.adjudication_evaluability == "UNEVALUABLE"
    assert result.payload.scientific_status is None
    assert result.payload.admission_status is None
    assert result.payload.scientific_reason_codes == ("ADJUDICATION_RECEIPT_MISSING",)
    assert result.payload.adjudication_artifact_id is None
    assert result.payload.adjudication_receipt_id is None


def test_type_malformed_adjudication_emits_no_verdict_after_operational_success(
    tmp_path: Path,
) -> None:
    api, _external, _catalog = _api(
        tmp_path,
        executor=_AdjudicationFieldMutatingExecutor("adjudication_id", 7),
    )

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.succeeded
    assert result.payload is not None
    assert result.payload.operational_status == "SUCCEEDED"
    assert result.payload.adjudication_state == "NOT_ADJUDICATED"
    assert result.payload.scientific_status is None
    assert result.payload.admission_status is None
    assert result.payload.scientific_reason_codes == ("ADJUDICATION_RECORD_INVALID",)


def test_adjudication_lineage_substitution_emits_no_verdict(
    tmp_path: Path,
) -> None:
    api, _external, _catalog = _api(
        tmp_path,
        executor=_AdjudicationFieldMutatingExecutor(
            "input_materialization_ids",
            ["materialization.substituted"],
        ),
    )

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.succeeded
    assert result.payload is not None
    assert result.payload.operational_status == "SUCCEEDED"
    assert result.payload.adjudication_state == "NOT_ADJUDICATED"
    assert result.payload.scientific_status is None
    assert result.payload.admission_status is None
    assert result.payload.scientific_reason_codes == ("ADJUDICATION_LINEAGE_MISMATCH",)


def test_obsolete_provider_summary_injection_cannot_override_receipt_record(
    tmp_path: Path,
) -> None:
    class _InjectedSummaryProvider(ReferenceCampaignRuntimeProvider):
        def scientific_summary(self, _plan):  # type: ignore[no-untyped-def]
            raise AssertionError("obsolete provider summary must never be called")

    api, _external, _catalog = _api(
        tmp_path,
        provider=_InjectedSummaryProvider(registry=load_authoring(PACKAGE).registry),
    )

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.succeeded
    assert result.payload is not None
    assert result.payload.scientific_status == "NOT_SUPPORTED"
    assert result.payload.admission_status == "EMPTY"


def test_resume_reconciles_receipts_and_status_is_read_only(tmp_path: Path) -> None:
    api, _external, catalog = _api(tmp_path)
    first = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))
    assert first.succeeded
    resumed = api.resume_campaign(ResumeCampaignRequest('synthetic-reference-run-plan', confirmed=True))
    before_status = tuple(
        (path.name, path.read_bytes()) for path in sorted(catalog.parent.glob(f"{catalog.name}*"))
    )
    status = api.campaign_status(
        CampaignStatusRequest('synthetic-reference-run-plan', include_attempt_history=True)
    )

    assert resumed.succeeded
    assert resumed.payload is not None
    assert len(resumed.payload.receipt_ids) == 6
    assert status.succeeded
    assert status.payload is not None
    assert status.payload.operational_status == "SUCCEEDED"
    assert status.payload.succeeded_task_ids == resumed.payload.completed_task_ids
    assert len(status.payload.attempt_history) == 6
    assert {attempt.ordinal for attempt in status.payload.attempt_history} == {1}
    assert (
        tuple(
            (path.name, path.read_bytes())
            for path in sorted(catalog.parent.glob(f"{catalog.name}*"))
        )
        == before_status
    )


def test_child_validate_compile_run_and_resume_reject_authorization_store_drift(
    tmp_path: Path,
) -> None:
    api, _external, _catalog = _api(tmp_path)
    first = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))
    assert first.succeeded
    service = api._execution_service
    assert service is not None and service.approval_service is not None
    approval_store = cast(Any, service.approval_service)._store
    record = next(iter(approval_store._records.values()))
    approval_store._records[record.authorization_id] = replace(
        record,
        reason_codes=("COMPLETE_ENVELOPE_APPROVED", "STORE_DRIFT"),
    )

    decoded = api.validate_document(DocumentRequest(PACKAGE))
    validated = api.validate_campaign_document(DocumentRequest(PACKAGE))
    compiled = api.compile_campaign(CompileCampaignRequest(PACKAGE))
    executed = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))
    resumed = api.resume_campaign(ResumeCampaignRequest('synthetic-reference-run-plan', confirmed=True))

    assert decoded.succeeded
    for rejected in (validated, compiled, executed, resumed):
        assert rejected.status.value == "INVALID"
        assert rejected.reason_codes == ("DURABLE_AUTHORIZATION_REPLAY_FAILED",)

    approval_store._records[record.authorization_id] = record
    proposal_store = cast(Any, service.approval_service)._proposal_store
    frozen = next(iter(proposal_store._records.values()))
    proposal_store._records[frozen.frozen_proposal_id] = replace(
        frozen,
        proposal=replace(
            frozen.proposal,
            expected_discrimination="Substituted stored proposal content.",
        ),
    )
    proposal_drift_resume = api.resume_campaign(
        ResumeCampaignRequest('synthetic-reference-run-plan', confirmed=True)
    )
    assert proposal_drift_resume.status.value == "INVALID"
    assert proposal_drift_resume.reason_codes == ("DURABLE_AUTHORIZATION_REPLAY_FAILED",)


def test_partial_immutable_plan_pair_maps_to_public_conflict(tmp_path: Path) -> None:
    api, external, _catalog = _api(tmp_path)
    first = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))
    assert first.succeeded
    external.joinpath(
        'runs/synthetic-reference-run-plan/plans/campaign-package.json.manifest.json'
    ).unlink()

    conflicted = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert conflicted.status.value == "CONFLICT"
    assert conflicted.reason_codes == ("IMMUTABLE_ARTIFACT_CONFLICT",)
    assert conflicted.errors[0].category.value == "CONFLICT"


def test_mount_loss_fails_closed_without_local_fallback(tmp_path: Path) -> None:
    api, external, catalog = _api(tmp_path)
    assert api._execution_service is not None
    api._execution_service.artifact_plane.root.inspector = _InactiveFilesystemInspector(external)

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.status.value == "BLOCKED"
    assert result.reason_codes == ("EXECUTION_CUSTODY_OR_STORAGE_FAILED",)
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


def test_resource_preflight_blocks_before_any_scientific_or_catalog_write(
    tmp_path: Path,
) -> None:
    api, external, catalog = _api(tmp_path)
    service = api._execution_service
    assert service is not None
    service.resource_admitter = LocalExecutionResourceAdmitter(
        ExecutionResourceCapacity(
            cpu_cores=1,
            memory_bytes=1,
            gpu_devices=0,
            enforced_cpu_limit=True,
            enforced_address_space_limit=True,
            enforced_no_network=False,
        )
    )

    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.status.value == "BLOCKED"
    assert result.reason_codes == ("EXECUTION_RESOURCE_UNAVAILABLE",)
    assert result.errors[0].category.value == "AUTHORITY"
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


def test_database_outage_is_normalized_after_external_plan_custody(
    tmp_path: Path,
) -> None:
    api, external, catalog = _api(tmp_path)
    assert api._execution_service is not None

    def unavailable():  # type: ignore[no-untyped-def]
        raise RuntimeError("injected database outage")

    api._execution_service.engine_factory = unavailable
    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.status.value == "BLOCKED"
    assert result.reason_codes == ("EXECUTION_CUSTODY_OR_STORAGE_FAILED",)
    assert external.joinpath('runs/synthetic-reference-run-plan/plans/campaign-package.json').is_file()
    assert not tuple(external.glob('runs/synthetic-reference-run-plan/receipts/**/*.json'))
    assert not catalog.exists()


@pytest.mark.parametrize(
    ("injected", "expected_type"),
    (
        (LiveLeaseError("active lease /secret"), CampaignConcurrencyError),
        (
            SchedulerConsistencyError("drift /secret"),
            CampaignValidationError,
        ),
        (TaskProcessError("worker /secret"), CampaignTaskError),
        (RuntimeError("unexpected /secret"), CampaignInternalError),
    ),
)
def test_service_normalizes_execution_boundaries_once(
    tmp_path: Path,
    injected: Exception,
    expected_type: type[Exception],
) -> None:
    api, _external, _catalog = _api(tmp_path)
    service = api._execution_service
    assert service is not None

    def fail(**_kwargs):  # type: ignore[no-untyped-def]
        raise injected

    service._execute_plan = fail
    package, run_plan, execution_plan = api._compile_path(PACKAGE)
    with pytest.raises(expected_type) as captured:
        service.execute(
            package=package,
            run_plan=run_plan,
            execution_plan=execution_plan,
        )
    assert "/secret" not in str(captured.value)


def test_unexpected_execution_error_is_redacted_in_public_envelope(tmp_path: Path) -> None:
    api, external, catalog = _api(tmp_path)
    service = api._execution_service
    assert service is not None

    def fail(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("secret path /private/provider/token")

    service._execute_plan = fail
    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.status.value == "FAILED"
    assert result.reason_codes == ("INTERNAL_EXECUTION_ERROR",)
    assert result.errors[0].category.value == "INTERNAL"
    assert "/private/provider/token" not in str(result.to_mapping())
    assert not external.joinpath("runs").exists()
    assert not catalog.exists()


def test_unexpected_status_error_is_redacted_in_public_envelope(tmp_path: Path) -> None:
    api, _external, catalog = _api(tmp_path)
    service = api._execution_service
    assert service is not None
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_bytes(b"catalog-present")

    def fail():  # type: ignore[no-untyped-def]
        raise RuntimeError("secret status path /private/catalog/token")

    service.read_only_engine_factory = fail
    result = api.campaign_status(CampaignStatusRequest('synthetic-reference-run-plan'))

    assert result.status.value == "FAILED"
    assert result.reason_codes == ("INTERNAL_EXECUTION_ERROR",)
    assert result.errors[0].category.value == "INTERNAL"
    assert "/private/catalog/token" not in str(result.to_mapping())


def test_exhausted_worker_failure_is_operational_not_scientific(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, external, catalog = _api(tmp_path)
    service = api._execution_service
    assert service is not None

    class _FailingExecutor:
        def execute(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise TaskProcessError(
                "injected worker failure /protected/outcome",
                child_exception_type="IntegrityBoundaryError",
                child_message="truth=protected /protected/outcome",
            )

    service.executor = EnforcingTestTaskExecutor(_FailingExecutor())
    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))

    assert result.status.value == "FAILED"
    assert result.reason_codes == ("TASK_EXECUTION_FAILED",)
    assert result.payload is not None
    assert result.payload.operational_status == "FAILED"
    assert result.payload.adjudication_state == "NOT_ADJUDICATED"
    assert result.payload.adjudication_evaluability == "UNEVALUABLE"
    assert result.payload.scientific_status is None
    assert result.payload.admission_status is None
    assert result.payload.scientific_reason_codes == ("EXECUTION_NOT_SUCCEEDED",)
    # A full plan can exceed the compact-receipt bound. Lower that bound in
    # this small fixture instead of allocating a 27 MiB scientific roster.
    plan_path = external / 'runs/synthetic-reference-run-plan/plans/execution-plan.json'
    compact_limit = plan_path.stat().st_size - 1
    assert compact_limit > 1024
    monkeypatch.setattr("empirical_lawhood.api.execution.MAX_CONTROL_PLANE_JSON_BYTES", compact_limit)
    normal_status = api.campaign_status(
        CampaignStatusRequest(
            'synthetic-reference-run-plan',
            include_attempt_history=True,
        )
    )
    assert normal_status.succeeded
    assert normal_status.payload is not None
    failed_attempts = tuple(
        value for value in normal_status.payload.attempt_history if value.disposition == "FAILED"
    )
    assert tuple(value.failure_class for value in failed_attempts) == (
        OperationalFailureClass.CHILD_PROCESS_FAILURE.value,
        OperationalFailureClass.RETRY_EXHAUSTION.value,
    )
    assert tuple(value.retryable for value in failed_attempts) == (True, False)
    assert {value.sanitized_exception_type for value in failed_attempts} == {
        "IntegrityBoundaryError"
    }
    assert all(value.diagnostic_sha256 is not None for value in failed_attempts)
    assert all(
        value.recovery_event_schema == TaskRecoveryEvent.SCHEMA for value in failed_attempts
    )
    assert normal_status.payload.status_source == "RECOVERY_TERMINAL"
    assert normal_status.payload.implementation_commit is not None
    assert normal_status.payload.execution_plan_id == 'execution.synthetic-reference-run-plan'
    assert normal_status.payload.recovery_index_relative_path is not None
    first_path = failed_attempts[0].recovery_event_relative_path
    assert first_path is not None
    retained = decode_task_recovery_event(external.joinpath(first_path).read_bytes())
    assert retained.failure is not None
    assert retained.failure.failure_class is OperationalFailureClass.CHILD_PROCESS_FAILURE
    assert b"protected" not in retained.canonical_bytes()
    assert b"truth" not in retained.canonical_bytes()

    catalog.unlink()
    for suffix in ("-shm", "-wal"):
        catalog.with_name(f"{catalog.name}{suffix}").unlink(missing_ok=True)
    recovered_status = api.campaign_status(
        CampaignStatusRequest(
            'synthetic-reference-run-plan',
            include_attempt_history=True,
        )
    )
    assert recovered_status.succeeded
    assert recovered_status.payload == normal_status.payload


@pytest.mark.parametrize(
    ("error", "failure_class"),
    (
        (SchedulerConsistencyError("contract /secret"), OperationalFailureClass.CONTRACT_REFUSAL),
        (ResourceAdmissionError("resource /secret"), OperationalFailureClass.RESOURCE_REFUSAL),
        (RuntimeError("provider /secret"), OperationalFailureClass.PROVIDER_RUNNER_DEFECT),
        (
            TaskProcessError(
                "worker /secret",
                child_exception_type="ValueError",
                child_message="outcome=secret /secret",
            ),
            OperationalFailureClass.CHILD_PROCESS_FAILURE,
        ),
        (OSError("temporary /secret"), OperationalFailureClass.TRANSIENT_INFRASTRUCTURE_FAILURE),
    ),
)
def test_operational_failure_classes_are_closed_redacted_and_policy_owned(
    error: BaseException,
    failure_class: OperationalFailureClass,
) -> None:
    diagnostic = operational_failure_diagnostic(error)

    assert diagnostic.failure_class is failure_class
    assert diagnostic.retryable is failure_class.retryable
    assert diagnostic.scientific_evidence is False
    assert b"secret" not in diagnostic.canonical_bytes()


def test_terminal_operational_failure_classes_are_never_retryable() -> None:
    source = TaskProcessError("worker failure")
    exhausted = operational_failure_diagnostic(
        source,
        failure_class=OperationalFailureClass.RETRY_EXHAUSTION,
    )
    obstructed = operational_failure_diagnostic(
        RunRecoveryError("recovery obstruction"),
        failure_class=OperationalFailureClass.RECOVERY_OBSTRUCTION,
    )

    assert not exhausted.retryable
    assert not obstructed.retryable
    assert set(OperationalFailureClass) == {
        OperationalFailureClass.CONTRACT_REFUSAL,
        OperationalFailureClass.RESOURCE_REFUSAL,
        OperationalFailureClass.PROVIDER_RUNNER_DEFECT,
        OperationalFailureClass.CHILD_PROCESS_FAILURE,
        OperationalFailureClass.TRANSIENT_INFRASTRUCTURE_FAILURE,
        OperationalFailureClass.RETRY_EXHAUSTION,
        OperationalFailureClass.RECOVERY_OBSTRUCTION,
    }


def test_deterministic_provider_defect_is_not_automatically_retried(tmp_path: Path) -> None:
    api, _external, _catalog = _api(tmp_path)
    service = api._execution_service
    assert service is not None

    class _DefectiveExecutor:
        def execute(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("deterministic provider defect /protected/outcome")

    service.executor = EnforcingTestTaskExecutor(_DefectiveExecutor())
    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))
    status = api.campaign_status(
        CampaignStatusRequest('synthetic-reference-run-plan', include_attempt_history=True)
    )

    assert result.status.value == "FAILED"
    assert status.succeeded
    assert status.payload is not None
    failed = tuple(
        value for value in status.payload.attempt_history if value.disposition == "FAILED"
    )
    assert len(failed) == 1
    assert failed[0].failure_class == OperationalFailureClass.PROVIDER_RUNNER_DEFECT.value
    assert failed[0].retryable is False
    assert "protected" not in str(status.to_mapping())


def test_scheduler_blocked_execution_returns_typed_not_adjudicated_boundary(
    tmp_path: Path,
) -> None:
    api, _external, _catalog = _api(tmp_path)
    service = api._execution_service
    assert service is not None

    class _BlockingExecutor:
        def execute(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise ResourceAdmissionError("injected task computability boundary")

    service.executor = EnforcingTestTaskExecutor(_BlockingExecutor())
    result = api.run_campaign(RunCampaignRequest(PACKAGE, confirmed=True))
    status = api.campaign_status(
        CampaignStatusRequest(
            'synthetic-reference-run-plan',
            include_attempt_history=True,
        )
    )

    assert result.status.value == "BLOCKED"
    assert result.payload is not None
    assert result.payload.operational_status == "BLOCKED"
    assert result.payload.adjudication_state == "NOT_ADJUDICATED"
    assert result.payload.adjudication_evaluability == "UNEVALUABLE"
    assert result.payload.scientific_status is None
    assert result.payload.admission_status is None
    assert result.payload.scientific_reason_codes == ("EXECUTION_NOT_SUCCEEDED",)
    assert status.succeeded
    assert status.payload is not None
    blocked = tuple(
        attempt for attempt in status.payload.attempt_history if attempt.disposition == "BLOCKED"
    )
    assert blocked
    assert {attempt.block_kind for attempt in blocked} == {"RETRYABLE"}
    resource = tuple(
        attempt
        for attempt in blocked
        if attempt.reason_code == "resource-computability-unavailable"
    )
    assert len(resource) == 1
    assert resource[0].failure_class == OperationalFailureClass.RESOURCE_REFUSAL.value
    assert resource[0].retryable is False
    assert resource[0].diagnostic_sha256 is not None
