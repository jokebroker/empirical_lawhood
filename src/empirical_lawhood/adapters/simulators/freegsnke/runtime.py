"""Fixed isolated-process boundary for FreeGSNKE episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import os
from pathlib import Path
from typing import Final

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .contracts import FreeGsnkeProcessRequest, FreeGsnkeProcessResponse, FreeGsnkeSavedPreparation, FreeGsnkeTargetActionObservation, FreeGsnkeTargetCurrentSnapshot, FreeGsnkeTargetEpisodeStatus, FreeGsnkeTargetEpisode, FreeGsnkeTargetProcessResponse, FreeGsnkeTargetReceiverSnapshot, FreeGsnkeTargetWorkerInput
from .worker_process import FreeGsnkeProgressSink, run_freegsnke_canonical_process


FREEGSNKE_MAX_REQUEST_BYTES: Final = 2 * 1024**2
FREEGSNKE_MAX_RESPONSE_BYTES: Final = 32 * 1024**2
FREEGSNKE_MAX_STDERR_BYTES: Final = 2 * 1024**2
FREEGSNKE_PREPARATION_TIMEOUT_SECONDS: Final = 6 * 60 * 60
FREEGSNKE_TARGET_WORKER_TIMEOUT_SECONDS: Final = 12 * 60 * 60
# Compatibility alias for callers that execute target episodes.
FREEGSNKE_WORKER_TIMEOUT_SECONDS: Final = FREEGSNKE_TARGET_WORKER_TIMEOUT_SECONDS
FREEGSNKE_ALLOWED_ENVIRONMENT_KEYS: Final = (
    "HOME",
    "MPLBACKEND",
    "MPLCONFIGDIR",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "TMPDIR",
)


@dataclass(frozen=True, slots=True)
class FreeGsnkeWorkerInvocation:
    """Trusted composition-root invocation; never serialized as science config."""

    python_executable: Path
    worker_script: Path
    working_directory: Path
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        for value, label in (
            (self.python_executable, "python executable"),
            (self.worker_script, "worker script"),
            (self.working_directory, "working directory"),
        ):
            if not value.is_absolute():
                raise ValueError(f"FreeGSNKE {label} must be an absolute installed path")
        if set(self.environment) - set(FREEGSNKE_ALLOWED_ENVIRONMENT_KEYS):
            raise ValueError("FreeGSNKE worker environment contains an unallowlisted key")
        if self.environment.get("PYTHONNOUSERSITE") != "1":
            raise ValueError("FreeGSNKE worker must disable user site packages")
        if self.environment.get("MPLBACKEND") != "Agg":
            raise ValueError("FreeGSNKE worker must use the noninteractive backend")


def task_local_freegsnke_invocation(
    invocation: FreeGsnkeWorkerInvocation,
) -> FreeGsnkeWorkerInvocation:
    """Place one follow-up worker inside its already bounded task scratch."""

    working_directory = Path.cwd().resolve(strict=True)
    environment = dict(invocation.environment)
    environment.update(
        {
            "HOME": os.fspath(working_directory),
            "MPLCONFIGDIR": os.fspath(working_directory),
            "TMPDIR": os.fspath(working_directory),
        }
    )
    return FreeGsnkeWorkerInvocation(
        python_executable=invocation.python_executable,
        worker_script=invocation.worker_script,
        working_directory=working_directory,
        environment=environment,
    )


def decode_freegsnke_response(
    *, request: FreeGsnkeProcessRequest, payload: bytes
) -> FreeGsnkeProcessResponse:
    """Decode a bounded exact response and bind it to the initiating request."""

    response = decode_canonical_bytes(
        payload,
        FreeGsnkeProcessResponse,
        maximum_bytes=FREEGSNKE_MAX_RESPONSE_BYTES,
    )
    expected = ObjectIdentity.from_record(request.request_id, request)
    if response.request != expected:
        raise ValueError("FreeGSNKE worker response is bound to a different request")
    episode = response.episode
    if episode.episode_id != request.request_id.replace("request.", "episode.", 1):
        raise ValueError("FreeGSNKE worker episode identity differs from request")
    if episode.physical_preparation_id != request.preparation.preparation_id:
        raise ValueError("FreeGSNKE worker changed the independent preparation")
    if episode.phase is not request.phase:
        raise ValueError("FreeGSNKE worker changed the request phase")
    for identity, record, label in (
        (episode.source_binding, request.source_binding, "source binding"),
        (episode.preparation, request.preparation, "preparation"),
        (episode.numerical_view, request.numerical_view, "numerical view"),
        (episode.action_chart, request.action_chart, "action chart"),
    ):
        expected_identity = ObjectIdentity.from_record(_record_id(record), record)
        if identity != expected_identity:
            raise ValueError(f"FreeGSNKE worker changed the {label}")
    if episode.action_ledger.requested_port_increments != request.requested_port_increments:
        raise ValueError("FreeGSNKE worker changed the requested native action")
    return response


def _record_id(record: CanonicalRecord) -> str:
    for attribute in ("binding_id", "preparation_id", "view_id", "chart_id"):
        value = getattr(record, attribute, None)
        if isinstance(value, str):
            return value
    raise TypeError("unknown FreeGSNKE source record")


def target_response_from_process_response(
    *,
    request: FreeGsnkeProcessRequest,
    response: FreeGsnkeProcessResponse,
) -> FreeGsnkeTargetProcessResponse:
    """Lift one qualified strict response into the adverse-outcome-capable schema."""

    response = decode_freegsnke_response(
        request=request,
        payload=response.canonical_bytes(),
    )
    episode = response.episode
    ledger = episode.action_ledger
    status = (
        FreeGsnkeTargetEpisodeStatus.OBSERVED_COMPLETE
        if episode.valid and all(value.valid for value in episode.receiver_snapshots)
        else FreeGsnkeTargetEpisodeStatus.OBSERVED_INVALID
    )
    return FreeGsnkeTargetProcessResponse(
        response_id=f"target.{response.response_id}",
        request=response.request,
        episode=FreeGsnkeTargetEpisode(
            episode_id=f"target.{episode.episode_id}",
            source_binding=episode.source_binding,
            preparation=episode.preparation,
            numerical_view=episode.numerical_view,
            action_chart=episode.action_chart,
            physical_preparation_id=episode.physical_preparation_id,
            phase=episode.phase,
            causal_state=episode.causal_state,
            action_observation=FreeGsnkeTargetActionObservation(
                observation_id=f"target.{ledger.ledger_id}",
                request_clock_s=ledger.request_clock_s,
                acceptance_clock_s=ledger.acceptance_clock_s,
                requested_port_increments=ledger.requested_port_increments,
                accepted_port_increments=ledger.accepted_coil_increments,
                applied_samples=ledger.applied_samples,
                clipping_flags=ledger.clipping_flags,
                realized_currents=tuple(
                    FreeGsnkeTargetCurrentSnapshot(
                        snapshot_id=value.snapshot_id,
                        receiver_clock_s=value.receiver_clock_s,
                        active_currents=value.active_currents,
                        passive_currents=value.passive_currents,
                        plasma_current=value.plasma_current,
                    )
                    for value in ledger.realized_currents
                ),
            ),
            receiver_snapshots=tuple(
                FreeGsnkeTargetReceiverSnapshot(
                    snapshot_id=value.snapshot_id,
                    receiver_clock_s=value.receiver_clock_s,
                    values=value.values,
                    topology_id=value.topology_id,
                    valid=value.valid,
                    reason_codes=(),
                )
                for value in episode.receiver_snapshots
            ),
            status=status,
            error_code=episode.error_code,
        ),
    )


def target_worker_failure_response(
    *,
    request: FreeGsnkeProcessRequest,
    status: FreeGsnkeTargetEpisodeStatus,
    error_code: str,
) -> FreeGsnkeTargetProcessResponse:
    """Preserve one issued unit after a typed timeout/nonzero worker outcome."""

    if status not in {
        FreeGsnkeTargetEpisodeStatus.WORKER_FAILED,
        FreeGsnkeTargetEpisodeStatus.WORKER_TIMED_OUT,
    }:
        raise ValueError("FreeGSNKE worker failure response status differs")
    return FreeGsnkeTargetProcessResponse(
        response_id=f"target-response.{request.request_id}.{status.value.lower()}",
        request=ObjectIdentity.from_record(request.request_id, request),
        episode=FreeGsnkeTargetEpisode(
            episode_id=f"target-episode.{request.request_id}.{status.value.lower()}",
            source_binding=ObjectIdentity.from_record(
                request.source_binding.binding_id,
                request.source_binding,
            ),
            preparation=ObjectIdentity.from_record(
                request.preparation.preparation_id,
                request.preparation,
            ),
            numerical_view=ObjectIdentity.from_record(
                request.numerical_view.view_id,
                request.numerical_view,
            ),
            action_chart=ObjectIdentity.from_record(
                request.action_chart.chart_id,
                request.action_chart,
            ),
            physical_preparation_id=request.preparation.preparation_id,
            phase=request.phase,
            causal_state=None,
            action_observation=FreeGsnkeTargetActionObservation(
                observation_id=f"target-action.{request.request_id}",
                request_clock_s=Decimal(0),
                acceptance_clock_s=None,
                requested_port_increments=request.requested_port_increments,
                accepted_port_increments=(),
                applied_samples=(),
                clipping_flags=(),
                realized_currents=(),
            ),
            receiver_snapshots=(),
            status=status,
            error_code=error_code,
        ),
    )


def decode_freegsnke_target_response(
    *,
    request: FreeGsnkeProcessRequest,
    payload: bytes,
) -> FreeGsnkeTargetProcessResponse:
    response = decode_canonical_bytes(
        payload,
        FreeGsnkeTargetProcessResponse,
        maximum_bytes=FREEGSNKE_MAX_RESPONSE_BYTES,
    )
    if response.request != ObjectIdentity.from_record(request.request_id, request):
        raise ValueError("FreeGSNKE target response is bound to another request")
    episode = response.episode
    if (
        episode.source_binding
        != ObjectIdentity.from_record(request.source_binding.binding_id, request.source_binding)
        or episode.preparation
        != ObjectIdentity.from_record(request.preparation.preparation_id, request.preparation)
        or episode.numerical_view
        != ObjectIdentity.from_record(request.numerical_view.view_id, request.numerical_view)
        or episode.action_chart
        != ObjectIdentity.from_record(request.action_chart.chart_id, request.action_chart)
        or episode.physical_preparation_id != request.preparation.preparation_id
        or episode.phase is not request.phase
        or episode.action_observation.requested_port_increments != request.requested_port_increments
    ):
        raise ValueError("FreeGSNKE target response changed a frozen request operand")
    return response


def run_freegsnke_worker(
    *, request: FreeGsnkeProcessRequest, invocation: FreeGsnkeWorkerInvocation
) -> tuple[FreeGsnkeProcessResponse, bytes]:
    """Run the repository-owned worker with canonical stdin/stdout only."""

    payload = request.canonical_bytes()
    if len(payload) > FREEGSNKE_MAX_REQUEST_BYTES:
        raise ValueError("FreeGSNKE request exceeds its bounded process contract")
    command: Sequence[str] = (
        os.fspath(invocation.python_executable),
        "-S",
        os.fspath(invocation.worker_script),
    )
    returncode, stdout, stderr = run_freegsnke_canonical_process(
        command=command,
        payload=payload,
        working_directory=invocation.working_directory,
        environment=invocation.environment,
        timeout_seconds=FREEGSNKE_WORKER_TIMEOUT_SECONDS,
        maximum_stdout_bytes=FREEGSNKE_MAX_RESPONSE_BYTES,
        maximum_stderr_bytes=FREEGSNKE_MAX_STDERR_BYTES,
        progress_subject_id=request.request_id,
    )
    if returncode != 0:
        excerpt = stderr[-4096:].decode("utf-8", errors="replace")
        raise RuntimeError(f"FreeGSNKE worker failed ({returncode}): {excerpt}")
    return decode_freegsnke_response(request=request, payload=stdout), stderr


def run_freegsnke_target_worker(
    *,
    request: FreeGsnkeProcessRequest,
    saved_preparation: FreeGsnkeSavedPreparation,
    invocation: FreeGsnkeWorkerInvocation,
    progress_sink: FreeGsnkeProgressSink | None = None,
) -> tuple[FreeGsnkeTargetProcessResponse, bytes]:
    "Run the independent substrate grounding target worker's exact adverse-outcome-capable contract.\n\n    The completed single-preparation evidence worker uses a historical raw\n    wrapper and emits :class:`FreeGsnkeProcessResponse`.  The independent substrate grounding target must\n    not silently reuse that incompatible process boundary: it accepts the\n    canonical request directly and emits the target response envelope so\n    invalid topology, incomplete observations and action-fibre failures remain\n    scientific outcomes rather than decoder failures.\n    "

    worker_input = FreeGsnkeTargetWorkerInput(
        input_id=f"worker-input.{request.request_id}",
        request=request,
        saved_preparation=saved_preparation,
    )
    payload = worker_input.canonical_bytes()
    if len(payload) > FREEGSNKE_MAX_REQUEST_BYTES:
        raise ValueError("FreeGSNKE target request exceeds its bounded process contract")
    task_invocation = task_local_freegsnke_invocation(invocation)
    command: Sequence[str] = (
        os.fspath(task_invocation.python_executable),
        "-S",
        os.fspath(task_invocation.worker_script),
    )
    returncode, stdout, stderr = run_freegsnke_canonical_process(
        command=command,
        payload=payload,
        working_directory=task_invocation.working_directory,
        environment=task_invocation.environment,
        timeout_seconds=FREEGSNKE_WORKER_TIMEOUT_SECONDS,
        maximum_stdout_bytes=FREEGSNKE_MAX_RESPONSE_BYTES,
        maximum_stderr_bytes=FREEGSNKE_MAX_STDERR_BYTES,
        progress_subject_id=request.request_id,
        progress_sink=progress_sink,
    )
    if returncode != 0:
        excerpt = stderr[-4096:].decode("utf-8", errors="replace")
        raise RuntimeError(f"FreeGSNKE target worker failed ({returncode}): {excerpt}")
    return (
        decode_freegsnke_target_response(request=request, payload=stdout),
        stderr,
    )


__all__ = [
    "FREEGSNKE_ALLOWED_ENVIRONMENT_KEYS",
    "FREEGSNKE_MAX_REQUEST_BYTES",
    "FREEGSNKE_MAX_RESPONSE_BYTES",
    "FREEGSNKE_PREPARATION_TIMEOUT_SECONDS",
    "FREEGSNKE_TARGET_WORKER_TIMEOUT_SECONDS",
    "FREEGSNKE_WORKER_TIMEOUT_SECONDS",
    "FreeGsnkeWorkerInvocation",
    "decode_freegsnke_target_response",
    "decode_freegsnke_response",
    "run_freegsnke_worker",
    "run_freegsnke_target_worker",
    "task_local_freegsnke_invocation",
    "target_response_from_process_response",
    "target_worker_failure_response",
]
