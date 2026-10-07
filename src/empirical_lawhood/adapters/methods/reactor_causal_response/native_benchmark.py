"""Frozen submission assembly and typed port to the unchanged upstream verifier."""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Protocol, ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .campaign_records import EmpiricalNativeBenchmarkResult, EmpiricalStudySource
from .experiment_records import EmpiricalQualificationTerminal, EmpiricalDiscoveryEnvelope
from .serialization import read_fit
from .config import NATIVE_BENCHMARK_ARMS
from .export import export_source
from empirical_lawhood.adapters.simulators.reactor_causal_response.campaign import prerequisite


class NativeVerifierPort(Protocol):
    environment: ObjectIdentity

    def verify(self, arm: str, submission: bytes) -> tuple[str, str, int, str]: ...


def submission_source(
    arm: str,
    qualification: EmpiricalQualificationTerminal,
    discovery: EmpiricalDiscoveryEnvelope,
    source: EmpiricalStudySource,
) -> bytes:
    if (
        prerequisite(qualification) is not None
        or discovery.recipe != qualification.recipe
    ):
        raise ValueError(
            "native submission requires the same eligible frozen primary parent"
        )
    if arm in ("REF", "EKF", "SCHEDULED_BACKOFF_ZERO", "SCHEDULED_BACKOFF_HALF"):
        content = source.comparators.source(arm)
        if arm in ("SCHEDULED_BACKOFF_ZERO", "SCHEDULED_BACKOFF_HALF"):
            content += f"\n_PublishedController = Controller\nclass Controller(_PublishedController):\n    def __init__(self):\n        super().__init__(back_off={0.0 if arm == 'SCHEDULED_BACKOFF_ZERO' else 0.5!r})\n".encode()
        return content
    report = qualification.result
    assert report is not None and report.payload.q is not None
    saved = json.loads(discovery.result_json)
    model = (
        read_fit(json.loads(report.payload.model_json))
        if arm not in ("F0", "F1")
        else read_fit(saved["fits"][saved["rivals"][int(arm[-1])]])
    )
    return export_source(
        model,
        float(report.payload.q),
        arm=arm,
        scheduled_source=source.comparators.source("SCHEDULED_BACKOFF_HALF") if arm == "FIXED" else None,
    )


def native_benchmark(
    arm: str,
    qualification: EmpiricalQualificationTerminal,
    discovery: EmpiricalDiscoveryEnvelope,
    source: EmpiricalStudySource,
    verifier: NativeVerifierPort,
) -> EmpiricalNativeBenchmarkResult:
    if arm not in NATIVE_BENCHMARK_ARMS:
        raise ValueError(
            "native benchmark arm is deferred by the eight-root allocation"
        )
    if verifier.environment != source.native_environment:
        raise ValueError(
            "native verifier environment differs from the pre-acquisition source freeze"
        )
    qid = ObjectIdentity.from_record("reactor-empirical-qualification", qualification)
    stop = prerequisite(qualification)
    if stop:
        return EmpiricalNativeBenchmarkResult(
            arm, qid, None, verifier.environment, None, None, None, stop
        )
    submission = submission_source(arm, qualification, discovery, source)
    report, output, reward, execution = verifier.verify(arm, submission)
    return EmpiricalNativeBenchmarkResult(
        arm,
        qid,
        sha256(submission).hexdigest(),
        verifier.environment,
        report,
        output,
        reward,
        None,
        execution,
    )


@dataclass(frozen=True, slots=True)
class EmpiricalNativeEnvironment(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/reactor-causal-response/empirical-native-environment'
    )
    VERSION: ClassVar[str] = '1.0.0'
    environment_id: str
    image_id: str
    source_commit: str
    source_files: tuple[tuple[str, str], ...]
    python_version: str = "3.12.13"
    numpy_version: str = "2.1.3"
    scipy_version: str = "1.14.1"
    pytest_version: str = "8.4.1"

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import (
            validate_stable_id,
            validate_relative_locator,
        )

        validate_stable_id(self.environment_id, field_name="environment_id")
        if not self.image_id.startswith("sha256:"):
            raise ValueError("native verifier requires an explicit image digest")
        validate_sha256(self.image_id.removeprefix("sha256:"))
        if self.source_commit != "749bc764b667502d47d2d99de988ee6e089b6aa4" or (
            self.python_version,
            self.numpy_version,
            self.scipy_version,
            self.pytest_version,
        ) != ("3.12.13", "2.1.3", "1.14.1", "8.4.1"):
            raise ValueError(
                "native verifier environment differs from the inspected pin"
            )
        if (
            not self.source_files
            or tuple(sorted(set(self.source_files))) != self.source_files
            or len({name for name, _ in self.source_files}) != len(self.source_files)
        ):
            raise ValueError("native verifier source census differs")
        for name, digest in self.source_files:
            validate_relative_locator(name)
            validate_sha256(digest)
