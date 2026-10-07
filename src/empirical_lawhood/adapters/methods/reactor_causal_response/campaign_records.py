"""Fixed comparative census and separately revealed prospective owner products."""

from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
import json
from typing import ClassVar
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.controller_evaluation_trajectory import TrajectoryUnitEvaluation, TrajectoryCohortEvaluation
from .config import ARMS, COMPARATOR_SOURCES, CONFIRMATION_ROOTS, NATIVE_BENCHMARK_ARMS
from .experiment_records import EmpiricalAcquisitionEnvelope
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource


@dataclass(frozen=True, slots=True)
class EmpiricalComparatorSources(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/reactor-causal-response/empirical-comparator-sources'
    )
    sources: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        expected = {arm: digest for arm, _, digest in COMPARATOR_SOURCES}
        if tuple(a for a, _ in self.sources) != tuple(sorted(expected)) or any(
            len(s.encode()) > 65536 or sha256(s.encode()).hexdigest() != expected[a]
            for a, s in self.sources
        ):
            raise ValueError("comparator source bytes or closed roster differ")

    def source(self, arm: str) -> bytes:
        return dict(self.sources)[arm].encode()


@dataclass(frozen=True, slots=True)
class EmpiricalStudySource(CanonicalRecord):
    """Pinned public executable inputs only; protected benchmark fixtures are absent."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-study-source'
    batch: ReactorBatchSource
    comparators: EmpiricalComparatorSources
    native_environment: ObjectIdentity

    def __post_init__(self) -> None:
        if self.native_environment.object_schema not in {
            'empirical-lawhood/methods/reactor-causal-response/empirical-native-environment',
        }:
            raise ValueError("native environment must be frozen with the study source")
        if self.batch.reference_controller.encode() != self.comparators.source("REF"):
            raise ValueError("study and reference source identities differ")


@dataclass(frozen=True, slots=True)
class ReactorConfirmationEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/reactor-response/reactor-confirmation-envelope'
    )
    root: str
    qualification: ObjectIdentity
    discovery: ObjectIdentity
    acquisition: EmpiricalAcquisitionEnvelope | None
    # Each callback has its compiled programme, actual tick and prospective link.
    children: tuple[tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity], ...]
    owner_cpu_seconds: tuple[Decimal, ...]
    prerequisite: str | None

    def __post_init__(self) -> None:
        if self.root not in CONFIRMATION_ROOTS:
            raise ValueError("unassigned confirmation root")
        if self.prerequisite not in (
            None,
            "IDENTIFICATION_NOT_SUPPORTED",
            "CALIBRATION_UNUSABLE",
            "LOCAL_LAW_NOT_SUPPORTED",
            "USEFULNESS_NOT_SUPPORTED",
        ):
            raise ValueError("undeclared control prerequisite")
        if self.prerequisite is not None:
            if self.acquisition is not None or self.children or self.owner_cpu_seconds:
                raise ValueError("unentered control cannot acquire evidence")
            return
        if (
            self.acquisition is None
            or self.acquisition.root != self.root
            or self.acquisition.role != "confirmation"
        ):
            raise ValueError("confirmation raw root differs")
        if (
            not 0 <= len(self.children) <= 2880
            or len(self.children) != len(self.owner_cpu_seconds)
            or any(not d.is_finite() or d < 0 for d in self.owner_cpu_seconds)
        ):
            raise ValueError("confirmation callback/cost census differs")


@dataclass(frozen=True, slots=True)
class EmpiricalEpisodeTiming(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-episode-timing'
    arm: str
    callback_wall_seconds: tuple[Decimal, ...]
    episode_wall_seconds: Decimal
    worker_peak_rss_bytes: int

    def __post_init__(self) -> None:
        if (
            self.arm not in ARMS
            or len(self.callback_wall_seconds) > 2880
            or any(
                not v.is_finite() or v < 0
                for v in (*self.callback_wall_seconds, self.episode_wall_seconds)
            )
            or type(self.worker_peak_rss_bytes) is not int
            or self.worker_peak_rss_bytes <= 0
        ):
            raise ValueError("episode timing census or measured resource value differs")
        if sum(self.callback_wall_seconds) > self.episode_wall_seconds:
            raise ValueError("callback wall times exceed the enclosing episode")


@dataclass(frozen=True, slots=True)
class TimedReactorConfirmationEnvelope(ReactorConfirmationEnvelope):
    "Causal-response records with separately measured cost telemetry."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/reactor-response/timed-reactor-confirmation-envelope'
    )
    telemetry: tuple[EmpiricalEpisodeTiming, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        super(TimedReactorConfirmationEnvelope, self).__post_init__()
        if self.prerequisite is not None:
            if self.telemetry:
                raise ValueError(
                    "unentered confirmation cannot invent timing measurements"
                )
        elif tuple(t.arm for t in self.telemetry) != ARMS:
            raise ValueError("confirmation requires all nine timing slots")


@dataclass(frozen=True, slots=True)
class EmpiricalRootAnalysis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-root-analysis'
    root: str
    confirmation: ObjectIdentity
    prospective_evaluation: TrajectoryUnitEvaluation | None
    endpoints_json: str
    contact_json: str
    prerequisite: str | None

    def __post_init__(self) -> None:
        if (
            len(self.endpoints_json.encode()) > 65536
            or len(self.contact_json.encode()) > 1024**2
        ):
            raise ValueError("root analysis exceeds bound")
        rows, contacts = json.loads(self.endpoints_json), json.loads(self.contact_json)
        if not isinstance(rows, list) or not isinstance(contacts, dict):
            raise ValueError("root analysis structures differ")
        if self.prerequisite is None:
            if (
                self.prospective_evaluation is None
                or self.prospective_evaluation.root != self.root
                or tuple(r["arm"] for r in rows) != ARMS
                or any(r["root"] != self.root for r in rows)
            ):
                raise ValueError("root analysis pairing/census differs")
        elif self.prospective_evaluation is not None or rows or contacts:
            raise ValueError("unentered analysis cannot manufacture an outcome")


@dataclass(frozen=True, slots=True)
class EmpiricalContributionResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/reactor-causal-response/empirical-contribution-result'
    )
    roots: tuple[EmpiricalRootAnalysis, ...]
    prospective_evaluation: TrajectoryCohortEvaluation | None
    contrasts_json: str
    matrix_json: str
    prerequisite: str | None

    def __post_init__(self) -> None:
        if tuple(r.root for r in self.roots) != CONFIRMATION_ROOTS:
            raise ValueError("contribution lost complete assigned root roster")
        if (
            len(self.contrasts_json.encode()) > 131072
            or len(self.matrix_json.encode()) > 131072
        ):
            raise ValueError("contribution exceeds bounded summary size")
        if self.prerequisite is None:
            if (
                any(r.prerequisite is not None for r in self.roots)
                or self.prospective_evaluation is None
                or len(json.loads(self.contrasts_json)) != 32
            ):
                raise ValueError(
                    "entered contribution requires all paired slots and owner cohort"
                )
        elif (
            self.prospective_evaluation is not None
            or any(r.prerequisite != self.prerequisite for r in self.roots)
            or json.loads(self.contrasts_json)
        ):
            raise ValueError("unentered contribution has invented outcomes")


@dataclass(frozen=True, slots=True)
class EmpiricalNativeBenchmarkResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/reactor-causal-response/empirical-native-benchmark-result'
    )
    arm: str
    qualification: ObjectIdentity
    submission_sha256: str | None
    environment: ObjectIdentity
    report_json: str | None
    verifier_json: str | None
    reward: int | None
    prerequisite: str | None
    execution_json: str | None = None

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_sha256

        if self.arm not in NATIVE_BENCHMARK_ARMS or self.reward not in (None, 0, 1):
            raise ValueError("native benchmark treatment or reward differs")
        if self.prerequisite is not None:
            if any(
                v is not None
                for v in (
                    self.submission_sha256,
                    self.report_json,
                    self.verifier_json,
                    self.reward,
                    self.execution_json,
                )
            ):
                raise ValueError("unentered native verifier cannot report outcomes")
        else:
            if (
                self.submission_sha256 is None
                or self.report_json is None
                or self.verifier_json is None
                or self.reward is None
                or self.execution_json is None
            ):
                raise ValueError(
                    "entered native verifier must retain report, exact output and reward"
                )
            validate_sha256(self.submission_sha256, field_name="submission_sha256")
            if (
                len(self.report_json.encode()) > 1024**2
                or len(self.verifier_json.encode()) > 16 * 1024**2
            ):
                raise ValueError("native verifier output exceeds bound")

            if self.execution_json is None or len(self.execution_json.encode()) > 65536:
                raise ValueError("native execution accounting is missing or oversized")
            execution = json.loads(self.execution_json)
            if set(execution) != {"public", "verifier"}:
                raise ValueError("native execution stage census differs")
            import math

            for stage, maximum in (("public", 5), ("verifier", 52)):
                cost = execution[stage]
                if (
                    cost["stage"] != stage
                    or type(cost["plant_calls"]) is not int
                    or not 0 <= cost["plant_calls"] <= maximum
                    or any(
                        not math.isfinite(cost[key]) or cost[key] < 0
                        for key in (
                            "python_cpu_s",
                            "wall_s",
                            "parent_peak_rss_kib",
                            "child_peak_rss_kib",
                        )
                    )
                ):
                    raise ValueError(
                        "native measured effort or simulator allocation differs"
                    )
