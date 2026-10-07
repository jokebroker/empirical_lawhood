"Closed native substreams of the already committed finite response-law PCG64 stage seeds.\n\nParent, word and numerical view never select a future seed. The two views\nconsume the same coarse and bridge innovations through the existing marcher.\nJump indices separate numerical substreams without changing stage seed keys.\n"

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.finite_response_law.science import seed_for
from empirical_lawhood.kernel.serialization import CanonicalRecord

NATIVE_PURPOSES = ("prefix", "parent", "future-1", "future-2")
SUBSTREAMS = (
    "coarse",
    "bridge",
    "post-ramp-coarse",
    "post-ramp-bridge",
    "passive-probes",
)


def _assigned_stage_unit(stage_unit: str, stage: str, count: int) -> bool:
    """Recognize assignment syntax; the source config must bind the assignment."""

    if not isinstance(stage_unit, str) or len(stage_unit) > 256:
        return False
    namespace, marker, index = stage_unit.rpartition(".r")
    prefix = f"empirical-lawhood.finite-response-law.{stage}."
    return bool(
        marker
        and namespace.startswith(prefix)
        and len(namespace) <= 128
        and namespace != prefix
        and ".r" not in namespace
        and re.fullmatch(r"[a-z0-9][a-z0-9._-]*", namespace)
        and not namespace.endswith(".")
        and re.fullmatch(r"[0-9]{3}", index)
        and int(index) < count
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRNGStream(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-rng-stream'
    stage_unit: str
    purpose: str
    substream: str
    committed_seed_decimal: str
    jump_index: int
    generator: str = "numpy.random.PCG64"
    derivation: str = "explicit-scientific-seed-commitment"
    splitting: str = "numpy.PCG64(seed).jumped(substream-index)"

    def valid_stage_purpose(self) -> bool:
        units = ("canary.r000", *(f"supplemental-development.r{i:03d}" for i in range(16)))
        return self.stage_unit in units and (
            self.stage_unit == "canary.r000" or self.purpose == "future-2"
        )

    def __post_init__(self) -> None:
        if (
            not self.valid_stage_purpose()
            or self.purpose not in NATIVE_PURPOSES
            or self.substream not in SUBSTREAMS
            or (self.substream in SUBSTREAMS[2:] and self.purpose != "prefix")
            or self.committed_seed_decimal
            != str(seed_for(self.purpose, self.stage_unit, committed_seed=int(self.committed_seed_decimal)))
            or type(self.jump_index) is not int
            or self.jump_index != SUBSTREAMS.index(self.substream)
            or self.generator != "numpy.random.PCG64"
            or self.derivation
            != "explicit-scientific-seed-commitment"
            or self.splitting != "numpy.PCG64(seed).jumped(substream-index)"
        ):
            raise ValueError(
                "Finite response-law native RNG differs from its frozen stage/purpose/substream"
            )

    def generator_instance(self) -> np.random.Generator:
        return np.random.Generator(
            np.random.PCG64(int(self.committed_seed_decimal)).jumped(self.jump_index)
        )

    @property
    def initial_state_sha256(self) -> str:
        return sha256(
            canonical_rng_state(self.generator_instance()).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationRNGStream(FiniteResponseLawRNGStream):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-calibration-rng-stream'
    )
    VERSION: ClassVar[str] = '1.0.0'

    def valid_stage_purpose(self) -> bool:
        return self.stage_unit in tuple(
            f"calibration.r{i:03d}" for i in range(32)
        ) or _assigned_stage_unit(self.stage_unit, "calibration", 32)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationRNGStream(FiniteResponseLawRNGStream):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-evaluation-rng-stream'
    )

    def valid_stage_purpose(self) -> bool:
        return self.stage_unit in tuple(
            f"prospective-evaluation.r{i:03d}" for i in range(64)
        ) or _assigned_stage_unit(self.stage_unit, "prospective-evaluation", 64)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyRandomStream(FiniteResponseLawRNGStream):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-policy-random-stream'

    def valid_stage_purpose(self) -> bool:
        stage, _, suffix = self.stage_unit.partition(".r")
        counts = {"preparation-screening": 24, "preparation-calibration": 32}
        return (
            stage in counts and suffix.isdigit() and int(suffix) in range(counts[stage])
        )


def native_rng(
    stage_unit: str, purpose: str, substream: str, *, committed_seed: int | None = None
) -> FiniteResponseLawRNGStream:
    if substream not in SUBSTREAMS:
        raise ValueError("undeclared finite response-law RNG substream")
    record_type = (
        FiniteResponseLawPreparationPolicyRandomStream
        if stage_unit.startswith(("preparation-screening.", "preparation-calibration."))
        else FiniteResponseLawCalibrationRNGStream
        if stage_unit.startswith(("calibration.", "empirical-lawhood.finite-response-law.calibration."))
        else FiniteResponseLawEvaluationRNGStream
        if stage_unit.startswith(("prospective-evaluation.", "empirical-lawhood.finite-response-law.prospective-evaluation."))
        else FiniteResponseLawRNGStream
    )
    return record_type(
        stage_unit,
        purpose,
        substream,
        str(seed_for(purpose, stage_unit, committed_seed=committed_seed)),
        SUBSTREAMS.index(substream),
    )


def canonical_rng_state(rng: np.random.Generator) -> str:
    if type(rng.bit_generator) is not np.random.PCG64:
        raise ValueError("Finite response-law checkpoint requires its declared PCG64 generator")
    return json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"))


def restore_rng(state: str) -> np.random.Generator:
    if type(state) is not str or len(state) > 1024:
        raise ValueError("Finite response-law RNG state exceeds its closed checkpoint bound")
    value = json.loads(state)
    if not isinstance(value, dict) or set(value) != {
        "bit_generator",
        "state",
        "has_uint32",
        "uinteger",
    }:
        raise ValueError("Finite response-law RNG state changes its closed fields")
    rng = np.random.Generator(np.random.PCG64(0))
    rng.bit_generator.state = value
    if canonical_rng_state(rng) != state:
        raise ValueError("Finite response-law RNG state is not canonical PCG64")
    return rng
