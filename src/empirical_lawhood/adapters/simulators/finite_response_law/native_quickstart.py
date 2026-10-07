"""Excluded Tier 1 native canary through the retained finite-lawhood marcher."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_receiver
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .contracts import CANARY_PARENT, FiniteResponseLawNativeConfig, native_invocations
from .discovery import SOURCE_CAPABILITY
from .source import execute_native_phase, frozen_prefix_frame


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCanary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-canary'
    config_id: str
    magnitude: Decimal
    direction_index: int
    sign: int
    stage: str = "native-canary"
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            not self.config_id.startswith("empirical-lawhood-finite-response-law-canary-")
            or self.stage != "native-canary"
            or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
            or self.sign == 0
        ):
            raise ValueError("finite canary requires its excluded development role")
        word = PreparedForceWord(self.magnitude, self.direction_index, self.sign)
        if word.direction_index not in (0, 1):
            raise ValueError("finite canary requires a declared two-port word")


def run_finite_canary(config: FiniteResponseLawCanary) -> dict[str, object]:
    """Exercise one excluded root in both views and both future streams."""

    if platform.python_version() != "3.11.14" or np.__version__ != "2.4.6":
        raise ValueError("Finite response-law canary requires CPython 3.11.14 and NumPy 2.4.6")
    source = FiniteResponseLawNativeConfig(
        "native-canary",
        FiniteResponseLawScienceSpec(),
        ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY),
        None,
        (),
    )
    invocations = native_invocations(source)
    prefix_task = next(task for task in invocations if task.phase == "prefix")
    parent_task = next(task for task in invocations if task.phase == "parent")
    prefixes = tuple(
        execute_native_phase(source, prefix_task, refinement, incoming=None, frame=None)
        for refinement in (1, 2)
    )
    if any(value.checkpoint is None for value in prefixes):
        raise RuntimeError(
            "Finite response-law canary prefix lacks a complete paired checkpoint"
        )
    assert prefixes[0].checkpoint is not None
    frame = frozen_prefix_frame(prefixes[0].checkpoint)
    if frame is None:
        raise RuntimeError("Finite response-law canary primary prefix lacks causal ports")
    word = PreparedForceWord(config.magnitude, config.direction_index, config.sign)
    hold = PreparedForceWord(Decimal(0), 0, 0)
    views = []
    for refinement, prefix in zip((1, 2), prefixes, strict=True):
        parent = execute_native_phase(
            source, parent_task, refinement, incoming=prefix.checkpoint, frame=frame
        )
        if parent.checkpoint is None:
            raise RuntimeError("Finite response-law canary parent lacks a native handoff")
        futures = []
        for purpose in ("future-1", "future-2"):
            pair = []
            for selected in (hold, word):
                task = next(
                    task
                    for task in invocations
                    if task.phase == "future"
                    and task.purpose == purpose
                    and task.word == selected
                )
                value = execute_native_phase(
                    source, task, refinement, incoming=parent.checkpoint, frame=frame
                )
                if value.checkpoint is None:
                    raise RuntimeError("Finite response-law canary future did not complete")
                pair.append(value)
            reference, applied = pair
            receiver = prepared_receiver(
                frame, applied.positions[-1, 0] - reference.positions[-1, 0]
            )
            futures.append(
                {
                    "purpose": purpose,
                    "requested_word": word.word_id,
                    "accepted": applied.delivery.accepted,
                    "applied_force_kicks": applied.delivery.applied_force_kicks,
                    "nonzero_force_intervals": applied.delivery.nonzero_force_intervals,
                    "realized_impulse": tuple(
                        str(v) for v in applied.delivery.realized_impulse
                    ),
                    "hold_realized_impulse": tuple(
                        str(v) for v in reference.delivery.realized_impulse
                    ),
                    "paired_receiver": tuple(float(v) for v in receiver),
                }
            )
        views.append(
            {
                "refinement": refinement,
                "native_timestep": str(Decimal("0.001") / refinement),
                "prefix_end_tick": int(prefix.ticks[-1]),
                "parent_end_tick": int(parent.ticks[-1]),
                "receiver_end_tick": 4560,
                "futures": futures,
            }
        )
    return {
        "config_id": config.config_id,
        "source_spec_sha256": source.fingerprint(),
        "independent_roots": 1,
        "nested_numerical_views": 2,
        "parent": CANARY_PARENT,
        "force_unit": "dimensionless-native-force",
        "clock_unit": "dimensionless-langevin-time",
        "evidence_role": config.evidence_role,
        "development_only": True,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
        "views": views,
    }


__all__ = ['FiniteResponseLawCanary', "run_finite_canary"]
