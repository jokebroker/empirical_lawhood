"Target-owned, excluded prepared response native canary through the retained phase mechanics."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .contracts import CONTEXTS, PARENTS, PreparedForceWord, PreparedNativeSpec, PreparedRootRandomness, prepared_native_member, prepared_numerical_view, validate_prepared_seed_census
from .instruments import prepared_receiver
from .source import bind_prepared_common_start, bind_prepared_native_handoff, execute_native_future, execute_native_parent, prepare_native_prefix


@dataclass(frozen=True, slots=True)
class PreparedResponsePreparedCanary(CanonicalRecord):
    "One public development root; the historical source qualification/dependent refinement/evaluation rosters stay separate."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/prepared-response/prepared-response-prepared-canary'
    )
    config_id: str
    seed_label: str
    context: str
    parent: str
    magnitude: Decimal
    direction_index: int
    sign: int
    source_seed_sha256: str
    stage: str = "excluded-canary"
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
    root_seed_census: tuple[PreparedRootRandomness, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.seed_label, field_name="seed_label")
        if (
            not self.config_id.startswith("empirical-lawhood-prepared-response-")
            or not self.seed_label.startswith("empirical-lawhood-prepared-response-")
            or self.context not in CONTEXTS
            or self.parent not in PARENTS
            or self.stage != 'excluded-canary'
            or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        ):
            raise ValueError(
                "prepared response canary requires a target development identity and excluded stage"
            )
        if self.sign == 0:
            raise ValueError("prepared response canary requires a nonzero comparison word")
        PreparedForceWord(self.magnitude, self.direction_index, self.sign)
        object.__setattr__(self, "root_seed_census", validate_prepared_seed_census(
            self.stage, self.source_seed_sha256, self.root_seed_census,
        ))


def _native_spec(config: PreparedResponsePreparedCanary) -> PreparedNativeSpec:
    """Bind a diagnostic source to its actual local code and lock bytes."""

    source_path = Path(__file__).with_name("source.py")
    lock_path = Path(__file__).parents[5] / "uv.lock"
    # An installed wheel has no checkout lock. The wheel diagnostic still uses
    # its own explicit development marker; no clean-Git identity is asserted.
    lock_digest = (
        sha256(lock_path.read_bytes()).hexdigest()
        if lock_path.is_file()
        else sha256(b"installed-wheel-development-canary-no-lock").hexdigest()
    )
    return PreparedNativeSpec(
        'excluded-canary',
        config.source_seed_sha256,
        config.fingerprint(),
        lock_digest,
        ObjectIdentity(
            "prepared-response-development-native-code",
            'empirical-lawhood/simulators/prepared-response/development-native-code',
            "1.0.0",
            sha256(source_path.read_bytes()).hexdigest(),
        ),
        prepared_native_member(),
        tuple(prepared_numerical_view(r) for r in (1, 2)),
        None,
        root_seed_census=config.root_seed_census,
    )


def run_prepared_canary(config: PreparedResponsePreparedCanary) -> dict[str, object]:
    """Execute one root's paired hold/action chart with no scientific promotion."""

    import platform

    if platform.python_version() != "3.11.14" or np.__version__ != "2.4.6":
        raise ValueError("prepared response native canary requires CPython 3.11.14 and NumPy 2.4.6")
    spec = _native_spec(config)
    root = next(r for r in spec.roots if r.context == config.context and r.index == 0)
    prefixes = tuple(prepare_native_prefix(spec, root, refinement=r) for r in (1, 2))
    if any(p.checkpoint is None for p in prefixes):
        raise RuntimeError("prepared response prefix did not supply both native checkpoints")
    common = bind_prepared_common_start(
        prefixes[0].checkpoint, prefixes[1].checkpoint
    )
    if common.frame is None:
        raise RuntimeError("prepared response causal pre-parent ports are unresolved")
    word = PreparedForceWord(config.magnitude, config.direction_index, config.sign)
    hold_word = PreparedForceWord(Decimal(0), 0, 0)
    views = []
    for refinement in (1, 2):
        parent = execute_native_parent(
            spec, common, parent=config.parent, refinement=refinement
        )
        if parent.checkpoint is None:
            raise RuntimeError("prepared response parent did not supply a native handoff")
        handoff = bind_prepared_native_handoff(parent)
        hold = execute_native_future(
            spec,
            common,
            handoff,
            parent=config.parent,
            word=hold_word,
            purpose='common-response',
        )
        action = execute_native_future(
            spec, common, handoff, parent=config.parent, word=word, purpose='common-response'
        )
        if hold.checkpoint is None or action.checkpoint is None:
            raise RuntimeError(
                "prepared response hold/action future lacks a complete native delivery"
            )
        receiver = prepared_receiver(
            common.frame, action.positions[-1, 0] - hold.positions[-1, 0]
        )
        views.append(
            {
                "refinement": refinement,
                "native_timestep": str(spec.numerical_views[refinement - 1].timestep),
                "prefix_end_tick": prefixes[refinement - 1].delivery.requested_end_tick,
                "parent_end_tick": parent.delivery.requested_end_tick,
                "receiver_end_tick": action.delivery.requested_end_tick,
                "requested_word": word.word_id,
                "accepted": action.delivery.accepted,
                "applied_force_kicks": action.delivery.applied_force_kicks,
                "nonzero_force_intervals": action.delivery.nonzero_force_intervals,
                "realized_impulse": tuple(
                    str(v) for v in action.delivery.realized_impulse
                ),
                "hold_realized_impulse": tuple(
                    str(v) for v in hold.delivery.realized_impulse
                ),
                "paired_receiver": tuple(float(v) for v in receiver),
            }
        )
    return {
        "config_id": config.config_id,
        "source_spec_sha256": spec.fingerprint(),
        "independent_roots": 1,
        "nested_numerical_views": 2,
        "context": config.context,
        "parent": config.parent,
        "force_unit": "dimensionless-native-force",
        "clock_unit": "dimensionless-langevin-time",
        "evidence_role": config.evidence_role,
        "development_only": True,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
        "views": views,
    }


__all__ = ['PreparedResponsePreparedCanary', "run_prepared_canary"]
