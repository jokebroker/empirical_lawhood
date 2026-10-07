"Target-blind parent decisions from the dependent refinement-frozen policy library."

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedNativeSpec, PreparedRoot
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedCommonStart
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_mechanism_sketch, prepared_observation_history
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import state_from_checkpoint
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.serialization import require_sorted_unique_strings

from .development_policy import POLICIES, PreparedResponseDevelopmentContextPolicy, PreparedResponseDevelopmentPolicyLibrary
from .development_selection import PreparedResponseDevelopmentNominalLibrary


@dataclass(frozen=True, slots=True)
class PreparedPolicyDecisionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-policy-decision-config'
    config_id: str
    native_spec: PreparedNativeSpec
    development_library: PreparedResponseDevelopmentNominalLibrary
    policy_ids: tuple[str, ...] = POLICIES
    decision_cutoff: str = "PRE_PARENT_COMMON_START_PRIMARY_NUMERICAL_VIEW"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.native_spec.stage not in ('calibration', 'conditional-risk-calibration', 'prospective-evaluation')
            or self.development_library.disposition != "NOMINATED_FOR_FRESH_CALIBRATION"
            or self.development_library.policy_library is None
            or self.development_library.config.fit.projection.native_spec.selected_amplitude
            != self.native_spec.selected_amplitude
            or self.development_library.config.fit.projection.native_spec.member != self.native_spec.member
            or self.development_library.config.fit.projection.native_spec.numerical_views
            != self.native_spec.numerical_views
            or self.development_library.config.fit.projection.native_spec.words != self.native_spec.words
            or self.policy_ids != POLICIES
            or self.decision_cutoff != "PRE_PARENT_COMMON_START_PRIMARY_NUMERICAL_VIEW"
        ):
            raise ValueError("prepared policy decision changes its protected-stage roster")

    @property
    def policy_library(self) -> PreparedResponseDevelopmentPolicyLibrary:
        library = self.development_library.policy_library
        if library is None:
            raise ValueError("prepared policy decision lacks its dependent refinement-frozen policy library")
        return library


@dataclass(frozen=True, slots=True)
class PreparedParentDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-parent-decision'
    decision_id: str
    config: ObjectIdentity
    root: PreparedRoot
    policy_id: str
    common_start: ObjectIdentity | None
    context_policy: ObjectIdentity
    selected_parent: str | None
    adequacy_probabilities: tuple[Decimal, ...]
    preparent_instrument_sha256: str
    disposition: str
    reason_codes: tuple[str, ...]
    target_blind: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        validate_sha256(
            self.preparent_instrument_sha256,
            field_name="preparent_instrument_sha256",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.config.object_schema != PreparedPolicyDecisionConfig.SCHEMA
            or self.root.stage not in ('calibration', 'conditional-risk-calibration', 'prospective-evaluation')
            or self.policy_id not in POLICIES
            or self.common_start is not None
            and self.common_start.object_schema != PreparedCommonStart.SCHEMA
            or self.context_policy.object_schema
            != PreparedResponseDevelopmentContextPolicy.SCHEMA
            or any(
                not isinstance(value, Decimal)
                or not value.is_finite()
                or not Decimal(0) < value < Decimal(1)
                for value in self.adequacy_probabilities
            )
            or not self.target_blind
        ):
            raise ValueError("prepared parent decision changes policy, cutoff or adequacy output")
        if self.disposition == "DECIDED":
            if (
                self.common_start is None
                or self.selected_parent not in PARENTS
                or len(self.adequacy_probabilities) != len(PARENTS)
                or self.reason_codes
            ):
                raise ValueError("decided prepared policy lacks its parent or adequacy forecast")
        elif self.disposition == "NONATTEMPT":
            if self.selected_parent is not None or self.adequacy_probabilities or not self.reason_codes:
                raise ValueError("unentered prepared policy cannot fabricate a parent decision")
        else:
            raise ValueError("prepared parent decision has an unknown disposition")


def prepare_parent_decision(
    config: PreparedPolicyDecisionConfig,
    root: PreparedRoot,
    common: PreparedCommonStart | None,
    policy_id: str,
) -> PreparedParentDecision:
    """Apply one frozen policy to the primary pre-parent instrument only."""
    if root not in config.native_spec.roots or policy_id not in POLICIES:
        raise ValueError("prepared parent decision is outside its assigned root/policy roster")
    if common is not None and common.root != root:
        raise ValueError("prepared parent decision received another root's common start")
    context_policy = next(
        value for value in config.policy_library.contexts if value.context == root.context
    )
    if common is None or common.frame is None:
        return PreparedParentDecision(
            f"{root.root_id}.policy.{policy_id}.decision.result",
            ObjectIdentity.from_record(config.config_id, config),
            root,
            policy_id,
            None if common is None else ObjectIdentity.from_record(common.common_start_id, common),
            ObjectIdentity.from_record(context_policy.policy_id, context_policy),
            None,
            (),
            sha256(b"").hexdigest(),
            "NONATTEMPT",
            ("PREFIX_UNAVAILABLE",)
            if common is None
            else ("PORT_FRAME_UNRESOLVED",),
        )
    checkpoint = common.checkpoints[0]
    positions = _decode(checkpoint.history_positions_base64, (31, 2, 3, 4, 4))
    momenta = _decode(checkpoint.history_momenta_base64, (31, 2, 3, 4, 4))
    history = prepared_observation_history(
        frame=common.frame,
        ticks=checkpoint.history_ticks,
        positions=positions,
        momenta=momenta,
        cutoff_tick=common.root.landmark,
    )
    sketch = None
    if context_policy.instrument_tier == "I1":
        state, _ = state_from_checkpoint(checkpoint.native)
        sketch = prepared_mechanism_sketch(state, common.frame)
    selected = context_policy.choose_parent(policy_id, history, sketch)
    probabilities = context_policy.adequacy_probabilities(history, sketch)
    instrument_bytes = np.asarray(history, dtype="<f8").tobytes()
    if sketch is not None:
        instrument_bytes += np.asarray(sketch, dtype="<f8").tobytes()
    return PreparedParentDecision(
        f"{root.root_id}.policy.{policy_id}.decision.result",
        ObjectIdentity.from_record(config.config_id, config),
        root,
        policy_id,
        ObjectIdentity.from_record(common.common_start_id, common),
        ObjectIdentity.from_record(context_policy.policy_id, context_policy),
        selected,
        tuple(Decimal(format(float(value), ".17g")) for value in probabilities),
        sha256(instrument_bytes).hexdigest(),
        "DECIDED",
        (),
    )
