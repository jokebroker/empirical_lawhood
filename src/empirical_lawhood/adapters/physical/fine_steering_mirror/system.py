"""Physical system identity for the CubeSpec fine-steering-mirror benchmark."""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.systems import (
    IndependentUnitSpec,
    RelationalIdentity,
    SystemBoundaryKind,
    SystemSpec,
)
from empirical_lawhood.kernel.time import (
    AvailabilitySpec,
    CausalPhase,
    ClockLabelSemantics,
    ClockSpec,
    HoldSemantics,
    HorizonSpec,
    SamplingSemantics,
)
from empirical_lawhood.kernel.worlds import WorldKind, WorldSpec


def fine_steering_mirror_system() -> SystemSpec:
    """Return the fixed `L(D,H,A,R,tau)` identity used by R10.

    The denominator deliberately includes the signal generator, voltage
    amplifier, piezoelectric actuation chain, mirror, probes and prepared
    optical-table environment. The source action is measured before the
    amplifier, so this object cannot represent an intrinsic mirror law.
    """

    clock = ClockSpec(
        clock_id="fsm-acquisition-clock",
        label="CubeSpec FSM acquisition clock",
        time_unit="s",
        coordinate_frame="source-period-start",
        sampling=SamplingSemantics.REGULAR,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.INSTANT,
        nominal_period=Decimal("0.00015625"),
        alignment_tolerance=Decimal("0"),
    )

    def availability(phase: CausalPhase, access: OutcomeAccess) -> AvailabilitySpec:
        return AvailabilitySpec(clock_id=clock.clock_id, phase=phase, outcome_access=access)

    quantities = (
        QuantitySpec(
            quantity_id="fsm-action-generator-rms",
            label="Source-defined orthogonal-multisine generator RMS level",
            kind=QuantityKind.ACTION,
            dimension="electric-potential",
            native_unit="V",
            coordinate_frame="pre-amplifier-generator-output",
            clock_id=clock.clock_id,
            availability=availability(CausalPhase.ACTION_APPLIED, OutcomeAccess.OUTCOME_BLIND),
        ),
        QuantitySpec(
            quantity_id="fsm-boundary-optical-table-environment",
            label="Prepared optical-table and ambient boundary",
            kind=QuantityKind.BOUNDARY,
            dimension="prepared-boundary-identity",
            native_unit="1",
            coordinate_frame="cubespec-optical-table",
            clock_id=clock.clock_id,
            availability=availability(CausalPhase.PREPARATION, OutcomeAccess.OUTCOME_BLIND),
        ),
        QuantitySpec(
            quantity_id="fsm-denominator-delivery-chain-and-mirror",
            label="Prepared generator, amplifier, piezo, mirror and probe denominator",
            kind=QuantityKind.DENOMINATOR,
            dimension="prepared-system-identity",
            native_unit="1",
            coordinate_frame="complete-cubespec-fsm-test-setup",
            clock_id=clock.clock_id,
            availability=availability(CausalPhase.PREPARATION, OutcomeAccess.OUTCOME_BLIND),
        ),
        QuantitySpec(
            quantity_id="fsm-history-periodic-steady-state",
            label="Retained periodic waveform and delivery-chain history",
            kind=QuantityKind.HISTORY,
            dimension="waveform-history",
            native_unit="1",
            coordinate_frame="preceding-source-period",
            clock_id=clock.clock_id,
            availability=availability(CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND),
        ),
        *tuple(
            QuantitySpec(
                quantity_id=f"fsm-observation-probe-{index}-displacement",
                label=f"Capacitive probe {index} displacement",
                kind=QuantityKind.OBSERVATION,
                dimension="length",
                native_unit="m",
                coordinate_frame=f"capacitive-probe-{index}",
                clock_id=clock.clock_id,
                availability=availability(
                    CausalPhase.RECEIVER,
                    OutcomeAccess.EVALUATION_SEALED,
                ),
            )
            for index in range(1, 4)
        ),
        *tuple(
            QuantitySpec(
                quantity_id=f"fsm-receiver-probe-{index}-frm-peak-frequency",
                label=f"Probe {index} dominant 750--950 Hz FRM row-norm peak",
                kind=QuantityKind.RECEIVER,
                dimension="frequency",
                native_unit="Hz",
                coordinate_frame=f"capacitive-probe-{index}-frequency-response",
                clock_id=clock.clock_id,
                availability=availability(
                    CausalPhase.POST_OUTCOME,
                    OutcomeAccess.EVALUATOR_REVEAL,
                ),
                response_direction=ResponseDirection.SIGNED_VECTOR,
            )
            for index in range(1, 4)
        ),
    )
    relation = RelationalIdentity(
        relation_id="fsm-amplitude-conditioned-frequency-response",
        denominator_quantity_ids=(
            "fsm-boundary-optical-table-environment",
            "fsm-denominator-delivery-chain-and-mirror",
        ),
        history_quantity_ids=("fsm-history-periodic-steady-state",),
        memoryless=False,
        action_quantity_ids=("fsm-action-generator-rms",),
        receiver_quantity_ids=tuple(
            f"fsm-receiver-probe-{index}-frm-peak-frequency" for index in range(1, 4)
        ),
        horizon=HorizonSpec(
            horizon_id="fsm-one-source-period",
            clock_id=clock.clock_id,
            duration=Decimal("1.28"),
            time_unit="s",
        ),
    )
    authority = AuthorityPolicy(
        policy_id='fine-steering-mirror-nonactuating-policy',
        delegator_id="user-handoff",
        delegate_id="policy-bound-approval-gate",
        scope_ids=("fsm-public-source", "fsm-sealed-evaluation"),
        allowed_world_kinds=frozenset({WorldKind.PHYSICAL_EXPERIMENT}),
        allowed_actions=frozenset(
            {
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.PUBLIC_SOURCE_ACQUISITION,
                AuthorityAction.READ_ONLY_EXPLORATION,
                AuthorityAction.REPOSITORY_IMPLEMENTATION,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.OFFICIAL_OPEN_PUBLIC}),
        required_gate_ids=(
            "clean-commit",
            "custody-separation",
            "official-open-source",
            "resource-envelope",
        ),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.HUMAN_OR_ANIMAL_INTERVENTION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.PAID_OR_EXTERNALLY_BILLED_RESOURCE,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
        ),
        budget_ceiling=ResourceBudget(
            cpu_cores=4,
            memory_bytes=4_000_000_000,
            gpu_devices=0,
            wall_time_seconds=3_600,
            source_scan_bytes=30_000_000,
            output_bytes=10_000_000,
        ),
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    world = WorldSpec(
        world_id="cubespec-fsm-public-physical-experiment",
        label="CubeSpec FSM public physical experiment",
        kind=WorldKind.PHYSICAL_EXPERIMENT,
        represented_physics=(
            "capacitive-probe-displacement-observation",
            "delivery-chain-plus-mirror-frequency-response",
            "source-defined-orthogonal-multisine-excitation",
        ),
        unrepresented_physics=(
            "fresh-preparation-variation",
            "intrinsic-post-amplifier-voltage",
            "spaceflight-environment",
        ),
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.ADMISSION,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.OUTCOME_BLIND,
            }
        ),
    )
    return SystemSpec(
        system_id="cubespec-fsm-delivery-chain-physical-system",
        label="CubeSpec fine-steering-mirror delivery-chain physical system",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=tuple(sorted(quantities, key=lambda item: item.quantity_id)),
        independent_unit=IndependentUnitSpec(
            unit_id="fsm-orthogonal-three-realization-block",
            label="Orthogonal three-realization physical block experiment",
            grouping_key="fsm-block-id",
        ),
        authority_policy=authority,
        ports=(),
    )
