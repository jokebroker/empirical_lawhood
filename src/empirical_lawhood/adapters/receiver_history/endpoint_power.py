"""Pure endpoint/coordinate development-power classification for receiver-history."""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.adapters.receiver_history.contracts import (
    ReceiverHistoryCoordinateKind,
    ReceiverHistoryDisorderFamily,
    ReceiverHistoryEndpoint,
    ReceiverHistoryPowerState,
    ReceiverHistoryScientificState,
    ReceiverHistoryTargetedCoordinateAdjudication,
)
from empirical_lawhood.adapters.receiver_history.runtime_contracts import (
    ReceiverHistoryDevelopmentLedger,
    ReceiverHistoryEndpointCoordinatePowerAtlas,
    ReceiverHistoryPowerCell,
)


_ENDPOINT_ATTRIBUTE = {
    ReceiverHistoryEndpoint.DYNAMICAL: "dynamical_state",
    ReceiverHistoryEndpoint.TARGET_DECISION: "target_state",
    ReceiverHistoryEndpoint.SINK_DECISION: "sink_state",
}


def _coordinate_label(
    kind: ReceiverHistoryCoordinateKind, depth: int, budget: Decimal | None
) -> str:
    if kind is ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH:
        return f"k{depth}"
    if budget is None:
        raise ValueError("receiver-history budget power cell lacks its budget")
    return f"b{str(budget).replace('.', 'p')}"


def build_endpoint_power_atlas(
    ledger: ReceiverHistoryDevelopmentLedger,
) -> ReceiverHistoryEndpointCoordinatePowerAtlas:
    """Classify all cells using only the complete typed development ledger."""

    # The complete coordinate ledger lives in the twelve enclosing bundles at
    # workflow construction time.  DevelopmentLedger now carries it directly
    # so classification cannot consult nominations, arrays or caller masks.
    targeted = ledger.targeted_adjudications
    grouped: dict[
        tuple[
            ReceiverHistoryEndpoint,
            str,
            ReceiverHistoryCoordinateKind,
            int,
            Decimal | None,
            Decimal,
            ReceiverHistoryDisorderFamily,
            int,
        ],
        list[ReceiverHistoryTargetedCoordinateAdjudication],
    ] = {}
    for row in targeted:
        for endpoint in ReceiverHistoryEndpoint:
            key = (
                endpoint,
                row.coordinate_id,
                row.coordinate_kind,
                row.depth,
                row.budget,
                row.resolution_epsilon,
                row.family,
                row.scale_cells,
            )
            grouped.setdefault(key, []).append(row)
    cells: list[ReceiverHistoryPowerCell] = []
    for key in sorted(
        grouped,
        key=lambda value: (
            value[0].value,
            value[6].value,
            value[7],
            value[1],
        ),
    ):
        endpoint, coordinate_id, kind, depth, budget, epsilon, family, scale = key
        values = grouped[key]
        if len(values) != 4:
            raise ValueError("receiver-history power cell lost a development unit")
        attribute = _ENDPOINT_ATTRIBUTE[endpoint]
        valid_rows = tuple(
            value for value in values if value.valid and bool(value.generator_observer_agreement)
        )
        states = tuple(getattr(value, attribute) for value in valid_rows)
        witness = sum(value is ReceiverHistoryScientificState.OPPOSED for value in states)
        informative = sum(
            value is ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE for value in states
        )
        limited = sum(
            value is ReceiverHistoryScientificState.TARGETABILITY_LIMITED for value in states
        )
        state = (
            ReceiverHistoryPowerState.INVALID
            if len(valid_rows) != 4
            else ReceiverHistoryPowerState.WITNESS_CAPABLE
            if witness >= 3
            else ReceiverHistoryPowerState.NONADVERSITY_CAPABLE
            if informative >= 3
            else ReceiverHistoryPowerState.MIXED_CAPABLE
            if witness >= 2 and informative >= 2
            else ReceiverHistoryPowerState.UNPOWERED
        )
        label = _coordinate_label(kind, depth, budget)
        cell_id = (
            f"power-cell.{endpoint.value.lower().replace('_', '-')}"
            f".{family.value}.n{scale}.{coordinate_id}"
        )
        cells.append(
            ReceiverHistoryPowerCell(
                cell_id=cell_id,
                endpoint=endpoint,
                coordinate_id=coordinate_id,
                coordinate_label=label,
                depth=depth,
                resolution_epsilon=epsilon,
                family=family,
                scale_cells=scale,
                requested_unit_count=4,
                valid_unit_count=len(valid_rows),
                witness_unit_count=witness,
                informative_nonadverse_unit_count=informative,
                targetability_limited_unit_count=limited,
                power_state=state,
            )
        )
    cells_tuple = tuple(sorted(cells, key=lambda value: value.cell_id))

    def complete_role(
        endpoint: ReceiverHistoryEndpoint,
        label: str,
        required: ReceiverHistoryPowerState,
    ) -> bool:
        selected = tuple(
            value
            for value in cells_tuple
            if value.endpoint is endpoint
            and value.coordinate_label == label
            and value.resolution_epsilon == Decimal("0.005")
        )
        return len(selected) == 9 and all(value.power_state is required for value in selected)

    questions: list[str] = []
    for endpoint in ReceiverHistoryEndpoint:
        slug = endpoint.value.lower().replace("_", "-")
        if complete_role(endpoint, "k0", ReceiverHistoryPowerState.WITNESS_CAPABLE):
            questions.append(f"question.{slug}.k0")
        if complete_role(
            endpoint, "k4", ReceiverHistoryPowerState.WITNESS_CAPABLE
        ) and complete_role(endpoint, "k8", ReceiverHistoryPowerState.NONADVERSITY_CAPABLE):
            questions.append(f"question.{slug}.fixed-k4-k8")
        if complete_role(
            endpoint, "b0p5", ReceiverHistoryPowerState.WITNESS_CAPABLE
        ) and complete_role(endpoint, "b1", ReceiverHistoryPowerState.NONADVERSITY_CAPABLE):
            questions.append(f"question.{slug}.normalized-b0p5-b1")
    capable = {
        ReceiverHistoryPowerState.WITNESS_CAPABLE,
        ReceiverHistoryPowerState.NONADVERSITY_CAPABLE,
        ReceiverHistoryPowerState.MIXED_CAPABLE,
    }
    return ReceiverHistoryEndpointCoordinatePowerAtlas(
        atlas_id="receiver-history.endpoint-coordinate-power-atlas",
        development_ledger_sha256=ledger.fingerprint(),
        cells=cells_tuple,
        eligible_cell_ids=tuple(
            value.cell_id for value in cells_tuple if value.power_state in capable
        ),
        eligible_question_ids=tuple(sorted(questions)),
        invalid_cell_count=sum(
            value.power_state is ReceiverHistoryPowerState.INVALID for value in cells_tuple
        ),
        outcome_count_at_freeze=0,
    )


__all__ = ["build_endpoint_power_atlas"]
