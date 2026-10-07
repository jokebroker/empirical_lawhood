"""Published causal observer plus held-tape forecasts, comparator only."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_prefix_response.causal_feed_forecast import causal_feed_forecast
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import COORDINATES, WORDS, ZERO, Coordinate, assignment
from .records import FrontierContext


@dataclass(frozen=True, slots=True)
class FrontierForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-forecast'
    root: str
    context: ObjectIdentity
    values: tuple[tuple[Coordinate, D, D, D], ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        if (
            self.context.object_schema != FrontierContext.SCHEMA
            or (not self.values and not self.reasons)
            or (
                self.values
                and (
                    self.reasons
                    or len(self.values) != 18
                    or tuple(row[0] for row in self.values)
                    != tuple(c for c in COORDINATES if c.context == self.values[0][0].context)
                )
            )
            or any(type(v) is not D or not v.is_finite() for _, *row in self.values for v in row)
        ):
            raise ValueError("mechanistic forecast lost its finite causal coordinate census")


def forecast_context(
    source: ReactorBatchSource, context: FrontierContext
) -> FrontierForecast:
    parent = ObjectIdentity.from_record(f"{context.root}.{context.context}.causal", context)

    grids, reasons = causal_feed_forecast(
        source,
        context.root,
        context,
        tuple((w.word_id, tuple(w.request(i * 10) for i in range(12))) for w in WORDS),
    )
    if reasons:
        return FrontierForecast(context.root, parent, (), reasons)
    values = tuple(
        (
            c,
            D(
                repr(
                    float(
                        grids[ZERO.word_id][: c.horizon_s + 1, 1].max()
                        - grids[c.pulse.word_id][: c.horizon_s + 1, 1].max()
                    )
                )
            ),
            D(repr(float(grids[c.pulse.word_id][:, 1].max()))),
            D(repr(float(grids[ZERO.word_id][:, 1].max()))),
        )
        for c in COORDINATES
        if c.context == context.context
    )
    return FrontierForecast(context.root, parent, values, ())
