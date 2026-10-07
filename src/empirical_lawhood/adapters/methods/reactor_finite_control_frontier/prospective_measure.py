"""Keep the independently delivered owner branch separate from its audit chart."""

from dataclasses import dataclass
from typing import ClassVar
import numpy as np

from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.prospective import FrontierNativeRoot
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.controller_runtime import TickDisposition
from .config import Coordinate
from .measurement import FrontierGuard, FrontierMeasuredRoot, FrontierObservation, measure_coordinate, measure_root, measure_guard
from .prospective_lock import FrontierFrozenRoot, FrontierFrozenUse
from .records import FrontierPreparation
from .selection import FrontierUseRequest


@dataclass(frozen=True, slots=True)
class FrontierMeasuredUse(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-measured-use'
    request: FrontierUseRequest
    observation: FrontierObservation | None
    owner_delivered: bool
    guard: FrontierGuard | None


@dataclass(frozen=True, slots=True)
class FrontierMeasuredProspective(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-measured-prospective'
    root: str
    native: ObjectIdentity
    chart: FrontierMeasuredRoot
    uses: tuple[FrontierMeasuredUse, ...]


def measure_prospective(
    preparation: FrontierPreparation,
    frozen: FrontierFrozenRoot,
    native: FrontierNativeRoot,
    frozen_uses: tuple[FrontierFrozenUse, ...],
) -> FrontierMeasuredProspective:
    if (
        len(frozen_uses) != len(frozen.use_archives)
        or any(
            archive.subject
            != ObjectIdentity.from_record(f"{frozen.root}.{use.request.request_id}.frozen-use", use)
            for archive, use in zip(frozen.use_archives, frozen_uses, strict=True)
        )
        or native.frozen != ObjectIdentity.from_record(frozen.record_id, frozen)
        or frozen.preparation != ObjectIdentity.from_record(preparation.record_id, preparation)
        or tuple(u.request for u in native.uses) != tuple(u.request for u in frozen_uses)
    ):
        raise ValueError("prospective measurement changed the frozen use census")
    chart = measure_root(preparation, native.assay)
    audit = native.assay.arrays.unpack()
    uses = []
    for selected, actual in zip(frozen_uses, native.uses, strict=True):
        if selected.projection is None:
            if actual.native_calls or actual.tick is not None:
                raise ValueError("a nonattempt acquired a native owner")
            uses.append(FrontierMeasuredUse(selected.request, None, False, None))
            continue
        request, projection = selected.request, selected.projection
        prefix = f"{request.context}_{projection.pulse.word_id}_"
        # Never let a missing owner window borrow the separately acquired audit pulse.
        data = {k: v for k, v in audit.items() if not k.startswith(prefix)}
        data.update({f"{prefix}{k}": v for k, v in actual.arrays.unpack().items()})
        row = measure_coordinate(
            preparation, Coordinate(request.context, projection.pulse, request.horizon_s), data
        )
        delivered = (
            actual.tick is not None
            and actual.tick.disposition is TickDisposition.ACTION_DELIVERED
            and actual.tick.commitment == selected.commitment
            and len(actual.native_deliveries) == 12
        )
        arrays = actual.arrays.unpack()
        if delivered:
            for i, native_delivery in enumerate(actual.native_deliveries):
                d = native_delivery
                delivered &= all(k in arrays for k in ("v0_requests", "v0_stages", "v0_exposure"))
                if delivered:
                    delivered &= (
                        np.array_equal(
                            arrays["v0_requests"][i],
                            (float(d.command.feed_kg_s), float(d.command.jacket_k)),
                        )
                        and np.array_equal(
                            arrays["v0_stages"][i],
                            tuple(
                                float(v)
                                for v in (
                                    d.accepted_feed_kg_s,
                                    d.accepted_jacket_k,
                                    d.applied_feed_kg_s,
                                    d.applied_jacket_k,
                                )
                            ),
                        )
                        and np.array_equal(
                            arrays["v0_exposure"][i],
                            tuple(
                                tuple(
                                    float(v)
                                    for v in (e.time_s, e.duration_s, e.feed_kg_s, e.jacket_k)
                                )
                                for e in d.exposures
                            ),
                        )
                    )
        uses.append(
            FrontierMeasuredUse(
                request,
                row,
                bool(delivered),
                measure_guard(preparation, request.context, projection.pulse, data),
            )
        )
    return FrontierMeasuredProspective(
        preparation.root, ObjectIdentity.from_record(native.record_id, native), chart, tuple(uses)
    )
