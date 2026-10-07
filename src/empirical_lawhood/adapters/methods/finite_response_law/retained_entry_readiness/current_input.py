"""Current authenticated preparation operands on the retained readiness mathematics.

The old saved census still defaults to ten known failures. Current inputs keep
all actual failures and never use that historical count as a qualification gate.
"""

from dataclasses import dataclass
from collections import Counter
from typing import ClassVar
import numpy as np

from empirical_lawhood.kernel.matrix_inputs import MatrixAllocation, SavedMatrixArrays
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from .science import enriched_inputs
from .screen import compute
from .control import compatibility
from .posthoc_summary import summarize
from .verification import validate_retained_inputs, check_arrays


@dataclass(frozen=True, slots=True)
class CurrentPreparationReadinessReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/finite-response-law/current-preparation-readiness-report"
    report_id: str
    input_manifest: ObjectIdentity
    allocation: ObjectIdentity
    original_f: ObjectIdentity
    nomination: str | None
    known_failure_cells: int
    compatibility_checked: bool
    follow_on_status: str = "UNENTERED"
    exposure: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self):
        validate_stable_id(self.report_id)
        if (self.input_manifest.object_schema != SavedMatrixArrays.SCHEMA
            or self.allocation.object_schema != MatrixAllocation.SCHEMA
            or self.known_failure_cells < 0 or self.follow_on_status != "UNENTERED"
            or self.exposure != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"):
            raise ValueError("current readiness changes its input/evidence ceiling")


def current_requests(allocation):
    """Preserve the readiness owner's consumer-major draw order on explicit seeds."""
    direction = np.empty((24, 256, 2), dtype=np.int64)
    requirement = np.empty((24, 256, 2), dtype=np.float64)
    if len(allocation.roots) != 24:
        raise ValueError("readiness requests require all24 independent assigned roots")
    for index, root in enumerate(allocation.roots):
        rng = np.random.Generator(np.random.PCG64(root.seed_for("requests")))
        direction[index] = rng.integers(0, 4, (256, 2))
        requirement[index, :, 0] = rng.uniform(.02, .12, 256)
        requirement[index, :, 1] = rng.uniform(.02, .06, 256)
    return direction, requirement


def current_readiness(*, manifest, allocation, arrays, lower, bridge_fits, report_id):
    """Consume already authenticated complete arrays; never acquire native inputs.

    Only the outer radial operators are reused. The retained owner fits its
    required excluded-root baseline/inner uncertainty operands and verifies them
    independently, preserving all covariance cross terms and negative gates.
    """
    if (manifest.allocation != allocation.identity or manifest.original_f != lower.identity
        or tuple(root.cohort for root in allocation.roots) != ("q2",)*8 + ("cir1",)*16):
        raise ValueError("readiness changes current source kinds, allocation or original F")
    required = ("x", "z", "y", "work", "features", "positions", "momenta", "requested_ticks",
                "center", "scale", "operator", "mean", "width", "upper_z", "upper_mean", "conjuncts")
    if any(name not in arrays for name in required):
        raise ValueError("UNEVALUABLE: current readiness lacks its complete required operands")
    m = dict(arrays)
    # Readiness declares six applicability conjuncts, and evaluates realized
    # coverage separately as actual_q. I4's saved seven-column maximum census
    # retains its coverage column; it is not an extra readiness entry gate.
    if m["conjuncts"].shape == (24, 9, 7):
        m["conjuncts"] = m["conjuncts"][..., :6]
    if m["conjuncts"].shape != (24, 9, 6):
        raise ValueError("readiness applicability conjunction census differs")
    ticks = arrays["requested_ticks"]
    expected_ticks = 4096 + 16*np.arange(26, dtype=np.int64)
    if ticks.shape != (24, 26) or ticks.dtype != np.int64 or not np.array_equal(ticks, np.broadcast_to(expected_ticks, (24, 26))):
        raise ValueError("readiness current native trajectory clocks differ")
    trajectory = {name: arrays[name] for name in ("features", "positions", "momenta", "center", "scale", "operator")}
    trajectory["ticks"] = expected_ticks
    bridge = {name: arrays[name] for name in ("radial.kernel", "radial.all.forecast_delta", "radial.all.native")}
    validate_retained_inputs(m, trajectory, lower)
    enriched = enriched_inputs(m["x"], trajectory["positions"], trajectory["momenta"], trajectory["ticks"])
    direction, requirement = current_requests(allocation)
    store = compute(m, trajectory, bridge, bridge_fits, lower, enriched, direction, requirement,
                    expected_known_failure_count=None)
    witnesses = tuple(compatibility(store.arrays["mean"][c], store.arrays["width"][c],
        store.arrays["entry"][c], direction, requirement, store.arrays["eligible"][c],
        sigma=store.arrays["sigma"][c], q=store.arrays["q"][c]) for c in range(4))
    result = summarize(store.arrays, m, all(row["mapped"] for row in witnesses))
    # The independent owner expects JSON's list-valued gate representation.
    checking_result = dict(result, candidates={key: dict(row, gates=None if row["gates"] is None else list(row["gates"]))
                                              for key, row in result["candidates"].items()})
    checked = check_arrays(store.arrays, {"result": checking_result, "fits": store.records,
                          "fit_counts": dict(Counter(row["kind"] for row in store.records))},
                          m, trajectory, bridge, bridge_fits, lower,
                          expected_known_failure_count=None, current_requests=(direction, requirement))
    report = CurrentPreparationReadinessReport(report_id,
        ObjectIdentity.from_record(manifest.operand_id, manifest), allocation.identity, lower.identity,
        result["nomination"], int(store.arrays["known"].sum()), bool(checked["verified"]))
    return report, store, checking_result, witnesses, checked
