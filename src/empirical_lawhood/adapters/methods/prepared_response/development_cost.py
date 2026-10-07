"""Excluded-canary instrument/prediction cost for the existing dependent refinement selector.

This operation consumes already prepared canary inputs. It never advances a
native trajectory, fits coefficients, publishes evidence or qualifies a law.
The operator retains the input artifacts and charges canary preparation and
payload decoding separately in the complete stage resource measurement.
"""

from statistics import median
from time import process_time_ns

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedCommonStart, PreparedNativeHandoff

from .coefficients import PreparedBilinearPredictorCoefficients
from .development_selection import PreparedResponseDevelopmentCandidateCost
from .projection import project_prepared_handoff


def measure_prepared_response_development_candidate_cost(
    *,
    cost_id: str,
    common: PreparedCommonStart,
    handoffs: tuple[PreparedNativeHandoff, ...],
    coefficients: PreparedBilinearPredictorCoefficients,
) -> PreparedResponseDevelopmentCandidateCost:
    """Median of eleven complete five-parent, nine-word root-view evaluations.

One untimed warm-up precedes the paired measurements. Instrument time includes
authenticated checkpoint-to-public-observation projection at all five handoffs;
prediction time includes both the finite mean and its conditional scale. The
pre-parent policy, other numerical view and artifact I/O have separate census
costs. Every candidate must use the same frozen canary root/view and this rule.
    """
    if (
        common.root.stage != 'excluded-canary'
        or common.frame is None
        or coefficients.model_spec.context != common.root.context
        or tuple(handoff.delivery.parent for handoff in handoffs) != PARENTS
        or len({handoff.delivery.refinement for handoff in handoffs}) != 1
    ):
        raise ValueError("dependent refinement cost requires one complete excluded-canary parent chart/view")
    predictor, scale = coefficients.predictors()
    tier = "I1" if coefficients.structure == "mechanism-i1" else "I0"
    instrument_times, predictor_times = [], []
    for repetition in range(12):
        start = process_time_ns()
        observed = tuple(
            project_prepared_handoff(common, handoff, instrument_tier=tier)
            for handoff in handoffs
        )
        history = np.stack([value.history for value in observed])
        sketch = (
            np.stack([value.sketch for value in observed if value.sketch is not None])
            if tier == "I1" else None
        )
        after_instrument = process_time_ns()
        prediction = predictor.predict(history, sketch)
        uncertainty = scale.predict(history, sketch)
        after_prediction = process_time_ns()
        if prediction.shape != (5, 9, 5, 7) or uncertainty.shape != prediction.shape:
            raise ValueError("dependent refinement cost measurement did not evaluate the complete output chart")
        if repetition:
            instrument_times.append(after_instrument - start)
            predictor_times.append(after_prediction - after_instrument)
    return PreparedResponseDevelopmentCandidateCost(
        cost_id,
        coefficients.structure,
        tier,
        ObjectIdentity.from_record(common.common_start_id, common),
        11,
        int(median(instrument_times)),
        int(median(predictor_times)),
    )
