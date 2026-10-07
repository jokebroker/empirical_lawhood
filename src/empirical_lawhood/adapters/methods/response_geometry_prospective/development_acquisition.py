"""Finite development-only hypothetical acquisition contact, conditional on shared development evidence.

All development observations/qualification are already common evidence at this cutoff.
Only model-generated continuations are queried here. This produces no native
acquisition, action admission, certificate, or observed acquisition-efficiency claim.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np
from scipy.special import erf

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.methods.receiver_conditioned_io import CanonicalMatrix
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS
from empirical_lawhood.adapters.methods.causal_access_tournament import (
    FeatureMap,
    StandardizedRidge,
    fit_standardized_ridge,
)
from .development_assessment import _support_features
from .development_geometry import response_geometry_development_geometry_steps, response_geometry_development_model_ranks, response_geometry_development_state_scales
from .development_geometry_assessment import ResponseGeometryDevelopmentGeometryCell, ResponseGeometryDevelopmentGeometryReport
from .development_models import DEVELOPMENT_DELTA, DT, FORCES, REPRESENTATIONS, ResponseGeometryDevelopmentAffineModel, ResponseGeometryDevelopmentRootSeries, ResponseGeometryDevelopmentMeasuredView, FloatArray, response_geometry_development_endpoint_coefficients, fit_response_geometry_development_latent_drift, propagate_response_geometry_development_latent, development_series, organize_response_geometry_development_views, root_series
from .development_records import ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentSupportResult, read_response_geometry_development_fit, report_identities
from .development_terminal import ResponseGeometryDevelopmentQualificationConfig


TARGETS = (-DEVELOPMENT_DELTA, DEVELOPMENT_DELTA)
NEAR_OPTIMAL_LOSS = 0.01
MATERIAL_LOSS = 1e-4
PSEUDO_SEED = 2026090702


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("offline ranking cannot serialize a nonfinite score")
    return Decimal(str(float(value)))


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentOfflineRank(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-offline-rank'
    representation: str
    root_index: int
    parent_ids: tuple[str, ...]
    severities: tuple[int, ...]
    scores: tuple[Decimal | None, ...]
    witness_uncertainties: tuple[tuple[Decimal, Decimal, Decimal] | None, ...]
    b_ranking: tuple[str, ...]
    a_ranking: tuple[str, ...]
    placebo_ranking: tuple[str, ...]
    antithetic_selections: tuple[tuple[str, str], ...]
    material_stable_contact: bool
    pseudo_packet_count: int
    coefficient_refit_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.representation not in REPRESENTATIONS
            or self.root_index not in range(32, 64)
            or self.parent_ids != tuple(sorted(PARENTS))
        ):
            raise ValueError("offline development ranking changes its representation/root/query roster")
        if (
            len(self.scores) != 5
            or len(self.severities) != 5
            or len(self.witness_uncertainties) != 5
        ):
            raise ValueError("offline development ranking drops a candidate query")
        if any(v not in (0, 1, 2, 3) for v in self.severities) or any(
            v is not None and not v.is_finite() for v in self.scores
        ):
            raise ValueError("offline development ranking has an invalid score or severity")
        if any(
            w is not None and (len(w) != 3 or any(not v.is_finite() or v < 0 for v in w))
            for w in self.witness_uncertainties
        ):
            raise ValueError("offline development witness uncertainty must be finite and nonnegative")
        for ranking in (self.a_ranking, self.b_ranking, self.placebo_ranking):
            if ranking and (len(ranking) != 5 or set(ranking) != set(self.parent_ids)):
                raise ValueError("offline development ranking must retain all queries or an explicit refusal")
        if any(self.scores[i] is None for i in range(5)):
            if self.a_ranking or self.b_ranking or self.placebo_ranking or not self.reason_codes:
                raise ValueError("unevaluable offline objective cannot produce a ranking")
        if (
            not 0 <= self.pseudo_packet_count <= 40
            or not 0 <= self.coefficient_refit_count <= self.pseudo_packet_count * 4
            or (
                self.a_ranking
                and (self.pseudo_packet_count, self.coefficient_refit_count) != (40, 160)
            )
        ):
            raise ValueError("offline development changes the eight-pseudo/four-refit budget per query")
        if self.a_ranking and len(self.antithetic_selections) != 2:
            raise ValueError("offline development ranking lacks the two antithetic checks")
        if self.material_stable_contact:
            if (
                not self.a_ranking
                or self.a_ranking[0] == self.b_ranking[0]
                or len(self.antithetic_selections) != 2
            ):
                raise ValueError("offline contact lacks its distinct stable selections")
            if any(
                pair != (self.a_ranking[0], self.b_ranking[0])
                for pair in self.antithetic_selections
            ):
                raise ValueError("offline contact is unstable between antithetic halves")
            a = self.scores[self.parent_ids.index(self.a_ranking[0])]
            b = self.scores[self.parent_ids.index(self.b_ranking[0])]
            if a is None or b is None or abs(a - b) < Decimal(".0001"):
                raise ValueError("offline contact lacks a material common-objective difference")


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentOfflineAcquisitionReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-offline-acquisition-report'
    context: str
    config: ObjectIdentity
    fit_result: ObjectIdentity
    calibration_result: ObjectIdentity
    support_result: ObjectIdentity
    geometry_report: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    rows: tuple[ResponseGeometryDevelopmentOfflineRank, ...]
    ensemble_training_roots: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]
    actual_new_native_updates: int = 0
    evidence_cutoff: str = "POST_DEVELOPMENT_COMMON_EVIDENCE"

    def __post_init__(self) -> None:
        if (
            self.context not in ("assembling", "prepared")
            or self.actual_new_native_updates != 0
            or self.evidence_cutoff != "POST_DEVELOPMENT_COMMON_EVIDENCE"
        ):
            raise ValueError(
                "offline development contact cannot claim native acquisition or a hidden-development cutoff"
            )
        if tuple((r.representation, r.root_index) for r in self.rows) != tuple(
            (r, i) for r in REPRESENTATIONS for i in range(32, 64)
        ):
            raise ValueError("offline development contact must retain all 160 representation/root cases")
        if tuple(r for r, _ in self.ensemble_training_roots) != REPRESENTATIONS:
            raise ValueError("offline development changes its complete ensemble roster")
        for _, folds in self.ensemble_training_roots:
            if len(folds) != 4 or any(
                roots != tuple(sorted(set(roots)))
                or (roots and len(roots) < 8)
                or any(i not in range(16) or i % 4 == fold for i in roots)
                for fold, roots in enumerate(folds)
            ):
                raise ValueError("offline ensemble includes a foreign or omitted-fold fitting root")
        if (
            self.config.object_schema != ResponseGeometryDevelopmentQualificationConfig.SCHEMA
            or self.fit_result.object_schema != ResponseGeometryDevelopmentFitResult.SCHEMA
            or self.calibration_result.object_schema != ResponseGeometryDevelopmentCalibrationResult.SCHEMA
            or self.support_result.object_schema != ResponseGeometryDevelopmentSupportResult.SCHEMA
            or self.geometry_report.object_schema != ResponseGeometryDevelopmentGeometryReport.SCHEMA
        ):
            raise ValueError("offline development contact changes its typed evidence lineage")
        expected = tuple(
            sorted(
                f"report.response-geometry-development.{self.context}.r{i:02d}.r{r}"
                for i in (*range(16), *range(32, 64))
                for r in (1, 2)
            )
        )
        if tuple(r.object_id for r in self.input_reports) != expected:
            raise ValueError("offline development contact changes its complete shared input roster")

    @property
    def report_id(self) -> str:
        return f"response-geometry-development.offline-acquisition.{self.context}"


@dataclass(frozen=True)
class ResponseGeometryDevelopmentConditionalEnsemble:
    models: tuple[ResponseGeometryDevelopmentAffineModel, ...]
    latent_training: tuple[FloatArray, ...]
    training_roots: tuple[tuple[int, ...], ...]


def conditional_ensemble(
    model: ResponseGeometryDevelopmentAffineModel, series: dict[int, ResponseGeometryDevelopmentRootSeries]
) -> ResponseGeometryDevelopmentConditionalEnsemble:
    """Coefficient variability conditional on the shared fitted feature map/grid."""
    if tuple(sorted(series)) != model.feature_map.training_roots or any(
        s.index != i or s.context != model.context or s.parent != model.parent
        for i, s in series.items()
    ):
        raise ValueError("development conditional coefficients require the exact common fitting roots")
    rows, models, roots = [], [], []
    for fold in range(4):
        indices = tuple(i for i in sorted(series) if i % 4 != fold)
        if len(indices) < 8:
            raise ValueError("development conditional ensemble lacks eight distinct fitting roots")
        latent = np.stack(
            [
                model.feature_map.transform(series[i].scalar, series[i].features, model.dimension)
                for i in indices
            ]
        )
        drift, diffusion, singular, nulls = fit_response_geometry_development_latent_drift(
            latent, ridge=model.ridge, split=model.split_at_force_off
        )
        # feature_map.training_roots still names the common basis fit. The actual
        # coefficient-refit roots are separately retained in this ensemble/report.
        models.append(
            replace(
                model,
                drift=drift,
                diffusion=diffusion,
                design_singular_values=singular,
                unidentified_directions=nulls,
            )
        )
        rows.append(latent)
        roots.append(indices)
    return ResponseGeometryDevelopmentConditionalEnsemble(tuple(models), tuple(rows), tuple(roots))


def pseudo_packet(
    model: ResponseGeometryDevelopmentAffineModel, initial: FloatArray, standard_noise: FloatArray
) -> FloatArray:
    """One hypothetical paired packet; common innovations are a model assumption."""
    if initial.shape != (2, model.dimension) or standard_noise.shape != (7, model.dimension):
        raise ValueError("development hypothetical packet changes its two-view clock/latent geometry")
    states = np.empty((2, 3, 8, model.dimension))
    states[:, :, 0] = initial[:, None, :]
    for k, dt in enumerate(DT):
        phase = int(model.split_at_force_off and k >= 2)
        coefficient, covariance = model.drift[phase], model.diffusion[phase]
        eigenvalues, vectors = np.linalg.eigh(covariance)
        if eigenvalues.min() < -1e-10 * max(1.0, float(np.linalg.norm(covariance))):
            raise ValueError("development hypothetical innovations are not positive semidefinite")
        noise = np.sqrt(dt) * ((vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ standard_noise[k])
        current = states[:, :, k]
        states[:, :, k + 1] = (
            current
            + dt
            * (current @ coefficient[:-2] + FORCES[:, k, None] * coefficient[-2] + coefficient[-1])
            + noise
        )
    if not np.isfinite(states).all():
        raise FloatingPointError("development hypothetical continuation is nonfinite")
    return states


def _predictions(
    ensembles: tuple[ResponseGeometryDevelopmentConditionalEnsemble, ...], initial: tuple[FloatArray, ...]
) -> tuple[FloatArray, FloatArray]:
    mean = np.empty((4, 5, 3, 2))
    deviation = np.empty((4, 5))
    for q, ensemble in enumerate(ensembles):
        for m, model in enumerate(ensemble.models):
            for v in range(2):
                prediction, sigma = propagate_response_geometry_development_latent(
                    initial[q][v], model.drift, model.diffusion
                )
                mean[m, q, :, v] = prediction[:, -1]
                deviation[m, q] = sigma[-1]
    return mean, deviation


def _decision(
    mean: FloatArray, deviation: FloatArray, support: FloatArray, quantile: float, target: float
) -> tuple[int, int] | None:
    allowed = np.all(np.isfinite(support) & (support >= 0.9), axis=0) & np.all(
        quantile * deviation <= 2 * DEVELOPMENT_DELTA, axis=0
    )
    loss = paired_expected_loss(mean, deviation, target).mean(axis=0)
    loss[~allowed, :] = np.inf
    index = np.unravel_index(int(np.argmin(loss)), loss.shape)
    # Refusal has fixed loss one, so excessive target loss never buys an action.
    return (int(index[0]), int(index[1])) if loss[index] < 1 - 1e-8 else None


def _world_loss(
    mean: FloatArray,
    deviation: FloatArray,
    world: int,
    decision: tuple[int, int] | None,
    target: float,
) -> float:
    if decision is None:
        return 1.0
    q, sign = decision
    return float(paired_expected_loss(mean, deviation, target)[world, q, sign])


def paired_expected_loss(mean: FloatArray, deviation: FloatArray, target: float) -> FloatArray:
    """Exact expected worst-view squared loss under shared Gaussian endpoint noise.

    With x/y the two mean errors and the same N in both views,
    E max((x+N)^2,(y+N)^2) = (x²+y²)/2 + sigma²
    + |x-y| E|N+(x+y)/2|. Shared innovations are an explicit model assumption,
    not empirically qualified joint coverage. This retains stochastic variance.
    """
    errors = (mean - target) / DEVELOPMENT_DELTA
    sigma = deviation[..., None] / DEVELOPMENT_DELTA
    middle = errors.mean(axis=-1)
    positive = sigma > 0
    safe_sigma = np.where(positive, sigma, 1.0)
    absolute = np.where(
        positive,
        safe_sigma * np.sqrt(2 / np.pi) * np.exp(-0.5 * (middle / safe_sigma) ** 2)
        + middle * erf(middle / (np.sqrt(2) * safe_sigma)),
        np.abs(middle),
    )
    return np.asarray(
        np.mean(errors**2, axis=-1) + sigma**2 + np.abs(errors[..., 0] - errors[..., 1]) * absolute,
        dtype=np.float64,
    )


def _rank(
    scores: FloatArray,
    severity: tuple[int, ...],
    witnesses: tuple[tuple[float, float, float] | None, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parents = tuple(sorted(PARENTS))
    eligible = [q for q in range(5) if severity[q] == min(severity)]
    best = max(float(scores[q]) for q in eligible)
    # Quantized tie buckets are not used: compare the actual frozen tolerance.
    ordered = sorted(range(5), key=lambda q: (severity[q], -float(scores[q]), parents[q]))
    tied = [q for q in eligible if best - scores[q] <= 1e-8 * max(1.0, abs(best))]
    b = min(tied, key=lambda q: parents[q])
    ordered.remove(b)
    ordered.insert(0, b)
    near = [q for q in eligible if best - scores[q] <= NEAR_OPTIMAL_LOSS]
    applicable = [q for q in near if witnesses[q] is not None]
    a = b
    if len(applicable) == len(near):
        # Only the witness tuple differs. B knows the same diagnostics and gets
        # charged the same shared computations, but does not rank by them.
        for axis in range(3):
            coordinates = {q: w[axis] for q in applicable if (w := witnesses[q]) is not None}
            maximum = max(coordinates.values())
            applicable = [
                q for q in applicable if maximum - coordinates[q] <= 1e-8 * max(1.0, abs(maximum))
            ]
        best_remaining = max(float(scores[q]) for q in applicable)
        a = min(
            (
                q
                for q in applicable
                if best_remaining - scores[q] <= 1e-8 * max(1.0, abs(best_remaining))
            ),
            key=lambda q: parents[q],
        )
    a_order = [a, *(q for q in ordered if q != a)]
    return tuple(parents[q] for q in a_order), tuple(parents[q] for q in ordered)


def witness_uncertainty(
    ensemble: ResponseGeometryDevelopmentConditionalEnsemble, cell: ResponseGeometryDevelopmentGeometryCell, *, receiver_coordinate: int = 0
) -> tuple[float, float, float] | None:
    """Variability of full-matrix summaries in the baseline's qualified metric.

    These powers are model diagnostics, not repeated measured interventions.
    The existing member/Jacobi/nonnormal owners decide baseline applicability.
    """
    if (
        cell.design is None
        or cell.result is None
        or not cell.result.applicable
        or cell.result.selected_order is None
    ):
        return None
    values = []
    dimension = ensemble.models[0].dimension
    for model in ensemble.models:
        a, b, _, _ = response_geometry_development_endpoint_coefficients(dimension, model.drift, model.diffusion)
        scales = response_geometry_development_state_scales(dimension)
        a, b = a * scales[None, :] / scales[:, None], b / scales
        if (
            np.linalg.norm(a, 2) > 4
            or np.linalg.norm((a - a.T) / 2, 2) > 1
            or np.linalg.norm(b) <= 1e-12
        ):
            return None
        steps = response_geometry_development_geometry_steps(model, cell.design)
        receiver = np.eye(dimension)[receiver_coordinate]
        if receiver_coordinate:
            original = steps[1].receiver_map
            shadow_receiver = CanonicalMatrix.from_array(
                matrix_id=f"{original.matrix_id}.placebo",
                row_coordinate_ids=original.row_coordinate_ids,
                column_coordinate_ids=original.column_coordinate_ids,
                values=receiver[None, :],
            )
            steps = (steps[0], replace(steps[1], receiver_map=shadow_receiver))
        ranks = response_geometry_development_model_ranks(steps)
        symmetric = (a + a.T) / 8  # Norm-four common normalization, including the factor 1/2.
        power = receiver.copy()
        moments = []
        for _ in range(2 * cell.result.selected_order):
            power = symmetric @ power
            moments.append(float(receiver @ power))
        values.append(
            (
                abs(float(receiver @ b)) / float(np.linalg.norm(b)),
                ranks.reachable_observable_quotient_rank / dimension,
                *moments,
            )
        )
    array = np.asarray(values)
    spread = np.ptp(array, axis=0)
    return (
        float(spread[0]),
        float(spread[1]),
        float(np.linalg.norm(spread[2:]) / np.sqrt(array.shape[1] - 2)),
    )


def rank_hypothetical_root(
    *,
    representation: str,
    root_index: int,
    ensembles: tuple[ResponseGeometryDevelopmentConditionalEnsemble, ...],
    initial: tuple[FloatArray, ...],
    support: FloatArray,
    quantile: float,
    witnesses: tuple[tuple[float, float, float] | None, ...],
    placebo: tuple[tuple[float, float, float] | None, ...],
    context: str,
) -> ResponseGeometryDevelopmentOfflineRank:
    if (
        len(ensembles) != 5
        or len(initial) != 5
        or support.shape != (4, 5)
        or not np.isfinite(quantile)
        or quantile < 0
        or len(witnesses) != 5
        or len(placebo) != 5
        or any(len(e.models) != 4 or len(e.latent_training) != 4 for e in ensembles)
        or any(not np.isfinite(v).all() for v in initial)
    ):
        raise ValueError("development offline objective changes its finite ensemble/query/calibration roster")
    try:
        mean, deviation = _predictions(ensembles, initial)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return _refusal(representation, root_index, "CONDITIONAL_PREDICTION_NUMERIC_REFUSAL")
    decisions = tuple(_decision(mean, deviation, support, quantile, target) for target in TARGETS)
    before = np.mean(
        [
            _world_loss(mean, deviation, world, decision, target)
            for world in range(4)
            for target, decision in zip(TARGETS, decisions, strict=True)
        ]
    )
    severity = []
    for q in range(5):
        p = support[:, q]
        if not np.isfinite(p).all() or (p.min() < 0.9 <= p.max()):
            severity.append(0)
        elif any(
            len(set(np.argmin(np.max(((mean[:, q] - t) / DEVELOPMENT_DELTA) ** 2, axis=-1), axis=-1))) > 1
            for t in TARGETS
        ):
            severity.append(1)
        else:
            severity.append(2 if float(np.ptp(mean[:, q], axis=0).max()) >= DEVELOPMENT_DELTA / 64 else 3)
    losses = np.empty((5, 4, 2))
    packets, refits = 0, 0
    seed_text = f"{PSEUDO_SEED}.{context}.{representation}.r{root_index}"
    generator = np.random.Generator(
        np.random.PCG64DXSM(int.from_bytes(sha256(seed_text.encode()).digest()[:16], "big"))
    )
    noises = generator.normal(size=(4, 7, ensembles[0].models[0].dimension))
    for query, ensemble in enumerate(ensembles):
        for world, teacher in enumerate(ensemble.models):
            for half, sign in enumerate((-1, 1)):
                packets += 1
                try:
                    pseudo = pseudo_packet(teacher, initial[query], sign * noises[world])
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    return _refusal(
                        representation, root_index, "PSEUDO_PACKET_NUMERIC_REFUSAL", packets, refits
                    )
                updated = []
                for m, student in enumerate(ensemble.models):
                    latent = np.concatenate((ensemble.latent_training[m], pseudo[None]), axis=0)
                    refits += 1
                    try:
                        drift, diffusion, singular, nulls = fit_response_geometry_development_latent_drift(
                            latent, ridge=student.ridge, split=student.split_at_force_off
                        )
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        return _refusal(
                            representation,
                            root_index,
                            "PSEUDO_REFIT_NUMERIC_REFUSAL",
                            packets,
                            refits,
                        )
                    updated.append(
                        replace(
                            student,
                            drift=drift,
                            diffusion=diffusion,
                            design_singular_values=singular,
                            unidentified_directions=nulls,
                        )
                    )
                after_mean, after_deviation = mean.copy(), deviation.copy()
                for m, student in enumerate(updated):
                    for v in range(2):
                        try:
                            prediction, sigma = propagate_response_geometry_development_latent(
                                initial[query][v], student.drift, student.diffusion
                            )
                        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                            return _refusal(
                                representation,
                                root_index,
                                "PSEUDO_PREDICTION_NUMERIC_REFUSAL",
                                packets,
                                refits,
                            )
                        after_mean[m, query, :, v] = prediction[:, -1]
                        after_deviation[m, query] = sigma[-1]
                if not np.isfinite(after_mean).all() or not np.isfinite(after_deviation).all():
                    return _refusal(
                        representation, root_index, "PSEUDO_PREDICTION_NONFINITE", packets, refits
                    )
                losses[query, world, half] = np.mean(
                    [
                        _world_loss(
                            mean,
                            deviation,
                            world,
                            _decision(after_mean, after_deviation, support, quantile, target),
                            target,
                        )
                        for target in TARGETS
                    ]
                )
    scores = before - losses.mean(axis=(1, 2))
    a, b = _rank(scores, tuple(severity), witnesses)
    shadow, _ = _rank(scores, tuple(severity), placebo)
    halves = tuple(
        (
            _rank(before - losses[:, :, h].mean(axis=1), tuple(severity), witnesses)[0][0],
            _rank(before - losses[:, :, h].mean(axis=1), tuple(severity), witnesses)[1][0],
        )
        for h in range(2)
    )
    parents = tuple(sorted(PARENTS))
    material = (
        a[0] != b[0]
        and abs(scores[parents.index(a[0])] - scores[parents.index(b[0])]) >= MATERIAL_LOSS
        and all(pair == (a[0], b[0]) for pair in halves)
    )
    return ResponseGeometryDevelopmentOfflineRank(
        representation,
        root_index,
        parents,
        tuple(severity),
        tuple(_decimal(v) for v in scores),
        tuple(
            None if w is None else (_decimal(w[0]), _decimal(w[1]), _decimal(w[2]))
            for w in witnesses
        ),
        b,
        a,
        shadow,
        halves,
        material,
        40,
        160,
        (
            "MODEL_CONDITIONAL_HYPOTHETICAL_CONTACT_ONLY",
            "SUPPORT_PROBABILITIES_ARE_NOT_ADMISSION_BOUNDS",
        ),
    )


def _refusal(
    representation: str, index: int, reason: str, packets: int = 0, refits: int = 0
) -> ResponseGeometryDevelopmentOfflineRank:
    return ResponseGeometryDevelopmentOfflineRank(
        representation,
        index,
        tuple(sorted(PARENTS)),
        (0,) * 5,
        (None,) * 5,
        (None,) * 5,
        (),
        (),
        (),
        (),
        False,
        packets,
        refits,
        (reason,),
    )


def assess_response_geometry_development_offline_acquisition(
    *,
    config: ResponseGeometryDevelopmentQualificationConfig,
    fit: ResponseGeometryDevelopmentFitResult,
    fit_payload: bytes,
    calibration: ResponseGeometryDevelopmentCalibrationResult,
    support: ResponseGeometryDevelopmentSupportResult,
    geometry: ResponseGeometryDevelopmentGeometryReport,
    fit_views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    validation_views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
) -> ResponseGeometryDevelopmentOfflineAcquisitionReport:
    """Finite contact conditional on all development evidence; no held-response pseudo truth.

    Features, hyperparameters, calibration and nested conformance labels are
    shared and fixed. The four coefficient fits are a conditional sensitivity
    ensemble, not a posterior or a calibrated confidence set. Hypothetical
    observations update response coefficients only. information-acquisition qualification must separately qualify
    the complete acquisition rule and its action/refusal constraints.
    """
    method_id = ObjectIdentity.from_record(config.method.config_id, config.method)
    fit_id = ObjectIdentity.from_record(fit.result_id, fit)
    calibration_id = ObjectIdentity.from_record(calibration.result_id, calibration)
    fit_reports = report_identities(fit_views)
    validation_reports = report_identities(validation_views)
    if (
        fit.config != method_id
        or calibration.config != method_id
        or support.config != method_id
        or geometry.config != ObjectIdentity.from_record(config.config_id, config)
        or any(
            v != fit_id for v in (calibration.fit_result, support.fit_result, geometry.fit_result)
        )
        or support.calibration_result != calibration_id
        or support.calibration_input_reports != calibration.input_reports
        or fit.input_reports != fit_reports
        or support.input_reports != fit_reports
        or geometry.input_reports != validation_reports
        or any(
            c != fit.context
            for c in (calibration.calibration.context, support.context, geometry.context)
        )
        or any(
            v.report.projection_config != config.method.projection_config
            for v in (*fit_views, *validation_views)
        )
    ):
        raise ValueError("development offline acquisition changes authenticated common evidence lineage")
    native_roots = {r.root_id: r for r in config.source.roots}
    if any(
        native_roots.get(v.report.root.root_id) != v.report.root
        for v in (*fit_views, *validation_views)
    ):
        raise ValueError("development offline acquisition changes a native root")
    groups = read_response_geometry_development_fit(fit, fit_payload)
    fitting = organize_response_geometry_development_views(fit_views, context=fit.context, role="fit")
    queries = organize_response_geometry_development_views(validation_views, context=fit.context, role="validation")
    support_rows = {(s.representation, s.parent): s for s in support.models}
    geometry_cells = {(c.representation, c.parent, c.invocation_offset): c for c in geometry.cells}
    rows, training = [], []
    parents = tuple(sorted(PARENTS))
    quantile = calibration.calibration.quantile
    for group in groups:
        ensembles: tuple[ResponseGeometryDevelopmentConditionalEnsemble, ...] = ()
        predictors = []
        series = development_series(fitting, group.representation)
        models = {m.parent: m for m in group.models}
        witness_cache: dict[tuple[str, int, int], tuple[float, float, float] | None] = {}
        reason = "CONDITIONAL_MODEL_OR_CALIBRATION_UNAVAILABLE"
        if len(models) == 5 and quantile is not None:
            try:
                ensembles = tuple(
                    conditional_ensemble(
                        models[p], {i: series[p][i] for i in models[p].feature_map.training_roots}
                    )
                    for p in parents
                )
                # These labels are already nested out-of-fold. This secondary
                # coefficient sensitivity leaves one fold out, conditional on
                # those shared labels and the common frozen feature map.
                for parent, ensemble in zip(parents, ensembles, strict=True):
                    row = support_rows[group.representation, parent]
                    labels = dict(zip(row.roots, row.labels, strict=True))
                    folds: list[StandardizedRidge | None] = []
                    for roots in ensemble.training_roots:
                        known = tuple(i for i in roots if labels[i] is not None)
                        if len(known) < 8:
                            folds.append(None)
                            continue
                        folds.append(
                            fit_standardized_ridge(
                                np.stack(
                                    [
                                        _support_features(models[parent], series[parent][i])
                                        for i in known
                                    ]
                                ),
                                np.asarray([labels[i] for i in known], dtype=float),
                                feature_map=FeatureMap.LINEAR,
                                ridge_alpha=1.0,
                            )
                        )
                    predictors.append(tuple(folds))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                ensembles = ()
                reason = "CONDITIONAL_ENSEMBLE_NUMERIC_REFUSAL"
        training.append(
            (group.representation, ensembles[0].training_roots if ensembles else ((),) * 4)
        )
        for index, pair in queries.items():
            if not ensembles:
                rows.append(_refusal(group.representation, index, reason))
                continue
            initial, probabilities = [], np.full((4, 5), np.nan)
            witnesses: list[tuple[float, float, float] | None] = []
            placebo: list[tuple[float, float, float] | None] = []
            missing = False
            for q, parent in enumerate(parents):
                query = root_series(pair, parent, group.representation)
                if query is None:
                    missing = True
                    break
                model = models[parent]
                initial.append(
                    model.feature_map.transform(
                        query.scalar[:, 0, 0], query.features[:, 0, 0], model.dimension
                    )
                )
                features = _support_features(model, query)[None]
                for fold, predictor in enumerate(predictors[q]):
                    if predictor is not None:
                        probabilities[fold, q] = float(
                            np.clip(predictor.predict(features)[0], 0, 1)
                        )
                cell = geometry_cells[
                    group.representation, parent, pair[0].report.root.invocation_offset
                ]
                for coordinate, destination in ((0, witnesses), (1, placebo)):
                    key = (parent, cell.invocation_offset, coordinate)
                    if key not in witness_cache:
                        witness_cache[key] = witness_uncertainty(
                            ensembles[q], cell, receiver_coordinate=coordinate
                        )
                    destination.append(witness_cache[key])
            if missing:
                rows.append(
                    _refusal(group.representation, index, "CAUSAL_QUERY_INITIALIZATION_UNAVAILABLE")
                )
                continue
            assert quantile is not None
            rows.append(
                rank_hypothetical_root(
                    representation=group.representation,
                    root_index=index,
                    ensembles=ensembles,
                    initial=tuple(initial),
                    support=probabilities,
                    quantile=float(quantile),
                    witnesses=tuple(witnesses),
                    placebo=tuple(placebo),
                    context=fit.context,
                )
            )
    return ResponseGeometryDevelopmentOfflineAcquisitionReport(
        fit.context,
        ObjectIdentity.from_record(config.config_id, config),
        fit_id,
        calibration_id,
        ObjectIdentity.from_record(support.result_id, support),
        ObjectIdentity.from_record(geometry.report_id, geometry),
        tuple(sorted((*fit_reports, *validation_reports), key=lambda r: r.object_id)),
        tuple(rows),
        tuple(training),
    )
