"""preparation-policy development root-level screen using the unchanged qualified lower payload."""

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .fitting import DELTA
from .intervals import oriented_interval
from .law_payloads import FiniteResponseLawLowerPayload
from .science import FiniteResponseLawScienceSpec, development_requests
from .preparation_policy_fitting import NU, AdequacyProvider, DevelopmentGates, PreparationPolicyFeatures, PreparationPolicyPanel, UpperPointFit, UpperWidthFit, coefficient_folds, coefficient_oof, composed_prediction, development_gates, fit_adequacy_provider, fit_upper_points, fit_upper_widths, outer_folds, provisional_quantile, provisional_scores
from .preparation_policy_projection import FiniteResponseLawPreparationPolicyRootPanel
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import PREPARATION_POLICY_SCHEDULE_IDS

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AdequacyOperands:
    conjuncts: BoolArray  # root,schedule,S/V/E/W/P/K
    maximum_error: Array
    maximum_width: Array
    maximum_numerical_difference: Array
    maximum_preservation_ratio: Array
    maximum_work: Array

    @property
    def event(self) -> BoolArray:
        return np.asarray(self.conjuncts.all(axis=2))


def panel_arrays(
    panels: tuple[FiniteResponseLawPreparationPolicyRootPanel, ...],
) -> tuple[PreparationPolicyFeatures, PreparationPolicyPanel, Array]:
    if len(panels) != 24:
        raise ValueError("preparation-policy development requires exactly 24 authenticated root panels")
    x = np.empty((24, 24, 2))
    z = np.empty((24, 9, 24, 2))
    y = np.full((24, 9, 4, 8, 2, 2), np.nan)
    observed = np.zeros_like(y, dtype=bool)
    valid = np.zeros_like(y, dtype=bool)
    work = np.full((24, 9, 2), np.nan)
    root_ids = []
    cohorts = []
    cohort_indices = np.empty(24, dtype=np.int64)
    for r, report in enumerate(panels):
        root = report.root
        root_ids.append(f"{root.cohort}.prepared.r{root.index:03d}")
        cohorts.append(root.cohort)
        cohort_indices[r] = root.index
        x[r] = np.asarray(report.prefix_features, dtype=float).T
        z[r] = np.asarray(report.handoff_features, dtype=float).transpose(0, 2, 1)
        work[r] = np.asarray(report.parent_work, dtype=float)
        for cell in report.cells:
            p = PREPARATION_POLICY_SCHEDULE_IDS.index(cell.schedule_id)
            f = ("future-1", "future-2").index(cell.future)
            v = cell.refinement - 1
            y[r, p, cell.pair_index, :, f, v] = np.asarray(cell.values, dtype=float)
            observed[r, p, cell.pair_index, :, f, v] = cell.observed
            valid[r, p, cell.pair_index, :, f, v] = cell.valid
    features = PreparationPolicyFeatures(tuple(root_ids), tuple(cohorts), cohort_indices, x, z)
    panel = PreparationPolicyPanel(y, observed, valid)
    return features, panel, work


def full_menu_adequacy(
    lower: FiniteResponseLawLowerPayload,
    features: PreparationPolicyFeatures,
    panel: PreparationPolicyPanel,
    work: Array,
) -> AdequacyOperands:
    if lower.q is None or work.shape != (24, 9, 2):
        raise ValueError("preparation-policy development adequacy requires calibrated lower law and paired work")
    actual = lower.predict(features.z[:, :, :, 0].reshape(-1, 24))
    mean = actual.mean.reshape(24, 9, 4, 8)
    sigma = actual.sigma.reshape(24, 9, 4, 8)
    support = actual.supported.reshape(24, 9)
    h = float(lower.q) * sigma + NU[None, None, None, :]
    measured = panel.observed & panel.valid & np.isfinite(panel.y)
    complete = measured.all(axis=(2, 3, 4, 5))
    numerical_difference = np.max(
        np.abs(panel.y[..., 0] - panel.y[..., 1]), axis=(2, 3, 4)
    )
    numerical = numerical_difference <= float(np.max(NU))
    # Per-cell thresholds differ; retain the scalar diagnostic separately from the exact event.
    numerical = (
        np.abs(panel.y[..., 0] - panel.y[..., 1])
        <= NU[None, None, None, :, None]
    ).all(axis=(2, 3, 4))
    error = np.abs(panel.y - mean[..., None, None])
    error_event = (error <= DELTA[None, None, None, :, None, None]).all(
        axis=(2, 3, 4, 5)
    )
    width_event = (h <= DELTA[None, None, None, :]).all(axis=(2, 3))
    caps = np.tile(np.asarray(FiniteResponseLawScienceSpec().preservation, dtype=float), 2)
    preservation_values = panel.y[:, :, :, 2:]
    preservation = (
        preservation_values <= caps[None, None, None, :, None, None]
    ).all(axis=(2, 3, 4, 5))
    preservation_ratio = np.max(
        preservation_values / caps[None, None, None, :, None, None],
        axis=(2, 3, 4, 5),
    )
    work_event = np.isfinite(work).all(axis=2) & (
        work <= float(FiniteResponseLawScienceSpec().parent_work_maximum)
    ).all(axis=2)
    conjuncts = np.stack(
        (support, complete & numerical, error_event, width_event, preservation, work_event),
        axis=2,
    )
    return AdequacyOperands(
        np.asarray(conjuncts),
        np.max(error / DELTA[None, None, None, :, None, None], axis=(2, 3, 4, 5)),
        np.max(h / DELTA[None, None, None, :], axis=(2, 3)),
        numerical_difference,
        preservation_ratio,
        np.max(work, axis=2),
    )


def _single_policy_choices(
    prediction: Array,
    sigma: Array,
    q: float,
    support: BoolArray,
    direction: IntArray,
    requirement: Array,
) -> IntArray:
    n = len(prediction)
    h = q * sigma + NU[None, None, :]
    low, high = prediction - h, prediction + h
    low[..., 2:] = np.maximum(low[..., 2:], 0)
    high[..., 2:] = np.maximum(high[..., 2:], 0)
    precise = np.asarray((h <= DELTA[None, None, :]).all(axis=2), dtype=np.bool_)
    caps = np.tile(np.asarray(FiniteResponseLawScienceSpec().preservation, dtype=float), 2)
    feasible = np.zeros((n, 256, 2, 8), dtype=bool)
    spec = FiniteResponseLawScienceSpec()
    for word in range(8):
        pair = word // 2
        lo, hi = oriented_interval(low, high, word)
        for consumer in (0, 1):
            axis = direction[:, :, consumer] // 2
            positive = direction[:, :, consumer] % 2 == 0
            a = np.where(axis == 0, lo[:, None, 0], lo[:, None, 1])
            b = np.where(axis == 0, hi[:, None, 0], hi[:, None, 1])
            longitudinal_low = np.where(positive, a, -b)
            longitudinal_high = np.where(positive, b, -a)
            transverse = np.where(
                axis == 0,
                np.maximum(abs(lo[:, None, 1]), abs(hi[:, None, 1])),
                np.maximum(abs(lo[:, None, 0]), abs(hi[:, None, 0])),
            )
            feasible[:, :, consumer, word] = (
                support[:, None]
                & precise[:, None, pair]
                & (hi[:, None, 2:] <= caps[None, None, :]).all(axis=2)
                & (longitudinal_low >= requirement[:, :, consumer])
                & (longitudinal_high <= float(spec.upper[consumer]))
                & (transverse <= float(spec.transverse[consumer]))
            )
    return np.where(feasible.any(axis=3), feasible.argmax(axis=3), -1).astype(np.int64)


def actual_joint(
    panel: PreparationPolicyPanel,
    work: Array,
    roots: IntArray,
    schedules: IntArray,
    selected: IntArray,
    direction: IntArray,
    requirement: Array,
) -> BoolArray:
    spec = FiniteResponseLawScienceSpec()
    result = np.zeros((len(roots), 256), dtype=bool)
    caps = np.tile(np.asarray(spec.preservation, dtype=float), 2)
    for i, (root, schedule) in enumerate(zip(roots, schedules, strict=True)):
        for request in range(256):
            success = []
            for consumer in (0, 1):
                word = int(selected[i, request, consumer])
                if word < 0:
                    success.append(False)
                    continue
                pair, sign = word // 2, (-1 if word % 2 == 0 else 1)
                values = panel.y[root, schedule, pair]
                source = (
                    panel.observed[root, schedule, pair]
                    & panel.valid[root, schedule, pair]
                    & np.isfinite(values)
                ).all()
                numerical = (
                    np.abs(values[..., 0] - values[..., 1])
                    <= NU[:, None]
                ).all()
                response = sign * values[:2]
                axis = direction[i, request, consumer] // 2
                polarity = 1 if direction[i, request, consumer] % 2 == 0 else -1
                longitudinal = polarity * response[axis]
                transverse = polarity * response[1 - axis]
                success.append(
                    bool(
                        source
                        and numerical
                        and np.isfinite(work[root, schedule]).all()
                        and (work[root, schedule] <= float(spec.parent_work_maximum)).all()
                        and (values[2:] <= caps[:, None, None]).all()
                        and (longitudinal >= requirement[i, request, consumer]).all()
                        and (longitudinal <= float(spec.upper[consumer])).all()
                        and (np.abs(transverse) <= float(spec.transverse[consumer])).all()
                    )
                )
            result[i, request] = all(success)
    return result


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyUpperPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-upper-payload'
    payload_id: str
    lower: ObjectIdentity
    lower_artifact: ArtifactIdentity
    screen_config: ObjectIdentity
    panels: tuple[ObjectIdentity, ...]
    schedule_ids: tuple[str, ...]
    best_fixed_schedule_id: str
    provisional_q: Decimal
    normalizer_center: tuple[Decimal, ...]
    normalizer_scale: tuple[Decimal, ...]
    upper_operators: tuple[Decimal, ...]
    base_scale: tuple[Decimal, ...]
    multipliers: tuple[Decimal, ...]
    provider_operators: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if (
            self.lower.object_schema != FiniteResponseLawLowerPayload.SCHEMA
            or self.screen_config.object_schema
            != 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-screen-config'
            or len(self.panels) != 24
            or any(
                panel.object_schema != 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-root-panel'
                for panel in self.panels
            )
            or self.schedule_ids != PREPARATION_POLICY_SCHEDULE_IDS
            or self.best_fixed_schedule_id not in self.schedule_ids
            or not self.provisional_q.is_finite()
            or self.provisional_q < 0
            or len(self.normalizer_center) != 24
            or len(self.normalizer_scale) != 24
            or len(self.upper_operators) != 9 * 25 * 24
            or len(self.base_scale) != 4 * 8
            or len(self.multipliers) != 9 * 25
            or len(self.provider_operators) != 9 * 25
        ):
            raise ValueError("preparation-policy upper package changes the frozen preparation-policy development geometry")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.payload_id, self)


def _decimals(value: Array) -> tuple[Decimal, ...]:
    return tuple(Decimal(format(float(item), ".17g")) for item in value.ravel())


@dataclass(frozen=True)
class PreparationScreenComputation:
    adequacy: AdequacyOperands
    selected: IntArray
    fixed: IntArray
    selected_joint: BoolArray
    fixed_joint: BoolArray
    gates: DevelopmentGates
    final_points: UpperPointFit | None
    final_widths: UpperWidthFit | None
    final_provider: AdequacyProvider | None
    final_q: float | None
    dependencies: tuple['FiniteResponseLawPreparationPolicyFitDependency', ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyFitDependency(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-preparation-policy-fit-dependency'
    object_id: str
    role: str
    training_root_ids: tuple[str, ...]
    held_root_ids: tuple[str, ...]
    scientific_dependencies: tuple[str, ...]
    output_sha256: str

    def __post_init__(self) -> None:
        if (
            self.role
            not in ("upper-point", "coefficient-oof", "upper-width", "provider", "quantile")
            or not self.training_root_ids
            or set(self.training_root_ids) & set(self.held_root_ids)
            or len(self.output_sha256) != 64
        ):
            raise ValueError("preparation-policy fit dependency loses its root exclusion or output identity")


def _fit_digest(*arrays: object) -> str:
    digest = sha256()
    for value in arrays:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def compute_preparation_screen_screen(
    panels: tuple[FiniteResponseLawPreparationPolicyRootPanel, ...], lower: FiniteResponseLawLowerPayload
) -> PreparationScreenComputation:
    features, panel, work = panel_arrays(panels)
    adequacy = full_menu_adequacy(lower, features, panel, work)
    folds = outer_folds(features)
    selected_policy = np.empty(24, dtype=np.int64)
    fixed_policy = np.empty(24, dtype=np.int64)
    selected_joint = np.zeros((24, 256), dtype=bool)
    fixed_joint = np.zeros((24, 256), dtype=bool)
    requests = development_requests(FiniteResponseLawScienceSpec())
    request_rows = np.r_[np.arange(8), 16 + np.arange(16)]
    direction = np.asarray(requests["direction"][request_rows], dtype=np.int64)
    requirement = np.asarray(requests["lower"][request_rows], dtype=np.float64)
    known = np.ones((24, 9), dtype=bool)
    dependencies: list[FiniteResponseLawPreparationPolicyFitDependency] = []
    for fold in range(4):
        train = np.flatnonzero(folds != fold).astype(np.int64)
        held = np.flatnonzero(folds == fold).astype(np.int64)
        points = fit_upper_points(features, train)
        oof_prediction, oof_support, coefficient_fits = coefficient_oof(
            features, train, lower
        )
        widths = fit_upper_widths(panel, features, train, oof_prediction, points.normalizer)
        oof_sigma, _ = widths.sigma(points.normalizer, features.x[train, :, 0])
        score = provisional_scores(
            panel, train, oof_prediction, oof_sigma, oof_support
        )
        rank, q = provisional_quantile(score, expected_n=18)
        if rank != 18:
            raise ValueError("preparation-policy development outer uncertainty does not use rank 18")
        provider = fit_adequacy_provider(
            features, train, adequacy.event, known, PREPARATION_POLICY_SCHEDULE_IDS
        )
        prefix = f"preparation-policy.outer-{fold}"
        train_ids = tuple(features.root_ids[index] for index in train)
        held_ids = tuple(features.root_ids[index] for index in held)
        dependencies.append(
            FiniteResponseLawPreparationPolicyFitDependency(
                f"{prefix}.upper-point",
                "upper-point",
                train_ids,
                held_ids,
                ("authenticated-prefix", "authenticated-nine-handoffs", "frozen-ridge-10"),
                _fit_digest(points.normalizer.center, points.normalizer.scale, points.operators),
            )
        )
        inner = coefficient_folds(features, train)
        for inner_fold, fit in enumerate(coefficient_fits):
            inner_held = train[inner == inner_fold]
            dependencies.append(
                FiniteResponseLawPreparationPolicyFitDependency(
                    f"{prefix}.coefficient-{inner_fold}",
                    "coefficient-oof",
                    tuple(features.root_ids[index] for index in fit.roots),
                    tuple(features.root_ids[index] for index in inner_held),
                    ("unchanged-lower", "cohort-ranked-three-fold", "native-U2-output"),
                    _fit_digest(
                        fit.normalizer.center, fit.normalizer.scale, fit.operators
                    ),
                )
            )
        dependencies.extend(
            (
                FiniteResponseLawPreparationPolicyFitDependency(
                    f"{prefix}.width",
                    "upper-width",
                    train_ids,
                    held_ids,
                    tuple(
                        f"{prefix}.coefficient-{inner_fold}"
                        for inner_fold in range(3)
                    ),
                    _fit_digest(
                        widths.base,
                        widths.multipliers,
                        widths.complete_targets,
                        widths.rms_eligible,
                    ),
                ),
                FiniteResponseLawPreparationPolicyFitDependency(
                    f"{prefix}.provider",
                    "provider",
                    train_ids,
                    held_ids,
                    ("full-menu-A-only", "no-request-input", "frozen-ridge-10"),
                    _fit_digest(
                        provider.normalizer.center,
                        provider.normalizer.scale,
                        provider.operators,
                    ),
                ),
                FiniteResponseLawPreparationPolicyFitDependency(
                    f"{prefix}.q",
                    "quantile",
                    train_ids,
                    held_ids,
                    (f"{prefix}.width", "all-nine-schedule-root-maximum", "rank-18"),
                    _fit_digest(score, np.asarray((q,))),
                ),
            )
        )
        policies, _ = provider.choose(features.x[held, :, 0])
        fixed = np.full(len(held), provider.best_fixed, dtype=np.int64)
        prediction, support, _ = composed_prediction(points, features, held, lower)
        sigma, _ = widths.sigma(points.normalizer, features.x[held, :, 0])
        for policies_here, destination in (
            (policies, selected_joint),
            (fixed, fixed_joint),
        ):
            rows = np.arange(len(held))
            choices = _single_policy_choices(
                prediction[rows, policies_here],
                sigma[rows, policies_here],
                q,
                support[rows, policies_here],
                direction[held],
                requirement[held],
            )
            destination[held] = actual_joint(
                panel,
                work,
                held,
                policies_here,
                choices,
                direction[held],
                requirement[held],
            )
        selected_policy[held] = policies
        fixed_policy[held] = fixed
    gates = development_gates(
        adequacy.event,
        selected_policy,
        fixed_policy,
        selected_joint,
        fixed_joint,
    )
    if not gates.passed:
        return PreparationScreenComputation(
            adequacy,
            selected_policy,
            fixed_policy,
            selected_joint,
            fixed_joint,
            gates,
            None,
            None,
            None,
            None,
            tuple(dependencies),
        )
    roots = np.arange(24, dtype=np.int64)
    final_points = fit_upper_points(features, roots)
    final_oof, final_support, _ = coefficient_oof(features, roots, lower)
    final_widths = fit_upper_widths(
        panel, features, roots, final_oof, final_points.normalizer
    )
    final_sigma, _ = final_widths.sigma(final_points.normalizer, features.x[:, :, 0])
    final_scores = provisional_scores(
        panel, roots, final_oof, final_sigma, final_support
    )
    rank, final_q = provisional_quantile(final_scores, expected_n=24)
    if rank != 23:
        raise ValueError("preparation-policy development final provisional uncertainty does not use rank 23")
    final_provider = fit_adequacy_provider(
        features, roots, adequacy.event, known, PREPARATION_POLICY_SCHEDULE_IDS
    )
    return PreparationScreenComputation(
        adequacy,
        selected_policy,
        fixed_policy,
        selected_joint,
        fixed_joint,
        gates,
        final_points,
        final_widths,
        final_provider,
        final_q,
        tuple(dependencies),
    )
