# SPDX-License-Identifier: MPL-2.0

"""Independent retained preparation-policy development arithmetic.

Preserves explicit fitting, root folds, rank-18 calibration, signed interval
choices and the three conjunctive development gates. Its supplied frozen lower
evaluator is distinct from new qualification. Call only after input authentication
and separate analysis authorization.
"""

from typing import Any
import numpy as np
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from .law_payloads import FiniteResponseLawLowerPayload
from .science import FiniteResponseLawScienceSpec, development_requests
from .preparation_policy_projection import FiniteResponseLawPreparationPolicyRootPanel
from .preparation_screen_results import FiniteResponseLawPreparationScreenResult, FiniteResponseLawPreparationPolicyScreenConfig, FiniteResponseLawPreparationPolicyUpperPayloadFreeze
from .preparation_policy_screen import FiniteResponseLawPreparationPolicyFitDependency
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import PREPARATION_POLICY_SCHEDULE_IDS


def independent(panels: tuple[FiniteResponseLawPreparationPolicyRootPanel, ...], lower: Any) -> dict[str, Any]:
    """Audit the complete exposed development denominator, with explicit loops."""
    spec = FiniteResponseLawScienceSpec()
    delta = np.asarray(spec.delta, dtype=float)
    nu = delta / 8
    caps = np.tile(np.asarray(spec.preservation, dtype=float), 2)
    x = np.asarray([p.prefix_features[0] for p in panels], dtype=float)
    z = np.asarray([[s[0] for s in p.handoff_features] for p in panels], dtype=float)
    work = np.asarray([p.parent_work for p in panels], dtype=float)
    y = np.full((24, 9, 4, 8, 2, 2), np.nan)
    measured = np.zeros_like(y, dtype=bool)
    for r, panel in enumerate(panels):
        for cell in panel.cells:
            index = (
                r,
                PREPARATION_POLICY_SCHEDULE_IDS.index(cell.schedule_id),
                cell.pair_index,
                slice(None),
                int(cell.future[-1]) - 1,
                cell.refinement - 1,
            )
            y[index] = np.asarray(cell.values, dtype=float)
            measured[index] = cell.observed and cell.valid
    if not measured.all() or not np.isfinite(y).all():
        raise ValueError(
            "Independent evaluable audit requires complete valid retained panels"
        )

    a = np.zeros((24, 9, 6), dtype=bool)
    ratios = np.empty((24, 9, 5))
    for r in range(24):
        prediction = lower.predict(z[r])
        for p in range(9):
            error = abs(y[r, p] - prediction.mean[p, :, :, None, None])
            width = float(lower.q) * prediction.sigma[p] + nu
            numerical = abs(y[r, p, ..., 0] - y[r, p, ..., 1])
            preservation = y[r, p, :, 2:]
            a[r, p] = (
                prediction.supported[p],
                (numerical <= nu[None, :, None]).all(),
                (error <= delta[None, :, None, None]).all(),
                (width <= delta).all(),
                (preservation <= caps[None, :, None, None]).all(),
                np.isfinite(work[r, p]).all()
                and (work[r, p] <= float(spec.parent_work_maximum)).all(),
            )
            ratios[r, p] = (
                np.max(error / delta[None, :, None, None]),
                np.max(width / delta),
                np.max(numerical),
                np.max(preservation / caps[None, :, None, None]),
                np.max(work[r, p]),
            )
    adequate = np.asarray(a.all(axis=-1), dtype=bool)
    folds = np.asarray([p.root.index % 4 for p in panels])
    requests = development_requests(spec)
    rows = np.r_[np.arange(8), 16 + np.arange(16)]
    directions = requests["direction"][rows]
    requirements = requests["lower"][rows]
    selected = np.empty(24, dtype=int)
    fixed = np.empty(24, dtype=int)
    successes = np.zeros((2, 24, 256), dtype=bool)
    admitted = np.zeros((2, 24, 256, 2), dtype=bool)
    fit_records: list[dict[str, Any]] = []

    def design(values: Any, center: Any, scale: Any) -> Any:
        return np.column_stack(((values - center) / scale, np.ones(len(values))))

    def ridge(matrix: Any, target: Any) -> Any:
        penalty = 10 * np.eye(25)
        penalty[-1, -1] = 0
        return np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)

    def upper(train: Any, held: Any) -> tuple[Any, Any, Any, Any, Any]:
        center = x[train].mean(axis=0)
        scale = np.maximum(np.sqrt(((x[train] - center) ** 2).mean(axis=0)), 1e-8)
        matrix = design(x[train], center, scale)
        operators = np.stack([ridge(matrix, z[train, p]) for p in range(9)])
        native = np.stack(
            [design(x[held], center, scale) @ op for op in operators], axis=1
        )
        evaluated = lower.predict(native.reshape(-1, 24))
        prediction = evaluated.mean.reshape(len(held), 9, 4, 8)
        prefix_support = abs((x[held] - center) / scale).max(axis=1) <= 6
        support = evaluated.supported.reshape(len(held), 9) & prefix_support[:, None]
        return center, scale, operators, prediction, support

    for fold in range(4):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        center, scale, operators, prediction, support = upper(train, held)
        inner = np.empty(18, dtype=int)
        for cohort in ("prepared-response", "information-response-prediction"):
            positions = [
                i for i, r in enumerate(train) if panels[r].root.cohort == cohort
            ]
            inner[positions] = np.arange(len(positions)) % 3
        oof = np.empty((18, 9, 4, 8))
        oof_support = np.empty((18, 9), dtype=bool)
        inner_records = []
        for part in range(3):
            c, s, op, pred, sup = upper(train[inner != part], train[inner == part])
            oof[inner == part], oof_support[inner == part] = pred, sup
            inner_records.append(
                {
                    "train": train[inner != part].tolist(),
                    "held": train[inner == part].tolist(),
                    "center": c.tolist(),
                    "scale": s.tolist(),
                    "operators": op.tolist(),
                }
            )
        residual = y[train] - oof[..., None, None]
        base = np.maximum(
            delta[None, :] / 20, np.sqrt((residual[..., 0] ** 2).mean(axis=(0, 1, 4)))
        )
        targets = np.log(
            np.maximum(
                0.25,
                (abs(residual) / base[None, None, :, :, None, None]).max(
                    axis=(2, 3, 4, 5)
                ),
            )
        )
        matrix = design(x[train], center, scale)
        multipliers = np.stack([ridge(matrix, targets[:, p, None]) for p in range(9)])

        def sigma(roots: Any) -> Any:
            logg = np.column_stack(
                [design(x[roots], center, scale) @ op for op in multipliers]
            )
            return (
                base[None, None]
                * np.exp(np.clip(logg, np.log(0.25), np.log(4)))[:, :, None, None]
            )

        scaled = (
            np.maximum(abs(residual) - nu[None, None, None, :, None, None], 0)
            / sigma(train)[..., None, None]
        )
        parent_scores = scaled.max(axis=(2, 3, 4, 5))
        numerical = (
            abs(y[train, ..., 0] - y[train, ..., 1]) <= nu[None, None, None, :, None]
        ).all(axis=(2, 3, 4))
        parent_scores[~(numerical & oof_support)] = np.inf
        root_scores = parent_scores.max(axis=1)
        q = float(sorted(root_scores)[17])
        provider = np.stack(
            [ridge(matrix, adequate[train, p, None].astype(float)) for p in range(9)]
        )
        scores = np.clip(
            np.column_stack([design(x[held], center, scale) @ op for op in provider]),
            0,
            1,
        )
        selected[held] = scores.argmax(axis=1)
        fixed[held] = adequate[train].mean(axis=0).argmax()
        widths = q * sigma(held) + nu
        fit_records.append(
            {
                "fold": fold,
                "train": train.tolist(),
                "held": held.tolist(),
                "center": center.tolist(),
                "scale": scale.tolist(),
                "operators": operators.tolist(),
                "inner": inner_records,
                "base": base.tolist(),
                "multipliers": multipliers.tolist(),
                "complete_targets": targets.tolist(),
                "provider": provider.tolist(),
                "root_scores": [None if np.isinf(v) else float(v) for v in root_scores],
                "rank": 18,
                "q": None if np.isinf(q) else q,
            }
        )
        for local, held_root in enumerate(held):
            r = int(held_root)
            for policy, schedule in enumerate((selected[r], fixed[r])):
                for request in range(256):
                    joint = []
                    for consumer in range(2):
                        axis = int(directions[r, request, consumer]) // 2
                        polarity = (
                            1 if int(directions[r, request, consumer]) % 2 == 0 else -1
                        )
                        minimum = float(requirements[r, request, consumer])
                        choice = None
                        for word in range(8):
                            pair, sign = word // 2, (-1 if word % 2 == 0 else 1)
                            mean, h = (
                                prediction[local, schedule, pair],
                                widths[local, schedule, pair],
                            )
                            lower_edge, upper_edge = mean - h, mean + h
                            long_low = min(
                                sign * polarity * lower_edge[axis],
                                sign * polarity * upper_edge[axis],
                            )
                            long_high = max(
                                sign * polarity * lower_edge[axis],
                                sign * polarity * upper_edge[axis],
                            )
                            if (
                                support[local, schedule]
                                and (h <= delta).all()
                                and (np.maximum(upper_edge[2:], 0) <= caps).all()
                                and long_low >= minimum
                                and long_high <= float(spec.upper[consumer])
                                and max(
                                    abs(lower_edge[1 - axis]), abs(upper_edge[1 - axis])
                                )
                                <= float(spec.transverse[consumer])
                            ):
                                choice = (pair, sign)
                                break
                        admitted[policy, r, request, consumer] = choice is not None
                        if choice is None:
                            joint.append(False)
                            continue
                        pair, sign = choice
                        actual = y[r, schedule, pair]
                        longitudinal = sign * polarity * actual[axis]
                        joint.append(
                            bool(
                                (
                                    abs(actual[..., 0] - actual[..., 1]) <= nu[:, None]
                                ).all()
                                and (actual[2:] <= caps[:, None, None]).all()
                                and (
                                    work[r, schedule] <= float(spec.parent_work_maximum)
                                ).all()
                                and (longitudinal >= minimum).all()
                                and (longitudinal <= float(spec.upper[consumer])).all()
                                and (
                                    abs(actual[1 - axis])
                                    <= float(spec.transverse[consumer])
                                ).all()
                            )
                        )
                    successes[policy, r, request] = all(joint)
    headroom = int((~adequate[:, 0] & adequate.any(axis=1)).sum())
    net = int(adequate[np.arange(24), selected].sum()) - int(
        adequate[np.arange(24), fixed].sum()
    )
    improvement = (int(successes[0].sum()) - int(successes[1].sum())) / (24 * 256)
    return {
        "conjuncts": a.tolist(),
        "ratios": ratios.tolist(),
        "selected": selected.tolist(),
        "fixed": fixed.tolist(),
        "joint_successes": successes.sum(axis=2).tolist(),
        "admissions": admitted.sum(axis=(1, 2, 3)).tolist(),
        "headroom_roots": headroom,
        "adequacy_net_roots": net,
        "joint_improvement": improvement,
        "gates_passed": headroom >= 6 and net >= 4 and improvement >= 0.05,
        "fits": fit_records,
    }


def verify_screen_readout(
    config: FiniteResponseLawPreparationPolicyScreenConfig,
    lower: FiniteResponseLawLowerPayload,
    panels: tuple[FiniteResponseLawPreparationPolicyRootPanel, ...],
    screen: FiniteResponseLawPreparationScreenResult,
    adjudication: ScientificAdjudicationRecord,
    freeze: FiniteResponseLawPreparationPolicyUpperPayloadFreeze,
) -> tuple[dict[str, Any], tuple[FiniteResponseLawPreparationPolicyFitDependency, ...]]:
    """Check the original complete screen and reconstruct its fit dependencies.

    Authenticate the frozen source/lower, 24 root panels, successful receipts,
    complete retained recovery census and separate reveal/analysis authority
    before entry. This exposed-data verifier performs independent affine fits
    and deterministic dependency reconstruction; it grants no qualification,
    launches no native acquisition and does not repair or publish a campaign.
    """
    from empirical_lawhood.kernel.provenance import ObjectIdentity

    if (
        screen.config != ObjectIdentity.from_record(config.config_id, config)
        or screen.lower_before != lower.identity
        or screen.lower_after != lower.identity
        or (
            screen.panels
            != tuple(
                sorted(
                    (ObjectIdentity.from_record(p.panel_id, p) for p in panels),
                    key=lambda i: i.object_id,
                )
            )
        )
        or (adjudication.scientific_status != screen.scientific_status)
        or (
            freeze.screen_result != ObjectIdentity.from_record(screen.result_id, screen)
        )
    ):
        raise ValueError("preparation-policy development screen, lower, panels, freeze or adjudication detached")
    audit = independent(panels, lower)
    for r, row in enumerate(screen.root_readouts):
        conjuncts = np.asarray(audit["conjuncts"][r], dtype=bool)
        selected_policy, fixed_policy = (audit["selected"][r], audit["fixed"][r])
        if (
            row.root_id != panels[r].root.stage_unit
            or row.adequacy_conjuncts
            != tuple((tuple(v) for v in audit["conjuncts"][r]))
            or row.adequacy_by_schedule != tuple(np.asarray(conjuncts.all(axis=1)))
            or (row.selected_schedule_id != PREPARATION_POLICY_SCHEDULE_IDS[selected_policy])
            or (row.fixed_schedule_id != PREPARATION_POLICY_SCHEDULE_IDS[fixed_policy])
            or (row.selected_joint_successes != audit["joint_successes"][0][r])
            or (row.fixed_joint_successes != audit["joint_successes"][1][r])
            or (
                row.joined_selected_successes
                != row.selected_joint_successes * bool(conjuncts[selected_policy].all())
            )
            or (
                row.joined_fixed_successes
                != row.fixed_joint_successes * bool(conjuncts[fixed_policy].all())
            )
        ):
            raise ValueError(f"Independent preparation-policy development A/J/C or policy mismatch at root {r}")
        published_ratios = np.asarray(
            (
                row.maximum_error_ratio,
                row.maximum_width_ratio,
                row.maximum_numerical_difference,
                row.maximum_preservation_ratio,
                row.maximum_work,
            ),
            dtype=float,
        ).T
        np.testing.assert_allclose(
            published_ratios, audit["ratios"][r], rtol=1e-12, atol=1e-12
        )
    if (
        screen.headroom_roots != audit["headroom_roots"]
        or screen.adequacy_net_roots != audit["adequacy_net_roots"]
        or abs(float(screen.joint_improvement) - audit["joint_improvement"]) > 1e-14
        or (screen.gates_passed != audit["gates_passed"])
    ):
        raise ValueError("Independent preparation-policy development gate mismatch")
    from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_screen import compute_preparation_screen_screen

    reconstruction = compute_preparation_screen_screen(panels, lower)
    dependencies = tuple(
        sorted(
            (
                ObjectIdentity.from_record(d.object_id, d)
                for d in reconstruction.dependencies
            ),
            key=lambda identity: identity.object_id,
        )
    )
    if dependencies != screen.fit_dependencies:
        raise ValueError(
            "preparation-policy development fit dependency reconstruction differs from receipted identities"
        )
    return (audit, reconstruction.dependencies)
