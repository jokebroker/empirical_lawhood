"""Deterministic non-RL selection policies for matched first-discovery worlds."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore
from sklearn.ensemble import ExtraTreesRegressor  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .contracts import (
    CandidateView,
    DiscoveryObservation,
    DiscoveryPolicyConfig,
    LabelState,
    ObservationOrigin,
    PolicyDecision,
    PolicyDecisionKind,
    PolicyKind,
    visible_prefix_sha256,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _LocalLaw:
    anchor_index: int
    intercept: float
    gradient: FloatArray
    radius: float
    residual_scale: float
    effective_rank: int


def _stable_rank(seed: int, round_index: int, candidate_id: str, namespace: str) -> bytes:
    return sha256(f"{namespace}\0{seed}\0{round_index}\0{candidate_id}".encode("ascii")).digest()


def _candidate_matrix(candidates: tuple[CandidateView, ...]) -> FloatArray:
    widths = {len(value.features) for value in candidates}
    if len(widths) != 1:
        raise ValueError("candidate feature widths differ")
    return np.asarray([value.features for value in candidates], dtype=np.float64)


def _observed_training(
    candidates_by_id: dict[str, CandidateView],
    observations: tuple[DiscoveryObservation, ...],
) -> tuple[FloatArray, FloatArray]:
    features: list[tuple[float, ...]] = []
    labels: list[float] = []
    for value in observations:
        if value.state not in {LabelState.MEASURED, LabelState.EXPLICIT_NEGATIVE}:
            continue
        candidate = candidates_by_id.get(value.candidate_id)
        if candidate is None or value.tc_kelvin is None:
            raise ValueError("visible observation lacks its candidate feature view")
        features.append(candidate.features)
        labels.append(float(value.tc_kelvin))
    if not features:
        width = len(next(iter(candidates_by_id.values())).features)
        return np.empty((0, width), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _farthest_order(
    matrix: FloatArray,
    reference: FloatArray,
    candidate_ids: tuple[str, ...],
    *,
    count: int,
    seed: int,
    round_index: int,
) -> tuple[int, ...]:
    if count <= 0 or len(matrix) == 0:
        return ()
    if len(reference):
        # Update one reference point at a time. The obvious three-dimensional
        # broadcast is exact but scales as pool x history x features and
        # exceeded 100 GiB on the real SC corpus. This formulation is equally
        # exact with working memory bounded by one pool-sized feature matrix.
        nearest_sq = np.full(len(matrix), np.inf, dtype=np.float64)
        for reference_row in reference:
            nearest_sq = np.minimum(
                nearest_sq,
                np.sum((matrix - reference_row) ** 2, axis=1),
            )
    else:
        centre = np.mean(matrix, axis=0, keepdims=True)
        nearest_sq = np.sum((matrix - centre) ** 2, axis=1)
    selected: list[int] = []
    available = set(range(len(matrix)))
    while available and len(selected) < count:
        best = max(
            available,
            key=lambda index: (
                float(nearest_sq[index]),
                _stable_rank(seed, round_index, candidate_ids[index], "maximin"),
            ),
        )
        selected.append(best)
        available.remove(best)
        distance_sq = np.sum((matrix - matrix[best]) ** 2, axis=1)
        nearest_sq = np.minimum(nearest_sq, distance_sq)
    return tuple(selected)


def _ensemble_predictions(
    train_x: FloatArray,
    train_y: FloatArray,
    pool_x: FloatArray,
    config: DiscoveryPolicyConfig,
) -> tuple[FloatArray, FloatArray]:
    if len(train_y) < max(8, config.ensemble_min_leaf * 2) or np.ptp(train_y) == 0:
        raise ValueError("SCALAR_MODEL_UNSUPPORTED_HISTORY")
    model = ExtraTreesRegressor(
        n_estimators=config.ensemble_members,
        min_samples_leaf=config.ensemble_min_leaf,
        max_features="sqrt",
        bootstrap=False,
        random_state=config.seed,
        n_jobs=1,
    )
    model.fit(train_x, train_y)
    members = np.asarray(
        [tree.predict(pool_x) for tree in model.estimators_],
        dtype=np.float64,
    )
    return np.mean(members, axis=0), np.std(members, axis=0, ddof=1)


def _bounded_bo_training(
    candidates_by_id: dict[str, CandidateView],
    observations: tuple[DiscoveryObservation, ...],
    config: DiscoveryPolicyConfig,
) -> tuple[FloatArray, FloatArray]:
    """Retain all acquired queries plus a deterministic four-stratum basis."""

    measured = tuple(
        value
        for value in observations
        if value.state in {LabelState.MEASURED, LabelState.EXPLICIT_NEGATIVE}
    )
    queried = tuple(value for value in measured if value.origin is ObservationOrigin.QUERY_RESPONSE)
    if len(queried) >= config.bo_training_limit:
        selected = queried[-config.bo_training_limit :]
    else:
        initial = tuple(
            value for value in measured if value.origin is ObservationOrigin.INITIAL_LABEL
        )
        remaining = config.bo_training_limit - len(queried)
        if initial and remaining:
            labels = np.asarray([float(value.tc_kelvin or 0) for value in initial])
            boundaries = np.quantile(labels, (0.25, 0.5, 0.75))
            strata = np.searchsorted(boundaries, labels, side="right")
            ranked_by_stratum = {
                stratum: sorted(
                    (value for index, value in enumerate(initial) if strata[index] == stratum),
                    key=lambda value: _stable_rank(
                        config.seed,
                        0,
                        value.candidate_id,
                        f"botorch-training-stratum-{stratum}",
                    ),
                )
                for stratum in range(4)
            }
            basis: list[DiscoveryObservation] = []
            while len(basis) < remaining and any(ranked_by_stratum.values()):
                for stratum in range(4):
                    values = ranked_by_stratum[stratum]
                    if values and len(basis) < remaining:
                        basis.append(values.pop())
            selected = (*basis, *queried)
        else:
            selected = queried
    if len(selected) < max(8, config.bo_projection_dimensions + 2):
        raise ValueError("BOTORCH_MODEL_UNSUPPORTED_HISTORY")
    x = np.asarray(
        [candidates_by_id[value.candidate_id].features for value in selected],
        dtype=np.float64,
    )
    y = np.asarray([float(value.tc_kelvin or 0) for value in selected], dtype=np.float64)
    return x, y


def _botorch_discrete_ucb_indices(
    *,
    candidates_by_id: dict[str, CandidateView],
    observations: tuple[DiscoveryObservation, ...],
    pool_x: FloatArray,
    config: DiscoveryPolicyConfig,
    count: int,
) -> tuple[int, ...]:
    """Fit the frozen CPU BoTorch GP and optimize UCB on exact candidates."""

    try:
        import torch  # type: ignore[import-not-found]
        from botorch.acquisition.analytic import UpperConfidenceBound  # type: ignore[import-not-found]
        from botorch.fit import fit_gpytorch_mll  # type: ignore[import-not-found]
        from botorch.models import SingleTaskGP  # type: ignore[import-not-found]
        from botorch.models.transforms import Normalize, Standardize  # type: ignore[import-not-found]
        from botorch.optim.optimize import optimize_acqf_discrete  # type: ignore[import-not-found]
        from gpytorch.mlls import (  # type: ignore
            ExactMarginalLogLikelihood,
        )
    except ImportError as error:
        raise ValueError("BOTORCH_DEPENDENCY_REQUIRED") from error

    from importlib.metadata import version

    if version("botorch") != "0.18.1":
        raise ValueError("BOTORCH_VERSION_DRIFT")
    if torch.__version__.split("+")[0] != "2.13.0":
        raise ValueError("BOTORCH_TORCH_VERSION_DRIFT")

    train_x, train_y = _bounded_bo_training(candidates_by_id, observations, config)
    combined = np.vstack((train_x, pool_x))
    scaled = StandardScaler().fit_transform(combined)
    projection = PCA(
        n_components=config.bo_projection_dimensions,
        svd_solver="randomized",
        random_state=config.seed,
    ).fit_transform(scaled)
    projected_train = projection[: len(train_x)]
    projected_pool = projection[len(train_x) :]

    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    train_tensor = torch.as_tensor(projected_train, dtype=torch.double, device="cpu")
    target_tensor = torch.as_tensor(train_y[:, None], dtype=torch.double, device="cpu")
    choices = torch.as_tensor(projected_pool, dtype=torch.double, device="cpu")
    model = SingleTaskGP(
        train_tensor,
        target_tensor,
        input_transform=Normalize(d=config.bo_projection_dimensions),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={"options": {"maxiter": config.bo_fit_max_iterations}},
    )
    acquisition = UpperConfidenceBound(model=model, beta=float(config.ucb_beta))
    remaining = choices
    remaining_indices = list(range(len(pool_x)))
    selected: list[int] = []
    for _ in range(min(count, len(pool_x))):
        candidate, _value = optimize_acqf_discrete(
            acquisition,
            q=1,
            choices=remaining,
            unique=True,
            max_batch_size=1024,
        )
        matches = torch.nonzero(
            torch.all(torch.isclose(remaining, candidate[0], rtol=0, atol=1e-12), dim=1),
            as_tuple=False,
        ).flatten()
        if not len(matches):
            raise ValueError("BOTORCH_DISCRETE_OPTIMIZER_RETURNED_UNKNOWN_CANDIDATE")
        local_index = int(matches[0].item())
        selected.append(remaining_indices.pop(local_index))
        mask = torch.ones(len(remaining), dtype=torch.bool)
        mask[local_index] = False
        remaining = remaining[mask]
    return tuple(selected)


def commit_hold(
    *,
    world_id: str,
    config: DiscoveryPolicyConfig,
    candidates: tuple[CandidateView, ...],
    observations: tuple[DiscoveryObservation, ...],
    round_index: int,
    reason: str,
) -> PolicyDecision:
    """Commit a typed no-query decision against the exact visible prefix."""

    candidates_by_id = {value.candidate_id: value for value in candidates}
    if len(candidates_by_id) != len(candidates):
        raise ValueError("candidate IDs repeat")
    queried = {value.candidate_id for value in observations}
    if not queried.issubset(candidates_by_id):
        raise ValueError("visible history contains an unknown candidate")
    pool_ids = tuple(
        sorted(
            value.candidate_id
            for value in candidates
            if value.valid and value.candidate_id not in queried
        )
    )
    prefix_hash = visible_prefix_sha256(observations)
    pool_hash = sha256(canonical_json_bytes(pool_ids)).hexdigest()
    state_hash = sha256(
        canonical_json_bytes((config.policy_kind.value, prefix_hash, pool_hash, reason))
    ).hexdigest()
    policy_slug = config.policy_kind.value.lower().replace("_", "-")
    return PolicyDecision(
        decision_id=f"decision.{world_id}.{policy_slug}.{round_index}",
        world_id=world_id,
        policy_config_sha256=config.fingerprint(),
        round_index=round_index,
        visible_prefix_sha256=prefix_hash,
        eligible_pool_sha256=pool_hash,
        selector_state_sha256=state_hash,
        kind=PolicyDecisionKind.HOLD,
        requested_candidate_ids=(),
        information_query_ids=(),
        incremental_cost=Decimal(0),
        hold_reason_id=reason,
    )


def _select_top(
    scores: FloatArray,
    candidate_ids: tuple[str, ...],
    *,
    count: int,
    seed: int,
    round_index: int,
    namespace: str,
) -> tuple[int, ...]:
    valid = np.flatnonzero(np.isfinite(scores))
    ranked = sorted(
        valid.tolist(),
        key=lambda index: (
            -float(scores[index]),
            _stable_rank(seed, round_index, candidate_ids[index], namespace),
        ),
    )
    return tuple(ranked[:count])


def _anchor_indices(
    train_x: FloatArray,
    train_y: FloatArray,
    config: DiscoveryPolicyConfig,
) -> tuple[int, ...]:
    threshold = float(np.quantile(train_y, float(config.local_anchor_quantile)))
    eligible = np.flatnonzero(train_y >= threshold)
    if not len(eligible):
        return ()
    ordered = sorted(eligible.tolist(), key=lambda index: (-float(train_y[index]), index))
    selected = [ordered[0]]
    nearest_sq = np.sum((train_x[eligible] - train_x[selected[0]]) ** 2, axis=1)
    eligible_positions = {index: position for position, index in enumerate(eligible.tolist())}
    nearest_sq[eligible_positions[selected[0]]] = -np.inf
    while len(selected) < min(config.local_anchor_count, len(ordered)):
        next_position = max(
            range(len(eligible)),
            key=lambda position: (float(nearest_sq[position]), -int(eligible[position])),
        )
        next_index = int(eligible[next_position])
        selected.append(next_index)
        distance_sq = np.sum((train_x[eligible] - train_x[next_index]) ** 2, axis=1)
        nearest_sq = np.minimum(nearest_sq, distance_sq)
        nearest_sq[[eligible_positions[index] for index in selected]] = -np.inf
    return tuple(selected)


def _queried_training(
    candidates_by_id: dict[str, CandidateView],
    observations: tuple[DiscoveryObservation, ...],
) -> FloatArray:
    """Return only acquired-query features for diversity state.

    The large initial labelled corpus trains predictive models, but it is not
    work performed by the query policy. Maximin coverage is defined over the
    policy's own acquisition history so the baseline remains bounded and its
    exploration cost is interpretable.
    """

    values = [
        candidates_by_id[value.candidate_id].features
        for value in observations
        if value.origin is ObservationOrigin.QUERY_RESPONSE
    ]
    if values:
        return np.asarray(values, dtype=np.float64)
    width = len(next(iter(candidates_by_id.values())).features)
    return np.empty((0, width), dtype=np.float64)


def _fit_local_laws(
    train_x: FloatArray,
    train_y: FloatArray,
    config: DiscoveryPolicyConfig,
) -> tuple[_LocalLaw, ...]:
    if len(train_y) < config.local_neighbor_count:
        return ()
    laws: list[_LocalLaw] = []
    ridge = float(config.local_ridge)
    for anchor in _anchor_indices(train_x, train_y, config):
        distance_sq = np.sum((train_x - train_x[anchor]) ** 2, axis=1)
        neighbors = np.argsort(distance_sq, kind="stable")[: config.local_neighbor_count]
        offsets = train_x[neighbors] - train_x[anchor]
        distances = np.sqrt(distance_sq[neighbors])
        positive = distances[distances > 0]
        if not len(positive):
            continue
        scale = max(float(np.quantile(positive, 0.9)), 1e-12)
        weights = np.exp(-((distances / scale) ** 2))
        design = np.column_stack((np.ones(len(neighbors)), offsets))
        weighted = design * np.sqrt(weights)[:, None]
        singular = np.linalg.svd(weighted[:, 1:], compute_uv=False)
        rank_floor = max(float(singular[0]) * 1e-8, 1e-12) if len(singular) else 1e-12
        effective_rank = int(np.count_nonzero(singular > rank_floor))
        if effective_rank < config.local_min_effective_rank:
            continue
        penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        gram = design.T @ (weights[:, None] * design) + penalty
        target = design.T @ (weights * train_y[neighbors])
        coefficient = np.linalg.solve(gram, target)
        residual = train_y[neighbors] - design @ coefficient
        residual_scale = max(
            float(np.sqrt(np.average(residual**2, weights=weights))),
            1e-6,
        )
        laws.append(
            _LocalLaw(
                anchor_index=anchor,
                intercept=float(coefficient[0]),
                gradient=np.asarray(coefficient[1:], dtype=np.float64),
                radius=scale * float(config.local_support_multiplier),
                residual_scale=residual_scale,
                effective_rank=effective_rank,
            )
        )
    return tuple(laws)


def _local_scores(
    train_x: FloatArray,
    train_y: FloatArray,
    pool_x: FloatArray,
    config: DiscoveryPolicyConfig,
) -> tuple[FloatArray, str]:
    laws = _fit_local_laws(train_x, train_y, config)
    if not laws:
        return np.full(len(pool_x), -np.inf), "LOCAL_LAW_NO_SUPPORTED_RECURRENCE"
    scores = np.full(len(pool_x), -np.inf, dtype=np.float64)
    for law in laws:
        offsets = pool_x - train_x[law.anchor_index]
        distances = np.linalg.norm(offsets, axis=1)
        admitted = distances <= law.radius
        prediction = law.intercept + offsets @ law.gradient
        relative = distances / max(law.radius, 1e-12)
        uncertainty = law.residual_scale * (1.0 + relative)
        score = (
            prediction
            + float(config.ucb_beta) * uncertainty
            + float(config.local_boundary_weight) * relative
        )
        scores = np.maximum(scores, np.where(admitted, score, -np.inf))
    reason = (
        "LOCAL_LAW_SUPPORTED_BOUNDARY"
        if np.any(np.isfinite(scores))
        else "LOCAL_LAW_CANDIDATES_OUTSIDE_SUPPORT"
    )
    return scores, reason


def select_batch(
    *,
    world_id: str,
    config: DiscoveryPolicyConfig,
    candidates: tuple[CandidateView, ...],
    observations: tuple[DiscoveryObservation, ...],
    round_index: int,
) -> PolicyDecision:
    """Select a matched batch from only the supplied visible history prefix."""

    if not candidates:
        raise ValueError("candidate universe is empty")
    if isinstance(round_index, bool) or round_index < 0:
        raise ValueError("round_index must be nonnegative")
    candidates_by_id = {value.candidate_id: value for value in candidates}
    if len(candidates_by_id) != len(candidates):
        raise ValueError("candidate IDs repeat")
    queried = {value.candidate_id for value in observations}
    if not queried.issubset(candidates_by_id):
        raise ValueError("visible history contains an unknown candidate")
    completed_queries = sum(
        value.origin is ObservationOrigin.QUERY_RESPONSE for value in observations
    )
    remaining_queries = config.total_query_budget - completed_queries
    pool = tuple(
        sorted(
            (value for value in candidates if value.valid and value.candidate_id not in queried),
            key=lambda value: value.candidate_id,
        )
    )
    prefix_hash = visible_prefix_sha256(observations)
    pool_hash = sha256(
        canonical_json_bytes(tuple(value.candidate_id for value in pool))
    ).hexdigest()
    policy_slug = config.policy_kind.value.lower().replace("_", "-")

    def hold(reason: str) -> PolicyDecision:
        return commit_hold(
            world_id=world_id,
            config=config,
            candidates=candidates,
            observations=observations,
            round_index=round_index,
            reason=reason,
        )

    if remaining_queries <= 0:
        return hold("hold.query-budget-exhausted")
    if not pool:
        return hold("hold.eligible-pool-empty")
    count = min(config.batch_size, remaining_queries, len(pool))
    pool_x = _candidate_matrix(pool)
    pool_ids = tuple(value.candidate_id for value in pool)
    train_x, train_y = _observed_training(candidates_by_id, observations)
    queried_x = _queried_training(candidates_by_id, observations)
    information_indices: tuple[int, ...] = ()

    if config.policy_kind is PolicyKind.STRATIFIED_RANDOM:
        by_stratum: dict[str, list[int]] = {}
        for index, value in enumerate(pool):
            by_stratum.setdefault(value.stratum_id, []).append(index)
        selected: list[int] = []
        strata = sorted(by_stratum)
        while len(selected) < count and strata:
            next_strata: list[str] = []
            for stratum in strata:
                choices = [index for index in by_stratum[stratum] if index not in selected]
                if not choices:
                    continue
                selected.append(
                    min(
                        choices,
                        key=lambda index: _stable_rank(
                            config.seed,
                            round_index,
                            pool_ids[index],
                            f"random-{stratum}",
                        ),
                    )
                )
                if len(selected) == count:
                    break
                if len(choices) > 1:
                    next_strata.append(stratum)
            strata = next_strata
        indices = tuple(selected)
        state_reason = "STRATIFIED_RANDOM_HASH_ORDER"
    elif config.policy_kind is PolicyKind.MAXIMIN:
        indices = _farthest_order(
            pool_x,
            queried_x,
            pool_ids,
            count=count,
            seed=config.seed,
            round_index=round_index,
        )
        state_reason = "MAXIMIN_VISIBLE_COMPOSITION_DISTANCE"
    elif config.policy_kind is PolicyKind.GREEDY_SCALAR_ENSEMBLE:
        try:
            mean, _scale = _ensemble_predictions(train_x, train_y, pool_x, config)
        except ValueError:
            indices = _farthest_order(
                pool_x,
                train_x,
                pool_ids,
                count=count,
                seed=config.seed,
                round_index=round_index,
            )
            state_reason = "ENSEMBLE_UNSUPPORTED_FALLBACK_MAXIMIN"
        else:
            indices = _select_top(
                mean,
                pool_ids,
                count=count,
                seed=config.seed,
                round_index=round_index,
                namespace=config.policy_kind.value.lower(),
            )
            state_reason = config.policy_kind.value
    elif config.policy_kind is PolicyKind.BOTORCH_DISCRETE_UCB:
        indices = _botorch_discrete_ucb_indices(
            candidates_by_id=candidates_by_id,
            observations=observations,
            pool_x=pool_x,
            config=config,
            count=count,
        )
        state_reason = "BOTORCH_0_18_1_DISCRETE_UCB"
    elif config.policy_kind is PolicyKind.LOCAL_LAW_BOUNDARY:
        scores, state_reason = _local_scores(train_x, train_y, pool_x, config)
        boundary_count = max(0, count - config.information_queries_per_batch)
        boundary = _select_top(
            scores,
            pool_ids,
            count=boundary_count,
            seed=config.seed,
            round_index=round_index,
            namespace="local-law-boundary",
        )
        remaining = tuple(index for index in range(len(pool)) if index not in boundary)
        info_count = min(config.information_queries_per_batch, count - len(boundary))
        if info_count:
            info_matrix = pool_x[np.asarray(remaining)]
            info_ids = tuple(pool_ids[index] for index in remaining)
            local_info = _farthest_order(
                info_matrix,
                queried_x,
                info_ids,
                count=info_count,
                seed=config.seed,
                round_index=round_index,
            )
            information_indices = tuple(remaining[index] for index in local_info)
        indices = (*boundary, *information_indices)
        if not indices:
            return hold("hold.local-law-no-supported-or-information-query")
    else:  # pragma: no cover - PolicyKind is a closed enum
        raise ValueError("unsupported first-discovery policy kind")

    requested = tuple(sorted(pool_ids[index] for index in indices))
    information = tuple(sorted(pool_ids[index] for index in information_indices))
    state_hash = sha256(
        canonical_json_bytes(
            {
                "information_query_ids": information,
                "policy": config.policy_kind.value,
                "reason": state_reason,
                "requested_candidate_ids": requested,
                "round_index": round_index,
                "visible_prefix_sha256": prefix_hash,
            }
        )
    ).hexdigest()
    return PolicyDecision(
        decision_id=f"decision.{world_id}.{policy_slug}.{round_index}",
        world_id=world_id,
        policy_config_sha256=config.fingerprint(),
        round_index=round_index,
        visible_prefix_sha256=prefix_hash,
        eligible_pool_sha256=pool_hash,
        selector_state_sha256=state_hash,
        kind=PolicyDecisionKind.QUERY,
        requested_candidate_ids=requested,
        information_query_ids=information,
        incremental_cost=config.query_cost * len(requested),
        hold_reason_id=None,
    )


__all__ = ["commit_hold", "select_batch"]
