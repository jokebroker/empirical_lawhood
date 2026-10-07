"""dependent refinement-frozen parent policies, adequacy forecast and joint residual scenarios.

The policy learner consumes only dependent refinement scalar projections and outer-held-out
predictions.  Deployment consumes the pre-parent instrument; it cannot read a
late target, root seed, native state or protected future.  The residual bank is
selected by root identity before inspecting outcomes and retains nonfinite
scenarios as explicit refusal mass.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import sha256
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.composition.prepared_response.task_charter import PreparedTaskCharterEntry
from empirical_lawhood.adapters.methods.response_formalization import (
    affine_prediction,
    fit_affine_operator,
)
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, READOUTS
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_stable_id,
)

from .development_fit import PreparedResponseDevelopmentModelFit
from .development_projection import PreparedResponseDevelopmentViewProjection
from .statistics import PreparedStatisticalSpec

if TYPE_CHECKING:
    from .development_selection import PreparedResponseDevelopmentSelectionConfig


Array = npt.NDArray[np.float64]
POLICIES = ("primary", "hold", "best-fixed", "conventional")
SCENARIO_COUNT = 16
PRESERVATION_CAPS = np.asarray((0.125, 0.05, 0.05), dtype=np.float64)
_ENCODING = "BASE64_LITTLE_ENDIAN_FLOAT64_C_ORDER_FIXED_SHAPES_NAN_UNKNOWN"


def _freeze(value: Array) -> Array:
    array = np.asarray(value, dtype="<f8")
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _encode(arrays: dict[str, Array]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            name,
            base64.b64encode(np.asarray(value, dtype="<f8").tobytes(order="C")).decode(
                "ascii"
            ),
        )
        for name, value in sorted(arrays.items())
    )


def _shapes(instrument_tier: str) -> dict[str, tuple[int, ...]]:
    features = 200 if instrument_tier == "I1" else 192
    shapes = {
        "adequacy_operator": (features + 1, len(PARENTS)),
        "conventional_feature_mean": (features,),
        "conventional_feature_scale": (features,),
        "conventional_operator": (features + 1, len(PARENTS)),
        "primary_feature_mean": (features,),
        "primary_feature_scale": (features,),
        "primary_operator": (features + 1, len(PARENTS)),
        "scenario_features": (SCENARIO_COUNT, features),
        "scenario_handoff_available": (SCENARIO_COUNT, len(PARENTS), 2),
        "scenario_handoff_displacement": (SCENARIO_COUNT, len(PARENTS), 2, 2),
        "scenario_history": (SCENARIO_COUNT, len(PARENTS), 2, 16, 12),
        "scenario_parent_work": (SCENARIO_COUNT, len(PARENTS), 2),
        "scenario_residuals": (SCENARIO_COUNT, len(PARENTS), 2, 9, 5, 7),
    }
    if instrument_tier == "I1":
        shapes["scenario_sketch"] = (SCENARIO_COUNT, len(PARENTS), 2, 8)
    return dict(sorted(shapes.items()))


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentContextPolicy(CanonicalRecord):
    """One context's target-blind parent decision and forecast operands."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-context-policy'
    policy_id: str
    context: str
    selected_fit: ObjectIdentity
    instrument_tier: str
    best_fixed_parent: str
    best_fixed_observed_utility: Decimal
    scenario_selection_sha256: str
    blocks: tuple[tuple[str, str], ...]
    encoding: str = _ENCODING
    primary_policy_rule: str = "DEVELOPMENT_OOF_UTILITY_WITH_ADEQUACY_IMPROVEMENT_OVER_HOLD"

    def __post_init__(self) -> None:
        validate_stable_id(self.policy_id, field_name="policy_id")
        shapes = _shapes(self.instrument_tier)
        if (
            self.context not in ("assembling", "prepared")
            or self.selected_fit.object_schema != PreparedResponseDevelopmentModelFit.SCHEMA
            or self.instrument_tier not in ("I0", "I1")
            or self.best_fixed_parent not in PARENTS
            or not isinstance(self.best_fixed_observed_utility, Decimal)
            or not self.best_fixed_observed_utility.is_finite()
            or not Decimal(0) <= self.best_fixed_observed_utility <= Decimal(1)
            or self.primary_policy_rule != "DEVELOPMENT_OOF_UTILITY_WITH_ADEQUACY_IMPROVEMENT_OVER_HOLD"
            or len(self.scenario_selection_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.scenario_selection_sha256)
            or self.encoding != _ENCODING
            or tuple(name for name, _ in self.blocks) != tuple(shapes)
        ):
            raise ValueError("prepared dependent refinement policy changes its context, policy or scenario contract")
        arrays = self.arrays()
        for name, value in arrays.items():
            if np.isinf(value).any():
                raise ValueError("prepared dependent refinement policy arrays require finite values or explicit NaN")
            if not name.startswith("scenario_") and not np.isfinite(value).all():
                raise ValueError("prepared dependent refinement policy coefficients must be finite")
        available = arrays["scenario_handoff_available"]
        if not np.isin(available, (0, 1)).all():
            raise ValueError("prepared dependent refinement handoff availability must retain its measured Boolean status")
        if not np.isnan(arrays["scenario_history"][available == 0]).all():
            raise ValueError("an unavailable dependent refinement scenario cannot fabricate a handoff history")
        if np.any(arrays["primary_feature_scale"] <= 0) or np.any(
            arrays["conventional_feature_scale"] <= 0
        ):
            raise ValueError("prepared dependent refinement policy feature scales must remain positive")

    def arrays(self, *, include_scenarios: bool = True) -> dict[str, Array]:
        shapes = _shapes(self.instrument_tier)
        result = {}
        for name, encoded in self.blocks:
            if not include_scenarios and name.startswith("scenario_"):
                continue
            expected = int(np.prod(shapes[name])) * 8
            if not isinstance(encoded, str) or len(encoded) != 4 * ((expected + 2) // 3):
                raise ValueError("prepared dependent refinement policy block changes its encoded size")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) != expected or base64.b64encode(raw).decode("ascii") != encoded:
                raise ValueError("prepared dependent refinement policy block is not canonical base64")
            result[name] = np.frombuffer(raw, dtype="<f8").reshape(shapes[name])
        return result

    def _features(self, history: Array, sketch: Array | None) -> Array:
        if history.shape != (16, 12) or not np.isfinite(history).all():
            raise ValueError("prepared parent policy requires one complete pre-parent I0 history")
        features = np.asarray(history, dtype=np.float64).reshape(1, 192)
        if self.instrument_tier == "I1":
            if sketch is None or sketch.shape != (8,) or not np.isfinite(sketch).all():
                raise ValueError("prepared I1 parent policy requires its bounded pre-parent sketch")
            features = np.column_stack((features, np.asarray(sketch)[None, :]))
        else:
            if sketch is not None:
                raise ValueError("prepared I0 parent policy cannot receive a private I1 sketch")
        return features

    def choose_parent(
        self, policy: str, history: Array, sketch: Array | None = None
    ) -> str:
        if policy not in POLICIES:
            raise ValueError("prepared parent policy is outside the frozen four-policy roster")
        if policy == "hold":
            return "hold"
        if policy == "best-fixed":
            return self.best_fixed_parent
        arrays = self.arrays(include_scenarios=False)
        inputs = self._features(history, sketch)
        if policy == "conventional":
            features = (inputs - arrays["conventional_feature_mean"]) / arrays[
                "conventional_feature_scale"
            ]
            operator = arrays["conventional_operator"]
        else:
            features = (inputs - arrays["primary_feature_mean"]) / arrays[
                "primary_feature_scale"
            ]
            operator = arrays["primary_operator"]
        scores = affine_prediction(operator, features)[0]
        if policy == "primary":
            probabilities = np.clip(
                affine_prediction(arrays["adequacy_operator"], features)[0], 0.000001, 0.999999
            )
            # HOLD is the baseline recipe. An active preparation must improve
            # predicted adequacy as well as compete on useful capability.
            eligible = probabilities > probabilities[PARENTS.index("hold")]
            eligible[PARENTS.index("hold")] = True
            scores = np.where(eligible, scores, -np.inf)
        return PARENTS[int(np.argmax(scores))]

    def adequacy_probabilities(self, history: Array, sketch: Array | None = None) -> Array:
        arrays = self.arrays(include_scenarios=False)
        primary = self._features(history, sketch)
        features = (primary - arrays["primary_feature_mean"]) / arrays[
            "primary_feature_scale"
        ]
        values = affine_prediction(arrays["adequacy_operator"], features)[0]
        return _freeze(np.clip(values, 0.000001, 0.999999))

    def scenario_weights(self, history: Array, sketch: Array | None = None) -> Array:
        arrays = self.arrays()
        primary = self._features(history, sketch)
        feature = (primary[0] - arrays["primary_feature_mean"]) / arrays[
            "primary_feature_scale"
        ]
        known = np.isfinite(arrays["scenario_features"]).all(axis=1)
        weights = np.full(SCENARIO_COUNT, 1 / SCENARIO_COUNT)
        if known.any():
            squared = np.mean((arrays["scenario_features"][known] - feature) ** 2, axis=1)
            inverse = 1 / np.maximum(squared, 1e-12)
            weights[known] = (inverse / inverse.sum()) * (int(known.sum()) / SCENARIO_COUNT)
        return _freeze(weights)


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentPolicyLibrary(CanonicalRecord):
    "The two-context dependent refinement policy/forecast freeze; it grants no law or authority."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-policy-library'
    library_id: str
    selection_config: ObjectIdentity
    contexts: tuple[PreparedResponseDevelopmentContextPolicy, ...]
    policy_ids: tuple[str, ...] = POLICIES
    threshold_family: tuple[Decimal, ...] = (
        Decimal("0.975"),
        Decimal("0.95"),
        Decimal("0.90"),
    )
    support_rule: str = "DEVELOPMENT_ONLY_PREPARENT_LINEAR_PROBABILITY_CLIPPED_1E-6"
    composition_rule: str = "SIXTEEN_NUMERIC_ROOT_SCENARIOS_KNOWN_INVERSE_DISTANCE_UNKNOWN_UNIFORM_MASS"
    grants_law_or_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.library_id, field_name="library_id")
        if (
            self.selection_config.object_schema
            != 'empirical-lawhood/methods/prepared-response/prepared-response-development-selection-config'
            or tuple(value.context for value in self.contexts) != ("assembling", "prepared")
            or self.policy_ids != POLICIES
            or len(set(self.policy_ids)) != len(self.policy_ids)
            or self.threshold_family != PreparedStatisticalSpec(
                "prepared-response.statistics"
            ).threshold_order
            or self.support_rule != "DEVELOPMENT_ONLY_PREPARENT_LINEAR_PROBABILITY_CLIPPED_1E-6"
            or self.composition_rule
            != "SIXTEEN_NUMERIC_ROOT_SCENARIOS_KNOWN_INVERSE_DISTANCE_UNKNOWN_UNIFORM_MASS"
            or self.grants_law_or_authority
        ):
            raise ValueError("prepared dependent refinement policy library changes its frozen roster or authority")


def _target_centers(charter: PreparedTaskCharterEntry, histories: Array) -> Array:
    velocities = histories[:, -1, (1, 10)]
    with localcontext() as context:
        context.prec = 28
        elapsed = Decimal("0.001") * (charter.parent_and_delay_ticks + charter.horizon_ticks)
        drift = float(1 - (-elapsed).exp())
        diagonal = float(Decimal(1) / Decimal(2).sqrt())
    directions = np.asarray(
        (
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-diagonal, -diagonal),
            (diagonal, diagonal),
            (-diagonal, diagonal),
            (diagonal, -diagonal),
        ),
        dtype=np.float64,
    )
    return velocities[:, None, :] * drift + float(charter.distance) * directions[None, :, :]


def _utilities(
    *,
    outputs: Array,
    halfwidths: Array,
    handoff_displacement: Array,
    parent_work: Array,
    targets: Array,
    epsilon: float,
) -> Array:
    """Target-integrated useful-chart fraction for every root and parent."""
    n = outputs.shape[0]
    result = np.zeros((n, len(PARENTS)), dtype=np.float64)
    for root in range(n):
        for parent in range(len(PARENTS)):
            if (
                not np.isfinite(parent_work[root, parent]).all()
                or np.any(parent_work[root, parent] < 0)
                or np.any(parent_work[root, parent] > 32)
                or not np.isfinite(handoff_displacement[root, parent]).all()
                or not np.isfinite(outputs[root, parent]).all()
                or not np.isfinite(halfwidths[root, parent]).all()
                or np.any(halfwidths[root, parent] < 0)
            ):
                continue
            endpoint = handoff_displacement[root, parent, :, None, None, :] + outputs[
                root, parent, :, :, :, :2
            ]
            preserved = (
                outputs[root, parent, :, :, :, 2:5]
                + halfwidths[root, parent, :, :, :, 2:5]
                <= PRESERVATION_CAPS
            ).all(axis=(0, 2, 3))
            sharp = (halfwidths[root, parent, :, :, :, :2] <= epsilon).all(
                axis=(0, 2, 3)
            )
            successes = 0
            for target in targets[root]:
                contact = (
                    abs(endpoint - target[None, None, None, :])
                    + halfwidths[root, parent, :, :, :, :2]
                    <= epsilon
                ).all(axis=(0, 2, 3))
                successes += int(np.any(contact & preserved & sharp))
            result[root, parent] = successes / len(targets[root])
    return result


def _adequacy_labels(
    observed: Array,
    predicted: Array,
    halfwidths: Array,
    valid: npt.NDArray[np.bool_],
    parent_work: Array,
    epsilon: float,
) -> Array:
    """Whole-chart held-out coverage/sharpness, independent of target contact."""
    complete = (
        valid.all(axis=2)
        & np.isfinite(observed).all(axis=(2, 3, 4, 5))
        & np.isfinite(predicted).all(axis=(2, 3, 4, 5))
        & np.isfinite(halfwidths).all(axis=(2, 3, 4, 5))
        & (halfwidths >= 0).all(axis=(2, 3, 4, 5))
        & np.isfinite(parent_work).all(axis=2)
        & ((parent_work >= 0) & (parent_work <= 32)).all(axis=2)
    )
    covered = (abs(observed - predicted) <= halfwidths).all(axis=(2, 3, 4, 5))
    sharp = (halfwidths[..., :2] <= epsilon).all(axis=(2, 3, 4, 5))
    return np.asarray(complete & covered & sharp, dtype=np.float64)


def _fit_policy(features: Array, targets: Array) -> tuple[Array, Array, Array]:
    observed = np.isfinite(features).all(axis=1)
    if not observed.any():
        raise ValueError("prepared dependent refinement policy has no observed pre-parent instrument rows")
    # Missing inputs cannot enter a regression. Their roots remain in the
    # utility denominator and the identity-selected scenario census.
    features, targets = features[observed], targets[observed]
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-12)
    operator = fit_affine_operator(
        (features - mean) / scale,
        targets,
        ridge=float(len(features)),
    )
    return mean, scale, operator


def build_prepared_response_development_policy_library(
    config: PreparedResponseDevelopmentSelectionConfig,
    selected_structure: str,
    projections: tuple[PreparedResponseDevelopmentViewProjection, ...],
    fits: tuple[PreparedResponseDevelopmentModelFit, ...],
) -> PreparedResponseDevelopmentPolicyLibrary:
    "Fit the frozen four-policy and composition operands from complete dependent refinement only."
    if selected_structure not in config.fit.model_spec("assembling").structures:
        raise ValueError("prepared dependent refinement policy builder requires the nominated finite structure")
    contexts = []
    charter = config.fit.projection.source_qualification.selected_charter
    assert charter is not None
    readout = READOUTS.index(charter.horizon_ticks)
    for context_name in ("assembling", "prepared"):
        reports = tuple(
            value
            for value in projections
            if value.root.context == context_name
        )
        roots = tuple(value for value in config.fit.projection.native_spec.roots if value.context == context_name)
        if tuple((value.root, value.refinement) for value in reports) != tuple(
            (root, refinement) for root in roots for refinement in (1, 2)
        ):
            raise ValueError("prepared dependent refinement policy requires every root/view projection in source order")
        selected_fit = next(
            (
                value
                for value in fits
                if value.context == context_name and value.structure == selected_structure
            ),
            None,
        )
        if selected_fit is None or selected_fit.failure_reason is not None:
            raise ValueError("prepared dependent refinement policy requires the eligible nominated context fit")
        arrays = [value.arrays() for value in reports]
        def stack(name: str, shape: tuple[int, ...]) -> Array:
            return np.asarray(np.stack([value[name] for value in arrays]), dtype=np.float64).reshape(shape)

        preparent_history = stack("preparent_history", (64, 2, 16, 12))
        preparent_sketch = stack("preparent_sketch", (64, 2, 8))
        handoff = stack("handoff_displacement", (64, 2, 5, 2)).transpose(0, 2, 1, 3)
        parent_work = stack("parent_absolute_density_work", (64, 2, 5)).transpose(0, 2, 1)
        observed = stack("observed", (64, 2, 5, 9, 5, 7)).transpose(0, 2, 1, 3, 4, 5)
        predicted, _ = selected_fit.arrays()
        halfwidth = selected_fit.outer_halfwidths()
        valid = np.asarray([value["fit_valid"] for value in arrays], dtype=bool).reshape(64, 2, 5)
        adequacy = _adequacy_labels(
            observed, predicted, halfwidth, valid.transpose(0, 2, 1),
            parent_work, float(charter.epsilon),
        )
        targets = _target_centers(charter, preparent_history[:, 0])
        predicted_utility = _utilities(
            outputs=predicted[:, :, :, :, readout : readout + 1],
            halfwidths=halfwidth[:, :, :, :, readout : readout + 1],
            handoff_displacement=handoff,
            parent_work=parent_work,
            targets=targets,
            epsilon=float(charter.epsilon),
        )
        observed_utility = _utilities(
            outputs=observed[:, :, :, :, readout : readout + 1],
            halfwidths=np.zeros_like(observed[:, :, :, :, readout : readout + 1]),
            handoff_displacement=handoff,
            parent_work=parent_work,
            targets=targets,
            epsilon=float(charter.epsilon),
        )
        tier = "I1" if selected_structure == "mechanism-i1" else "I0"
        primary_features = preparent_history[:, 0].reshape(64, 192)
        if tier == "I1":
            primary_features = np.column_stack((primary_features, preparent_sketch[:, 0]))
        # One multi-output solve shares the identical feature/ridge calculation.
        p_mean, p_scale, operators = _fit_policy(
            primary_features, np.column_stack((predicted_utility, observed_utility, adequacy)),
        )
        p_operator, c_operator, adequacy_operator = np.split(operators, 3, axis=1)
        fixed_index = int(np.argmax(observed_utility.mean(axis=0)))
        fixed = float(observed_utility[:, fixed_index].mean())
        # The scientific allocation is a numeric whole-root commitment. Public
        # root labels only describe custody and cannot choose residual scenarios.
        scenario_indices = dict(config.scenario_root_indices)[context_name]
        scenario_digest = sha256(
            "\n".join(roots[index].physical_unit_id for index in scenario_indices).encode()
        ).hexdigest()
        standardized = (primary_features - p_mean) / p_scale
        residuals = observed - predicted
        # Select whole roots once: handoff state, failure status and every
        # response residual retain their joint parent/view/scenario identity.
        selected = list(scenario_indices)
        scenario_blocks = {
            "scenario_features": standardized[selected],
            "scenario_handoff_available": np.asarray(
                [[handoff is not None for handoff in report.handoffs] for report in reports],
                dtype=np.float64,
            ).reshape(64, 2, 5).transpose(0, 2, 1)[selected],
            "scenario_handoff_displacement": handoff[selected],
            "scenario_history": stack("history", (64, 2, 5, 16, 12)).transpose(
                0, 2, 1, 3, 4
            )[selected],
            "scenario_parent_work": parent_work[selected],
            "scenario_residuals": residuals[selected],
        }
        if tier == "I1":
            scenario_blocks["scenario_sketch"] = stack("sketch", (64, 2, 5, 8)).transpose(
                0, 2, 1, 3
            )[selected]
        contexts.append(
            PreparedResponseDevelopmentContextPolicy(
                f"prepared-response.dependent-refinement.policy.{context_name}",
                context_name,
                ObjectIdentity.from_record(selected_fit.report_id, selected_fit),
                tier,
                PARENTS[fixed_index],
                Decimal(format(fixed, ".17g")),
                scenario_digest,
                _encode(
                    {
                        "adequacy_operator": adequacy_operator,
                        "conventional_feature_mean": p_mean,
                        "conventional_feature_scale": p_scale,
                        "conventional_operator": c_operator,
                        "primary_feature_mean": p_mean,
                        "primary_feature_scale": p_scale,
                        "primary_operator": p_operator,
                        **scenario_blocks,
                    }
                ),
            )
        )
    return PreparedResponseDevelopmentPolicyLibrary(
        "prepared-response.dependent-refinement.policy-library",
        ObjectIdentity.from_record(config.config_id, config),
        tuple(contexts),
    )
