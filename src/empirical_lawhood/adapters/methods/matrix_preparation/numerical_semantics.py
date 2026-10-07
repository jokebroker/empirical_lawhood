"""Native numerical qualification and separate privileged mechanism readouts."""

from decimal import Decimal
from hashlib import sha256

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT
from .data import Array, ProjectedContext, NUMERICAL_SEMANTICS_SCHEMA, write_method_arrays
from .records import PreparationNumericalSemanticsConfig, PreparationNumericalSemanticsReport


def finite_metrics(values: dict[str, tuple[float | int, str]]) -> tuple[NamedDecimal, ...]:
    """Missing numeric operands remain absent, never silently zeroed."""
    return tuple(
        NamedDecimal(key, Decimal(format(float(value), ".17g")), unit)
        for key, (value, unit) in sorted(values.items())
        if np.isfinite(value)
    )


def native_view_errors(response: Array) -> tuple[Array, Array]:
    if response.ndim != 6 or response.shape[1:3] != (5, 2) or response.shape[-2:] != (3, 5):
        raise ValueError("native agreement requires the paired action/readout chart")
    absolute = np.max(np.abs(response[:, :, 0] - response[:, :, 1]), axis=(3, 4))
    odd = (response[..., 2, :] - response[..., 0, :]) / 2
    return absolute, np.max(np.abs(odd[:, :, 0] - odd[:, :, 1]), axis=3)


def qualify_numerical_semantics(
    config: PreparationNumericalSemanticsConfig,
    observable: ProjectedContext,
    privileged: ProjectedContext,
) -> tuple[PreparationNumericalSemanticsReport, bytes]:
    if (
        observable.role != "observable"
        or privileged.role != "privileged"
        or observable.context != privileged.context
        or observable.roots != privileged.roots
        or any(
            r.projection_config != config.projection_config
            for r in (*observable.reports, *privileged.reports)
        )
        or any(
            a.native_result != b.native_result
            for a, b in zip(observable.reports, privileged.reports, strict=True)
        )
    ):
        raise ValueError("Numerical semantics changes its complete observable/privileged lineage")
    chosen = np.asarray([r.numerical_semantics for r in observable.roots])
    arrays = {key: value[chosen] for key, value in privileged.arrays.items()}
    absolute, odd = native_view_errors(observable.arrays["response"])
    arrays["native_absolute_view_error"] = absolute
    arrays["native_odd_view_error"] = odd
    arrays["numerical_semantics_root_indices"] = np.flatnonzero(chosen).astype(float)
    reasons = []
    if any(r.incomplete_native_occurrences for r in observable.reports):
        reasons.append("AUDIT_OR_PREPARATION_DELIVERY_UNRESOLVED")
    if not np.isfinite(absolute).all() or not np.isfinite(odd).all():
        reasons.append("NATIVE_VIEW_AGREEMENT_UNRESOLVED")
    elif np.max(absolute) > float(config.absolute_view_tolerance) or np.max(odd) > float(
        config.odd_view_tolerance
    ):
        reasons.append("NATIVE_VIEW_AGREEMENT_FAILED")
    for field in ("position", "momentum"):
        error, norm = arrays[f"secant_{field}_error"], arrays[f"secant_{field}_norm"]
        scaled = error / (
            float(config.secant_absolute_tolerance) + float(config.secant_relative_tolerance) * norm
        )
        arrays[f"secant_{field}_normalized_error"] = scaled
        if not np.isfinite(scaled).all():
            reasons.append(f"SECANT_{field.upper()}_IDENTITY_UNRESOLVED")
        elif np.max(scaled) > 1:
            reasons.append(f"SECANT_{field.upper()}_IDENTITY_FAILED")
    # These discrepancies address explanations. They do not redefine Gate 0.
    arrays["scalar_coupled_receiver_difference"] = (
        arrays["scalar_tangent"] - arrays["coupled_tangent"]
    )
    arrays["amplitude_receiver_tangent_difference"] = (
        arrays["amplitude_chi"] - arrays["coupled_tangent"][..., None, :]
    )
    arrays["finite_action_secant_tangent_difference"] = (
        arrays["secant_receiver"] - arrays["coupled_tangent"]
    )
    values: dict[str, tuple[float | int, str]] = {
        "retained-development-roots": (64, "1"),
        "mechanism-independent-roots": (int(chosen.sum()), "1"),
        "native-absolute-view-error-maximum": (float(np.max(absolute)), "hilbert-schmidt-native"),
        "native-odd-view-error-maximum": (float(np.max(odd)), "hilbert-schmidt-native"),
    }
    for key in ("secant_position_normalized_error", "secant_momentum_normalized_error"):
        values[key.replace("_", "-") + "-maximum"] = (float(np.max(arrays[key])), "1")
    identity = ObjectIdentity.from_record(config.config_id, config)
    payload = write_method_arrays(
        NUMERICAL_SEMANTICS_SCHEMA, observable.context, identity, {"mechanism": arrays}
    )
    report = PreparationNumericalSemanticsReport(
        f"{DEVELOPMENT}.numerical-semantics.{observable.context}",
        observable.context,
        identity,
        observable.identities,
        privileged.identities,
        finite_metrics(values),
        sha256(payload).hexdigest(),
        not reasons,
        tuple(sorted(reasons)),
    )
    return report, payload
