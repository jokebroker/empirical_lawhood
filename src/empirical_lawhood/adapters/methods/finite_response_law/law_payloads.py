"Bounded frozen F/U payloads; serialization and prediction grant no local-law status.\n\nLower evaluation has only native handoff features as input. Composition refers\nto the exact lower bytes and carries U and its own calibration, never a copy of\nF. Cached/direct packages retain their own boundary and uncertainty identities.\n"

from dataclasses import dataclass, field, fields
from decimal import Decimal
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.adapters.methods.response_formalization import affine_prediction
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT
from .assigned_prediction import AssignedPrediction
from .fitting import Array, Normalizer, PointFit, clamp
from .intervals import WidthFit
from .science import FiniteResponseLawScienceSpec, PROGRAMME
from .paired_assay import QUANTITIES

MAXIMUM_LAW_BYTES = 1024 * 1024
CALIBRATION_SCHEMA = 'empirical-lawhood/methods/finite-response-law/finite-response-law-boundary-calibration'


@dataclass(frozen=True, slots=True)
class FiniteResponseLawResponseChart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-response-chart'
    science: FiniteResponseLawScienceSpec = FiniteResponseLawScienceSpec()
    output_ids: tuple[str, ...] = QUANTITIES
    output_units: tuple[str, ...] = (
        "hilbert-schmidt-native",
        "hilbert-schmidt-native",
        "hilbert-schmidt-native",
        "1",
        "native-passive-transfer",
        "hilbert-schmidt-native",
        "1",
        "native-passive-transfer",
    )
    native_words: tuple[tuple[int, int, int], ...] = tuple(
        (magnitude, direction, sign)
        for magnitude in (8, 16)
        for direction in (0, 1)
        for sign in (-1, 1)
    )
    frame_id: str = f"{PROGRAMME}.frozen-preparent-two-port-frame"
    clock_id: str = f"{PROGRAMME}.reference-clock"
    horizon_ticks: int = 192
    receiver_definition: str = "selected-minus-opposite-over-two-with-both-branch-preservation"
    hold_reference: str = "actual-same-root-parent-purpose-view-native-HOLD"
    algebraic_origin: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if any(getattr(self, f.name) != f.default for f in fields(self)):
            raise ValueError(
                "Finite response-law payload cannot change native actions, paired receiver, units or horizon"
            )


def _values(array: Array) -> tuple[Decimal, ...]:
    return tuple(Decimal(format(float(v), ".17g")) for v in array.ravel())


def _array(values: tuple[Decimal, ...], shape: tuple[int, ...], *, positive: bool = False) -> Array:
    if len(values) != int(np.prod(shape)) or any(
        type(v) is not Decimal or not v.is_finite() for v in values
    ):
        raise ValueError("Finite response law coefficient axes or finite native values differ")
    array = np.asarray(values, dtype=np.float64).reshape(shape)
    if not np.isfinite(array).all() or positive and (array <= 0).any():
        raise ValueError("Finite response law coefficients exceed finite float64 or positive scales")
    return array


def _provenance(
    payload_id: str, instrument: ObjectIdentity, calibration: ObjectIdentity, q: Decimal | None
) -> None:
    validate_stable_id(payload_id, field_name="payload_id")
    if (
        instrument != REFERENCE_INSTRUMENT.identity
        or (calibration.object_schema, calibration.object_version)
        not in (
            (CALIBRATION_SCHEMA, "1.0.0"),
            ('empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-boundary-calibration', "1.0.0"),
        )
        or q is not None
        and (type(q) is not Decimal or not q.is_finite() or q < 0)
    ):
        raise ValueError("Finite response law changes its fixed instrument or calibration provenance")


def _features(values: Array) -> None:
    if values.ndim != 2 or values.shape[1] != 24 or values.dtype != np.float64:
        raise ValueError("Finite response law requires 24 native primary-view features")


def _supported(normalized: Array) -> NDArray[np.bool_]:
    return np.asarray(np.isfinite(normalized).all(axis=-1) & (np.abs(normalized).max(axis=-1) <= 6))


def _sigma(base: Array, log_multiplier: Array) -> Array:
    return (
        base[None, None]
        * np.exp(np.clip(log_multiplier, np.log(0.25), np.log(4)))[:, :, None, None]
    )


def _lower_response(operator: Array, normalized: Array) -> Array:
    if normalized.ndim != 3 or normalized.shape[1:] != (5, 24):
        raise ValueError("Finite response-law F evaluation changes its computational layout")
    return clamp(affine_prediction(operator, normalized.reshape(-1, 24)).reshape(-1, 5, 4, 8))


def _lower_prediction(
    handoff: Array, normalizer: Normalizer, operator: Array, base: Array, multiplier: Array
) -> AssignedPrediction:
    _features(handoff)
    normalized = normalizer.apply(handoff)
    # The same parent-free arithmetic is used before and after calibration.
    # Computational padding does not represent acquired parents or observations.
    padded = np.repeat(normalized[:, None, :], 5, axis=1)
    mean = _lower_response(operator, padded)
    logg = affine_prediction(multiplier, padded.reshape(-1, 24)).reshape(-1, 5)
    return AssignedPrediction(mean[:, 0], _sigma(base, logg)[:, 0], _supported(normalized))


def uncalibrated_lower_prediction(
    points: PointFit, width: WidthFit, handoff: Array
) -> AssignedPrediction:
    """Compute calibration operands without a fabricated calibration identity."""
    if points.recipe.dimension != 24 or width.boundary != "lower":
        raise ValueError("Finite response-law lower evaluation changes the frozen 24-feature lower response-law boundary")
    return _lower_prediction(handoff, points.nh, points.lower, width.base, width.multiplier)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawLowerPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-lower-payload'
    payload_id: str
    development_manifest: ArtifactIdentity
    frozen_coefficients: ArtifactIdentity
    instrument: ObjectIdentity
    calibration: ObjectIdentity
    q: Decimal | None
    center: tuple[Decimal, ...]
    scale: tuple[Decimal, ...]
    operator: tuple[Decimal, ...]
    base: tuple[Decimal, ...]
    multiplier: tuple[Decimal, ...]
    chart: FiniteResponseLawResponseChart = field(default=FiniteResponseLawResponseChart(), kw_only=True)

    def __post_init__(self) -> None:
        _provenance(self.payload_id, self.instrument, self.calibration, self.q)
        _array(self.center, (24,))
        _array(self.scale, (24,), positive=True)
        _array(self.operator, (25, 32))
        _array(self.base, (4, 8), positive=True)
        _array(self.multiplier, (25, 1))
        if len(self.canonical_bytes()) > MAXIMUM_LAW_BYTES:
            raise ValueError("Finite response-law lower payload exceeds its bounded decoder")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.payload_id, self)

    def normalize(self, handoff: Array) -> Array:
        return Normalizer(_array(self.center, (24,)), _array(self.scale, (24,))).apply(handoff)

    def response(self, normalized: Array) -> Array:
        """Existing F arithmetic, retaining the frozen five-slot evaluation layout."""
        return _lower_response(_array(self.operator, (25, 32)), normalized)

    def predict(self, handoff: Array) -> AssignedPrediction:
        return _lower_prediction(
            handoff,
            Normalizer(_array(self.center, (24,)), _array(self.scale, (24,))),
            _array(self.operator, (25, 32)),
            _array(self.base, (4, 8)),
            _array(self.multiplier, (25, 1)),
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawConditionalPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-conditional-payload'
    payload_id: str
    boundary: str
    development_manifest: ArtifactIdentity
    frozen_coefficients: ArtifactIdentity
    instrument: ObjectIdentity
    calibration: ObjectIdentity
    q: Decimal | None
    lower: ObjectIdentity | None
    center: tuple[Decimal, ...]
    scale: tuple[Decimal, ...]
    operator: tuple[Decimal, ...]
    base: tuple[Decimal, ...]
    multiplier: tuple[Decimal, ...]
    chart: FiniteResponseLawResponseChart = field(default=FiniteResponseLawResponseChart(), kw_only=True)

    def __post_init__(self) -> None:
        _provenance(self.payload_id, self.instrument, self.calibration, self.q)
        if self.boundary not in ("composed", "cached", "direct"):
            raise ValueError("Unknown frozen finite response-law conditional boundary")
        if (self.boundary == "composed") != (self.lower is not None) or (
            self.lower is not None and self.lower.object_schema != FiniteResponseLawLowerPayload.SCHEMA
        ):
            raise ValueError("Only composition binds the exact lower payload")
        _array(self.base, (4, 8), positive=True)
        if self.boundary == "cached":
            if self.center or self.scale:
                raise ValueError("Cached parent mean has no current-interface feature map")
            _array(self.operator, (5, 32))
            _array(self.multiplier, (5,))
        else:
            _array(self.center, (24,))
            _array(self.scale, (24,), positive=True)
            _array(self.operator, (5, 25, 24 if self.boundary == "composed" else 32))
            _array(self.multiplier, (5, 25, 1))
        if len(self.canonical_bytes()) > MAXIMUM_LAW_BYTES:
            raise ValueError("Finite response-law conditional payload exceeds its bounded decoder")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.payload_id, self)

    def predict(
        self, prefix: Array, parents: NDArray[np.int64], *, lower: FiniteResponseLawLowerPayload | None = None
    ) -> AssignedPrediction:
        _features(prefix)
        if (
            parents.shape != (len(prefix),)
            or parents.dtype != np.int64
            or ((parents < 0) | (parents >= 5)).any()
        ):
            raise ValueError("Finite response-law conditional evaluation changes its assigned native parent")
        if self.boundary == "composed":
            if (
                lower is None
                or lower.identity != self.lower
                or (
                    lower.development_manifest != self.development_manifest
                    or lower.frozen_coefficients != self.frozen_coefficients
                )
            ):
                raise ValueError("Finite response-law composition substitutes the frozen lower payload")
        elif lower is not None:
            raise ValueError("Comparator evaluation cannot acquire a lower-law dependency")
        n = len(prefix)
        if self.boundary == "cached":
            mean = clamp(np.broadcast_to(_array(self.operator, (5, 4, 8)), (n, 5, 4, 8)).copy())
            logg = np.broadcast_to(_array(self.multiplier, (5,)), (n, 5))
            support = np.ones((n, 5), dtype=bool)
        else:
            x, values = self._operator_values(prefix)
            support = np.broadcast_to(_supported(x)[:, None], (n, 5))
            if self.boundary == "composed":
                assert lower is not None
                normalized = lower.normalize(values)
                mean = lower.response(normalized)
                support = support & _supported(normalized)
            else:
                mean = clamp(values.reshape(-1, 5, 4, 8))
            multipliers = _array(self.multiplier, (5, 25, 1))
            logg = np.column_stack([affine_prediction(multipliers[p], x)[:, 0] for p in range(5)])
        rows = np.arange(n)
        return AssignedPrediction(
            mean[rows, parents],
            _sigma(_array(self.base, (4, 8)), logg)[rows, parents],
            support[rows, parents],
        )

    def _operator_values(self, prefix: Array) -> tuple[Array, Array]:
        """Preserve the frozen five-parent BLAS layout for U and direct prediction."""
        _features(prefix)
        if self.boundary not in ("composed", "direct"):
            raise ValueError("Cached prediction has no current-interface operator")
        x = Normalizer(_array(self.center, (24,)), _array(self.scale, (24,))).apply(prefix)
        operators = _array(self.operator, (5, 25, 24 if self.boundary == "composed" else 32))
        return x, np.stack([affine_prediction(operators[p], x) for p in range(5)], axis=1)

    def forecast_handoff(self, prefix: Array, parents: NDArray[np.int64]) -> Array:
        """Expose the same U output used inside composition, before preparation.

        This output alone grants neither support nor qualification; the complete
        law evaluation retains those checks.
        """
        if self.boundary != "composed" or (
            parents.shape != (len(prefix),)
            or parents.dtype != np.int64
            or ((parents < 0) | (parents >= 5)).any()
        ):
            raise ValueError("Only frozen U forecasts the declared parent's handoff")
        _, values = self._operator_values(prefix)
        return values[np.arange(len(prefix)), parents]


def lower_payload(
    points: PointFit,
    width: WidthFit,
    *,
    payload_id: str,
    development_manifest: ArtifactIdentity,
    frozen_coefficients: ArtifactIdentity,
    calibration: ObjectIdentity,
    q: Decimal | None,
) -> FiniteResponseLawLowerPayload:
    if points.recipe.dimension != 24 or width.boundary != "lower":
        raise ValueError("Finite response-law lower publication changes the frozen 24-feature lower response-law boundary")
    return FiniteResponseLawLowerPayload(
        payload_id,
        development_manifest,
        frozen_coefficients,
        REFERENCE_INSTRUMENT.identity,
        calibration,
        q,
        _values(points.nh.center),
        _values(points.nh.scale),
        _values(points.lower),
        _values(width.base),
        _values(width.multiplier),
    )


def conditional_payload(
    points: PointFit,
    width: WidthFit,
    *,
    payload_id: str,
    development_manifest: ArtifactIdentity,
    frozen_coefficients: ArtifactIdentity,
    calibration: ObjectIdentity,
    q: Decimal | None,
    lower: FiniteResponseLawLowerPayload | None = None,
) -> FiniteResponseLawConditionalPayload:
    if points.recipe.dimension != 24 or width.boundary not in ("composed", "cached", "direct"):
        raise ValueError("Finite response-law conditional publication changes its frozen boundary")
    operator = {"composed": points.upper, "cached": points.cached, "direct": points.direct}[
        width.boundary
    ]
    return FiniteResponseLawConditionalPayload(
        payload_id,
        width.boundary,
        development_manifest,
        frozen_coefficients,
        REFERENCE_INSTRUMENT.identity,
        calibration,
        q,
        None if lower is None else lower.identity,
        () if width.boundary == "cached" else _values(points.n0.center),
        () if width.boundary == "cached" else _values(points.n0.scale),
        _values(operator),
        _values(width.base),
        _values(width.multiplier),
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawLowerDecoder:
    extension_namespace: str = "finite-response-law-lower"
    decoder_schema: str = 'empirical-lawhood/methods/finite-response-law/lower-law-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = FiniteResponseLawLowerPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, FiniteResponseLawLowerPayload, maximum_bytes=min(maximum_bytes, MAXIMUM_LAW_BYTES)
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawConditionalDecoder:
    extension_namespace: str = "finite-response-law-conditional"
    decoder_schema: str = 'empirical-lawhood/methods/finite-response-law/conditional-law-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = FiniteResponseLawConditionalPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, FiniteResponseLawConditionalPayload, maximum_bytes=min(maximum_bytes, MAXIMUM_LAW_BYTES)
        )
