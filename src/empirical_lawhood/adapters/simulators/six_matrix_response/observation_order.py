"""Paired same-driver native acquisition for the observation order observation experiment."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
from typing import Any, ClassVar, Literal

import numpy as np

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.response_experiment_ports import NativeAcquisitionRequest

from .contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView, SixMatrixResponseSixMatrixSourceConfig
from .observation_order_scientific_inputs import (
    OBSERVATION_ORDER_STREAM_PURPOSES,
    MatrixObservationOrderScientificStreamInput,
    observation_order_scientific_stream_inputs,
)
from .history_preparation import FOUR_FAMILY_PREPARATION_IDS, four_family_preparation_couplings
from .model import SixMatrixState, hermiticity_residual, ideal_state
from .shooting import brownian_bridge_split
from .simulation import BAOABGradientCache, baoab_step_with_hermitian_noise, hermitian_noise

OBSERVATION_ORDER_HDF5_SCHEMA = 'empirical-lawhood/simulators/six-matrix-response/observation-order-paired-native-history-hdf5'
OBSERVATION_ORDER_HDF5_MEDIA_TYPE = "application/x-hdf5"
OBSERVATION_ORDER_CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
OBSERVATION_ORDER_SEED_DOMAIN = "empirical-lawhood/matrix-observation-order/scientific-stream-input"
OBSERVATION_ORDER_PRIMARY_STEPS = 1024
OBSERVATION_ORDER_FINE_STEPS = 2048
OBSERVATION_ORDER_RAMP_STEPS = 256
OBSERVATION_ORDER_DECODED_HDF5_BYTES = 9_517_104
OBSERVATION_ORDER_MAXIMUM_HDF5_BYTES = 16 * 1024 * 1024
OBSERVATION_ORDER_QUALIFICATION_FIXTURE_ID = "fixture.matrix-observation-order.full-chain"
OBSERVATION_ORDER_QUALIFICATION_SEED_ROOT = '79269e4590bbd7aad7a980fb701012a4597f748ae0382f00315770908170aad0'
OBSERVATION_ORDER_PRODUCTION_SEED_ROOT = 'ad6d0a0198f406ec4b4318226a5f537e3a865a91d07d17ebb8c5f9d791c34b3d'
_STATE_SHAPE = (2, 3, 4, 4)
_HDF5_KEYS = (
    '["/fine/alpha_tilde","/fine/momenta","/fine/positions","/fine/step_index",'
    '"/primary/alpha_tilde","/primary/momenta","/primary/positions",'
    '"/primary/step_index"]'
)
_HDF5_UNITS = (
    '{"/fine/alpha_tilde":"dimensionless-scaled-coupling",'
    '"/fine/momenta":"dimensionless-canonical-momentum",'
    '"/fine/positions":"dimensionless-matrix-coordinate",'
    '"/fine/step_index":"fine-integration-step",'
    '"/primary/alpha_tilde":"dimensionless-scaled-coupling",'
    '"/primary/momenta":"dimensionless-canonical-momentum",'
    '"/primary/positions":"dimensionless-matrix-coordinate",'
    '"/primary/step_index":"primary-equivalent-integration-step"}'
)
_HDF5_FRAMES = (
    '{"/fine/alpha_tilde":"six-matrix-response.frame.xy-coupling",'
    '"/fine/momenta":"six-matrix-response.frame.simultaneous-unitary-quotient",'
    '"/fine/positions":"six-matrix-response.frame.simultaneous-unitary-quotient",'
    '"/fine/step_index":"six-matrix-response.frame.integration-index",'
    '"/primary/alpha_tilde":"six-matrix-response.frame.xy-coupling",'
    '"/primary/momenta":"six-matrix-response.frame.simultaneous-unitary-quotient",'
    '"/primary/positions":"six-matrix-response.frame.simultaneous-unitary-quotient",'
    '"/primary/step_index":"six-matrix-response.frame.integration-index"}'
)
_HDF5_CLOCKS = (
    '{"/fine/alpha_tilde":"six-matrix-response.clock.fine-integration-step",'
    '"/fine/momenta":"six-matrix-response.clock.fine-integration-step",'
    '"/fine/positions":"six-matrix-response.clock.fine-integration-step",'
    '"/fine/step_index":"six-matrix-response.clock.fine-integration-step",'
    '"/primary/alpha_tilde":"six-matrix-response.clock.primary-equivalent-step",'
    '"/primary/momenta":"six-matrix-response.clock.primary-equivalent-step",'
    '"/primary/positions":"six-matrix-response.clock.primary-equivalent-step",'
    '"/primary/step_index":"six-matrix-response.clock.primary-equivalent-step"}'
)


class MatrixObservationOrderAcquisitionDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    PROVIDER_EXCEPTION = "PROVIDER_EXCEPTION"


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderHistorySlot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/matrix-observation-order-history-slot'

    history_index: int
    history_id: str
    family_id: str
    physical_independent_unit_id: str
    preparation_instance_id: str
    acquisition_group_id: str
    primary_view_id: str
    fine_view_id: str

    def __post_init__(self) -> None:
        if not 0 <= self.history_index < 256:
            raise ValueError("observation order history index differs")
        for name in (
            "history_id",
            "family_id",
            "physical_independent_unit_id",
            "preparation_instance_id",
            "acquisition_group_id",
            "primary_view_id",
            "fine_view_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.family_id != FOUR_FAMILY_PREPARATION_IDS[self.history_index // 64]:
            raise ValueError("observation order history family allocation differs")


def build_observation_order_history_slots() -> tuple[MatrixObservationOrderHistorySlot, ...]:
    values = []
    for index in range(256):
        stem = f"matrix-observation-order.h{index:03d}"
        values.append(
            MatrixObservationOrderHistorySlot(
                history_index=index,
                history_id=f"history.{stem}",
                family_id=FOUR_FAMILY_PREPARATION_IDS[index // 64],
                physical_independent_unit_id=f"unit.{stem}",
                preparation_instance_id=f"preparation.{stem}",
                acquisition_group_id=f"acquisition.{stem}",
                primary_view_id=f"view.{stem}.primary",
                fine_view_id=f"view.{stem}.fine",
            )
        )
    return tuple(values)


def build_observation_order_qualification_slots() -> tuple[MatrixObservationOrderHistorySlot, ...]:
    """Return one outcome-independent plumbing history per frozen family."""

    production = build_observation_order_history_slots()
    return tuple(production[index] for index in (0, 64, 128, 192))


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderPairedHistorySourceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/six-matrix-response/matrix-observation-order-paired-history-source-config'
    )

    config_id: str
    science_specification: ObjectIdentity
    source_design: ObjectIdentity
    member: SixMatrixResponseModelFamilyMember
    primary_view: SixMatrixResponseNumericalView
    fine_view: SixMatrixResponseNumericalView
    seed_root_hex: str
    seed_domain: str
    scientific_stream_inputs: tuple[MatrixObservationOrderScientificStreamInput, ...]
    slots: tuple[MatrixObservationOrderHistorySlot, ...]
    target_alpha_tilde_x: Decimal
    target_alpha_tilde_y: Decimal
    decreasing_coupling_start: Decimal
    ramp_steps: int
    primary_total_steps: int
    fine_integrator_multiplier: int
    decoded_hdf5_bytes: int
    maximum_hdf5_bytes: int
    qualification_fixture_id: str | None
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.seed_root_hex, field_name="seed_root_hex")
        expected_inputs = observation_order_scientific_stream_inputs(
            qualification=self.qualification_fixture_id is not None
        )
        if self.scientific_stream_inputs != expected_inputs:
            raise ValueError(
                "observation-order complete original scientific stream inputs are required"
            )
        if self.qualification_fixture_id is not None:
            validate_stable_id(
                self.qualification_fixture_id,
                field_name="qualification_fixture_id",
            )
        require_sorted_unique_ids(
            self.slots, attribute="history_id", field_name="slots"
        )
        for name in ("target_alpha_tilde_x", "target_alpha_tilde_y", "decreasing_coupling_start"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if (
            self.source_design.object_schema != SixMatrixResponseSixMatrixSourceConfig.SCHEMA
            or self.member.member_id != "six-matrix-response.member.mass-0p5.cross-coupling-1"
            or self.member.fingerprint()
            != "1f15e5757d5f1adef0bb6611d83a6210dd5e1b2d94b5a1adef1639124192c287"
            or self.primary_view.view_id != "six-matrix-response.view.baoab-dt-0p001"
            or self.primary_view.fingerprint()
            != "8abd68021c070d1fe82e32c6a0b488b2e559f213be000c5873a18742eafb4d6a"
            or self.fine_view.view_id != "six-matrix-response.view.baoab-dt-0p0005"
            or self.fine_view.fingerprint()
            != "bf595a377ad9a4f9a064b1640bee26b864f1bc8e2256a88ca8cf4e6fa55bce99"
            or self.science_specification.object_schema
            != "empirical-lawhood/scientific-input/matrix-observation-order-science-freeze"
            or self.seed_domain != OBSERVATION_ORDER_SEED_DOMAIN
            or self.target_alpha_tilde_x != Decimal("0.6666666666666666666666666667")
            or self.target_alpha_tilde_y != Decimal("7.333333333333333333333333334")
            or self.decreasing_coupling_start != Decimal(8)
            or self.ramp_steps != OBSERVATION_ORDER_RAMP_STEPS
            or self.primary_total_steps != OBSERVATION_ORDER_PRIMARY_STEPS
            or self.fine_integrator_multiplier != 2
            or self.decoded_hdf5_bytes != OBSERVATION_ORDER_DECODED_HDF5_BYTES
            or self.maximum_hdf5_bytes != OBSERVATION_ORDER_MAXIMUM_HDF5_BYTES
            or (
                self.qualification_fixture_id is None
                and (
                    self.seed_root_hex != OBSERVATION_ORDER_PRODUCTION_SEED_ROOT
                    or self.slots != build_observation_order_history_slots()
                )
            )
            or (
                self.qualification_fixture_id is not None
                and (
                    self.qualification_fixture_id != OBSERVATION_ORDER_QUALIFICATION_FIXTURE_ID
                    or self.seed_root_hex != OBSERVATION_ORDER_QUALIFICATION_SEED_ROOT
                    or self.slots != build_observation_order_qualification_slots()
                )
            )
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("observation order source config differs from its frozen contract")


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderPairedHistoryRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/matrix-observation-order-paired-history-request'

    request_id: str
    task_id: str
    source_config: ObjectIdentity
    neutral_request: NativeAcquisitionRequest
    slot: MatrixObservationOrderHistorySlot
    evidence_ceiling: EvidenceCeiling
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.task_id, field_name="task_id")
        if (
            self.neutral_request.acquisition_group_id != self.slot.acquisition_group_id
            or self.neutral_request.physical_independent_unit_id
            != self.slot.physical_independent_unit_id
            or self.evidence_ceiling is not EvidenceCeiling.ORDER_RELATION
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("observation order paired request changes lineage or authority")


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderPairedHistoryResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/matrix-observation-order-paired-history-result'

    result_id: str
    request: ObjectIdentity
    source_config: ObjectIdentity
    history_id: str
    family_id: str
    physical_independent_unit_id: str
    acquisition_group_id: str
    primary_view_id: str
    fine_view_id: str
    disposition: MatrixObservationOrderAcquisitionDisposition
    completed_primary_steps: int
    completed_fine_steps: int
    hdf5_payload_schema: str
    hdf5_size_bytes: int
    hdf5_sha256: str
    decoded_dataset_bytes: int
    seed_receipts: tuple[tuple[str, str], ...]
    reason_codes: tuple[str, ...]
    scientific_verdict_constructed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "result_id",
            "history_id",
            "family_id",
            "physical_independent_unit_id",
            "acquisition_group_id",
            "primary_view_id",
            "fine_view_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.hdf5_sha256, field_name="hdf5_sha256")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        purposes = tuple(value[0] for value in self.seed_receipts)
        if tuple(sorted(set(purposes))) != purposes:
            raise ValueError("observation order seed receipt roster differs")
        for _, digest in self.seed_receipts:
            validate_sha256(digest, field_name="seed_receipt")
        complete = self.disposition is MatrixObservationOrderAcquisitionDisposition.COMPLETE
        if (
            self.family_id not in FOUR_FAMILY_PREPARATION_IDS
            or not 0 <= self.completed_primary_steps <= OBSERVATION_ORDER_PRIMARY_STEPS
            or not 0 <= self.completed_fine_steps <= OBSERVATION_ORDER_FINE_STEPS
            or self.hdf5_payload_schema != OBSERVATION_ORDER_HDF5_SCHEMA
            or not 0 < self.hdf5_size_bytes <= OBSERVATION_ORDER_MAXIMUM_HDF5_BYTES
            or self.decoded_dataset_bytes != OBSERVATION_ORDER_DECODED_HDF5_BYTES
            or complete != (not self.reason_codes)
            or (
                complete
                and (
                    self.completed_primary_steps != OBSERVATION_ORDER_PRIMARY_STEPS
                    or self.completed_fine_steps != OBSERVATION_ORDER_FINE_STEPS
                )
            )
            or self.scientific_verdict_constructed
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("observation order paired result differs from native disposition")


def _seed(
    config: MatrixObservationOrderPairedHistorySourceConfig,
    slot: MatrixObservationOrderHistorySlot,
    purpose: str,
) -> tuple[np.random.Generator, str]:
    if slot not in config.slots or purpose not in OBSERVATION_ORDER_STREAM_PURPOSES:
        raise ValueError("observation-order scientific stream input key is unsupported")
    ordinal = OBSERVATION_ORDER_STREAM_PURPOSES.index(purpose)
    stream = next(
        (
            value
            for value in config.scientific_stream_inputs
            if value.history_index == slot.history_index
            and value.purpose_ordinal == ordinal
        ),
        None,
    )
    if stream is None:
        raise ValueError("observation-order scientific stream input is missing")
    digest = bytes.fromhex(stream.scientific_seed_sha256)
    return np.random.Generator(
        np.random.PCG64DXSM(int.from_bytes(digest[:16], "big"))
    ), digest.hex()


def _initial(slot: MatrixObservationOrderHistorySlot, start: float) -> SixMatrixState:
    de_anneal = slot.family_id == "matrix-history.joint-decreasing-coupling"
    return ideal_state(
        q=2,
        alpha_tilde_x=start if de_anneal else 0.0,
        alpha_tilde_y=start if de_anneal else 0.0,
        constitution="11" if de_anneal else "00",
    )


def _empty_arrays() -> dict[str, np.ndarray[Any, Any]]:
    return {
        "primary_positions": np.full((1025, *_STATE_SHAPE), np.nan, dtype="<c16"),
        "primary_momenta": np.full((1025, *_STATE_SHAPE), np.nan, dtype="<c16"),
        "primary_alpha": np.full((1025, 2), np.nan, dtype="<f8"),
        "fine_positions": np.full((2049, *_STATE_SHAPE), np.nan, dtype="<c16"),
        "fine_momenta": np.full((2049, *_STATE_SHAPE), np.nan, dtype="<c16"),
        "fine_alpha": np.full((2049, 2), np.nan, dtype="<f8"),
    }


def _store(
    arrays: dict[str, np.ndarray[Any, Any]],
    view: Literal["primary", "fine"],
    index: int,
    state: SixMatrixState,
) -> None:
    arrays[f"{view}_positions"][index] = state.positions
    arrays[f"{view}_momenta"][index] = state.momenta
    arrays[f"{view}_alpha"][index] = (state.alpha_tilde_x, state.alpha_tilde_y)


def _hdf5(
    config: MatrixObservationOrderPairedHistorySourceConfig,
    request: MatrixObservationOrderPairedHistoryRequest,
    arrays: dict[str, np.ndarray[Any, Any]],
    disposition: MatrixObservationOrderAcquisitionDisposition,
) -> bytes:
    stream = BytesIO()
    h5py = importlib.import_module("h5py")
    with h5py.File(stream, "w", libver="earliest", track_order=False) as handle:
        attrs = {
            "acquisition_group_id": request.slot.acquisition_group_id,
            "compression": "NONE",
            "disposition": disposition.value,
            "history_id": request.slot.history_id,
            "empirical_lawhood_clocks": _HDF5_CLOCKS,
            "empirical_lawhood_frames": _HDF5_FRAMES,
            "empirical_lawhood_keys": _HDF5_KEYS,
            "empirical_lawhood_payload_schema": OBSERVATION_ORDER_HDF5_SCHEMA,
            "empirical_lawhood_units": _HDF5_UNITS,
            "physical_independent_unit_id": request.slot.physical_independent_unit_id,
            "request_fingerprint": request.fingerprint(),
            "source_config_fingerprint": config.fingerprint(),
        }
        for key, value in sorted(attrs.items()):
            encoded = str(value).encode("utf-8")
            handle.attrs.create(key, encoded, dtype=f"S{len(encoded)}")
        for view, count in (("primary", 1025), ("fine", 2049)):
            group = handle.create_group(view, track_order=False)
            group.create_dataset(
                "alpha_tilde", data=arrays[f"{view}_alpha"], track_times=False
            )
            group.create_dataset(
                "momenta", data=arrays[f"{view}_momenta"], track_times=False
            )
            group.create_dataset(
                "positions", data=arrays[f"{view}_positions"], track_times=False
            )
            group.create_dataset(
                "step_index",
                data=np.arange(count, dtype="<i8"),
                track_times=False,
            )
    payload = stream.getvalue()
    if len(payload) > config.maximum_hdf5_bytes:
        raise ValueError("observation order paired HDF5 exceeds its frozen byte bound")
    return payload


def execute_observation_order_paired_history(
    config: MatrixObservationOrderPairedHistorySourceConfig, request: MatrixObservationOrderPairedHistoryRequest
) -> tuple[MatrixObservationOrderPairedHistoryResult, bytes]:
    """Integrate one complete history and its same-driver fine nested view."""

    gradient_cache_fine = BAOABGradientCache()
    gradient_cache_primary = BAOABGradientCache()
    if request.source_config != ObjectIdentity.from_record(config.config_id, config):
        raise ValueError("observation order source request substitutes its issued config")
    arrays = _empty_arrays()
    completed_primary = completed_fine = 0
    disposition = MatrixObservationOrderAcquisitionDisposition.COMPLETE
    reasons: tuple[str, ...] = ()
    receipts: list[tuple[str, str]] = []
    try:
        preparation_rng, digest = _seed(
            config, request.slot, "preparation-coarse-driver"
        )
        receipts.append(("preparation-coarse-driver", digest))
        autonomous_rng, digest = _seed(config, request.slot, "autonomous-coarse-driver")
        receipts.append(("autonomous-coarse-driver", digest))
        preparation_bridge, digest = _seed(
            config, request.slot, "preparation-fine-bridge"
        )
        receipts.append(("preparation-fine-bridge", digest))
        autonomous_bridge, digest = _seed(
            config, request.slot, "autonomous-fine-bridge"
        )
        receipts.append(("autonomous-fine-bridge", digest))
        primary = _initial(request.slot, float(config.decreasing_coupling_start))
        fine = _initial(request.slot, float(config.decreasing_coupling_start))
        _store(arrays, "primary", 0, primary)
        _store(arrays, "fine", 0, fine)
        half_decay = float(np.exp(-float(config.primary_view.friction_gamma) * 0.0005))
        for coarse_step in range(1, config.primary_total_steps + 1):
            in_preparation = coarse_step <= config.ramp_steps
            coarse_rng = preparation_rng if in_preparation else autonomous_rng
            bridge_rng = preparation_bridge if in_preparation else autonomous_bridge
            coarse_noise = hermitian_noise(rng=coarse_rng, q=2)
            bridge_noise = hermitian_noise(rng=bridge_rng, q=2)
            first, second = brownian_bridge_split(
                coarse_noise=coarse_noise,
                bridge_noise=bridge_noise,
                half_decay=half_decay,
            )
            primary_xy = four_family_preparation_couplings(
                request.slot.family_id,
                coarse_step,
                ramp_steps=config.ramp_steps,
                integrator_multiplier=1,
                target_x=float(config.target_alpha_tilde_x),
                target_y=float(config.target_alpha_tilde_y),
                decreasing_coupling_start=float(config.decreasing_coupling_start),
            )
            primary = baoab_step_with_hermitian_noise(
                primary,
                member=config.member,
                numerical_view=config.primary_view,
                next_alpha_tilde_x=primary_xy[0],
                next_alpha_tilde_y=primary_xy[1],
                standardized_noise=coarse_noise,
                gradient_cache=gradient_cache_primary,
            )
            _store(arrays, "primary", coarse_step, primary)
            completed_primary = coarse_step
            for half, noise in enumerate((first, second), 1):
                fine_step = (coarse_step - 1) * 2 + half
                fine_xy = four_family_preparation_couplings(
                    request.slot.family_id,
                    fine_step,
                    ramp_steps=config.ramp_steps,
                    integrator_multiplier=2,
                    target_x=float(config.target_alpha_tilde_x),
                    target_y=float(config.target_alpha_tilde_y),
                    decreasing_coupling_start=float(config.decreasing_coupling_start),
                )
                fine = baoab_step_with_hermitian_noise(
                    fine,
                    member=config.member,
                    numerical_view=config.fine_view,
                    next_alpha_tilde_x=fine_xy[0],
                    next_alpha_tilde_y=fine_xy[1],
                    standardized_noise=noise,
                    gradient_cache=gradient_cache_fine,
                )
                _store(arrays, "fine", fine_step, fine)
                completed_fine = fine_step
            if (
                not primary.finite
                or not fine.finite
                or max(
                    hermiticity_residual(primary.positions),
                    hermiticity_residual(fine.positions),
                )
                > 1e-10
            ):
                raise FloatingPointError("observation order integration became nonfinite")
    except FloatingPointError:
        disposition = MatrixObservationOrderAcquisitionDisposition.INVALID
        reasons = ("NATIVE_NUMERICAL_INVALID",)
    except Exception:
        disposition = MatrixObservationOrderAcquisitionDisposition.PROVIDER_EXCEPTION
        reasons = ("NATIVE_PROVIDER_EXCEPTION",)
    if disposition is MatrixObservationOrderAcquisitionDisposition.COMPLETE and (
        completed_primary != OBSERVATION_ORDER_PRIMARY_STEPS or completed_fine != OBSERVATION_ORDER_FINE_STEPS
    ):
        disposition = MatrixObservationOrderAcquisitionDisposition.PARTIAL
        reasons = ("NATIVE_HISTORY_PARTIAL",)
    payload = _hdf5(config, request, arrays, disposition)
    result = MatrixObservationOrderPairedHistoryResult(
        result_id=f"paired-history-result.{request.slot.history_id}",
        request=ObjectIdentity.from_record(request.request_id, request),
        source_config=ObjectIdentity.from_record(config.config_id, config),
        history_id=request.slot.history_id,
        family_id=request.slot.family_id,
        physical_independent_unit_id=request.slot.physical_independent_unit_id,
        acquisition_group_id=request.slot.acquisition_group_id,
        primary_view_id=request.slot.primary_view_id,
        fine_view_id=request.slot.fine_view_id,
        disposition=disposition,
        completed_primary_steps=completed_primary,
        completed_fine_steps=completed_fine,
        hdf5_payload_schema=OBSERVATION_ORDER_HDF5_SCHEMA,
        hdf5_size_bytes=len(payload),
        hdf5_sha256=sha256(payload).hexdigest(),
        decoded_dataset_bytes=OBSERVATION_ORDER_DECODED_HDF5_BYTES,
        seed_receipts=tuple(sorted(receipts)),
        reason_codes=reasons,
        scientific_verdict_constructed=False,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )
    return result, payload


def read_observation_order_paired_history(
    payload: bytes, result: MatrixObservationOrderPairedHistoryResult, view: Literal["primary", "fine"]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Validate hostile HDF5 structure before returning one bounded view."""

    if (
        len(payload) != result.hdf5_size_bytes
        or sha256(payload).hexdigest() != result.hdf5_sha256
    ):
        raise ValueError("observation order HDF5 custody identity differs")
    if len(payload) > OBSERVATION_ORDER_MAXIMUM_HDF5_BYTES or view not in {"primary", "fine"}:
        raise ValueError("observation order HDF5 request exceeds its bound")
    h5py = importlib.import_module("h5py")
    with h5py.File(BytesIO(payload), "r") as handle:
        if set(handle) != {"fine", "primary"}:
            raise ValueError("observation order HDF5 root inventory differs")
        attrs = {
            key: value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for key, value in handle.attrs.items()
        }
        expected_attrs = {
            "acquisition_group_id": result.acquisition_group_id,
            "compression": "NONE",
            "disposition": result.disposition.value,
            "history_id": result.history_id,
            "empirical_lawhood_clocks": _HDF5_CLOCKS,
            "empirical_lawhood_frames": _HDF5_FRAMES,
            "empirical_lawhood_keys": _HDF5_KEYS,
            "empirical_lawhood_payload_schema": OBSERVATION_ORDER_HDF5_SCHEMA,
            "empirical_lawhood_units": _HDF5_UNITS,
            "physical_independent_unit_id": result.physical_independent_unit_id,
            "request_fingerprint": result.request.object_fingerprint,
            "source_config_fingerprint": result.source_config.object_fingerprint,
        }
        if attrs != expected_attrs:
            raise ValueError("observation order HDF5 metadata differs")
        expected_count = 1025 if view == "primary" else 2049
        group = handle[view]
        if set(group) != {"alpha_tilde", "momenta", "positions", "step_index"}:
            raise ValueError("observation order HDF5 view inventory differs")
        specifications = {
            "alpha_tilde": ((expected_count, 2), np.dtype("<f8")),
            "momenta": ((expected_count, *_STATE_SHAPE), np.dtype("<c16")),
            "positions": ((expected_count, *_STATE_SHAPE), np.dtype("<c16")),
            "step_index": ((expected_count,), np.dtype("<i8")),
        }
        for name, (shape, dtype) in specifications.items():
            dataset = group[name]
            if (
                dataset.shape != shape
                or dataset.dtype != dtype
                or dataset.chunks is not None
                or dataset.compression is not None
            ):
                raise ValueError("observation order HDF5 dataset contract differs")
        if not np.array_equal(
            group["step_index"][:], np.arange(expected_count, dtype="<i8")
        ):
            raise ValueError("observation order HDF5 native step index differs")
        positions = np.asarray(group["positions"][:], dtype="<c16")
        momenta = np.asarray(group["momenta"][:], dtype="<c16")
        alpha = np.asarray(group["alpha_tilde"][:], dtype="<f8")
    return positions, momenta, alpha


__all__ = [
    'MatrixObservationOrderAcquisitionDisposition',
    'MatrixObservationOrderHistorySlot',
    'MatrixObservationOrderPairedHistoryRequest',
    'MatrixObservationOrderPairedHistoryResult',
    'MatrixObservationOrderPairedHistorySourceConfig',
    "OBSERVATION_ORDER_CANONICAL_MEDIA_TYPE",
    "OBSERVATION_ORDER_DECODED_HDF5_BYTES",
    "OBSERVATION_ORDER_HDF5_MEDIA_TYPE",
    "OBSERVATION_ORDER_HDF5_SCHEMA",
    "OBSERVATION_ORDER_MAXIMUM_HDF5_BYTES",
    "OBSERVATION_ORDER_PRODUCTION_SEED_ROOT",
    "OBSERVATION_ORDER_QUALIFICATION_FIXTURE_ID",
    "OBSERVATION_ORDER_QUALIFICATION_SEED_ROOT",
    'build_observation_order_history_slots',
    'build_observation_order_qualification_slots',
    'execute_observation_order_paired_history',
    'read_observation_order_paired_history',
]
