"Metadata-complete Gym--TORAX request and episode contracts.\n\nThe retained diagnostic-corpus decoder remains separate. Fresh Gym-TORAX source\nevidence uses these metadata-complete records; metadata repair does not rewrite\nthe meaning of the retained diagnostic records.\n"

from __future__ import annotations

import base64
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
import hashlib
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .diagnostic_contracts import GymToraxActionDelivery, GymToraxDeliveryDisposition, GymToraxNativeSchedule, GymToraxNumericalDisposition, GymToraxNumericalMember, GymToraxObservationDisposition, GymToraxOperatorSourceSummary, GymToraxPreparation, GymToraxSourceDisposition
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest


class GymToraxRequestRole(StrEnum):
    METADATA_CANARY = "METADATA_CANARY"
    SCIENTIFIC_EPISODE = "SCIENTIFIC_EPISODE"


@dataclass(frozen=True, slots=True)
class GymToraxFieldMetadataEpisodeRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-field-metadata-episode-request'

    request_id: str
    request_role: GymToraxRequestRole
    source_qualification: ObjectIdentity | None
    extraction_manifest: ObjectIdentity
    field_metadata_manifest: ObjectIdentity
    preparation: GymToraxPreparation
    numerical_member: GymToraxNumericalMember
    schedule: GymToraxNativeSchedule
    maximum_output_bytes: int
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        if self.request_role is GymToraxRequestRole.METADATA_CANARY:
            if self.source_qualification is not None:
                raise ValueError("metadata canary must precede source qualification")
            if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
                raise ValueError("metadata canary must remain nonpromotable")
        elif self.source_qualification is None:
            raise ValueError("scientific source episode requires source qualification")
        if self.extraction_manifest.object_schema != GymToraxBoundedExtractionManifest.SCHEMA:
            raise ValueError("Gym--TORAX metadata-complete request requires the surgical extraction manifest")
        if self.field_metadata_manifest.object_schema != GymToraxFieldMetadataManifest.SCHEMA:
            raise ValueError("Gym--TORAX metadata-complete request requires the field metadata manifest")
        if self.maximum_output_bytes <= 0 or self.maximum_output_bytes > 512 * 1024**2:
            raise ValueError("Gym--TORAX episode output bound is outside 512 MiB")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("Gym--TORAX request has an unsupported outcome-access lane")
        if self.evidence_ceiling not in {
            EvidenceCeiling.NON_PROMOTABLE,
            EvidenceCeiling.MEASUREMENT,
        }:
            raise ValueError("Gym--TORAX source request cannot claim response-law truth")


@dataclass(frozen=True, slots=True)
class GymToraxFieldMetadataFloat64Block(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-field-metadata-float64-block'

    block_id: str
    category: str
    native_field_id: str
    native_unit: str
    native_frame_id: str
    field_metadata_id: str | None
    dimension_ids: tuple[str, ...]
    shape: tuple[int, ...]
    clock_values: tuple[int, ...]
    data_base64: str
    data_sha256: str
    finite_value_count: int
    nonfinite_value_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.block_id, field_name="block_id")
        validate_stable_id(self.category, field_name="category")
        validate_nonempty(self.native_field_id, field_name="native_field_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.native_unit == "source-unit-unspecified":
            raise ValueError("Gym--TORAX metadata-complete blocks cannot carry unspecified units")
        validate_stable_id(self.native_frame_id, field_name="native_frame_id")
        is_source = self.category.startswith("source-")
        if is_source != (self.field_metadata_id is not None):
            raise ValueError("source blocks require exactly one field metadata entry")
        if self.field_metadata_id is not None:
            validate_stable_id(self.field_metadata_id, field_name="field_metadata_id")
        if not self.dimension_ids or any(not value for value in self.dimension_ids):
            raise ValueError("dimension IDs must contain explicit ordered axes")
        if len(set(self.dimension_ids)) != len(self.dimension_ids):
            raise ValueError("dimension IDs must be unique")
        if any(value.startswith("source-axis-") for value in self.dimension_ids):
            raise ValueError("synthetic fallback axes are prohibited in metadata-complete episodes")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError("Gym--TORAX block shape must be positive")
        if self.finite_value_count < 0 or self.nonfinite_value_count < 0:
            raise ValueError("Gym--TORAX block counts must be nonnegative")
        if self.finite_value_count + self.nonfinite_value_count != int(np.prod(self.shape)):
            raise ValueError("Gym--TORAX block counts differ from shape")
        payload = base64.b64decode(self.data_base64, validate=True)
        validate_sha256(self.data_sha256, field_name="data_sha256")
        if len(payload) != int(np.prod(self.shape)) * 8:
            raise ValueError("Gym--TORAX block byte count differs from float64")
        if hashlib.sha256(payload).hexdigest() != self.data_sha256:
            raise ValueError("Gym--TORAX block digest differs")
        if self.clock_values != tuple(sorted(set(self.clock_values))):
            raise ValueError("Gym--TORAX block clocks must be ordered and unique")
        if self.clock_values and self.shape[0] != len(self.clock_values):
            raise ValueError("Gym--TORAX block clock axis differs from shape")

    def array(self) -> npt.NDArray[np.float64]:
        return np.frombuffer(base64.b64decode(self.data_base64), dtype="<f8").reshape(self.shape)


def encode_gym_torax_float64_block(
    *,
    block_id: str,
    category: str,
    native_field_id: str,
    native_unit: str,
    native_frame_id: str,
    field_metadata_id: str | None,
    dimension_ids: tuple[str, ...],
    values: npt.ArrayLike,
    clock_values: tuple[int, ...] = (),
) -> GymToraxFieldMetadataFloat64Block:
    """Encode one bounded little-endian array with explicit metadata lineage."""

    array = np.ascontiguousarray(values, dtype="<f8")
    if array.ndim == 0:
        array = array.reshape(1)
    payload = array.tobytes(order="C")
    finite_count = int(np.isfinite(array).sum())
    return GymToraxFieldMetadataFloat64Block(
        block_id=block_id,
        category=category,
        native_field_id=native_field_id,
        native_unit=native_unit,
        native_frame_id=native_frame_id,
        field_metadata_id=field_metadata_id,
        dimension_ids=dimension_ids,
        shape=tuple(int(value) for value in array.shape),
        clock_values=clock_values,
        data_base64=base64.b64encode(payload).decode("ascii"),
        data_sha256=hashlib.sha256(payload).hexdigest(),
        finite_value_count=finite_count,
        nonfinite_value_count=array.size - finite_count,
    )


@dataclass(frozen=True, slots=True)
class GymToraxFieldMetadataNativeEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-field-metadata-native-episode'

    episode_id: str
    request: ObjectIdentity
    preparation: ObjectIdentity
    numerical_member: ObjectIdentity
    action_word: ObjectIdentity
    extraction_manifest: ObjectIdentity
    field_metadata_manifest: ObjectIdentity
    source_reset_attempted: bool
    state_clocks: tuple[int, ...]
    missing_required_state_clocks: tuple[int, ...]
    last_valid_state_clock: int | None
    deliveries: tuple[GymToraxActionDelivery, ...]
    blocks: tuple[GymToraxFieldMetadataFloat64Block, ...]
    operator_source: GymToraxOperatorSourceSummary
    source_disposition: GymToraxSourceDisposition
    delivery_disposition: GymToraxDeliveryDisposition
    numerical_disposition: GymToraxNumericalDisposition
    observation_disposition: GymToraxObservationDisposition
    termination: bool
    truncation: bool
    backend: str
    precision: str
    python_version: str
    numpy_version: str
    scipy_version: str
    xarray_version: str
    gymtorax_version: str
    torax_version: str
    jax_version: str
    jaxlib_version: str
    runtime_seconds: Decimal
    peak_rss_bytes: int
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        if self.request.object_schema != GymToraxFieldMetadataEpisodeRequest.SCHEMA:
            raise ValueError("Gym--TORAX metadata-complete episode binds another request schema")
        if self.preparation.object_schema != GymToraxPreparation.SCHEMA:
            raise ValueError("Gym--TORAX episode binds another preparation schema")
        if self.numerical_member.object_schema != GymToraxNumericalMember.SCHEMA:
            raise ValueError("Gym--TORAX episode binds another member schema")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("Gym--TORAX episode requires an exact ActionWord")
        if self.extraction_manifest.object_schema != GymToraxBoundedExtractionManifest.SCHEMA:
            raise ValueError("Gym--TORAX metadata-complete episode binds another extraction schema")
        if self.field_metadata_manifest.object_schema != GymToraxFieldMetadataManifest.SCHEMA:
            raise ValueError("Gym--TORAX episode binds another metadata schema")
        if self.state_clocks and not self.source_reset_attempted:
            raise ValueError("observed state clocks require an attempted source reset")
        if self.state_clocks != tuple(sorted(set(self.state_clocks))):
            raise ValueError("Gym--TORAX state clocks must be ordered and unique")
        if self.missing_required_state_clocks != tuple(
            sorted(set(self.missing_required_state_clocks))
        ):
            raise ValueError("Gym--TORAX missing clocks must be sorted and unique")
        require_sorted_unique_ids(
            self.deliveries,
            attribute="delivery_id",
            field_name="deliveries",
        )
        require_sorted_unique_ids(self.blocks, attribute="block_id", field_name="blocks")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.last_valid_state_clock != (max(self.state_clocks) if self.state_clocks else None):
            raise ValueError("Gym--TORAX last-valid clock differs from its state roster")
        validate_decimal(self.runtime_seconds, field_name="runtime_seconds", minimum=Decimal(0))
        if self.peak_rss_bytes < 0:
            raise ValueError("Gym--TORAX peak RSS must be nonnegative")
        if (
            self.backend != "cpu"
            or self.precision != "float64"
            or self.python_version != "3.11.14"
            or self.numpy_version != "2.4.6"
            or self.scipy_version != "1.17.1"
            or self.xarray_version != "2026.7.0"
            or self.gymtorax_version != "1.1.1"
            or self.torax_version != "1.4.2"
            or self.jax_version != "0.10.2"
            or self.jaxlib_version != "0.10.2"
        ):
            raise ValueError("Gym--TORAX episode runtime denominator differs")
        if self.evidence_ceiling not in {
            EvidenceCeiling.NON_PROMOTABLE,
            EvidenceCeiling.MEASUREMENT,
        }:
            raise ValueError("Gym--TORAX episode cannot claim response-law truth")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("Gym--TORAX source episode cannot reveal outcomes")
        source_blocks = tuple(
            value for value in self.blocks if value.category.startswith("source-")
        )
        if self.source_disposition is GymToraxSourceDisposition.AVAILABLE and not source_blocks:
            raise ValueError("Gym--TORAX metadata-complete episode requires retained source blocks")
        if any(
            value.native_unit == "source-unit-unspecified"
            or value.field_metadata_id is None
            or not value.native_frame_id
            for value in source_blocks
        ):
            raise ValueError("Gym--TORAX metadata-complete source block failed the metadata barrier")


__all__ = [
    'GymToraxFieldMetadataEpisodeRequest',
    'GymToraxFieldMetadataFloat64Block',
    'GymToraxFieldMetadataNativeEpisode',
    'GymToraxRequestRole',
    'encode_gym_torax_float64_block',
]
