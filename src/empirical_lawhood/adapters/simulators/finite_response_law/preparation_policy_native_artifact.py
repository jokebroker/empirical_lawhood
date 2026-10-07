"""Closed paired-view artifact for the distinct preparation-policy clocks."""

from dataclasses import dataclass
from functools import lru_cache
import io
from typing import ClassVar

from .native_artifact import FiniteResponseLawNativePairArtifact, METADATA
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_text, response_hdf5_writer
from .preparation_policy_source import FiniteResponseLawPreparationPolicyNativeCheckpoint, FiniteResponseLawPreparationPolicyNativeDelivery

PREPARATION_POLICY_NATIVE_PAIR_SCHEMA = 'empirical-lawhood/simulators/finite-response-law/preparation-screen-native-pair-hdf5'
PREPARATION_POLICY_METADATA = {
    **METADATA,
    "empirical_lawhood_payload_schema": PREPARATION_POLICY_NATIVE_PAIR_SCHEMA,
    "empirical_lawhood_clocks": '{"ticks":"native-step/refinement;one-reference-tick=0.001-native-time","transfer":"trailing-32-reference-tick-operator","handoff":4496,"future-terminal":4688}',
}


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyNativePairArtifact(FiniteResponseLawNativePairArtifact):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-policy-native-pair-artifact'
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawPreparationPolicyNativeDelivery]] = FiniteResponseLawPreparationPolicyNativeDelivery  # type: ignore[assignment]
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawPreparationPolicyNativeCheckpoint]] = FiniteResponseLawPreparationPolicyNativeCheckpoint
    METADATA: ClassVar[dict[str, str]] = PREPARATION_POLICY_METADATA
    deliveries: tuple[FiniteResponseLawPreparationPolicyNativeDelivery, ...]  # type: ignore[assignment]


@lru_cache(maxsize=1)
def unentered_preparation_policy_native_bytes() -> bytes:
    stream = io.BytesIO()
    with response_hdf5_writer(stream) as artifact:
        for name, value in sorted(PREPARATION_POLICY_METADATA.items()):
            response_hdf5_text(artifact, name, value)
        response_hdf5_text(artifact, "native_phase_disposition", "NOT_ENTERED")
    return stream.getvalue()
