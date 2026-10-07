"""Current ordinary-source cohorts, custody and actual numerical allocations."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.preparation_applicability.exposure import PreparationApplicabilityExposure
from empirical_lawhood.kernel.matrix_inputs import MatrixAllocation
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from .records import PreparationApplicabilityDesign

ORIGINAL_F_SCHEMA = 'empirical-lawhood/methods/finite-response-law/original-finite-response-law'


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityUpstream(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/preparation-applicability/upstream'
    key: str
    artifact: ArtifactIdentity
    custody_receipt: ObjectIdentity
    source_run_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.key)
        validate_stable_id(self.source_run_id)


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/preparation-applicability/stage'
    config_id: str
    design: PreparationApplicabilityDesign
    phase: str
    allocation: MatrixAllocation
    upstream: tuple[PreparationApplicabilityUpstream, ...]
    exposure: PreparationApplicabilityExposure
    source: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        if (
            self.phase not in ('D', 'E')
            or len(self.allocation.roots) != (32 if self.phase == 'D' else 64)
            or any(root.cohort != 'q2' or root.source_prefix is not None for root in self.allocation.roots)
            or tuple(value.key for value in self.upstream) != ('lower',)
            or self.upstream[0].artifact.payload_schema != ORIGINAL_F_SCHEMA
        ):
            raise ValueError('ordinary screen changes its complete source/input census')
        self.exposure.check(self.allocation)

    @property
    def root_ids(self) -> tuple[str, ...]:
        return tuple(root.root_id for root in self.allocation.roots)

    @property
    def cpu_seconds(self) -> int:
        return len(self.root_ids) * 3550

    @property
    def wall_seconds(self) -> int:
        return len(self.root_ids) * 1750


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/preparation-applicability/native-config'
    stage: PreparationApplicabilityStage

    @property
    def config_id(self) -> str:
        return f'{self.stage.config_id}.native'


@dataclass(frozen=True, slots=True)
class PreparationApplicabilitySource(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/preparation-applicability/source'
    implementation_plan_sha256: str
    dependency_lock_sha256: str
    native_implementation: ObjectIdentity
    conformance_receipt: ObjectIdentity
    design: PreparationApplicabilityDesign = PreparationApplicabilityDesign()
    python_version: str = '3.11.14'
    numpy_version: str = '2.4.6'
    native_recipe: str = 'q2-prepared-baoab-unit-mass-skr24-returned-y400-two-futures'

    def __post_init__(self) -> None:
        validate_sha256(self.implementation_plan_sha256)
        validate_sha256(self.dependency_lock_sha256)
        if (self.python_version, self.numpy_version, self.native_recipe) != (
            '3.11.14', '2.4.6', 'q2-prepared-baoab-unit-mass-skr24-returned-y400-two-futures'
        ):
            raise ValueError('ordinary source changes its numerical denominator')


def preparation_run_id(stage) -> str:
    """Keep scientific IDs while separating exact allocated input custody."""
    return f"{stage.config_id}.run-{stage.fingerprint()[:32]}"
