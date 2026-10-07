"""Current constructed Q/E inputs and independently allocated physical roots."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.preparation_applicability.exposure import PreparationApplicabilityExposure
from empirical_lawhood.adapters.methods.preparation_applicability.config import PreparationApplicabilityUpstream, ORIGINAL_F_SCHEMA
from empirical_lawhood.kernel.matrix_inputs import MatrixAllocation
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from .records import ConstructedPreparationDesign

FIXED_PROBE_SEED = int('138298f06df15e5a8b7b6e15eab582a3067127dd015ce7c0bb145b321200ee21', 16)


@dataclass(frozen=True, slots=True)
class ConstructedPreparationStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/constructed-preparation-applicability/stage'
    config_id: str
    design: ConstructedPreparationDesign
    phase: str
    allocation: MatrixAllocation
    upstream: tuple[PreparationApplicabilityUpstream, ...]
    exposure: PreparationApplicabilityExposure
    source: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        if (
            self.phase not in ('Q', 'E')
            or len(self.allocation.roots) != (8 if self.phase == 'Q' else 32)
            or self.allocation.bootstrap_seed is None
            or self.allocation.exposure != 'PROPOSED_UNRUN'
            or any(root.cohort != 'constructed' or root.source_prefix is not None
                   or root.seed_for('passive-probes') != FIXED_PROBE_SEED
                   for root in self.allocation.roots)
            or tuple(value.key for value in self.upstream) != (('lower',) if self.phase == 'Q' else ('lower', 'qualification'))
            or self.upstream[0].artifact.payload_schema != ORIGINAL_F_SCHEMA
            or self.phase == 'E' and self.upstream[1].artifact.payload_schema != 'empirical-lawhood/constructed-preparation-applicability/report'
        ):
            raise ValueError('constructed phase changes its assigned source/prerequisite census')
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
class ConstructedPreparationNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/constructed-preparation-applicability/native-config'
    stage: ConstructedPreparationStage

    @property
    def config_id(self) -> str:
        return f'{self.stage.config_id}.native'


@dataclass(frozen=True, slots=True)
class ConstructedPreparationSource(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/constructed-preparation-applicability/source'
    implementation_plan_sha256: str
    dependency_lock_sha256: str
    native_implementation: ObjectIdentity
    conformance_receipt: ObjectIdentity
    design: ConstructedPreparationDesign = ConstructedPreparationDesign()
    python_version: str = '3.11.14'
    numpy_version: str = '2.4.6'
    native_recipe: str = 'fresh-ideal-q2-prescribed-r1-d016-bath-plus-.002-independent-noise-through-4496-ordinary-futures'

    def __post_init__(self) -> None:
        validate_sha256(self.implementation_plan_sha256)
        validate_sha256(self.dependency_lock_sha256)
        if self.python_version != '3.11.14' or self.numpy_version != '2.4.6' or self.native_recipe != type(self).__dataclass_fields__['native_recipe'].default:
            raise ValueError('constructed source changes its frozen numerical recipe')
