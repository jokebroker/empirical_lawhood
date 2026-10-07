"""Closed history protocol, numeric inputs and exposed completion records.

The four families, independent histories and nested numerical views belong to
this scientific assay. None of these records grants qualification or authority.
"""

from dataclasses import dataclass, field, fields
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.matrix_inputs import MatrixArrayMember, MAXIMUM_MATRIX_ARRAY_BYTES
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.numerical_provenance import NumericalProducingProvenance
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactWriteResult

HISTORY_PURPOSES = (
    "autonomous-coarse-driver", "autonomous-fine-bridge",
    "preparation-coarse-driver", "preparation-fine-bridge",
)
HISTORY_FAMILIES = (
    "matrix-history.joint-increasing-coupling",
    "matrix-history.joint-decreasing-coupling",
    "matrix-history.x-first-increasing-coupling",
    "matrix-history.y-first-increasing-coupling",
)
EXPOSED = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
PUBLIC_HISTORY_MASTER_SEED = 706256
HISTORY_TRANSPORT_SCHEMA = "empirical-lawhood/source/checksummed-uninterpreted-public-bytes"


@dataclass(frozen=True, slots=True)
class MatrixHistoryProtocol(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/protocol"
    history_count: int = 256
    histories_per_family: int = 64
    ramp_ticks: int = 256
    total_ticks: int = 1024
    fine_multiplier: int = 2
    primary_dt: Decimal = Decimal("0.001")
    target_x: Decimal = Decimal("0.6666666666666666666666666667")
    target_y: Decimal = Decimal("7.333333333333333333333333334")
    decreasing_start: Decimal = Decimal(8)
    algebra_stride: int = 16
    dense_family_index: int = 1
    dense_family_slot: int = 18
    cohort_origins: tuple[int, ...] = (256, 512, 768, 864)
    dense_origins: tuple[int, ...] = (256, 512, 768, 832, 864, 880, 896, 912, 928, 944)
    dense_horizons: tuple[int, ...] = (16, 32, 64, 128)
    dense_kappas: tuple[Decimal, ...] = (Decimal("0.25"), Decimal("0.5"), Decimal(1))
    reference_math_sha256: str = "a23d92fb4866e451ad62959469ffcef773ee337d757f594e8cdde8be29bab563"
    historical_source_closure_verified: bool = False
    original_selected_excursion_reproduced: bool = False

    def __post_init__(self) -> None:
        if any(type(getattr(self, f.name)) is not type(f.default) or getattr(self, f.name) != f.default for f in fields(self)):
            raise ValueError("History protocol changes its owned clocks, sample roles or mathematical census")


@dataclass(frozen=True, slots=True)
class MatrixHistorySeed(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/seed"
    purpose: str
    digest_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.digest_sha256)
        if self.purpose not in HISTORY_PURPOSES:
            raise ValueError("History purpose is outside the complete driver/bridge menu")

    @property
    def effective_seed(self) -> int:
        return int(self.digest_sha256[:32], 16)


@dataclass(frozen=True, slots=True)
class MatrixHistoryRootAllocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/root-allocation"
    history_id: str
    history_index: int
    seeds: tuple[MatrixHistorySeed, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.history_id)
        if type(self.history_index) is not int or not 0 <= self.history_index < 256 or tuple(s.purpose for s in self.seeds) != HISTORY_PURPOSES:
            raise ValueError("History root loses its scientific slot or complete ordered stream census")
        if len({s.effective_seed for s in self.seeds}) != 4:
            raise ValueError("History purposes collide in consumed PCG64DXSM bits")

    @property
    def family_id(self) -> str:
        return HISTORY_FAMILIES[self.history_index // 64]

    @property
    def dense_role(self) -> bool:
        return self.history_index == 82

    def seed_for(self, purpose: str) -> int:
        return next(s.effective_seed for s in self.seeds if s.purpose == purpose)


@dataclass(frozen=True, slots=True)
class MatrixHistoryAllocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/allocation"
    allocation_id: str
    roots: tuple[MatrixHistoryRootAllocation, ...]
    exposure: str = EXPOSED
    prior_effective_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.allocation_id)
        values = tuple(s.effective_seed for root in self.roots for s in root.seeds)
        if tuple(root.history_index for root in self.roots) != tuple(range(256)) or len({r.history_id for r in self.roots}) != 256 or len(set(values)) != len(values):
            raise ValueError("History allocation needs four ordered families of64 independent roots and disjoint effective streams")
        if self.exposure not in (EXPOSED, "PROPOSED_UNRUN") or self.prior_effective_seeds != tuple(sorted(set(self.prior_effective_seeds))) or any(type(s) is not int or not 0 <= s < 2**128 for s in self.prior_effective_seeds):
            raise ValueError("History exposure declaration differs")
        if self.exposure == "PROPOSED_UNRUN" and set(values).intersection(self.prior_effective_seeds):
            raise ValueError("Declared exposed effective streams cannot be proposed as new histories")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.allocation_id, self)


@dataclass(frozen=True, slots=True)
class MatrixHistorySourceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/source-config"
    config_id: str
    allocation: ObjectIdentity
    protocol: MatrixHistoryProtocol
    code_sources_sha256: str
    dependency_lock_sha256: str
    evidence_role: str = EXPOSED

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        validate_sha256(self.code_sources_sha256)
        validate_sha256(self.dependency_lock_sha256)
        if self.allocation.object_schema != MatrixHistoryAllocation.SCHEMA or self.evidence_role != EXPOSED:
            raise ValueError("History source changes its allocation or descriptive evidence ceiling")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.config_id, self)


@dataclass(frozen=True, slots=True)
class MatrixHistoryAnalysisConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/analysis-config"
    config_id: str
    mode: str
    source: MatrixHistorySourceConfig
    compare_commutant_altered: bool = True
    evidence_role: str = EXPOSED

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        if self.mode not in ("ALGEBRA", "PASSIVE") or type(self.compare_commutant_altered) is not bool or self.evidence_role != EXPOSED:
            raise ValueError("History analysis selector is outside the descriptive closed assay")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.config_id, self)


@dataclass(frozen=True, slots=True)
class MatrixHistoryArrayInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/array-input"
    operand_id: str
    source: ObjectIdentity
    root: MatrixHistoryRootAllocation
    kind: str
    array_artifact: ArtifactIdentity
    members: tuple[MatrixArrayMember, ...]
    evidence_role: str = EXPOSED

    def __post_init__(self) -> None:
        validate_stable_id(self.operand_id)
        if self.source.object_schema not in (MatrixHistorySourceConfig.SCHEMA, MatrixHistoryAnalysisConfig.SCHEMA) or self.kind not in ("NATIVE_HISTORY", "ALGEBRA", "PASSIVE") or self.array_artifact.payload_schema != HISTORY_TRANSPORT_SCHEMA or self.array_artifact.media_type != "application/zip" or not 0 < self.array_artifact.size_bytes <= MAXIMUM_MATRIX_ARRAY_BYTES or not self.members or tuple(m.name for m in self.members) != tuple(sorted({m.name for m in self.members})) or sum(m.size_bytes for m in self.members) > MAXIMUM_MATRIX_ARRAY_BYTES or self.evidence_role != EXPOSED:
            raise ValueError("History arrays change their typed source, complete members or expansion bound")


@dataclass(frozen=True, slots=True)
class MatrixHistoryEnvironment(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/environment"
    VERSION: ClassVar[str] = "2.0.0"
    provenance: NumericalProducingProvenance = field(kw_only=True)
    python_version: str
    numpy_version: str
    scipy_version: str
    dependency_lock_sha256: str
    code_sources_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.dependency_lock_sha256)
        validate_sha256(self.code_sources_sha256)
        import json
        observed = json.loads(self.provenance.runtime_observation_json)
        if (tuple(observed[key] for key in ("python", "numpy", "scipy")) != (
                self.python_version, self.numpy_version, self.scipy_version)
            or self.provenance.dependency_lock_sha256 != self.dependency_lock_sha256
            or self.provenance.code_sources_sha256 != self.code_sources_sha256):
            raise ValueError("History environment differs from its exact producing provenance")
        if not all((self.python_version, self.numpy_version, self.scipy_version)):
            raise ValueError("History environment omits its actual executing versions")


@dataclass(frozen=True, slots=True)
class MatrixHistorySourceReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/source-receipt"
    VERSION: ClassVar[str] = "2.0.0"
    receipt_id: str
    source: MatrixHistorySourceConfig
    root: MatrixHistoryRootAllocation
    arrays: MatrixHistoryArrayInput
    arrays_publication: ArtifactWriteResult
    operand_publication: ArtifactWriteResult
    source_publication: ArtifactWriteResult
    allocation_publication: ArtifactWriteResult
    environment: MatrixHistoryEnvironment
    disposition: str
    completed_primary_steps: int
    completed_fine_steps: int
    reason: str | None = None
    acquisition_origin: str = "CURRENT_NATIVE_PRODUCER"

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id)
        if self.source_publication.logical.content_sha256 != self.source.fingerprint() or self.allocation_publication.logical.content_sha256 != self.source.allocation.object_fingerprint:
            raise ValueError("History source receipt omits actual frozen source/allocation bytes")
        if self.arrays.source != self.source.identity or self.arrays.root != self.root or self.arrays.kind != "NATIVE_HISTORY" or self.arrays_publication.materialization.physical_sha256 != self.arrays.array_artifact.sha256 or self.operand_publication.logical.payload_schema != MatrixHistoryArrayInput.SCHEMA or self.environment.code_sources_sha256 != self.source.code_sources_sha256 or self.environment.dependency_lock_sha256 != self.source.dependency_lock_sha256 or self.acquisition_origin not in ("CURRENT_NATIVE_PRODUCER", "SUPPLIED_EXPOSED_ARRAYS") or self.disposition not in ("COMPLETE", "NUMERICAL_INVALID") or not 0 <= self.completed_primary_steps <= 1024 or not 0 <= self.completed_fine_steps <= 2048 or (self.disposition == "COMPLETE") != (self.completed_primary_steps == 1024 and self.completed_fine_steps == 2048 and self.reason is None):
            raise ValueError("History receipt loses source/array/census/environment completion joins")


@dataclass(frozen=True, slots=True)
class MatrixHistoryAnalysisRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/root-result"
    root_index: int
    history_id: str
    disposition: str
    reason: str | None
    source_receipt: ObjectIdentity | None
    arrays: MatrixHistoryArrayInput | None
    arrays_publication: ArtifactWriteResult | None
    operand_publication: ArtifactWriteResult | None
    requested_samples: int
    retained_samples: int
    identifiable_y_samples: int
    metrics: tuple[tuple[str, Decimal | None], ...] = ()
    source_receipt_path: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.history_id)
        if type(self.root_index) is not int or not 0 <= self.root_index < 256 or self.disposition not in ("COMPLETE", "UNEVALUABLE", "UNENTERED") or not 0 <= self.retained_samples <= self.requested_samples or not 0 <= self.identifiable_y_samples <= self.retained_samples:
            raise ValueError("History result drops its root, missingness or nested sample census")
        if (self.disposition == "COMPLETE") != (self.reason is None) or (self.arrays is None) != (self.arrays_publication is None) or (self.arrays is None) != (self.operand_publication is None):
            raise ValueError("History result completion and actual output publications differ")
        if self.disposition == "COMPLETE" and (self.arrays is None or self.source_receipt is None or self.retained_samples != self.requested_samples):
            raise ValueError("A complete history result requires all actual requested output/source publications")
        if (self.source_receipt is None) != (self.source_receipt_path is None):
            raise ValueError("History result omits its actual source receipt locator")
        if self.source_receipt_path is not None:
            from empirical_lawhood.kernel.serialization import validate_relative_locator
            validate_relative_locator(self.source_receipt_path)


@dataclass(frozen=True, slots=True)
class MatrixHistoryAnalysisReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-history-analysis/analysis-receipt"
    VERSION: ClassVar[str] = "2.0.0"
    receipt_id: str
    config: MatrixHistoryAnalysisConfig
    allocation: ObjectIdentity
    environment: MatrixHistoryEnvironment
    roots: tuple[MatrixHistoryAnalysisRoot, ...]
    family_medians: tuple[tuple[str, tuple[tuple[str, Decimal | None], ...]], ...]
    column_contract_publication: ArtifactWriteResult
    config_publication: ArtifactWriteResult
    allocation_publication: ArtifactWriteResult
    source_receipt_locators: tuple[str, ...]
    disposition: str
    evidence_role: str = EXPOSED

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id)
        if self.config_publication.logical.content_sha256 != self.config.fingerprint() or self.allocation_publication.logical.content_sha256 != self.allocation.object_fingerprint:
            raise ValueError("History analysis receipt omits actual input/allocation publications")
        from empirical_lawhood.kernel.serialization import validate_relative_locator
        if len(self.source_receipt_locators) > 256 or len(set(self.source_receipt_locators)) != len(self.source_receipt_locators):
            raise ValueError("History result changes its explicit input locator census")
        for path in self.source_receipt_locators:
            validate_relative_locator(path)
        if tuple(r.root_index for r in self.roots) != tuple(range(256)) or len({r.history_id for r in self.roots}) != 256 or self.allocation != self.config.source.allocation or self.environment.code_sources_sha256 != self.config.source.code_sources_sha256 or self.environment.dependency_lock_sha256 != self.config.source.dependency_lock_sha256 or self.disposition != ("COMPLETE" if all(r.disposition == "COMPLETE" for r in self.roots) else "UNEVALUABLE") or self.evidence_role != EXPOSED or tuple(f for f, _ in self.family_medians) != HISTORY_FAMILIES:
            raise ValueError("History analysis completion drops roots or promotes descriptive evidence")
