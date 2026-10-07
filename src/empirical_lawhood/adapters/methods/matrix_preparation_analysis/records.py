"""Static declarations for the fixed preparation analyses.

Paths and byte custody are supplied to the public API separately. These records
never discover a source, allocate an RNG, fit a model or grant authority.
"""

from dataclasses import dataclass, field, fields
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.numerical_provenance import NumericalProducingProvenance
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import TransientBridgeSpec


@dataclass(frozen=True, slots=True)
class PreparationAnalysisSpecification(CanonicalRecord):
    """Operative current baseline, independent of frozen historical documents."""
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/specification"
    specification_id: str = "matrix-preparation-analysis.current-baseline"
    contexts: tuple[tuple[str, int, int], ...] = (("assembling", 1024, 64), ("prepared", 4096, 64))
    numerical_roots_per_context: int = 16
    development_roles_per_context: tuple[int, ...] = (32, 16, 16)
    tangent_parent_ticks: int = 144
    tangent_pulse_ticks: int = 64
    tangent_readouts: tuple[int, ...] = (64, 128, 192, 256, 320)
    tangent_amplitudes: tuple[Decimal, ...] = (Decimal("0.5"), Decimal(1), Decimal(2), Decimal(8))
    absolute_view_tolerance: Decimal = Decimal(1) / Decimal(128)
    odd_view_tolerance: Decimal = Decimal(1) / Decimal(256)
    secant_absolute_tolerance: Decimal = Decimal("0.00000001")
    secant_relative_tolerance: Decimal = Decimal("0.00001")
    transient: TransientBridgeSpec = TransientBridgeSpec()
    outer_folds: int = 4
    inner_cohort_folds: int = 3
    support_feature_count: int = 24
    support_bound_inclusive: Decimal = Decimal(6)
    qualification: str = "NONE"

    def __post_init__(self):
        if any(getattr(self, item.name) != item.default for item in fields(self)):
            raise ValueError("Preparation analysis changes its closed current scientific baseline")

    @property
    def identity(self):
        return ObjectIdentity.from_record(self.specification_id, self)


SPECIFICATION = PreparationAnalysisSpecification()


@dataclass(frozen=True, slots=True)
class TransientAnalysisConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/transient-analysis-config"
    analysis_id: str
    summary_population: str = "all"
    specification: ObjectIdentity = field(default=SPECIFICATION.identity, kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        if self.summary_population not in ("all", "q2", "cir1") or self.specification != SPECIFICATION.identity:
            raise ValueError("Transient summary population must select all, q2 or cir1; all roots remain retained")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.analysis_id, self)


@dataclass(frozen=True, slots=True)
class BaselineSupportConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/baseline-support-config"
    analysis_id: str
    model: str = "radial.all"
    specification: ObjectIdentity = field(default=SPECIFICATION.identity, kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        if self.model not in ("radial.all", "nonrestoring.all") or self.specification != SPECIFICATION.identity:
            raise ValueError("Baseline decomposition selects one of the two fixed all-regime forecasts")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.analysis_id, self)


@dataclass(frozen=True, slots=True)
class PreparationAnalysisMetric(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/metric"
    name: str
    value: Decimal | None

    def __post_init__(self) -> None:
        validate_stable_id(self.name, field_name="name")
        if self.value is not None and not self.value.is_finite():
            raise ValueError("Undefined diagnostics use None, never a nonfinite Decimal")


@dataclass(frozen=True, slots=True)
class PreparationAnalysisReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/report"
    VERSION: ClassVar[str] = "2.0.0"
    report_id: str
    configuration: ObjectIdentity
    inputs: tuple[ArtifactIdentity, ...]
    arrays: ArtifactIdentity
    root_ids: tuple[str, ...]
    incomplete_roots: tuple[str, ...]
    metrics: tuple[PreparationAnalysisMetric, ...]
    disposition: str
    qualification: str = "NONE"
    visibility: str = "OUTCOME_VISIBLE_EXPLORATION"
    independent_unit: str = "whole-stochastic-root"
    implementation_sources_sha256: str = field(kw_only=True)
    provenance: NumericalProducingProvenance = field(kw_only=True)
    specification: ObjectIdentity = field(default=SPECIFICATION.identity, kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.implementation_sources_sha256, field_name="implementation_sources_sha256")
        if self.implementation_sources_sha256 != self.provenance.code_sources_sha256:
            raise ValueError("Preparation report changes its actual producing source provenance")
        if (not self.root_ids or len(set(self.root_ids)) != len(self.root_ids)
                or not set(self.incomplete_roots).issubset(self.root_ids)
                or tuple(sorted(set(self.incomplete_roots))) != self.incomplete_roots
                or self.disposition not in ("COMPLETE", "UNEVALUABLE")
                or (self.disposition == "COMPLETE") != (not self.incomplete_roots)
                or self.qualification != "NONE"
                or self.visibility != "OUTCOME_VISIBLE_EXPLORATION"
                or self.independent_unit != "whole-stochastic-root"
                or self.specification != SPECIFICATION.identity
                or tuple(metric.name for metric in self.metrics) != tuple(sorted(set(metric.name for metric in self.metrics)))):
            raise ValueError("Preparation analysis changes its complete population or exploratory claim")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.report_id, self)


@dataclass(frozen=True, slots=True)
class PreparationAnalysisInputs(CanonicalRecord):
    """Complete validated input identities; never an issue or qualification grant."""
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/inputs"
    input_id: str
    configuration: ObjectIdentity
    analysis_kind: str
    artifacts: tuple[ArtifactIdentity, ...]
    roles: tuple[str, ...]
    upstream_members: tuple[ArtifactIdentity, ...] = ()

    def __post_init__(self):
        validate_stable_id(self.input_id, field_name="input_id")
        if (self.analysis_kind not in ("TANGENT_SOURCE", "TANGENT_ANALYSIS", "TRANSIENT_ANALYSIS", "BASELINE_SUPPORT")
                or not 0 < len(self.artifacts) <= 1024 or len(self.roles) != len(self.artifacts)
                or len(set(self.roles)) != len(self.roles) or len(self.upstream_members) > 16384):
            raise ValueError("Preparation input manifest changes its complete selected roles")
        for role in self.roles:
            validate_stable_id(role, field_name="input_role")
