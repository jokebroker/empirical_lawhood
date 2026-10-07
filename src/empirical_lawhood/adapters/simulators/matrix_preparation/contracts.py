"""Strict native declarations; retained source records require verified exports."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)


from .scientific_inputs import preparation_scientific_root_inputs
from .retained_exports import PreparationRetainedSourceExport
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import ResponseGeometryDevelopmentNativeRoot


CAMPAIGN = "matrix-preparation-adequacy"
DEVELOPMENT = f"{CAMPAIGN}.development"
PREFIX_EXPORT_RUN = f"{CAMPAIGN}.retained-prefix-exports"
NATIVE_SCHEMA = 'empirical-lawhood/simulators/matrix-preparation/native-observations-hdf5'
CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
CONTEXTS = ("assembling", "prepared")
PARENTS = ("hold", "x-negative", "x-positive", "y-negative", "y-positive")
RESPONSE_BUNDLES = ("common-response", "independent-response-1", "independent-response-2", "independent-response-3")
READOUTS = (64, 128, 192, 256, 320)
SEED_SHA256 = '64055b9c820c2015b6ed837642725b2b59425f9d80b2624cccb7dd9649b895ed'
PREFIX_BUNDLE_SCHEMA = 'empirical-lawhood/simulators/matrix-preparation/preparation-prefix-bundle'
PREFIX_BUNDLE_MAXIMUM_BYTES = 3 * 1024**2


@dataclass(frozen=True, slots=True)
class PreparationRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-root'
    context: str
    index: int

    def __post_init__(self) -> None:
        if self.context not in CONTEXTS or type(self.index) is not int or not 0 <= self.index < 64:
            raise ValueError("preparation development retains exactly the 128 exposed stochastic roots")

    @property
    def root_id(self) -> str:
        return f"{DEVELOPMENT}.{self.context}.r{self.index:02d}"

    @property
    def physical_unit_id(self) -> str:
        return f"{CAMPAIGN}.original-stochastic-root.{self.context}.r{self.index:02d}"

    @property
    def landmark(self) -> int:
        return 1024 if self.context == "assembling" else 4096

    @property
    def handoff(self) -> int:
        return self.landmark + 144

    @property
    def numerical_semantics(self) -> bool:
        return preparation_scientific_root_inputs(self.context, self.index).numerical_semantics_rank < 16

    @property
    def development_role(self) -> str:
        rank = preparation_scientific_root_inputs(self.context, self.index).development_rank
        return "fit" if rank < 32 else "interval" if rank < 48 else "screen"


def preparation_roots() -> tuple[PreparationRoot, ...]:
    return tuple(PreparationRoot(context, i) for context in CONTEXTS for i in range(64))


@dataclass(frozen=True, slots=True)
class RetainedPreparationPrefix(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/retained-preparation-prefix'
    root: PreparationRoot
    source_config: ObjectIdentity
    artifact: ArtifactIdentity
    relative_path: str
    manifest_sha256: str
    publication_commit_sha256: str
    task_receipt: ObjectIdentity

    source_export: PreparationRetainedSourceExport | None = None

    def __post_init__(self) -> None:
        if type(self.source_export) is not PreparationRetainedSourceExport:
            raise ValueError("preparation prefix requires a separately verified original-source export and current target custody before native work")
        self.source_export.validate_target(
            self.artifact, self.relative_path, self.task_receipt,
            self.manifest_sha256, self.publication_commit_sha256,
            (self.root.physical_unit_id,),
        )
        task = f"{self.root.root_id}.prefix-source"
        validate_relative_locator(self.relative_path)
        for name in ("manifest_sha256", "publication_commit_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            self.source_config.object_schema != 'empirical-lawhood/simulators/six-matrix-response/response-geometry-development-native-config'
            or self.source_config.object_version != "1.0.0"
            or self.artifact.artifact_id != f"artifact.{PREFIX_EXPORT_RUN}.{task}.native-result"
            or self.artifact.payload_schema != 'empirical-lawhood/simulators/six-matrix-response/response-geometry-development-native-segment-result'
            or self.artifact.media_type != CANONICAL_MEDIA_TYPE
            or not 0 < self.artifact.size_bytes <= 512 * 1024
            or self.relative_path != f"runs/{PREFIX_EXPORT_RUN}/outputs/{task}/native-result.canonical.json"
            or self.task_receipt.object_id != f"receipt.{PREFIX_EXPORT_RUN}.{task}.attempt-001"
        ):
            raise ValueError("preparation prefix export changes its current root/custody identity")

    @property
    def source_root_id(self) -> str:
        return ResponseGeometryDevelopmentNativeRoot(self.root.context, self.root.landmark, self.root.index).root_id

    @property
    def root_id(self) -> str:
        return self.root.root_id

    @property
    def imported_artifact_id(self) -> str:
        return f"retained.{self.root.root_id}.prefix"


@dataclass(frozen=True, slots=True)
class PreparationPrefixBundleRef(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-prefix-bundle-ref'
    context: str
    group_index: int
    artifact: ArtifactIdentity

    def __post_init__(self) -> None:
        if (
            self.context not in CONTEXTS
            or type(self.group_index) is not int
            or not 0 <= self.group_index < 16
            or self.artifact.artifact_id != f"artifact.{self.bundle_id}"
            or self.artifact.payload_schema != PREFIX_BUNDLE_SCHEMA
            or self.artifact.media_type != CANONICAL_MEDIA_TYPE
            or not 0 < self.artifact.size_bytes <= PREFIX_BUNDLE_MAXIMUM_BYTES
        ):
            raise ValueError("retained prefix bundle changes its exact four-root materialization")

    @property
    def bundle_id(self) -> str:
        return f"{DEVELOPMENT}.prefix-bundle.{self.context}.g{self.group_index:02d}"

    @property
    def imported_artifact_id(self) -> str:
        return f"retained.{self.bundle_id}"

    @property
    def roots(self) -> tuple[PreparationRoot, ...]:
        return tuple(PreparationRoot(self.context, self.group_index * 4 + i) for i in range(4))


@dataclass(frozen=True, slots=True)
class PreparationSourceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-source-config'
    config_id: str
    implementation_plan_sha256: str
    imported_inventory: ObjectIdentity
    prefix_source_config: ObjectIdentity
    prefixes: tuple[RetainedPreparationPrefix, ...]
    prefix_bundles: tuple[PreparationPrefixBundleRef, ...]
    design_sha256: str
    seed_sha256: str = SEED_SHA256
    parent_end_tick: int = 128
    post_parent_delay_ticks: int = 16
    pulse_ticks: int = 64
    readouts: tuple[int, ...] = READOUTS
    refinements: tuple[int, ...] = (1, 2)
    response_bundles: tuple[str, ...] = RESPONSE_BUNDLES
    mode_policy: str = "PRIMARY_PREPARENT_MODE_FIXED_ACROSS_PARENTS_AND_VIEWS"
    inference_scope: str = "EXPOSED_DEVELOPMENT_ONLY"
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.implementation_plan_sha256, field_name="implementation_plan_sha256")
        validate_sha256(self.design_sha256, field_name="design_sha256")
        require_sorted_unique_ids(self.prefixes, attribute="root_id", field_name="prefixes")
        if (
            self.config_id != f"{DEVELOPMENT}.source-config"
            or tuple(p.root for p in self.prefixes) != preparation_roots()
            or tuple((b.context, b.group_index) for b in self.prefix_bundles)
            != tuple((c, i) for c in CONTEXTS for i in range(16))
            or self.imported_inventory.object_schema
            != 'empirical-lawhood/simulators/matrix-preparation/preparation-source-import-inventory'
            or self.prefix_source_config.object_schema
            != 'empirical-lawhood/simulators/six-matrix-response/response-geometry-development-native-config'
            or any(row.source_config != self.prefix_source_config for row in self.prefixes)
            or self.seed_sha256 != SEED_SHA256
            or type(self.parent_end_tick) is not int
            or self.parent_end_tick != 128
            or type(self.post_parent_delay_ticks) is not int
            or self.post_parent_delay_ticks != 16
            or type(self.pulse_ticks) is not int
            or self.pulse_ticks != 64
            or self.readouts != READOUTS
            or any(type(v) is not int for v in self.readouts)
            or self.refinements != (1, 2)
            or any(type(v) is not int for v in self.refinements)
            or self.response_bundles != RESPONSE_BUNDLES
            or self.mode_policy != "PRIMARY_PREPARENT_MODE_FIXED_ACROSS_PARENTS_AND_VIEWS"
            or self.inference_scope != "EXPOSED_DEVELOPMENT_ONLY"
            or self.grants_authority is not False
        ):
            raise ValueError("preparation source changes the frozen finite development design")

    @property
    def roots(self) -> tuple[PreparationRoot, ...]:
        return tuple(p.root for p in self.prefixes)

    @property
    def maximum_native_updates(self) -> int:
        return sum(native_updates_for_root(root) for root in self.roots)


@dataclass(frozen=True, slots=True)
class PreparationAction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-action'
    root: PreparationRoot
    parent: str
    bundle: str
    sign: int
    magnitude: Decimal

    def __post_init__(self) -> None:
        if (
            self.parent not in PARENTS
            or self.bundle not in (*RESPONSE_BUNDLES, "prospective-task")
            or type(self.sign) is not int
            or self.sign not in (-1, 0, 1)
            or self.magnitude not in (Decimal("0.5"), Decimal(1), Decimal(2), Decimal(8))
            or (
                self.magnitude != 8
                and (not self.root.numerical_semantics or self.bundle != "common-response" or self.sign == 0)
            )
        ):
            raise ValueError("preparation occurrence is outside its issued finite chart")

    @property
    def action_id(self) -> str:
        sign = {-1: "neg", 0: "hold", 1: "pos"}[self.sign]
        magnitude = {Decimal("0.5"): "a05", Decimal(1): "a1", Decimal(2): "a2", Decimal(8): "a8"}[
            self.magnitude
        ]
        return f"{self.root.root_id}.{self.parent}.{self.bundle}.{magnitude}.{sign}"

    @property
    def retain_native_path(self) -> bool:
        return self.root.numerical_semantics and self.bundle == "common-response" and self.magnitude == 8


def preparation_actions(root: PreparationRoot) -> tuple[PreparationAction, ...]:
    actions = [
        PreparationAction(root, parent, bundle, sign, Decimal(8))
        for parent in PARENTS
        for bundle in (*RESPONSE_BUNDLES, "prospective-task")
        for sign in (-1, 0, 1)
    ]
    if root.numerical_semantics:
        actions.extend(
            PreparationAction(root, parent, "common-response", sign, amplitude)
            for parent in PARENTS
            for amplitude in (Decimal("0.5"), Decimal(1), Decimal(2))
            for sign in (-1, 1)
        )
    return tuple(sorted(actions, key=lambda a: (
        PARENTS.index(a.parent),
        (*RESPONSE_BUNDLES, "prospective-task").index(a.bundle),
        (Decimal("0.5"), Decimal(1), Decimal(2), Decimal(8)).index(a.magnitude),
        (0, -1, 1).index(a.sign),
    )))


def native_updates_for_root(root: PreparationRoot) -> int:
    # Imported prefix is not reacquired. Half-step work is two native updates.
    return 3 * (len(PARENTS) * 144 + len(preparation_actions(root)) * 320)


@dataclass(frozen=True, slots=True)
class PreparationDelivery(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-delivery'
    occurrence_id: str
    refinement: int
    requested_start_tick: int
    requested_end_tick: int
    accepted: bool
    completed_intervals: int
    force_evaluations: int
    nonzero_intervals: int
    signed_impulse: Decimal
    force_work: Decimal
    absolute_parent_density_work: Decimal
    interval_trace_sha256: str
    disposition: str
    reason: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.occurrence_id, field_name="occurrence_id")
        validate_sha256(self.interval_trace_sha256, field_name="interval_trace_sha256")
        if type(self.refinement) is not int or self.refinement not in (1, 2):
            raise ValueError("preparation delivery has another numerical view")
        if any(
            type(v) is not int or v < 0
            for v in (
                self.requested_start_tick,
                self.requested_end_tick,
                self.completed_intervals,
                self.force_evaluations,
                self.nonzero_intervals,
            )
        ):
            raise ValueError("preparation delivery clocks/counts must be nonnegative integers")
        expected = (self.requested_end_tick - self.requested_start_tick) * self.refinement
        if (
            expected <= 0
            or not 0 <= self.nonzero_intervals <= self.completed_intervals <= expected
            or self.force_evaluations != 2 * self.completed_intervals
            or type(self.accepted) is not bool
            or (not self.accepted and self.completed_intervals)
            or self.disposition
            not in ("COMPLETE", "UNENTERED", "NUMERICAL_FAILURE", "OBSERVATION_FAILURE")
            or (self.disposition == "COMPLETE") != (self.reason is None)
            or self.accepted != (self.disposition != "UNENTERED")
            or (self.disposition == "COMPLETE" and self.completed_intervals != expected)
            or any(
                not v.is_finite()
                for v in (self.signed_impulse, self.force_work, self.absolute_parent_density_work)
            )
            or self.absolute_parent_density_work < 0
            or (self.nonzero_intervals == 0 and self.signed_impulse != 0)
        ):
            raise ValueError("preparation delivery disagrees with completed native intervals")


@dataclass(frozen=True, slots=True)
class PreparationNativeResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-native-result'
    result_id: str
    source_config: ObjectIdentity
    root: PreparationRoot
    imported_prefix: ObjectIdentity
    mode_sha256: str | None
    deliveries: tuple[PreparationDelivery, ...]
    observations_sha256: str
    completed_native_updates: int

    def __post_init__(self) -> None:
        validate_sha256(self.observations_sha256, field_name="observations_sha256")
        if self.mode_sha256 is not None:
            validate_sha256(self.mode_sha256, field_name="mode_sha256")
        expected_ids = {
            f"{self.root.root_id}.{parent}.parent.r{r}" for parent in PARENTS for r in (1, 2)
        } | {f"{a.action_id}.r{r}" for a in preparation_actions(self.root) for r in (1, 2)}
        require_sorted_unique_ids(
            self.deliveries, attribute="occurrence_id", field_name="deliveries"
        )
        if (
            self.result_id != f"result.{self.root.root_id}.native"
            or self.source_config.object_schema != PreparationSourceConfig.SCHEMA
            or self.imported_prefix.object_schema != RetainedPreparationPrefix.SCHEMA
            or {d.occurrence_id for d in self.deliveries} != expected_ids
            or type(self.completed_native_updates) is not int
            or self.completed_native_updates != sum(d.completed_intervals for d in self.deliveries)
            or self.completed_native_updates > native_updates_for_root(self.root)
        ):
            raise ValueError(
                "preparation native result changes its complete root/branch denominator"
            )
