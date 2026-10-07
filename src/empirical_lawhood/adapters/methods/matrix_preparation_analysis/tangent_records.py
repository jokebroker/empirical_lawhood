"""Fresh numerical operands for the fixed 128-root tangent experiment.

Full 256-bit commitments remain provenance; PCG64DXSM consumes their leading
128 bits. Names never allocate a stream. The fixed protocol and historical
records retain separate identities.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import (
    PARENTS, PreparationDelivery, preparation_actions,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeViewSegment
from empirical_lawhood.kernel.matrix_inputs import MatrixArrayMember, MAXIMUM_MATRIX_ARRAY_BYTES, MATRIX_ARRAY_SCHEMA
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes, validate_sha256, validate_stable_id
from .records import SPECIFICATION


@dataclass(frozen=True, slots=True)
class TangentPreparationRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/tangent-root"
    namespace: str
    context: str
    index: int
    numerical_rank: int
    development_rank: int
    prefix_seed_sha256: tuple[str, ...]
    preparation_seed_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.namespace, field_name="namespace")
        if (self.context not in ("assembling", "prepared")
                or any(type(value) is not int or not 0 <= value < 64 for value in (self.index, self.numerical_rank, self.development_rank))
                or len(self.prefix_seed_sha256) != 5 or len(self.preparation_seed_sha256) != 12):
            raise ValueError("Tangent root changes contexts, pre-outcome ranks or complete numerical purposes")
        for seed in (*self.prefix_seed_sha256, *self.preparation_seed_sha256):
            validate_sha256(seed, field_name="scientific_seed_sha256")

    @property
    def root_id(self):
        return f"{self.namespace}.{self.context}.r{self.index:02d}"

    @property
    def physical_unit_id(self):
        return self.root_id

    @property
    def landmark(self):
        return 1024 if self.context == "assembling" else 4096

    @property
    def landmark_tick(self):
        return self.landmark

    @property
    def handoff(self):
        return self.landmark + 144

    @property
    def numerical_semantics(self):
        return self.numerical_rank < 16

    @property
    def development_role(self):
        return "fit" if self.development_rank < 32 else "interval" if self.development_rank < 48 else "screen"

    @property
    def covariance(self):
        return False

    @property
    def refinements(self):
        return (1, 2)

    @property
    def invocation_offset(self):
        return 144

    @property
    def horizon_ticks(self):
        return 320

    @property
    def probe_roster_seed_sha256(self):
        return self.prefix_seed_sha256[4]

    def native_seed(self, *, purpose_ordinal, counter):
        if purpose_ordinal not in range(4) or counter != 0:
            raise ValueError("Tangent prefix has no undeclared covariance or quarter view")
        return self.prefix_seed_sha256[purpose_ordinal]

    def preparation_seed(self, *, purpose_index, bridge):
        if purpose_index not in range(6) or type(bridge) is not bool:
            raise ValueError("Tangent preparation purpose differs")
        return self.preparation_seed_sha256[2 * purpose_index + int(bridge)]


@dataclass(frozen=True, slots=True)
class TangentPreparationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/tangent-preparation-config"
    VERSION: ClassVar[str] = "2.0.0"
    config_id: str
    roots: tuple[TangentPreparationRoot, ...]
    summary_context: str = "all"
    exposure: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
    code_sources_sha256: str = field(kw_only=True)
    dependency_lock_sha256: str = field(kw_only=True)
    specification: ObjectIdentity = field(default=SPECIFICATION.identity, kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.code_sources_sha256, field_name="code_sources_sha256")
        validate_sha256(self.dependency_lock_sha256, field_name="dependency_lock_sha256")
        expected = tuple((context, index) for context in ("assembling", "prepared") for index in range(64))
        if (tuple((root.context, root.index) for root in self.roots) != expected
                or len({root.root_id for root in self.roots}) != 128
                or self.summary_context not in ("all", "assembling", "prepared")
                or self.exposure != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
                or self.specification != SPECIFICATION.identity):
            raise ValueError("Tangent config requires its full ordered 128-root development population")
        for context in ("assembling", "prepared"):
            selected = tuple(root for root in self.roots if root.context == context)
            if sorted(root.numerical_rank for root in selected) != list(range(64)) or sorted(root.development_rank for root in selected) != list(range(64)):
                raise ValueError("Tangent pre-outcome ranks must be complete permutations per context")
        effective = [seed[:32] for root in self.roots for seed in (*root.prefix_seed_sha256, *root.preparation_seed_sha256)]
        if len(set(effective)) != len(effective):
            raise ValueError("Tangent independent PCG64DXSM effective numerical streams collide")

    @property
    def scientific_inputs(self):
        return self

    @property
    def seed_root_sha256(self):
        return sha256(canonical_json_bytes(tuple((root.prefix_seed_sha256, root.preparation_seed_sha256) for root in self.roots))).hexdigest()

    @property
    def seed_sha256(self):
        return self.seed_root_sha256

    def physical_unit_id(self, root):
        if root not in self.roots:
            raise ValueError("Tangent prefix root is outside the current allocation")
        return root.physical_unit_id

    @property
    def identity(self):
        return ObjectIdentity.from_record(self.config_id, self)


@dataclass(frozen=True, slots=True)
class TangentPrefixSegment(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/tangent-prefix-segment"
    root: TangentPreparationRoot
    phase: str = "prefix"
    parent: None = None

    def __post_init__(self):
        if self.phase != "prefix" or self.parent is not None:
            raise ValueError("Current tangent prefix is a complete pre-parent interval")

    @property
    def task_id(self):
        return self.root.root_id + ".prefix"

    @property
    def start_tick(self):
        return 0

    @property
    def end_tick(self):
        return self.root.landmark


@dataclass(frozen=True, slots=True)
class TangentPrefixResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/tangent-prefix-result"
    result_id: str
    configuration: ObjectIdentity
    root: TangentPreparationRoot
    views: tuple[ResponseGeometryAssayNativeViewSegment, ...]

    def __post_init__(self):
        if (self.result_id != self.root.root_id + ".prefix-result" or self.configuration.object_schema != TangentPreparationConfig.SCHEMA
                or tuple(view.refinement for view in self.views) != (1, 2)):
            raise ValueError("Current tangent prefix loses its root/config/view census")
        segment = TangentPrefixSegment(self.root)
        for view in self.views:
            end = self.root.landmark * view.refinement
            if (not 0 <= view.last_completed_native_step <= end
                    or view.delivery.completed_intervals != view.last_completed_native_step
                    or view.disposition == "COMPLETE" and view.last_completed_native_step != end):
                raise ValueError("Current tangent prefix changes requested/delivered complete clocks")
            if view.checkpoint is not None and (view.checkpoint.native.request != ObjectIdentity.from_record(segment.task_id, segment)
                    or view.checkpoint.native.total_steps != (self.root.landmark + 144 + 320) * view.refinement
                    or view.checkpoint.parent_id is not None or view.checkpoint.pulse is not None):
                raise ValueError("Current tangent prefix checkpoint changes its exact native request")


@dataclass(frozen=True, slots=True)
class TangentNativeResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/tangent-native-result"
    result_id: str
    source_config: ObjectIdentity
    root: TangentPreparationRoot
    prefix: ArtifactIdentity
    mode_sha256: str | None
    deliveries: tuple[PreparationDelivery, ...]
    observations_sha256: str
    completed_native_updates: int

    def __post_init__(self):
        validate_sha256(self.observations_sha256, field_name="observations_sha256")
        if self.mode_sha256 is not None:
            validate_sha256(self.mode_sha256, field_name="mode_sha256")
        expected = {f"{self.root.root_id}.{parent}.parent.r{view}" for parent in PARENTS for view in (1, 2)} | {f"{action.action_id}.r{view}" for action in preparation_actions(self.root) for view in (1, 2)}
        if (self.result_id != self.root.root_id + ".native-result" or self.source_config.object_schema != TangentPreparationConfig.SCHEMA
                or self.prefix.payload_schema != TangentPrefixResult.SCHEMA
                or tuple(delivery.occurrence_id for delivery in self.deliveries) != tuple(sorted(expected))
                or self.completed_native_updates != sum(delivery.completed_intervals for delivery in self.deliveries)):
            raise ValueError("Current tangent native result loses its exact complete branch denominator")
        actions = {action.action_id: action for action in preparation_actions(self.root)}
        for delivery in self.deliveries:
            action_id = delivery.occurrence_id.rsplit(".r", 1)[0]
            action = actions.get(action_id)
            expected_start = self.root.landmark if action is None else self.root.handoff
            expected_end = self.root.handoff if action is None else self.root.handoff + 320
            if (delivery.requested_start_tick, delivery.requested_end_tick) != (expected_start, expected_end):
                raise ValueError("Current tangent native delivery changes parent/response clocks")


@dataclass(frozen=True, slots=True)
class SavedTangentArrays(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/matrix-preparation-analysis/saved-tangent-arrays"
    VERSION: ClassVar[str] = "2.0.0"
    operand_id: str
    configuration: ObjectIdentity
    array_artifact: ArtifactIdentity
    members: tuple[MatrixArrayMember, ...]
    source_artifacts: tuple[ArtifactIdentity, ...]

    def __post_init__(self):
        validate_stable_id(self.operand_id, field_name="operand_id")
        if (self.configuration.object_schema != TangentPreparationConfig.SCHEMA
                or not 0 < len(self.members) <= 256
                or tuple(member.name for member in self.members) != tuple(sorted(set(member.name for member in self.members)))
                or sum(member.size_bytes for member in self.members) > MAXIMUM_MATRIX_ARRAY_BYTES
                or self.array_artifact.payload_schema != MATRIX_ARRAY_SCHEMA
                or self.array_artifact.media_type != "application/x-npz"
                or not 0 < self.array_artifact.size_bytes <= MAXIMUM_MATRIX_ARRAY_BYTES
                or len(self.source_artifacts) > 515):
            raise ValueError("Saved tangent arrays exceed bounded complete source contracts")

    @property
    def identity(self):
        return ObjectIdentity.from_record(self.operand_id, self)


def proposed_tangent_configuration(*, config_id: str, namespace: str, master_seed: int, code_sources_sha256: str, dependency_lock_sha256: str, summary_context="all") -> TangentPreparationConfig:
    """Explicit editable numerical allocation; no freshness/qualification claim."""
    if type(master_seed) is not int or not 0 <= master_seed < 2**256:
        raise ValueError("Tangent master must be an explicit unsigned 256-bit integer")
    roots = []
    for context in ("assembling", "prepared"):
        def digest(index, purpose):
            return sha256(canonical_json_bytes(("empirical-lawhood.current-tangent-allocation.v1", master_seed, context, index, purpose))).hexdigest()
        numerical_order = sorted(range(64), key=lambda index: (digest(index, "numerical-rank"), index))
        development_order = sorted(range(64), key=lambda index: (digest(index, "development-rank"), index))
        for index in range(64):
            roots.append(TangentPreparationRoot(namespace, context, index, numerical_order.index(index), development_order.index(index),
                                                tuple(digest(index, f"prefix-{purpose}") for purpose in range(5)),
                                                tuple(digest(index, f"preparation-{purpose}") for purpose in range(12))))
    return TangentPreparationConfig(config_id, tuple(roots), summary_context, code_sources_sha256=code_sources_sha256,
                                    dependency_lock_sha256=dependency_lock_sha256)
