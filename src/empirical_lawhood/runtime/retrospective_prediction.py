"""Noncontact finite prediction topology and exact held-input access checks."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.retrospective_prediction import RetrospectivePredictionExperiment, RetrospectivePredictionRole as Role, RetrospectivePredictionInputRole as InputRole
from .capabilities import CapabilityRegistry
from .plans import BarrierKind, ProtocolExecutionPlan, ScientificStage


@dataclass(frozen=True, slots=True)
class RetrospectivePredictionBinding(CanonicalRecord):
    """Installed archive/prediction binding; deliberately not a native actuator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-prediction-binding'
    binding_id: str
    prediction_carrier: ObjectIdentity
    installed_executable_binding: ObjectIdentity
    projection_owner: ObjectIdentity
    projection_config: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if (
            self.prediction_carrier.object_schema != RetrospectivePredictionExperiment.SCHEMA
            or self.installed_executable_binding.object_schema
            != 'empirical-lawhood/runtime/executable-capability-binding'
            or self.projection_owner.object_schema != 'empirical-lawhood/runtime/capability-manifest'
        ):
            raise ValueError("historical binding requires its exact carrier and installed owner")


_STAGES = {
    Role.TRAINING_PROJECTION: ScientificStage.TRANSFORM,
    Role.FEATURE_PROJECTION: ScientificStage.TRANSFORM,
    Role.FIT: ScientificStage.DEVELOP,
    Role.CALIBRATE: ScientificStage.DEVELOP,
    Role.PREDICT: ScientificStage.DEVELOP,
    Role.COMMIT: ScientificStage.FREEZE,
    Role.REVEAL: ScientificStage.REVEAL,
    Role.EVALUATE: ScientificStage.EVALUATE,
}


@dataclass(frozen=True, slots=True)
class RetrospectivePredictionStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-prediction-stage'
    task_id: str
    role: Role
    fold_id: str | None
    dependency_task_ids: tuple[str, ...]
    owner: ObjectIdentity
    config: ObjectIdentity
    data_input_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        if not isinstance(self.role, Role):
            raise ValueError("historical stage requires a declared role")
        if (self.role is Role.EVALUATE) != (self.fold_id is None):
            raise ValueError("only the terminal evaluator has no fold")
        if self.fold_id is not None:
            validate_stable_id(self.fold_id, field_name="fold_id")
        for name in ("dependency_task_ids", "data_input_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if self.task_id in self.dependency_task_ids:
            raise ValueError("historical stage cannot depend on itself")
        if self.owner.object_schema != 'empirical-lawhood/runtime/capability-manifest':
            raise ValueError("historical stage requires its exact capability owner")

    @property
    def stage(self) -> ScientificStage:
        return _STAGES[self.role]

    @property
    def barrier(self) -> BarrierKind:
        if self.role is Role.COMMIT:
            return BarrierKind.FREEZE
        if self.role in (Role.REVEAL, Role.EVALUATE):
            return BarrierKind.REVEAL
        return BarrierKind.NONE


@dataclass(frozen=True, slots=True)
class CompiledRetrospectivePrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-retrospective-prediction'
    extension_set: ObjectIdentity
    stages: tuple[RetrospectivePredictionStage, ...]

    def __post_init__(self) -> None:
        ids = tuple(s.task_id for s in self.stages)
        if (
            self.extension_set.object_schema != RetrospectivePredictionExperiment.SCHEMA
            or not 8 <= len(ids) <= 113
            or ids != tuple(sorted(set(ids)))
            or sum(s.role is Role.EVALUATE for s in self.stages) != 1
        ):
            raise ValueError("historical topology lacks its bounded exact census")


def derive_retained_prediction_topology(
    extension: RetrospectivePredictionExperiment,
) -> CompiledRetrospectivePrediction:
    owners = {o.role: o for o in extension.owners}
    result: list[RetrospectivePredictionStage] = []
    terminal_dependencies: list[str] = []
    all_commits = tuple(
        sorted(f"{extension.experiment_id}.{fold.fold_id}.commit" for fold in extension.folds)
    )
    for fold in extension.folds:
        ids = {
            role: f"{extension.experiment_id}.{fold.fold_id}.{role.value.lower().replace('_', '-')}"
            for role in Role
            if role is not Role.EVALUATE
        }
        dependencies = {
            Role.TRAINING_PROJECTION: (),
            Role.FEATURE_PROJECTION: (),
            Role.FIT: (ids[Role.TRAINING_PROJECTION],),
            Role.CALIBRATE: (ids[Role.FIT],),
            Role.PREDICT: tuple(sorted((ids[Role.CALIBRATE], ids[Role.FEATURE_PROJECTION]))),
            Role.COMMIT: (ids[Role.PREDICT],),
            # No fold's withheld labels enter before every forecast is durable.
            Role.REVEAL: all_commits,
        }
        input_roles = {
            Role.TRAINING_PROJECTION: InputRole.TRAINING,
            Role.FEATURE_PROJECTION: InputRole.TARGET_FEATURES,
            Role.CALIBRATE: InputRole.CALIBRATION,
            Role.REVEAL: InputRole.TARGET_LABELS,
        }
        for role, task_id in ids.items():
            owner = owners[role]
            inputs = tuple(
                i.input_id
                for i in extension.inputs
                if i.fold_id == fold.fold_id and i.role is input_roles.get(role)
            )
            result.append(
                RetrospectivePredictionStage(
                    task_id,
                    role,
                    fold.fold_id,
                    dependencies[role],
                    owner.owner,
                    owner.config,
                    inputs,
                )
            )
        terminal_dependencies.extend((ids[Role.COMMIT], ids[Role.REVEAL]))
    owner = owners[Role.EVALUATE]
    result.append(
        RetrospectivePredictionStage(
            f"{extension.experiment_id}.evaluate",
            Role.EVALUATE,
            None,
            tuple(sorted(terminal_dependencies)),
            owner.owner,
            owner.config,
            (),
        )
    )
    return CompiledRetrospectivePrediction(
        ObjectIdentity.from_record(extension.extension_set_id, extension),
        tuple(sorted(result, key=lambda s: s.task_id)),
    )


def retained_prediction_contract_reasons(
    extension: RetrospectivePredictionExperiment,
    compiled: CompiledRetrospectivePrediction,
    plan: ProtocolExecutionPlan,
    registry: CapabilityRegistry,
) -> tuple[str, ...]:
    """Check actual task ports as well as chronology: dependencies alone are insufficient."""
    expected = derive_retained_prediction_topology(extension)
    tasks = {t.task_id: t for t in plan.tasks}
    reasons: list[str] = []
    if compiled != expected or set(tasks) != {s.task_id for s in expected.stages}:
        reasons.append("HISTORICAL_PREDICTION_STAGE_ROSTER_DIFFERS")
    inputs = {i.input_id: i for i in extension.inputs}
    allowed_configs = {o.config for o in extension.owners}
    for stage in expected.stages:
        task = tasks.get(stage.task_id)
        if task is None:
            continue
        manifest = registry.resolve(
            task.capability.capability_key, task.capability.capability_version
        )
        if (
            task.stage is not stage.stage
            or task.dependency_task_ids != stage.dependency_task_ids
            or ObjectIdentity.from_record(manifest.capability_key, manifest) != stage.owner
            or task.capability.config.config_schema != stage.config.object_schema
            or task.capability.config.content_sha256 != stage.config.object_fingerprint
            or task.maximum_attempts != 1
            or task.barrier is not stage.barrier
        ):
            reasons.append(f"HISTORICAL_PREDICTION_STAGE_BINDING_DIFFERS:{stage.task_id}")
        actual: set[str] = set()
        for port in task.external_inputs:
            # Only this stage's exact config may accompany its declared data.
            if (
                port.expected_payload_schema == stage.config.object_schema
                and port.expected_content_sha256 == stage.config.object_fingerprint
                and stage.config in allowed_configs
            ):
                continue
            data = next(
                (
                    i
                    for i in inputs.values()
                    if i.artifact.artifact_id == port.logical_artifact_id
                    and i.input_id in stage.data_input_ids
                ),
                None,
            )
            if data is None:
                reasons.append(f"HISTORICAL_PREDICTION_UNDECLARED_INPUT:{stage.task_id}")
                continue
            actual.add(data.input_id)
            artifact = data.artifact
            if (
                port.expected_content_sha256 != artifact.sha256
                or port.expected_payload_schema != artifact.payload_schema
                or port.expected_media_type != artifact.media_type
                or port.expected_size_bytes != artifact.size_bytes
                or port.expected_outcome_access is not data.outcome_access
                or port.expected_visibility_ceiling is not data.visibility_ceiling
            ):
                reasons.append(f"HISTORICAL_PREDICTION_INPUT_IDENTITY_DIFFERS:{data.input_id}")
        if actual != set(stage.data_input_ids):
            reasons.append(f"HISTORICAL_PREDICTION_DATA_ROSTER_DIFFERS:{stage.task_id}")
    return tuple(sorted(set(reasons)))
