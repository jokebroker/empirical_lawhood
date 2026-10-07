"""Adapter-owned exact 397-task native/development production expansion."""

from dataclasses import replace
from typing import TypeVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.time import ClockCoordinate, ClockProjection
from empirical_lawhood.planning.native_source import NativeLawQualificationConfig, NativeLawQualificationExperiment, NativeSourceProfile
from empirical_lawhood.planning.study_authoring import CapabilitySelection
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref
from empirical_lawhood.adapters.methods.matrix_preparation.closeout import PreparationCloseoutConfig
from empirical_lawhood.adapters.methods.matrix_preparation.contracts import PreparationMethodConfig, PreparationProjectionConfig
from empirical_lawhood.adapters.methods.matrix_preparation.extension_bundle import METHOD_CAPABILITIES
from empirical_lawhood.adapters.methods.matrix_preparation.continuation import PreparationProjectionContinuation
from empirical_lawhood.adapters.methods.matrix_preparation.records import PreparationForecastConfig, PreparationDecisionConfig, PreparationTaskAssessmentConfig, PreparationAssessmentConfig, PreparationNumericalSemanticsConfig
from empirical_lawhood.adapters.methods.matrix_preparation.topology import DATA_OUTPUTS, METHOD_ROLES, OUTPUT_MIB, RECORD_OUTPUTS, STAGE_SCHEMA, WALL_SECONDS, preparation_task_declarations
from .contracts import CANONICAL_MEDIA_TYPE, DEVELOPMENT, NATIVE_SCHEMA, PreparationSourceConfig, native_updates_for_root
from .extension_bundle import SOURCE_CAPABILITY
from .continuation import PreparationDevelopmentContinuation, preparation_continuation
from .roster import preparation_native_declarations


Record = TypeVar("Record", bound=CanonicalRecord)


def one_preparation_record(records: tuple[CanonicalRecord, ...], kind: type[Record]) -> Record:
    selected = tuple(r for r in records if type(r) is kind)
    if len(selected) != 1:
        raise ValueError(f"preparation route requires one exact {kind.SCHEMA}")
    return selected[0]


def preparation_clock_reference(config: PreparationAssessmentConfig) -> ExecutableReference:
    manifest = METHOD_CAPABILITIES[METHOD_ROLES.index("description")]
    artifact = ArtifactIdentity(
        f"config-artifact.{config.config_id}",
        "preparation-authenticated-input",
        config.SCHEMA,
        config.fingerprint(),
        CANONICAL_MEDIA_TYPE,
        len(config.canonical_bytes()),
    )
    return ExecutableReference(
        f"{config.config_id}.native-clock-projector",
        manifest.capability_key,
        manifest.capability_version,
        "kernel.clock-transport.project",
        artifact,
        SafePayloadFormat.CANONICAL_JSON,
        ClockCoordinate.SCHEMA,
        ClockProjection.SCHEMA,
        True,
    )


def preparation_protocol_steps(
    source: PreparationSourceConfig,
    projection: PreparationProjectionConfig,
    numerical_semantics: PreparationNumericalSemanticsConfig,
    method: PreparationMethodConfig,
    assessment: PreparationAssessmentConfig,
    closeout: PreparationCloseoutConfig,
    continuation: PreparationDevelopmentContinuation | None = None,
) -> tuple[ProtocolStepTemplate, ...]:
    if continuation is not None:
        continuation.validate_source(source)
    configs = {
        "source": source,
        "projection": projection,
        "numerical-semantics": numerical_semantics,
        "fit": method,
        "description": assessment,
        "forecast": PreparationForecastConfig(f"{DEVELOPMENT}.forecast-config", method),
        "decision": PreparationDecisionConfig(f"{DEVELOPMENT}.decision-config", method),
        "task-assessment": PreparationTaskAssessmentConfig(
            f"{DEVELOPMENT}.task-assessment-config", method
        ),
        "evaluation": closeout,
    }
    manifests = {
        "source": SOURCE_CAPABILITY,
        **dict(zip(METHOD_ROLES, METHOD_CAPABILITIES, strict=True)),
    }
    config_sizes = {role: len(config.canonical_bytes()) for role, config in configs.items()}
    steps: dict[str, ProtocolStepTemplate] = {}
    for task in preparation_task_declarations():
        config, manifest = configs[task.role], manifests[task.role]
        config_id = str(getattr(config, "config_id"))
        stage = (
            ScientificStage.PREPARE
            if task.role == "source"
            else ScientificStage.TRANSFORM
            if task.role == "projection"
            else ScientificStage.QUALIFY
            if task.role in ("description", "numerical-semantics")
            else ScientificStage.EVALUATE
            if task.role == "evaluation"
            else ScientificStage.DEVELOP
        )
        scan = config_sizes[task.role] * (2 if task.role == "source" else 1) + sum(
            steps[d].resource_budget.output_bytes for d in task.dependencies
        )
        if task.role == "source":
            root = next(p.root for p in source.prefixes if task.task_id == f"{p.root_id}.native")
            scan += next(b.artifact.size_bytes for b in source.prefix_bundles if root in b.roots)
        output = OUTPUT_MIB[task.role] * 1024**2
        if (
            scan > manifest.resource_ceiling.source_scan_bytes
            or output > manifest.resource_ceiling.output_bytes
        ):
            raise ValueError(
                f"preparation task {task.task_id} exceeds its installed resource ceiling"
            )
        outputs = [
            OutputTemplate(
                name,
                kind.SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
                CANONICAL_MEDIA_TYPE,
                ".canonical.json",
            )
            for name, kind in RECORD_OUTPUTS[task.role]
        ]
        outputs.extend(
            OutputTemplate(name, schema, ArtifactProfile.AUDITED_HDF5, "application/x-hdf5", ".h5")
            for name, schema in DATA_OUTPUTS[task.role]
        )
        outputs.append(
            OutputTemplate(
                "stage-envelope" if task.role == "source" else "stage",
                STAGE_SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
                CANONICAL_MEDIA_TYPE,
                ".canonical.json",
            )
        )
        steps[task.task_id] = ProtocolStepTemplate(
            task.task_id,
            stage,
            manifest.capability_key,
            manifest.capability_version,
            _config_ref(config, ObjectIdentity.from_record(config_id, config), manifest),
            task.dependencies,
            tuple(sorted(outputs, key=lambda o: o.output_id)),
            manifest.permissions,
            OutcomeAccess.EVALUATION_REVEALED
            if task.role == "evaluation"
            else OutcomeAccess.DEVELOPMENT_VISIBLE,
            # Every acquired root descends from retained outcome-visible development
            # evidence. Carry that exposure through all method output plans.
            VisibilityCeiling.OUTCOME_VISIBLE,
            replace(
                manifest.resource_ceiling,
                source_scan_bytes=scan,
                output_bytes=output,
                wall_time_seconds=WALL_SECONDS[task.role],
            ),
            (),
            BarrierKind.REVEAL
            if task.role == "evaluation"
            else BarrierKind.NONE
            if task.role == "source"
            else BarrierKind.FREEZE,
            1,
            (f"{DEVELOPMENT}.single-terminal",)
            if task.role == "evaluation"
            else (f"custody.{task.task_id}",),
        )
    if continuation is not None:
        retained = set(continuation.retained_task_ids)
        steps = {
            key: replace(
                step,
                dependency_step_ids=tuple(
                    dep for dep in step.dependency_step_ids if dep not in retained
                ),
            )
            for key, step in steps.items()
            if key not in retained
        }
    return tuple(sorted(steps.values(), key=lambda s: s.step_id))


def validate_preparation_source_records(
    records: tuple[CanonicalRecord, ...], binding: ObjectIdentity
) -> PreparationSourceConfig:
    source = one_preparation_record(records, PreparationSourceConfig)
    profile = one_preparation_record(records, NativeSourceProfile)
    law_qualification = one_preparation_record(records, NativeLawQualificationConfig)
    extension = one_preparation_record(records, NativeLawQualificationExperiment)
    substrate = one_preparation_record(records, ResponseSubstrateBinding)
    identity = ObjectIdentity.from_record(source.config_id, source)
    if (
        law_qualification != extension.identification_config
        or law_qualification.source_pipeline_profile != ObjectIdentity.from_record(profile.profile_id, profile)
        or profile.source_config != identity
        or profile.source_selection
        != CapabilitySelection(
            SOURCE_CAPABILITY.capability_key,
            SOURCE_CAPABILITY.capability_version,
            SOURCE_CAPABILITY.implementation_sha256,
        )
        or profile.observation_schema != NATIVE_SCHEMA
        or profile.prerequisite_qualification != source.imported_inventory
        or profile.physical_independent_unit_ids
        != tuple(sorted(r.physical_unit_id for r in source.roots))
        or law_qualification.identification_admission_physical_independent_unit_ids != profile.physical_independent_unit_ids
        or profile.maximum_native_segment_count != 128
        or profile.maximum_native_update_count
        != sum(native_updates_for_root(r) for r in source.roots)
        or substrate.installed_executable_binding != binding
        or substrate.native_config != identity
    ):
        raise ValueError(
            "preparation source records changed their exact exposed-root/profile/custody lineage"
        )
    return source


def expand_preparation_protocol(
    *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate, binding: ObjectIdentity
) -> ProtocolTemplate:
    source = validate_preparation_source_records(records, binding)
    continuation = preparation_continuation(records)
    projection = one_preparation_record(records, PreparationProjectionConfig)
    if continuation is not None:
        imports = one_preparation_record(records, PreparationProjectionContinuation)
        if imports.continuation != ObjectIdentity.from_record(
            continuation.config_id, continuation
        ) or imports.projection_config != ObjectIdentity.from_record(
            projection.config_id, projection
        ):
            raise ValueError("projection and source continuation declarations disagree")
    elif any(isinstance(r, PreparationProjectionContinuation) for r in records):
        raise ValueError("projection import declaration lacks its source continuation")
    numerical_semantics = one_preparation_record(records, PreparationNumericalSemanticsConfig)
    method = one_preparation_record(records, PreparationMethodConfig)
    assessment = one_preparation_record(records, PreparationAssessmentConfig)
    closeout = one_preparation_record(records, PreparationCloseoutConfig)
    for kind in (
        PreparationForecastConfig,
        PreparationDecisionConfig,
        PreparationTaskAssessmentConfig,
    ):
        if one_preparation_record(records, kind).method != method:
            raise ValueError("preparation method role changes the single frozen scientific recipe")
    extension = one_preparation_record(records, NativeLawQualificationExperiment)
    law_qualification = extension.identification_config
    native = preparation_native_declarations(source, preparation_clock_reference(assessment))
    if (
        (
            law_qualification.physical_units,
            law_qualification.acquisition_groups,
            law_qualification.nested_views,
            law_qualification.projection_groups,
            law_qualification.action_chart,
        )
        != (native.units, native.acquisitions, native.views, native.projections, native.chart)
        or assessment.projection != projection
        or assessment.numerical_semantics != numerical_semantics
        or assessment.method != method
        or closeout.assessment != assessment
        or assessment.source_config != ObjectIdentity.from_record(source.config_id, source)
        or law_qualification.projection_config != ObjectIdentity.from_record(projection.config_id, projection)
        or law_qualification.method_config != ObjectIdentity.from_record(method.config_id, method)
        or law_qualification.law_finalizer_owner
        != ObjectIdentity.from_record(METHOD_CAPABILITIES[3].capability_key, METHOD_CAPABILITIES[3])
    ):
        raise ValueError(
            "preparation expansion changed its full native census or scientific owners"
        )
    return replace(
        template,
        template_id=f"{template.template_id}.{extension.fingerprint()[:16]}",
        steps=preparation_protocol_steps(
            source, projection, numerical_semantics, method, assessment, closeout, continuation
        ),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )
