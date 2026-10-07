"""development adapter-owned native expansion with separate fit/calibration/validation cuts."""

from dataclasses import replace
from math import ceil
from typing import TypeVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.native_source import NativeSourceProfile, NativeLawQualificationConfig, NativeLawQualificationExperiment
from empirical_lawhood.planning.study_authoring import CapabilitySelection
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_closeout import ResponseGeometryDevelopmentCloseoutConfig, ResponseGeometryDevelopmentContextResult, ResponseGeometryDevelopmentDevelopmentResult
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_geometry_assessment import ResponseGeometryDevelopmentGeometryReport
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_projection import DEVELOPMENT_DATA_SCHEMA, ResponseGeometryDevelopmentProjectionConfig, ResponseGeometryDevelopmentViewReport
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_records import DEVELOPMENT_FIT_SCHEMA, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentMethodConfig, ResponseGeometryDevelopmentSupportResult
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_terminal import DEVELOPMENT_VALIDATION_SCHEMA, ResponseGeometryDevelopmentQualificationConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_assessment_provider import response_geometry_development_assessment_references
from empirical_lawhood.adapters.methods.response_geometry_development.extension_bundle import DEVELOPMENT_CAPABILITIES, DEVELOPMENT_ROLES
from empirical_lawhood.adapters.simulators.response_geometry_development.extension_bundle import DEVELOPMENT_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_HDF5_SCHEMA, DEVELOPMENT_PANEL_ID, ResponseGeometryDevelopmentNativeConfig, ResponseGeometryDevelopmentNativeSegmentResult, development_segments

from .executable_binding import _config_ref, _outputs
from .discovery import CANONICAL_MEDIA_TYPE
from .development_roster import response_geometry_development_native_declarations


Record = TypeVar("Record", bound=CanonicalRecord)
_MIB = 1024**2
_STAGE = LinkedCampaignStageEnvelope.SCHEMA


def _one(records: tuple[CanonicalRecord, ...], kind: type[Record]) -> Record:
    selected = tuple(value for value in records if type(value) is kind)
    if len(selected) != 1:
        raise ValueError(f"development expansion requires one exact {kind.SCHEMA}")
    value = selected[0]
    assert isinstance(value, kind)
    return value


def expand_response_geometry_development_protocol(
    *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate, binding: ObjectIdentity
) -> ProtocolTemplate:
    """Compile the fixed native and method roster, without opening a source."""
    source = validate_response_geometry_development_source_records(records, binding)
    projection = _one(records, ResponseGeometryDevelopmentProjectionConfig)
    method = _one(records, ResponseGeometryDevelopmentMethodConfig)
    qualification = _one(records, ResponseGeometryDevelopmentQualificationConfig)
    closeout = _one(records, ResponseGeometryDevelopmentCloseoutConfig)
    extension = _one(records, NativeLawQualificationExperiment)
    substrate = _one(records, ResponseSubstrateBinding)
    identification_config = extension.identification_config
    clock, _ = response_geometry_development_assessment_references(
        qualification,
        DEVELOPMENT_CAPABILITIES[2],
        ArtifactIdentity(
            f"config-artifact.{qualification.config_id}",
            "development-authenticated-input",
            qualification.SCHEMA,
            qualification.fingerprint(),
            CANONICAL_MEDIA_TYPE,
            len(qualification.canonical_bytes()),
        ),
    )
    declared = response_geometry_development_native_declarations(source, clock)
    if (
        identification_config.physical_units,
        identification_config.acquisition_groups,
        identification_config.nested_views,
        identification_config.projection_groups,
        identification_config.action_chart,
    ) != (
        declared.units,
        declared.acquisitions,
        declared.views,
        declared.projections,
        declared.chart,
    ):
        raise ValueError("development carrier changes its exact unit/split/native-action/view declarations")
    source_id = ObjectIdentity.from_record(source.config_id, source)
    if (
        substrate.installed_executable_binding != binding
        or substrate.native_config != source_id
        or qualification.source != source
        or qualification.projection != projection
        or qualification.method != method
        or closeout.qualification != qualification
        or identification_config.projection_config != ObjectIdentity.from_record(projection.config_id, projection)
        or identification_config.method_config != ObjectIdentity.from_record(method.config_id, method)
        or identification_config.law_finalizer_owner
        != ObjectIdentity.from_record(DEVELOPMENT_CAPABILITIES[2].capability_key, DEVELOPMENT_CAPABILITIES[2])
        or set(identification_config.identification_admission_physical_independent_unit_ids)
        != {source.physical_unit_id(root) for root in source.roots}
    ):
        raise ValueError("development expansion changes its native/method/sole-owner lineage")
    expected_groups = {
        f"{root.root_id}.project.r{refinement}": (source.physical_unit_id(root), f"r{refinement}")
        for root in source.roots
        for refinement in (1, 2)
    }
    views = {view.view_id: view for view in identification_config.nested_views}
    if (
        len(identification_config.nested_views) != 3840
        or len(identification_config.acquisition_groups) != 1920
        or ({g.projection_task_id for g in identification_config.projection_groups} != set(expected_groups))
    ):
        raise ValueError("development expansion changes its paired acquisition/projection roster")
    for group in identification_config.projection_groups:
        if len(group.scientific_view_ids) != 15 or {
            (views[v].physical_independent_unit_id, views[v].numerical_member_id)
            for v in group.scientific_view_ids
        } != {expected_groups[group.projection_task_id]}:
            raise ValueError("development projection changes its exact root/member denominator")

    manifests = dict(zip(DEVELOPMENT_ROLES, DEVELOPMENT_CAPABILITIES, strict=True))
    steps: list[ProtocolStepTemplate] = []
    by_id: dict[str, ProtocolStepTemplate] = {}

    def add(
        task: str,
        stage: ScientificStage,
        manifest: CapabilityManifest,
        config: CanonicalRecord,
        config_id: str,
        dependencies: tuple[str, ...],
        reports: tuple[tuple[str, str], ...],
        data: tuple[str, str] | None,
        output_bytes: int,
        seconds: int,
        barrier: BarrierKind,
    ) -> None:
        config_size = len(config.canonical_bytes())
        scan_bytes = config_size * (
            2 if stage is ScientificStage.PREPARE and not dependencies else 1
        )
        scan_bytes += sum(by_id[d].resource_budget.output_bytes for d in dependencies)
        if scan_bytes > manifest.resource_ceiling.source_scan_bytes or (
            output_bytes > manifest.resource_ceiling.output_bytes
        ):
            raise ValueError(f"development task {task} exceeds its installed scan/output ceiling")
        step = ProtocolStepTemplate(
            task,
            stage,
            manifest.capability_key,
            manifest.capability_version,
            _config_ref(config, ObjectIdentity.from_record(config_id, config), manifest),
            tuple(sorted(dependencies)),
            _outputs(
                (
                    *reports,
                    ("stage-envelope" if stage is ScientificStage.PREPARE else "stage", _STAGE),
                ),
                data,
            ),
            manifest.permissions,
            OutcomeAccess.EVALUATION_REVEALED
            if stage is ScientificStage.EVALUATE
            else OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.PROSPECTIVE,
            replace(
                manifest.resource_ceiling,
                source_scan_bytes=scan_bytes,
                output_bytes=output_bytes,
                wall_time_seconds=seconds,
            ),
            (),
            barrier,
            1,
            (f"{DEVELOPMENT_PANEL_ID}.single-terminal",)
            if stage is ScientificStage.EVALUATE
            else (f"custody.{task}",),
        )
        steps.append(step)
        by_id[task] = step

    segments = development_segments()
    # Construction order is dependency order; canonical protocol order is by id.
    for phase in ("prefix", "parent", "inner"):
        for segment in (s for s in segments if s.phase == phase):
            predecessor = segment.predecessor
            add(
                segment.task_id,
                ScientificStage.PREPARE,
                DEVELOPMENT_SOURCE_CAPABILITY,
                source,
                source.config_id,
                () if predecessor is None else (predecessor.task_id,),
                (("native-result", ResponseGeometryDevelopmentNativeSegmentResult.SCHEMA),),
                ("native-observations", DEVELOPMENT_HDF5_SCHEMA),
                17 * _MIB,
                ceil(2 + segment.native_updates * 0.03),
                BarrierKind.NONE,
            )
    for root in source.roots:
        dependencies = tuple(s.task_id for s in segments if s.root == root)
        for refinement in (1, 2):
            add(
                f"{root.root_id}.project.r{refinement}",
                ScientificStage.TRANSFORM,
                manifests["projection"],
                projection,
                projection.config_id,
                dependencies,
                (("report", ResponseGeometryDevelopmentViewReport.SCHEMA),),
                ("data", DEVELOPMENT_DATA_SCHEMA),
                17 * _MIB,
                120,
                BarrierKind.FREEZE,
            )
    for context in ("assembling", "prepared"):

        def projected(indices: range) -> tuple[str, ...]:
            return tuple(
                f"{DEVELOPMENT_PANEL_ID}.{context}.r{i:02d}.project.r{r}" for i in indices for r in (1, 2)
            )

        fit, calibrate, support = (
            f"{DEVELOPMENT_PANEL_ID}.{role}.{context}" for role in ("fit", "calibrate", "support")
        )
        for task, dependencies, report, data, size in (
            (fit, projected(range(16)), ResponseGeometryDevelopmentFitResult, ("data", DEVELOPMENT_FIT_SCHEMA), 65 * _MIB),
            (calibrate, (fit, *projected(range(16, 32))), ResponseGeometryDevelopmentCalibrationResult, None, 128 * 1024),
            (support, (fit, calibrate, *projected(range(32))), ResponseGeometryDevelopmentSupportResult, None, _MIB),
        ):
            add(
                task,
                ScientificStage.DEVELOP,
                manifests["development"],
                method,
                method.config_id,
                dependencies,
                (("report", report.SCHEMA),),
                data,
                size,
                28800,
                BarrierKind.FREEZE,
            )
        add(
            f"{DEVELOPMENT_PANEL_ID}.assess.{context}",
            ScientificStage.QUALIFY,
            manifests["assessment"],
            qualification,
            qualification.config_id,
            (fit, calibrate, support, *projected(range(16)), *projected(range(32, 64))),
            (("report", ResponseGeometryDevelopmentContextResult.SCHEMA), ("geometry", ResponseGeometryDevelopmentGeometryReport.SCHEMA)),
            ("data", DEVELOPMENT_VALIDATION_SCHEMA),
            297 * _MIB,
            28800,
            BarrierKind.FREEZE,
        )
    add(
        f"{DEVELOPMENT_PANEL_ID}.evaluate",
        ScientificStage.EVALUATE,
        manifests["evaluation"],
        closeout,
        closeout.config_id,
        tuple(f"{DEVELOPMENT_PANEL_ID}.assess.{c}" for c in ("assembling", "prepared")),
        (
            ("report", ResponseGeometryDevelopmentDevelopmentResult.SCHEMA),
            ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
        ),
        None,
        8 * _MIB,
        120,
        BarrierKind.REVEAL,
    )
    if len(steps) != 2953:
        raise ValueError("development expansion differs from its complete 2953-task graph")
    return replace(
        template,
        template_id=f"{template.template_id}.{extension.fingerprint()[:16]}",
        steps=tuple(sorted(steps, key=lambda s: s.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def validate_response_geometry_development_source_records(
    records: tuple[CanonicalRecord, ...], binding: ObjectIdentity
) -> ResponseGeometryDevelopmentNativeConfig:
    """Join decoded native metadata before a provider can expose any runner."""
    source = _one(records, ResponseGeometryDevelopmentNativeConfig)
    profile = _one(records, NativeSourceProfile)
    qualification_config = _one(records, NativeLawQualificationConfig)
    extension = _one(records, NativeLawQualificationExperiment)
    substrate = _one(records, ResponseSubstrateBinding)
    identity = ObjectIdentity.from_record(source.config_id, source)
    segments = development_segments()
    if (
        qualification_config != extension.identification_config
        or qualification_config.source_pipeline_profile != ObjectIdentity.from_record(profile.profile_id, profile)
        or profile.source_config != identity
        or profile.source_selection
        != CapabilitySelection(
            DEVELOPMENT_SOURCE_CAPABILITY.capability_key,
            DEVELOPMENT_SOURCE_CAPABILITY.capability_version,
            DEVELOPMENT_SOURCE_CAPABILITY.implementation_sha256,
        )
        or profile.observation_schema != DEVELOPMENT_HDF5_SCHEMA
        or profile.prerequisite_qualification != source.assay_evaluation
        or profile.physical_independent_unit_ids
        != tuple(sorted(source.physical_unit_id(r) for r in source.roots))
        or qualification_config.identification_admission_physical_independent_unit_ids != profile.physical_independent_unit_ids
        or profile.maximum_native_segment_count != len(segments)
        or profile.maximum_native_update_count != sum(s.native_updates for s in segments)
        or substrate.installed_executable_binding != binding
        or substrate.native_config != identity
    ):
        raise ValueError("development issued source records change native profile/carrier/config lineage")
    return source
