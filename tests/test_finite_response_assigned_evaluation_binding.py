"""Synthetic assigned prospective evaluation and retained bindings, without execution or a result."""

from tests.finite_response_seed_fixtures import ASSIGNED_SEEDS
from decimal import Decimal

import pytest

from empirical_lawhood.adapters.composition.finite_response_law.assignment import FiniteResponseLawCohortAssignment
from empirical_lawhood.adapters.composition.finite_response_law.authoring import build_native_authoring
from empirical_lawhood.adapters.composition.finite_response_law.completion_authoring import build_completion_authoring
from empirical_lawhood.adapters.composition.finite_response_law.design import qualification_system
from empirical_lawhood.adapters.composition.finite_response_law.exposure import native_seed_ids
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation, FiniteResponseLawAssignedEvaluationCompletionConfig, FiniteResponseLawAssignedEvaluationProjectionConfig, FiniteResponseLawAssignedEvaluationViewObservation
from empirical_lawhood.adapters.methods.finite_response_law.completion.contracts import PROSPECTIVE_EVALUATION_COMPLETION_PREFIX, FiniteResponseLawAssignedRetainedCompletionConfig, FiniteResponseLawRetainedProjection
from empirical_lawhood.adapters.methods.finite_response_law.completion.executable_binding import ASSIGNED_BINDING as COMPLETION_BINDING
from empirical_lawhood.adapters.methods.finite_response_law.completion.extension_bundle import ASSIGNED_CAPABILITY as COMPLETION_CAPABILITY
from empirical_lawhood.adapters.methods.finite_response_law.control_records import FiniteResponseLawControlConfig
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_results import FiniteResponseLawEvaluationRevealConfig
from empirical_lawhood.adapters.methods.finite_response_law.method_records import FiniteResponseLawAssignedQualificationReport
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation import discovery, executable_binding
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_continuation import discovery as continuation_discovery
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_continuation import executable_binding as continuation_binding
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_retention import FiniteResponseLawEvaluationRetention
from empirical_lawhood.adapters.simulators.finite_response_law.native_artifact import NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.finite_response_law.roster import Q_CLOCK
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.models import ModelIntersectionSemantics, ModelMemberLawBinding, ModelSetSpec
from empirical_lawhood.kernel.obligations import ObligationStatus, ValiditySpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.planning.source_qualification import ProspectiveRetainedPredecessor, ProspectiveRetainedSourceUse, QualifiedSourceUseExperiment
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.source_qualification import ProspectiveRetainedSourceUseBinding, QualifiedSourceUseSubstrateBinding


def _identity(name: str, schema: str) -> ObjectIdentity:
    return ObjectIdentity(name, schema, "1.0.0", "0" * 64)


def _artifact(name: str, schema: str) -> ArtifactIdentity:
    return ArtifactIdentity(
        name,
        "synthetic-development",
        schema,
        "0" * 64,
        "application/vnd.empirical-lawhood.canonical+json",
        1,
    )


def _model_set(source: FiniteResponseLawAssignedEvaluationConfig) -> ModelSetSpec:
    member = ModelMemberLawBinding(
        "synthetic.member",
        "synthetic.denominator",
        "synthetic.relation",
        _identity("synthetic.law", 'empirical-lawhood/kernel/response-law'),
        _identity(
            "synthetic.qualification",
            'empirical-lawhood/kernel/law-qualification-result',
        ),
        ("synthetic.candidate",),
        ("synthetic.view",),
        ("synthetic.property",),
        ("synthetic.nontransported",),
    )
    return ModelSetSpec(
        "finite-response-law.composed.prospective-control.models",
        qualification_system(source).world.world_id,
        (member,),
        ("synthetic.relation",),
        "Synthetic binding only.",
        "Synthetic binding only.",
        ("synthetic.uncertainty",),
        ValiditySpec(
            "synthetic.validity",
            ("synthetic.domain",),
            ("synthetic.assumption",),
            (),
            ObligationStatus.REQUIRED,
        ),
        ModelIntersectionSemantics.QUALIFIED_INTERSECTION,
        EvidenceCeiling.ADMISSION,
        OutcomeAccess.OUTCOME_BLIND,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )


def _assigned_prospective_evaluation() -> tuple[
    FiniteResponseLawCohortAssignment, FiniteResponseLawAssignedEvaluationConfig, FiniteResponseLawControlConfig
]:
    science = FiniteResponseLawScienceSpec()
    assignment = FiniteResponseLawCohortAssignment(
        "prospective-evaluation",
        "empirical-lawhood.finite-response-law.prospective-evaluation.synthetic-binding",
        science.plan_sha256,
        64,
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    scientific_seeds=ASSIGNED_SEEDS["empirical-lawhood.finite-response-law.prospective-evaluation.synthetic-binding"],
    )
    source = FiniteResponseLawAssignedEvaluationConfig(
        "prospective-evaluation",
        science,
        ObjectIdentity.from_record(
            discovery.SOURCE_CAPABILITY.capability_key, discovery.SOURCE_CAPABILITY
        ),
        None,
        (),
        assignment.cohort_namespace,
        assignment.fingerprint(),
    scientific_seeds=assignment.scientific_seeds,
    )
    control = FiniteResponseLawControlConfig(
        source,
        _artifact("synthetic.report", FiniteResponseLawAssignedQualificationReport.SCHEMA),
        _artifact("synthetic.report.receipt", CanonicalTaskReceipt.SCHEMA),
        _identity("synthetic.report.receipt", CanonicalTaskReceipt.SCHEMA),
        _artifact("synthetic.native", FiniteResponseLawAssignedCalibrationNativeEvaluation.SCHEMA),
        _artifact("synthetic.native.receipt", CanonicalTaskReceipt.SCHEMA),
        _identity("synthetic.native.receipt", CanonicalTaskReceipt.SCHEMA),
        _model_set(source),
    )
    return assignment, source, control


def test_assigned_prospective_evaluation_authoring_and_every_registered_owner_bind() -> None:
    assignment, source, control = _assigned_prospective_evaluation()
    assert (
        decode_canonical_bytes(
            control.canonical_bytes(), FiniteResponseLawControlConfig, maximum_bytes=1024**2
        )
        == control
    )
    bundle = build_native_authoring(
        source=source,
        implementation_sha256="0" * 64,
        exposure=ObjectIdentity.from_record(
            "synthetic-development-assignment", assignment
        ),
        exposed_unit_ids=("synthetic-exposed-root",),
        exposed_seed_ids=("synthetic-exposed-seed",),
        new_seed_ids=native_seed_ids(source),
        control=control,
    )
    records = {type(record): record for record in bundle.payloads}
    assert len(records) == 7
    steps = {
        step.step_id: step
        for step in bundle.standard_context.base.templates[0].protocol.steps
    }
    assert len(steps) == 1796
    first_root = source.roots[0]
    first_tasks = tuple(
        task for task in native_invocations(source) if task.root == first_root
    )
    parent = next(task for task in first_tasks if task.phase == "parent")
    future = next(task for task in first_tasks if task.phase == "future")
    assert (
        f"{first_root.stage_unit}.lock-choices"
        in steps[parent.task_id].dependency_step_ids
    )
    assert (
        f"{first_root.stage_unit}.join-parent"
        in steps[future.task_id].dependency_step_ids
    )
    assert (
        type(records[FiniteResponseLawAssignedEvaluationProjectionConfig])
        is FiniteResponseLawAssignedEvaluationProjectionConfig
    )
    assert (
        type(records[FiniteResponseLawAssignedEvaluationCompletionConfig])
        is FiniteResponseLawAssignedEvaluationCompletionConfig
    )
    assert type(records[FiniteResponseLawEvaluationRevealConfig]) is FiniteResponseLawEvaluationRevealConfig
    assert (
        type(records[QualifiedSourceUseExperiment]) is QualifiedSourceUseExperiment
    )
    assert (
        type(records[QualifiedSourceUseSubstrateBinding])
        is QualifiedSourceUseSubstrateBinding
    )

    # Port objects are synthetic and no task runner is executed.
    class _SyntheticRuntime:
        prepared_store = object()

    runtime = _SyntheticRuntime()
    registry = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    for binding, manifest, required in (
        (
            executable_binding.SOURCE_BINDING,
            discovery.SOURCE_CAPABILITY,
            (
                source,
                records[QualifiedSourceUseExperiment],
                records[QualifiedSourceUseSubstrateBinding],
            ),
        ),
        (
            executable_binding.PROJECTION_BINDING,
            discovery.PROJECTION_CAPABILITY,
            (records[FiniteResponseLawAssignedEvaluationProjectionConfig],),
        ),
        (
            executable_binding.EVALUATION_BINDING,
            discovery.EVALUATION_CAPABILITY,
            (records[FiniteResponseLawAssignedEvaluationCompletionConfig],),
        ),
        (
            executable_binding.CONTROL_BINDING,
            discovery.CONTROL_CAPABILITY,
            (control,),
        ),
        (
            executable_binding.REVEAL_BINDING,
            discovery.REVEAL_CAPABILITY,
            (records[FiniteResponseLawEvaluationRevealConfig],),
        ),
    ):
        ports = tuple(
            ExecutablePlatformPort(key, runtime)
            for key in binding.required_platform_port_keys
        )
        provider = registry.provider_factory(binding.binding_id).build_provider(
            registry=CapabilityRegistry(
                f"synthetic.{manifest.capability_key}.registry", (manifest,)
            ),
            records=required,
            platform_ports=ports,
        )
        assert provider.manifest == manifest
    assert len(source.roots) == 64
    from empirical_lawhood.adapters.composition.finite_response_law.stage_input import inspect_consumer_providers

    all_keys = {
        key
        for binding in registry.aggregate.bindings
        if binding.capability_key in {step.capability_key for step in steps.values()}
        for key in binding.required_platform_port_keys
    }
    owners = inspect_consumer_providers(
        bundle, tuple(ExecutablePlatformPort(key, runtime) for key in sorted(all_keys))
    )
    assert len(owners) == 5
    assert sum(row["native_tasks_planned"] for row in owners) == 1280


def test_assigned_prospective_evaluation_continuation_reuses_all_original_roots_and_prefixes() -> None:
    assignment, source, control = _assigned_prospective_evaluation()
    prefixes = []
    for invocation in native_invocations(source):
        if invocation.phase != "prefix":
            continue
        stem = invocation.task_id
        artifacts = tuple(
            sorted(
                (
                    _artifact(
                        f"{stem}.result", FiniteResponseLawAssignedEvaluationTaskResult.SCHEMA
                    ),
                    _artifact(f"{stem}.stage", LinkedCampaignStageEnvelope.SCHEMA),
                    ArtifactIdentity(
                        f"{stem}.native",
                        "synthetic-development",
                        NATIVE_PAIR_SCHEMA,
                        "0" * 64,
                        "application/x-hdf5",
                        1,
                    ),
                ),
                key=lambda value: value.artifact_id,
            )
        )
        prefixes.append(
            ProspectiveRetainedPredecessor(
                stem,
                invocation.root.physical_unit_id,
                Q_CLOCK,
                Decimal(invocation.clocks[1]),
                tuple(f"{invocation.root.root_id}.flh-project.r{v}" for v in (1, 2)),
                artifacts,
                ArtifactIdentity(
                    f"{stem}.receipt",
                    "synthetic-development",
                    CanonicalTaskReceipt.SCHEMA,
                    "0" * 64,
                    "application/json",
                    1,
                ),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            )
        )
    retention = FiniteResponseLawEvaluationRetention(
        source,
        "synthetic-donor-run",
        ArtifactIdentity(
            "synthetic-donor-closeout",
            "synthetic-development",
            'empirical-lawhood/document/json',
            "0" * 64,
            "application/json",
            1,
        ),
        tuple(prefixes),
    )
    assert len(retention.prefixes) == 64
    assert (
        decode_canonical_bytes(
            retention.canonical_bytes(), FiniteResponseLawEvaluationRetention, maximum_bytes=1024**2
        )
        == retention
    )
    bundle = build_native_authoring(
        source=source,
        implementation_sha256="0" * 64,
        exposure=ObjectIdentity.from_record(
            "synthetic-development-assignment", assignment
        ),
        exposed_unit_ids=tuple(root.physical_unit_id for root in source.roots),
        exposed_seed_ids=native_seed_ids(source),
        new_seed_ids=(),
        control=control,
        retention=retention,
    )
    records = {type(record): record for record in bundle.payloads}
    assert (
        type(records[ProspectiveRetainedSourceUse]) is ProspectiveRetainedSourceUse
    )
    steps = {
        step.step_id: step
        for step in bundle.standard_context.base.templates[0].protocol.steps
    }
    assert len(steps) == 1796
    assert all(
        steps[row.segment_id].stage.value == "FREEZE" for row in retention.prefixes
    )
    assert len({row.physical_independent_unit_id for row in retention.prefixes}) == 64

    class _SyntheticRuntime:
        prepared_store = object()

    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.provider_factory(
        continuation_binding.SOURCE_BINDING.binding_id
    )
    provider = factory.build_provider(
        registry=CapabilityRegistry(
            "synthetic-continuation-source-registry",
            (continuation_discovery.SOURCE_CAPABILITY,),
        ),
        records=(
            source,
            records[ProspectiveRetainedSourceUse],
            records[ProspectiveRetainedSourceUseBinding],
        ),
        platform_ports=(
            ExecutablePlatformPort(
                continuation_binding.SOURCE_BINDING.required_platform_port_keys[0],
                _SyntheticRuntime(),
            ),
        ),
    )
    assert len(provider.invocations) == 1216

    projections = tuple(
        FiniteResponseLawRetainedProjection(
            f"{root.root_id}.flh-project.r{view}",
            _artifact(
                f"{root.root_id}.flh-project.r{view}.result",
                FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA,
            ),
            ArtifactIdentity(
                f"{root.root_id}.flh-project.r{view}.receipt",
                "synthetic-development",
                CanonicalTaskReceipt.SCHEMA,
                "0" * 64,
                "application/json",
                1,
            ),
            _identity(
                f"{root.root_id}.flh-project.r{view}.receipt",
                CanonicalTaskReceipt.SCHEMA,
            ),
        )
        for root in source.roots
        for view in (1, 2)
    )
    with pytest.raises(ValueError, match="original root/view custody"):
        FiniteResponseLawAssignedRetainedCompletionConfig(
            f"{PROSPECTIVE_EVALUATION_COMPLETION_PREFIX}.config",
            source,
            "synthetic-continuation-run",
            "0" * 40,
            projections[:-1],
            control,
            retention,
        )
    completion = FiniteResponseLawAssignedRetainedCompletionConfig(
        f"{PROSPECTIVE_EVALUATION_COMPLETION_PREFIX}.config",
        source,
        "synthetic-continuation-run",
        "0" * 40,
        projections,
        control,
        retention,
    )
    assert (
        decode_canonical_bytes(
            completion.canonical_bytes(),
            FiniteResponseLawAssignedRetainedCompletionConfig,
            maximum_bytes=1024**2,
        )
        == completion
    )
    completed = build_completion_authoring(
        config=completion,
        implementation_sha256="0" * 64,
        exposure=ObjectIdentity.from_record(
            "synthetic-development-assignment", assignment
        ),
        exposed_unit_ids=tuple(root.physical_unit_id for root in source.roots),
        exposed_seed_ids=native_seed_ids(source),
    )
    completion_steps = completed.standard_context.base.templates[0].protocol.steps
    assert len(completion_steps) == 3
    assert {step.step_id for step in completion_steps} == {
        completion.freeze_task_id,
        completion.aggregate_task_id,
        completion.adjudicate_task_id,
    }
    completion_provider = (
        EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.provider_factory(
            COMPLETION_BINDING.binding_id
        ).build_provider(
            registry=CapabilityRegistry(
                "synthetic-retained-completion-registry", (COMPLETION_CAPABILITY,)
            ),
            records=(completion,),
            platform_ports=(
                ExecutablePlatformPort(
                    COMPLETION_BINDING.required_platform_port_keys[0], object()
                ),
            ),
        )
    )
    assert completion_provider.config == completion
    from empirical_lawhood.adapters.composition.finite_response_law.stage_input import inspect_consumer_providers

    owners = inspect_consumer_providers(
        completed,
        (
            ExecutablePlatformPort(
                COMPLETION_BINDING.required_platform_port_keys[0], object()
            ),
        ),
    )
    assert len(owners) == 1
    assert owners[0]["native_tasks_planned"] == 0
