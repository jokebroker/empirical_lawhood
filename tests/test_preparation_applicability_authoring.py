"""Full-size current candidates and strict before-effect Q guards, no campaign."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from tests.preparation_applicability_fixtures import stage_and_source,allocation
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import ConstructedPreparationReport
from empirical_lawhood.adapters.methods.preparation_applicability.exposure import effective_seed_ids
from empirical_lawhood.adapters.composition.phase_authoring import PhaseContextProvider
from empirical_lawhood.api.authoring_handoff import create_authoring_api,write_exclusive_record
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.compiler import compile_preissue_run_plan,lower_run_plan
from empirical_lawhood.runtime.study_issue import project_standard_study_extensions


def qualified_report(stage,source,qualified=True):
    roots=allocation(8,constructed=True,phase="q")
    return ConstructedPreparationReport("Q",tuple(root.root_id for root in roots.roots),("a"*64,)*8,
        True,qualified,"QUALIFIED" if qualified else "CONSTRUCTOR_NOT_QUALIFIED",
        (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),Decimal(0),0,0,Decimal(1),Decimal(0),(Decimal(0),Decimal(0)),
        lower_sha256=stage.upstream[0].artifact.sha256,design_sha256=stage.design.fingerprint(),source_sha256=source.fingerprint(),
        allocation_sha256=roots.fingerprint(),effective_seed_ids=effective_seed_ids(roots),
        constructor_crossings=(qualified,)*8,numerical_maxima=(Decimal(1),)*24)


def ordinary_stage(phase):
    from empirical_lawhood.adapters.methods.preparation_applicability.config import PreparationApplicabilityStage,PreparationApplicabilitySource
    from empirical_lawhood.adapters.methods.preparation_applicability.records import PreparationApplicabilityDesign
    q,source=stage_and_source()
    source=PreparationApplicabilitySource(source.implementation_plan_sha256,source.dependency_lock_sha256,source.native_implementation,source.conformance_receipt)
    return PreparationApplicabilityStage(f"exposed.synthetic.ordinary.{phase.lower()}",PreparationApplicabilityDesign(),phase,
        allocation(32 if phase=="D" else 64,phase=phase.lower()),q.upstream,q.exposure,ObjectIdentity.from_record("exposed.synthetic.source",source)),source


@pytest.mark.parametrize("family,phase,count",[("constructed","Q",50),("constructed","E",194),("ordinary","D",194),("ordinary","E",386)])
def test_full_size_current_candidate_outputs_resources_and_factories(tmp_path,family,phase,count):
    if family=="constructed":
        from empirical_lawhood.adapters.composition.constructed_preparation_applicability.authoring import build_authoring
        from empirical_lawhood.adapters.composition.constructed_preparation_applicability.resources import resource_envelope
        stage,source=stage_and_source(phase)
        report=qualified_report(stage,source) if phase=="E" else None
        if report:
            stage=replace(stage,upstream=(stage.upstream[0],replace(stage.upstream[1],artifact=replace(stage.upstream[1].artifact,sha256=report.fingerprint(),size_bytes=len(report.canonical_bytes())))))
        bundle=build_authoring(stage,source,implementation_sha256="c"*64,specification_sha256="1"*64,qualified_report=report)
    else:
        from empirical_lawhood.adapters.composition.preparation_applicability.authoring import build_authoring
        from empirical_lawhood.adapters.composition.preparation_applicability.resources import resource_envelope
        stage,source=ordinary_stage(phase)
        bundle=build_authoring(stage,source,implementation_sha256="c"*64,specification_sha256="1"*64)
    authoring=write_exclusive_record(tmp_path,"authoring.json",bundle.authoring)
    payloads=tuple(write_exclusive_record(tmp_path,f"p-{i}.json",row) for i,row in enumerate(bundle.payloads))
    decoders=tuple(write_exclusive_record(tmp_path,f"d-{i}.json",row) for i,row in enumerate(bundle.decoder_registrations))
    api=create_authoring_api(repo_root=Path.cwd(),candidate_context_provider=PhaseContextProvider(bundle),candidate_capability_catalog=bundle.catalog)
    compiled=api.compile_candidate(CompileCandidateRequest(authoring,payloads,decoders))
    assert compiled.succeeded,(compiled.reason_codes,compiled.errors)
    report=compiled.payload.report
    candidate=report.candidate
    extensions,_,_=project_standard_study_extensions(candidate=candidate,extension_payload_bytes=tuple(row.canonical_bytes() for row in bundle.payloads),decoder_registrations=bundle.decoder_registrations,extension_materializations=report.extension_materializations)
    base=candidate.base_candidate.base_candidate
    resources=resource_envelope(stage,base.protocol,ObjectIdentity.from_record(extensions.issued_extension_set_id,extensions))
    projected=compile_preissue_run_plan(run_plan_id=stage.config_id,candidate_record=base,candidate=ObjectIdentity.from_record(candidate.candidate_id,candidate),registry=bundle.standard_context.base.registry,implementation_commit="0"*40,issued_extension_set=extensions,resource_envelope=resources,jit_census=None,jit_manifest=None)
    plan=lower_run_plan(projected,bundle.standard_context.base.registry)
    assert len(plan.tasks)==len(resources.task_cells)==count
    assert sum(cell.native_simulator_launch for cell in resources.task_cells)==3*len(stage.root_ids)
    assert {task.task_id for task in plan.tasks}=={cell.task_id for cell in resources.task_cells}
    from empirical_lawhood.adapters.composition.generated_executable_bindings import EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    for capability in bundle.standard_context.base.registry.capabilities:
        assert any(binding.capability_key==capability.capability_key and EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.factory(binding.binding_id).binding==binding for binding in EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.aggregate.bindings)
    if family=="constructed" and phase=="E":
        for task in plan.tasks:
            if task.task_id.startswith("ap.prefix."):
                assert any(edge.logical_artifact_id==stage.upstream[1].artifact.artifact_id for edge in task.scientific_inputs)


@pytest.mark.parametrize("failure",("missing","unqualified","numerical","wrong-lower","overlap"))
def test_native_E_guard_precedes_every_prefix_contact(monkeypatch,failure):
    from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.provider import ConstructedPreparationNativeRunner
    from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import ConstructedPreparationNativeConfig
    from empirical_lawhood.runtime.execution import WorkerInputKind
    stage,source=stage_and_source("E")
    report=qualified_report(stage,source,qualified=failure!="unqualified")
    if failure=="numerical":
        report=replace(report,numerical_maxima=(*report.numerical_maxima[:-1],Decimal("1.0001")),constructor_qualified=False,disposition="CONSTRUCTOR_NOT_QUALIFIED")
    elif failure=="wrong-lower":
        report=replace(report,lower_sha256="f"*64)
    elif failure=="overlap":
        report=replace(report,effective_seed_ids=effective_seed_ids(stage.allocation))
    module="empirical_lawhood.adapters.simulators.constructed_preparation_applicability.provider"
    def prerequisite(*args,**kwargs):
        if failure=="missing":
            raise ValueError("required qualification missing")
        return report
    monkeypatch.setattr(module+".upstream",prerequisite)
    monkeypatch.setattr(module+".config_input",lambda *args:None)
    def forbidden(*args,**kwargs):
        raise AssertionError("E prefix contacted before Q gate")
    monkeypatch.setattr(module+".acquire_prefix",forbidden)
    context=SimpleNamespace(task_id=f"ap.prefix.{stage.root_ids[0]}",input_ports=(SimpleNamespace(kind=WorkerInputKind.EXTERNAL,payload_schema=source.SCHEMA,read=source.canonical_bytes),))
    runner=ConstructedPreparationNativeRunner(ConstructedPreparationNativeConfig(stage),object(),SimpleNamespace(task=lambda task:nullcontext()))
    with pytest.raises(ValueError,match="qualification|qualified"):
        runner.execute(context)
