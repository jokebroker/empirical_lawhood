"""Actual full-size public custody/issue plumbing; fictional outcomes and signers.

Only dirty-checkout Git metadata and disposable physical-location seams are mocked.
Native conformance runs one short interval once. All Q reports/signers below
are explicitly software fixtures and can never qualify an actual study.
"""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import pytest

from empirical_lawhood.api.finite_operands import original_f_operand_export
from empirical_lawhood.api.preparation_conformance import preparation_native_conformance
from empirical_lawhood.api.preparation_inputs import publish_preparation_input, inspect_preparation_exposure, prepare_preparation_allocation, create_preparation_selection, bind_constructed_preparation_q, bind_preparation_report, read_preparation_report
from empirical_lawhood.api.preparation_result_access import preparation_result_access
from empirical_lawhood.api.preparation_inputs import bind_ordinary_preparation_panel
from empirical_lawhood.api.integration_handoffs import load_integration_handoff, prove_integration_handoff, integration_api_inputs
from empirical_lawhood.api.composition import create_cli_api
from empirical_lawhood.api.results import CompileCandidateRequest, OperationStatus
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import ConstructedPreparationReport
from empirical_lawhood.adapters.methods.preparation_applicability.exposure import effective_seed_ids
from empirical_lawhood.adapters.methods.preparation_applicability.config import preparation_run_id
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind
from empirical_lawhood.runtime.artifacts import ExternalRootContract
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile, OperatorStorageAccessMode
from tests.finite_response_rerun_fixtures import publish_current_issue, publish_current_step_output

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def actual_short_conformance(tmp_path_factory):
    directory=tmp_path_factory.mktemp("actual-preparation-conformance")
    ordinary=prepare_preparation_allocation(allocation_id="test.conformance.ordinary",cohort_namespace="test.conformance",phase="D",master_seed=93581)
    constructed=prepare_preparation_allocation(allocation_id="test.conformance.constructed",cohort_namespace="test.conformance-constructed",phase="Q",master_seed=93582,constructed=True)
    return preparation_native_conformance(roots=(ordinary.roots[0],replace(ordinary.roots[1],cohort="cir1"),constructed.roots[0]),
        output_dir=directory/"short",conformance_id="test.actual.short-native-conformance")


def _setup(tmp_path,monkeypatch,conformance):
    store=tmp_path/"software-store";store.mkdir()
    plane=ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract("synthetic.preparation-store",
        "fictional preparation software store",str(store),"/",OperatorStorageProfile.SCHEMA,1,None,None,())))
    lower_dir=store/"original-f";lower_dir.mkdir()
    lower=original_f_operand_export(lower_dir,payload_id="synthetic.preparation.original-f")
    inspection=inspect_preparation_exposure(prior_inputs=(),inspection_id="synthetic.complete-known-exposure",original_f=lower)
    publications={key:publish_preparation_input(record=value,key=key,logical_artifact_id=f"synthetic.input.{key}",
        relative_root=f"synthetic/inputs/{key}",receipt_id=f"synthetic.import.{key}",artifact_writer=plane)
        for key,value in (("lower",lower),("exposure",inspection),("conformance",conformance))}
    closure=ImplementationSourceClosure("synthetic.preparation.current-source",SourceClosureKind.CLEAN_GIT_COMMIT,
        "0"*40,"d"*64,"e"*64,True)
    from empirical_lawhood.api import preparation_applicability as ordinary, constructed_preparation_applicability as constructed
    for application in (ordinary,constructed):
        monkeypatch.setattr(application,"capture_clean_target_closure",lambda *args:closure)
        monkeypatch.setattr(application,"resolve_authoring_directory",lambda directory,**kwargs:(directory,plane.root.contract))
    from empirical_lawhood.api import composition, integration_handoffs
    from empirical_lawhood.infrastructure import source_origin
    run_command=source_origin.run_bounded_command
    def current_fixture_metadata(arguments,**kwargs):
        result=run_command(arguments,**kwargs)
        if arguments[:2]==("git","ls-files"):
            # The real executing-origin checks remain active. This source test
            # runs before the new owners are staged and asserts no clean HEAD.
            result=replace(result,stdout=b"\0".join(path.relative_to(ROOT).as_posix().encode()
                for path in sorted((ROOT/"src"/"empirical_lawhood").rglob("*.py")))+b"\0")
        return result
    monkeypatch.setattr(source_origin,"run_bounded_command",current_fixture_metadata)
    monkeypatch.setattr(integration_handoffs,"resolve_authoring_directory",lambda directory,**kwargs:(directory,plane.root.contract))
    monkeypatch.setattr(composition,"resolve_authoring_directory",lambda directory,**kwargs:(directory,plane.root.contract))
    monkeypatch.setattr(composition,"resolve_external_root_contract",lambda *args,**kwargs:plane.root.contract)
    scratch=store/"software-scratch";scratch.mkdir()
    monkeypatch.setattr(composition,"resolve_scientific_scratch_contract",lambda *args,**kwargs:replace(plane.root.contract,
        storage_root_id="synthetic.preparation-scratch",canonical_path=str(scratch)))
    def forbidden(*args,**kwargs):
        raise AssertionError("full-size author/load/provider/issue proof contacted native source")
    for family in ("preparation_applicability","constructed_preparation_applicability"):
        for owner in ("native","provider"):
            for function in ("acquire_prefix","acquire_phase"):
                monkeypatch.setattr(f"empirical_lawhood.adapters.simulators.{family}.{owner}.{function}",forbidden)
    return plane,publications,ordinary,constructed


def _selection(phase,constructed,publications,qualification=None):
    allocation=prepare_preparation_allocation(allocation_id=f"synthetic.{('constructed' if constructed else 'ordinary')}.{phase.lower()}.allocation",
        cohort_namespace="synthetic.current.constructed" if constructed else "synthetic.current.ordinary",
        phase=phase,master_seed=935862,constructed=constructed,proposed_unrun=constructed)
    return create_preparation_selection(root=ROOT,config_id=f"synthetic.current.{('constructed' if constructed else 'ordinary')}.{phase.lower()}",
        phase=phase,allocation=allocation,constructed=constructed,qualification=qualification,**publications)


def _author_load_prove(application,selection,plane,constructed):
    output=Path(plane.root.contract.canonical_path)/selection.stage.config_id
    author=application.author_constructed_preparation_applicability if constructed else application.author_preparation_applicability
    loader=application.load_constructed_preparation_applicability_handoff if constructed else application.load_preparation_applicability_handoff
    prover=application.prove_constructed_preparation_applicability_handoff if constructed else application.prove_preparation_applicability_handoff
    started=perf_counter()
    summary=author(root=ROOT,selection=selection,output_dir=output,artifact_writer=plane)
    authored=perf_counter()
    handoff=loader(directory=output,root=ROOT,storage_profile=None,artifact_writer=plane)
    loaded=perf_counter()
    proof=prover(handoff)
    proved=perf_counter()
    print('public_handoff_seconds',{'family':'constructed-preparation-applicability' if constructed else 'preparation-applicability',
        'phase':selection.stage.phase,'independent_roots':len(selection.stage.root_ids),
        'tasks':len(handoff.projection.tasks),'outputs':sum(len(task.outputs) for task in handoff.projection.tasks),
        'author':authored-started,'load':loaded-authored,'prove':proved-loaded},flush=True)
    assert prover(handoff)==proof
    assert summary["source_contacted"] is False and summary["native_tasks_executed"]==0
    assert proof["native_tasks_executed"]==0
    assert summary["task_count"]==proof["task_count"]==len(handoff.resources.task_cells)
    public=load_integration_handoff(directory=output,root=ROOT,storage_profile=None,artifact_writer=plane)
    public_proof=prove_integration_handoff(public)
    assert prove_integration_handoff(public)==public_proof
    assert public.selected.projection==handoff.projection
    assert public_proof["task_count"]==proof["task_count"]
    assert public_proof["scientific_execution_performed"] is False
    public_inputs=integration_api_inputs(public)
    assert public_inputs["study_bundle_registry"]==handoff.bundle.standard_context.base.registry
    assert public_inputs["executable_platform_ports"]==public.selected.platform_ports
    profile=OperatorStorageProfile("synthetic.preparation-cli","external-filesystem","1.0.0",
        "/dev/shm/synthetic-preparation-cli","/dev/shm","artifacts","scratch",OperatorStorageAccessMode.READ_WRITE,
        None,None,(),"strict-mount-contained-no-symlink",1,(),1,False)
    api=create_cli_api(repo_root=ROOT,operator_storage_profile=profile,authoring_dir=output)
    compiled=api.compile_candidate(CompileCandidateRequest(output/"authoring.json",
        tuple(output/f"payload-{index:02d}.json" for index in range(len(handoff.bundle.payloads))),
        tuple(output/f"decoder-{index:02d}.json" for index in range(len(handoff.bundle.decoder_registrations)))))
    assert compiled.status is OperationStatus.SUCCEEDED,compiled.errors
    assert compiled.payload.report.candidate.base_candidate.base_candidate.protocol==handoff.bundle.standard_context.base.templates[0].protocol
    return handoff


def _fictional_qualified_report(selection):
    stage=selection.stage
    return ConstructedPreparationReport("Q",stage.root_ids,tuple(sha256(root.encode()).hexdigest() for root in stage.root_ids),
        True,True,"QUALIFIED",(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),Decimal(0),0,0,Decimal(1),Decimal(0),(Decimal(0),Decimal(0)),
        lower_sha256=stage.upstream[0].artifact.sha256,design_sha256=stage.design.fingerprint(),source_sha256=selection.source.fingerprint(),
        allocation_sha256=stage.allocation.fingerprint(),effective_seed_ids=effective_seed_ids(stage.allocation),
        constructor_crossings=(True,)*8,numerical_maxima=(Decimal(1),)*24)


@pytest.mark.parametrize("constructed,phase,count",((False,"D",194),(False,"E",386),(True,"Q",50),(True,"E",194)))
def test_actual_public_full_size_author_load_provider_and_issue_route(tmp_path,monkeypatch,actual_short_conformance,constructed,phase,count):
    plane,publications,ordinary,boundary=_setup(tmp_path,monkeypatch,actual_short_conformance)
    qualification=None
    if constructed and phase=="E":
        q=_selection("Q",True,publications)
        q_handoff=_author_load_prove(boundary,q,plane,True)
        issue=publish_current_issue(plane,q_handoff,label="preparation-constructor-q")
        receipt,_,published=publish_current_step_output(plane,issue,label="fictional-constructor-report",record=_fictional_qualified_report(q))
        context=preparation_result_access(run_id=issue.publication.run_id,issued_study_id=issue.publication.issued_study.object_id,
            execution_authority_id=issue.publication.execution_authority.object_id,reveal_authority_id=issue.publication.reveal_authority.object_id,writer=plane)
        qualification=bind_constructed_preparation_q(run_id=issue.publication.run_id,receipt_id=receipt.receipt_id,artifact_writer=plane,result_access=context)
        assert qualification.record.constructor_qualified
        # No accuracy/support/coverage gate is added to constructor qualification.
        assert qualification.record.full_valid_counts==(0,0,0)
        retained=bind_preparation_report(run_id=issue.publication.run_id,receipt_id=receipt.receipt_id,constructed=True,artifact_writer=plane,result_access=context)
        assert read_preparation_report(retained,artifact_writer=plane)["independent_roots"]==8
        with pytest.raises(PermissionError,match="CURRENT_AUTHORITY"):
            bind_constructed_preparation_q(run_id=issue.publication.run_id,receipt_id=receipt.receipt_id,artifact_writer=plane,result_access=None)
        wrong=replace(context,execution_authority=replace(context.execution_authority,object_fingerprint="1"*64))
        with pytest.raises(PermissionError,match="IDENTITY"):
            bind_constructed_preparation_q(run_id=issue.publication.run_id,receipt_id=receipt.receipt_id,artifact_writer=plane,result_access=wrong)
        store=ExternalStudyOperationAuthorityStore(plane)
        expired=replace(store.load(context.reveal_authority.object_id),authority_id="synthetic.expired.preparation-reveal",
            issued_at_utc="2026-09-01T00:00:00Z",expires_at_utc="2026-09-02T00:00:00Z")
        store.persist(expired)
        from empirical_lawhood.api.preparation_result_access import authenticate_preparation_result_access
        with monkeypatch.context() as refusal:
            verify=plane.verify_manifest
            def before_protected_bytes(manifest,*args,**kwargs):
                if manifest.materialization.relative_path==published.materialization.relative_path:
                    raise AssertionError('invalid current grant contacted protected scientific bytes')
                return verify(manifest,*args,**kwargs)
            refusal.setattr(plane,'verify_manifest',before_protected_bytes)
            for identity in (context.execution_authority,context.reveal_authority):
                sidecar=plane.root.resolve(store._relative_path(identity.object_id)+'.manifest.json',for_write=False)
                preserved=sidecar.read_bytes()
                sidecar.unlink()
                try:
                    with pytest.raises(KeyError):
                        bind_constructed_preparation_q(run_id=issue.publication.run_id,receipt_id=receipt.receipt_id,
                            artifact_writer=plane,result_access=context)
                finally:
                    sidecar.write_bytes(preserved)
            with pytest.raises(PermissionError):
                bind_constructed_preparation_q(run_id=issue.publication.run_id,receipt_id=receipt.receipt_id,artifact_writer=plane,
                    result_access=replace(context,reveal_authority=ObjectIdentity.from_record(expired.authority_id,expired)))
            with pytest.raises(PermissionError,match='PREPARATION_RESULT_RUN_MISMATCH'):
                bind_constructed_preparation_q(run_id=issue.publication.run_id,receipt_id=receipt.receipt_id,artifact_writer=plane,
                    result_access=replace(context,run_id='synthetic.foreign-preparation-run'))
        # Preserve the public include_package return shape and reread after
        # restoring the actual stored grants; no successful authorization cache.
        plan,package=authenticate_preparation_result_access(context=context,receipt=receipt,
            logical=published.logical,writer=plane,include_package=True)
        assert plan.source_plan.object_id==package.run_plan_id==receipt.run_id
        assert bind_constructed_preparation_q(run_id=issue.publication.run_id,receipt_id=receipt.receipt_id,
            artifact_writer=plane,result_access=context).record==qualification.record
    selection=_selection(phase,constructed,publications,qualification)
    handoff=_author_load_prove(boundary if constructed else ordinary,selection,plane,constructed)
    assert len(handoff.projection.tasks)==count
    issue=publish_current_issue(plane,handoff,label=f"preparation-{('constructed' if constructed else 'ordinary')}-{phase.lower()}")
    assert issue.package.run_plan_id==preparation_run_id(selection.stage)
    assert issue.manifest.candidate.base_candidate.base_candidate.protocol==handoff.bundle.standard_context.base.templates[0].protocol
    if not constructed and phase=="D":
        _retained_ordinary_failure_fixture(selection,plane,issue)


def _retained_ordinary_failure_fixture(selection,plane,issue):
    """Authenticate actual custody for explicitly fictional incomplete cells."""
    from empirical_lawhood.adapters.simulators.preparation_applicability.records import PreparationApplicabilityNativePhase,PreparationApplicabilityStream,PreparationApplicabilityPrefix,PreparationApplicabilityPanel
    from empirical_lawhood.adapters.methods.preparation_applicability.seals import PreparationApplicabilityLowerSeal
    from empirical_lawhood.adapters.methods.preparation_applicability.records import PreparationApplicabilityMeasuredRoot
    from empirical_lawhood.adapters.methods.preparation_applicability.readout import PreparationApplicabilityScreenReport
    from empirical_lawhood.api.preparation_result_access import authenticate_preparation_result_access
    from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
    root=selection.stage.allocation.roots[0]
    phases=tuple(PreparationApplicabilityNativePhase(root.root_id,"prefix",None,None,None,view,
        selection.source.fingerprint(),None,0,4096,0,0,(Decimal(0),Decimal(0)),Decimal(0),Decimal(0),Decimal(0),
        "NUMERICAL_FAILURE","fictional unexecuted software cell","0"*64,
        tuple(PreparationApplicabilityStream(root.seed_for(purpose)) for purpose in ("prefix","prefix-bridge")),
        "",(),"","","",(),None,(),"","",allocation=root) for view in (1,2))
    prefix=PreparationApplicabilityPrefix(root.root_id,phases,None,())
    seal=PreparationApplicabilityLowerSeal(root.root_id,"b"*64,selection.stage.upstream[0].artifact.sha256,
        False,(),(),(),(),(),())
    panel=PreparationApplicabilityPanel(root.root_id,prefix.fingerprint(),"a"*64,(),seal.fingerprint())
    measurement=PreparationApplicabilityMeasuredRoot(root.root_id,prefix.fingerprint(),panel.fingerprint(),seal.lower_sha256,
        False,(),(),(),(),(),(),(),(),())
    receipts=tuple(publish_current_step_output(plane,issue,label=f"fictional-incomplete-{index}",record=record)[0].receipt_id
        for index,record in enumerate((prefix,panel,seal,measurement)))
    context=preparation_result_access(run_id=issue.publication.run_id,issued_study_id=issue.publication.issued_study.object_id,
        execution_authority_id=issue.publication.execution_authority.object_id,reveal_authority_id=issue.publication.reveal_authority.object_id,writer=plane)
    bound=bind_ordinary_preparation_panel(run_id=issue.publication.run_id,root_id=root.root_id,receipt_ids=receipts,
        artifact_writer=plane,result_access=context)
    bound.authenticate(plane,ExternalTaskReceiptStore(plane),result_access_authenticator=authenticate_preparation_result_access)
    assert not bound.measurement.complete
    with pytest.raises(PermissionError,match="CURRENT_AUTHORITY"):
        bind_ordinary_preparation_panel(run_id=issue.publication.run_id,root_id=root.root_id,receipt_ids=receipts,
            artifact_writer=plane,result_access=None)
    report=PreparationApplicabilityScreenReport("D",selection.stage.root_ids,tuple("0"*64 for _ in selection.stage.root_ids),
        False,"UNEVALUABLE",(),(),(),(),(),(),seal.lower_sha256,selection.stage.design.fingerprint(),selection.source.fingerprint(),selection.stage.allocation.fingerprint())
    receipt,_,_=publish_current_step_output(plane,issue,label="fictional-incomplete-ordinary-report",record=report)
    retained=bind_preparation_report(run_id=issue.publication.run_id,receipt_id=receipt.receipt_id,constructed=False,
        artifact_writer=plane,result_access=context)
    summary=read_preparation_report(retained,artifact_writer=plane)
    assert summary["independent_roots"]==32 and summary["disposition"]=="UNEVALUABLE" and not summary["complete"]


@pytest.mark.parametrize("constructed",(False,True))
@pytest.mark.parametrize("destination",("parent-traversal","parent-symlink","broken-symlink","existing","file-parent","outside"))
def test_author_destination_rejection_precedes_upstream_contact(tmp_path,monkeypatch,constructed,destination):
    from empirical_lawhood.api import preparation_applicability as ordinary, constructed_preparation_applicability as boundary
    application=boundary if constructed else ordinary
    store=tmp_path/"guarded";store.mkdir()
    plane=ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract("synthetic.destination-store",
        "fictional destination software store",str(store),"/",OperatorStorageProfile.SCHEMA,1,None,None,())))
    if destination=="parent-traversal":
        output=store/".."/"escaped"
    elif destination in ("parent-symlink","broken-symlink"):
        link=store/"link"
        link.symlink_to(tmp_path/("missing" if destination=="broken-symlink" else "outside"))
        output=link/"escaped"
    elif destination=="existing":
        output=store/"existing";output.mkdir()
    elif destination=="file-parent":
        parent=store/"file";parent.write_text("software fixture")
        output=parent/"escaped"
    else:
        output=tmp_path/"outside"
    def forbidden(*args,**kwargs):
        raise AssertionError("rejected author destination contacted upstream inputs")
    monkeypatch.setattr(application,"_authenticate",forbidden)
    author=application.author_constructed_preparation_applicability if constructed else application.author_preparation_applicability
    with pytest.raises((ValueError,FileExistsError,NotADirectoryError)):
        author(root=ROOT,selection=None,output_dir=output,artifact_writer=plane)
    assert not (tmp_path/"escaped").exists()
    assert not (tmp_path/"outside"/"escaped").exists()
