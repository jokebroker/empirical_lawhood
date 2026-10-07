"""Synthetic complete D32/E64 diagnostics; never native/qualification evidence."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from empirical_lawhood.api.finite_operands import original_f_operand_export, saved_matrix_arrays_export
from empirical_lawhood.api.preparation_screens import ordinary_preparation_screens
from empirical_lawhood.adapters.methods.preparation_applicability.headroom import opportunity, prefix_forecast
from empirical_lawhood.adapters.methods.preparation_applicability.records import numbers
from empirical_lawhood.adapters.methods.preparation_applicability.readout import PreparationApplicabilityScreenReport
from empirical_lawhood.kernel.provenance import ObjectIdentity
from tests.test_preparation_applicability_authoring import ordinary_stage


def _banks(tmp_path):
    lower_directory=tmp_path/"lower";lower_directory.mkdir()
    lower=original_f_operand_export(lower_directory,payload_id="test.synthetic.screens.original-f")
    rng=np.random.default_rng(501)
    for phase,count in (("D",32),("E",64)):
        stage,_=ordinary_stage(phase)
        stage=replace(stage,upstream=(replace(stage.upstream[0],artifact=replace(stage.upstream[0].artifact,
            sha256=lower.fingerprint(),size_bytes=len(lower.canonical_bytes()))),))
        prefix=rng.normal(size=(count,24,2))
        handoff=np.broadcast_to(np.asarray(lower.center,dtype=float)[None,None,:,None],(count,3,24,2)).copy()
        handoff[:,0,17,:]+=prefix[:,0,:]*float(lower.scale[17])
        maxima=np.zeros((count,3,7));work=np.zeros((count,3,2))
        arrays=dict(prefix=prefix,handoff=handoff,maxima=maxima,work=work)
        services=((),(),())
        if phase=="E":
            maxima[:4,0,0]=2
            arrays["success"]=np.ones((64,3,256,2),dtype=bool)
            services=((60*256,64*256,64*256),(64*256,)*3,(0,)*3)
        valid=tuple(map(int,(maxima<=1).all(axis=2).sum(axis=0)))
        support=tuple(map(int,(maxima[:,:,0]<=1).sum(axis=0)))
        summary=PreparationApplicabilityScreenReport(phase,stage.root_ids,
            tuple(sha256(root.encode()).hexdigest() for root in stage.root_ids),True,
            "EXPOSED_SCREEN_OPERANDS_READY",valid,support,*services,numbers(handoff),
            lower.fingerprint(),stage.design.fingerprint(),stage.source.object_fingerprint,stage.allocation.fingerprint())
        destination=tmp_path/phase;destination.mkdir()
        (destination/"stage.canonical.json").write_bytes(stage.canonical_bytes())
        (destination/"screen-report.canonical.json").write_bytes(summary.canonical_bytes())
        saved_matrix_arrays_export(destination,operand_id=f"test.synthetic.{phase.lower()}.screen",allocation=stage.allocation,
            arrays=arrays,producers=(ObjectIdentity.from_record(stage.config_id,stage),
                ObjectIdentity.from_record(f"{stage.config_id}.report",summary)),original_f=lower.identity)
    return lower


def test_public_complete_96_root_screens_keep_distinct_roles_and_oracle_ceiling(tmp_path):
    lower=_banks(tmp_path)
    result=ordinary_preparation_screens(development_directory=tmp_path/"D",evaluation_directory=tmp_path/"E",
        original_f=lower,output_dir=tmp_path/"result")
    assert result["ordinary_independent_roots"]==96
    assert result["development_roots"]==32 and result["evaluation_roots"]==64
    assert result["headroom"]["n_independent_roots"]==64
    assert result["headroom"]["full_valid_counts"]==[60,64,64]
    assert result["headroom"]["covered_service_counts"]==[60*256,64*256,64*256]
    assert result["headroom"]["oracle_uses_actual_outcomes_and_is_not_an_available_policy"]
    assert result["causal_prefix_forecast"]["training_only_scales"]
    assert result["ceiling"]=="EXPOSED_DEVELOPMENT_NONPROMOTABLE" and result["follow_on_status"]=="UNENTERED"
    assert (tmp_path/"result"/"ordinary-screens.json").is_file()


def test_loo_forecast_excludes_held_root_label_and_scales():
    rng=np.random.default_rng(604)
    prefix=rng.normal(size=(96,24))
    prefix[0]*=1000
    targets=rng.normal(size=96)
    result=prefix_forecast(prefix,targets,-1e10)
    forecast=result["actual_failure_root_forecasts"][0]
    training=prefix[1:]
    center,scale=training.mean(axis=0),training.std(axis=0)
    design=np.column_stack((np.ones(95),(training-center)/scale))
    penalty=np.eye(25)*10;penalty[0,0]=0
    expected=np.r_[1,(prefix[0]-center)/scale]@np.linalg.solve(design.T@design+penalty,design.T@targets[1:])
    assert forecast==pytest.approx(expected)
    changed=targets.copy();changed[0]+=10000
    again=prefix_forecast(prefix,changed,-1e10)
    assert again["actual_failure_root_forecasts"][0]==forecast
    assert again["actual_failure_root_forecasts"][1]!=result["actual_failure_root_forecasts"][1]


def test_existing_output_refuses_before_any_input_read_or_fit(tmp_path,monkeypatch):
    output=tmp_path/"existing";output.mkdir()
    def forbidden(*args,**kwargs):
        raise AssertionError("saved input read before output preflight")
    monkeypatch.setattr("empirical_lawhood.api.preparation_screens.saved_matrix_arrays_import",forbidden)
    with pytest.raises(FileExistsError,match="fresh"):
        ordinary_preparation_screens(development_directory=Path("absent"),evaluation_directory=Path("absent"),original_f=None,output_dir=output)


@pytest.mark.parametrize("fault",("incomplete","swapped-roles","wrong-law","unrecorded-service"))
def test_saved_screens_reject_missing_roles_law_and_modified_report(tmp_path,fault):
    lower=_banks(tmp_path)
    development,evaluation=tmp_path/"D",tmp_path/"E"
    if fault=="incomplete":
        (evaluation/"arrays.npz").unlink()
    elif fault=="swapped-roles":
        development,evaluation=evaluation,development
    elif fault=="wrong-law":
        lower=replace(lower,payload_id="test.substituted.lower")
    else:
        from empirical_lawhood.kernel.decoding import decode_canonical_bytes
        path=evaluation/"screen-report.canonical.json"
        summary=decode_canonical_bytes(path.read_bytes(),PreparationApplicabilityScreenReport,maximum_bytes=4*1024**2)
        path.write_bytes(replace(summary,joint_counts=(0,0,0),covered_counts=(0,0,0)).canonical_bytes())
    with pytest.raises((OSError,ValueError)):
        ordinary_preparation_screens(development_directory=development,evaluation_directory=evaluation,original_f=lower,output_dir=tmp_path/"result")
    assert not (tmp_path/"result").exists()


def test_opportunity_does_not_count_views_or_allow_nested_root_axes():
    with pytest.raises(ValueError,match="census"):
        opportunity(np.zeros((64,3,7,2)),np.zeros((64,3,256,2),dtype=bool))
    with pytest.raises(ValueError,match="census"):
        prefix_forecast(np.zeros((95,24)),np.zeros(95),1)
