"""Synthetic saved current readiness, never native or qualification evidence."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from empirical_lawhood.api.finite_operands import original_f_operand_export, preparation_operands_export, saved_matrix_arrays_export, saved_matrix_arrays_import, preparation_bridge_fits
from empirical_lawhood.api.preparation_readiness import current_preparation_readiness
from empirical_lawhood.adapters.methods.finite_response_law.retained_entry_readiness.current_input import current_requests, current_readiness
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import mechanical_kernel
from empirical_lawhood.kernel.provenance import ObjectIdentity
from tests.test_matrix_operand_transport import allocation as allocated
from tests.numerical_provenance_fixtures import PROJECT_ROOT, numerical_plane


def synthetic_operands(tmp_path):
    writer = numerical_plane(tmp_path)
    source_dir, derived_dir, lower_dir = (tmp_path / name for name in ("native", "derived", "lower"))
    for directory in (source_dir, derived_dir, lower_dir):
        directory.mkdir()
    lower = original_f_operand_export(lower_dir, payload_id="test.readiness.lower")
    allocation = allocated(24)
    allocation = replace(allocation, roots=tuple(replace(root, cohort="q2" if index < 8 else "cir1") for index, root in enumerate(allocation.roots)))
    rng = np.random.default_rng(77)
    center, scale = np.asarray(lower.center, dtype=float), np.asarray(lower.scale, dtype=float)
    prefix = center + .01*rng.normal(size=(24,24))*scale
    hold = center + .01*rng.normal(size=(24,24))*scale
    kernel, _ = mechanical_kernel()
    shift = np.einsum("stk,rkj->rstj", kernel, .0001*rng.normal(size=(24,4,24)))*scale
    baseline = prefix[:,None,None] + np.linspace(0,1,26)[None,None,:,None]*(hold-prefix)[:,None,None]
    features = np.repeat((baseline+shift)[:,:,None],2,axis=2)
    z = features[:,:,:,-1].transpose(0,1,3,2).copy()
    mean = lower.predict(z[...,0].reshape(-1,24)).mean.reshape(24,9,4,8)
    positions = np.broadcast_to(rng.normal(size=(24,1,2,1,2,3,4,4)),(24,9,2,26,2,3,4,4)).astype(np.complex128).copy()
    momenta = np.broadcast_to(rng.normal(size=(24,1,2,1,2,3,4,4)), positions.shape).astype(np.complex128).copy()
    arrays = dict(x=np.repeat(prefix[...,None],2,axis=-1), z=z,features=features,
        y=np.broadcast_to(mean[...,None,None],(24,9,4,8,2,2)).copy(),work=np.zeros((24,9,2)),
        positions=positions,momenta=momenta,requested_ticks=np.broadcast_to(4096+16*np.arange(26,dtype=np.int64),(24,26)).copy())
    producer = ObjectIdentity("test.synthetic.source", "empirical-lawhood/tests/synthetic-readiness", "1.0.0", sha256(b"exposed test fixture").hexdigest())
    saved_matrix_arrays_export(source_dir,operand_id="test.synthetic.native",allocation=allocation,arrays=arrays,producers=(producer,),original_f=lower.identity)
    preparation_operands_export(derived_dir,native_manifest_path=source_dir/"arrays.canonical.json",native_arrays_path=source_dir/"arrays.npz",allocation_path=source_dir/"allocation.canonical.json",original_f=lower,operand_id="test.synthetic.derived",artifact_writer=writer,project_root=PROJECT_ROOT)
    manifest, allocation, derived = saved_matrix_arrays_import(derived_dir/"arrays.canonical.json", arrays_path=derived_dir/"arrays.npz", allocation_path=derived_dir/"allocation.canonical.json",artifact_writer=writer)
    return lower,manifest,allocation,derived,derived_dir


def test_current_request_order_is_exact_readiness_draw_order():
    allocation = allocated(24)
    directions, requirements = current_requests(allocation)
    rng = np.random.Generator(np.random.PCG64(allocation.roots[0].seed_for("requests")))
    np.testing.assert_array_equal(directions[0],rng.integers(0,4,(256,2)))
    np.testing.assert_array_equal(requirements[0,:,0],rng.uniform(.02,.12,256))
    np.testing.assert_array_equal(requirements[0,:,1],rng.uniform(.02,.06,256))


def test_current_complete_screen_has_no_historical_failure_count_gate(tmp_path):
    lower,manifest,allocation,arrays,_ = synthetic_operands(tmp_path)
    report,store,result,_,checked = current_readiness(manifest=manifest,allocation=allocation,arrays=arrays,
        lower=lower,bridge_fits=preparation_bridge_fits(arrays),report_id="test.current.readiness")
    assert report.known_failure_cells == 0
    assert checked["verified"] and checked["model_solves"] == checked["native_calls"] == 0
    assert report.follow_on_status == "UNENTERED"
    assert result["follow_on_status"] == "UNENTERED"
    assert store.arrays["conjuncts"].shape == (24,9,6)
    assert arrays["conjuncts"].shape == (24,9,7)
    assert not store.arrays["entry"][store.arrays["known"]].any() if store.arrays["known"].any() else True


@pytest.mark.parametrize("fault",("cohort","tick","missing","lower"))
def test_current_readiness_rejects_input_substitution_before_fits(tmp_path,monkeypatch,fault):
    lower,manifest,allocation,arrays,_ = synthetic_operands(tmp_path)
    arrays=dict(arrays)
    if fault=="cohort":
        allocation=replace(allocation,roots=(replace(allocation.roots[0],cohort="cir1"),*allocation.roots[1:]))
    elif fault=="tick":
        arrays["requested_ticks"]=arrays["requested_ticks"].copy(); arrays["requested_ticks"][-1,-1]+=1
    elif fault=="missing":
        del arrays["momenta"]
    else:
        lower=replace(lower,payload_id="test.other.lower")
    def forbidden(*args,**kwargs):
        raise AssertionError("readiness fitting entered before input checks")
    monkeypatch.setattr("empirical_lawhood.adapters.methods.finite_response_law.retained_entry_readiness.current_input.compute",forbidden)
    with pytest.raises(ValueError):
        current_readiness(manifest=manifest,allocation=allocation,arrays=arrays,lower=lower,bridge_fits=preparation_bridge_fits(arrays),report_id="test.bad")


def test_public_current_screen_saves_complete_operands_without_native(tmp_path):
    lower,_,_,_,directory=synthetic_operands(tmp_path)
    output=tmp_path/"screen"
    result=current_preparation_readiness(manifest_path=directory/"arrays.canonical.json",arrays_path=directory/"arrays.npz",
        allocation_path=directory/"allocation.canonical.json",original_f=lower,output_dir=output,report_id="test.public.readiness",artifact_writer=numerical_plane(tmp_path),project_root=PROJECT_ROOT)
    assert result["native_calls"] == result["issue_calls"] == 0
    assert (output/"readiness-report.canonical.json").is_file()
    assert len(tuple(output.glob("arrays-*")))>=2
    with pytest.raises(FileExistsError):
        current_preparation_readiness(manifest_path=directory/"arrays.canonical.json",arrays_path=directory/"arrays.npz",
            allocation_path=directory/"allocation.canonical.json",original_f=lower,output_dir=output,report_id="test.public.readiness",artifact_writer=numerical_plane(tmp_path),project_root=PROJECT_ROOT)
