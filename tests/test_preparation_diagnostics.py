"""Independent posthoc controls; all numerical operands are exposed software fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest

from empirical_lawhood.adapters.methods.constructed_preparation_applicability import diagnostics as d
from empirical_lawhood.adapters.methods.preparation_applicability.science import CAPS, DELTA
from empirical_lawhood.api import preparation_diagnostics as api
from tests.test_preparation_applicability_reducers import _SyntheticLower


@pytest.mark.parametrize("n", [1, 8, 32])
def test_extreme_bounds_match_original_estimands_only_at_extreme_counts(n):
    bounds = d.root_probability_bounds(assigned=n, complete=n, false_admission_roots=0, full_valid_roots=n)
    assert bounds["upper_false_admission"] == pytest.approx(1 - .05**(1/n))
    assert bounds["lower_full_valid"] == pytest.approx(.05**(1/n))
    reversed_counts = d.root_probability_bounds(assigned=n, complete=n, false_admission_roots=n, full_valid_roots=0)
    assert reversed_counts["upper_false_admission"] == 1
    assert reversed_counts["lower_full_valid"] == 0


def test_intermediate_bounds_solve_actual_binomial_tail():
    from math import comb
    result = d.root_probability_bounds(assigned=32, complete=32, false_admission_roots=3, full_valid_roots=25)
    upper, lower = result["upper_false_admission"], result["lower_full_valid"]
    assert sum(comb(32,k)*upper**k*(1-upper)**(32-k) for k in range(4)) == pytest.approx(.05)
    assert sum(comb(32,k)*lower**k*(1-lower)**(32-k) for k in range(25,33)) == pytest.approx(.05)
    assert upper > 1-.05**(1/32) and lower < .05**(1/32)


def test_missing_empty_and_invalid_root_counts_never_infer_zero_failures():
    assert d.root_probability_bounds(assigned=0, complete=0, false_admission_roots=0, full_valid_roots=0)["status"] == "NOT_ESTIMATED"
    result = d.root_probability_bounds(assigned=32, complete=31, false_admission_roots=0, full_valid_roots=31)
    assert result["status"] == "UNEVALUABLE" and result["upper_false_admission"] is None
    with pytest.raises(ValueError):
        d.root_probability_bounds(assigned=32, complete=32, false_admission_roots=33, full_valid_roots=0)


def _operands(value=.04):
    y = np.zeros((3,4,8,2,2))
    y[:,:,0] = value
    work = np.zeros((3,2))
    direction = np.zeros((256,2),dtype=np.int64)
    required = np.full((256,2),.03)
    return y, work, direction, required


def test_signed_word_mapping_and_nonattempt_are_not_actual_deliveries():
    y,work,direction,required = _operands()
    feasible = d.feasible_words(y,work,direction,required)
    assert not feasible[:,:,:,0].any() and feasible[:,:,:,1].all()
    assert np.array_equal(d.aligned_words(direction),np.ones((3,256,2),dtype=int))
    assert np.array_equal(d.aligned_words(direction,16),np.full((3,256,2),5))
    assert not d.selected_success(feasible,np.full((3,256,2),-1)).any()
    y[:,:,0] = -.04
    feasible = d.feasible_words(y,work,direction,required)
    assert feasible[:,:,:,0].all() and not feasible[:,:,:,1].any()


def test_separate_receiver_limits_and_last_nested_view():
    y,work,direction,required = _operands(.12)
    feasible = d.feasible_words(y,work,direction,required)
    assert feasible[:,:,0,1].all() and not feasible[:,:,1,1].any()
    y[2,0,0,1,1] = .0
    feasible = d.feasible_words(y,work,direction,required)
    assert not feasible[2,:,:,1].any()


@pytest.mark.parametrize("change", ["preservation", "work", "numerical"])
def test_finite_word_availability_keeps_preservation_work_and_both_views(change):
    y,work,direction,required = _operands()
    if change == "preservation": y[1,0,2,1,1] = CAPS[0]*2
    elif change == "work": work[1,1] = 33
    else: y[1,0,7,1,1] = DELTA[7]/4
    assert not d.feasible_words(y,work,direction,required)[1,:,:,1].any()


def test_primary_both_and_discrepancy_masks_do_not_compensate_other_conjuncts():
    z = np.zeros((3,24,2)); conjuncts = np.ones((3,7),dtype=bool)
    z[0,0] = (5.999,6.001)
    z[1,0] = (5.999,5.995)
    conjuncts[2,5] = False
    _,masks = d.numerical_masks(_SyntheticLower(),z,conjuncts)
    assert masks["primary"].tolist() == [True,True,False]
    assert masks["both_views"].tolist() == [False,True,False]
    assert masks["view_envelope"].tolist() == [False,False,False]


def test_bypass_changes_only_support_and_keeps_original_choice_immutable():
    class Lower(_SyntheticLower):
        def predict(self,z):
            result=super().predict(z)
            result.mean[:,:,0] = .04
            return result
    y,work,direction,required=_operands()
    z=np.full((3,24,2),7.0)
    choices=np.full((3,256,2),-1)
    before=choices.copy()
    row,_=d.diagnose_root(lower=Lower(),z=z,y=y,work=work,direction=direction,requirement=required,choices=choices)
    assert row["actual"]["joint_pairs"] == [0,0,0]
    assert row["actual"]["false_admissions"] == [0,0,0]
    assert row["support_bypassed_F"]["joint_pairs"] == [256,256,256]
    assert tuple(map(float,row["support_threshold_sweep"])) == d.THRESHOLDS
    np.testing.assert_array_equal(choices,before)


def test_excluded_roots_contribute_zero_but_stay_in_whole_root_denominator():
    rows=[]
    for index in range(8):
        row={key:{"joint_pairs":[0,256,0]} for key in ("actual","support_bypassed_F","fixed8_unlicensed","fixed16_unlicensed","native_word_oracle")}
        row["mask_H_N_P"]={key:[True,index!=7,True] for key in ("primary","both_views","view_envelope")}
        rows.append(row)
    result=d.aggregate_rows(rows,bootstrap_seed=177)
    assert result["n_independent_roots"] == 8
    assert result["numerical_mask_diagnostics"]["primary"]["N_minus_H_fraction"] == 7/8
    assert result["counterfactuals"]["support_bypassed_F"]["N_minus_H_fraction"] == 1
    assert d.whole_root_interval(np.ones(8),seed=177) == [1,1]
    with pytest.raises(ValueError): d.aggregate_rows(rows[:-1],bootstrap_seed=177)


def test_output_refusal_precedes_any_protected_read(monkeypatch,tmp_path):
    class Root:
        def verify(self,**kwargs): pass
        def resolve(self,*args,**kwargs):return tmp_path
    monkeypatch.setattr(api,"analyze_preparation_diagnostics",lambda *args,**kwargs:pytest.fail("protected input touched"))
    with pytest.raises(FileExistsError):
        api.publish_preparation_diagnostics(SimpleNamespace(),artifact_writer=SimpleNamespace(root=Root()),relative_root="occupied")


def test_changed_analysis_owner_stops_before_phase_payloads(monkeypatch):
    monkeypatch.setattr(api,"_phase_read",lambda *args,**kwargs:pytest.fail("protected payload touched"))
    with pytest.raises(ValueError,match="source"):
        api.analyze_preparation_diagnostics(SimpleNamespace(analysis_source_sha256="0"*64),artifact_writer=None)


def test_owned_source_identity_is_actual_byte_bound():
    value=api.current_preparation_diagnostic_source_sha256()
    assert len(value)==64 and int(value,16)>0


def test_strict_result_census_and_analysis_ceiling():
    from empirical_lawhood.adapters.methods.constructed_preparation_applicability.diagnostic_records import PreparationDiagnosticReport,PreparationDiagnosticInput
    from empirical_lawhood.kernel.provenance import ObjectIdentity
    from empirical_lawhood.kernel.serialization import canonical_json_bytes
    identity=ObjectIdentity("test.input",PreparationDiagnosticInput.SCHEMA,"1.0.0","1"*64)
    with pytest.raises(ValueError,match="denominators"):
        PreparationDiagnosticReport("test.analysis",identity,(identity,),canonical_json_bytes({"Q":{"n_independent_roots":8},"E":{"n_independent_roots":31}}).decode())
    with pytest.raises(ValueError,match="ceiling"):
        PreparationDiagnosticReport("test.analysis",identity,(identity,),"{}",native_calls=1)


def test_opaque_diagnostic_tables_preserve_finite_numeric_json_without_weakening_inputs():
    import json
    from empirical_lawhood.adapters.methods.constructed_preparation_applicability.diagnostic_records import diagnostic_json_payload
    value = {"Q": {"n_independent_roots": 8, "interval": [0.125, 0.875]}, "E": {"n_independent_roots": 32}}
    encoded = diagnostic_json_payload(value)
    assert json.loads(encoded) == value and diagnostic_json_payload(json.loads(encoded)) == encoded
    with pytest.raises(ValueError):
        diagnostic_json_payload({"unexpected_nonfinite": float("nan")})
    from empirical_lawhood.kernel.serialization import canonical_json_bytes
    with pytest.raises(ValueError):
        canonical_json_bytes(0.125)
