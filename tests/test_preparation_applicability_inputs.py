"""Closed allocation/exposure metadata and before-payload access rejection."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from empirical_lawhood.api.preparation_inputs import prepare_preparation_allocation, _retained_output
from empirical_lawhood.adapters.methods.preparation_applicability.exposure import PreparationApplicabilityExposure, effective_seed_ids, public_historical_exposure
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import FIXED_PROBE_SEED
from empirical_lawhood.api.preparation_result_access import authenticate_preparation_result_access
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.matrix_inputs import MatrixAllocation, MATRIX_NATIVE_PURPOSES
from tests.preparation_applicability_fixtures import synthetic_identity


@pytest.mark.parametrize("constructed,phase,count",((False,"D",32),(False,"E",64),(True,"Q",8),(True,"E",32)))
def test_complete_explicit_roster_is_reproducible_and_supplies_no_freshness(constructed,phase,count):
    kwargs=dict(allocation_id="test.exposed.allocation",cohort_namespace="test.exposed.study",phase=phase,master_seed=78651,constructed=constructed)
    allocation=prepare_preparation_allocation(**kwargs)
    assert len(allocation.roots)==count and allocation.exposure=="EXPOSED_DEVELOPMENT_NONPROMOTABLE"
    assert prepare_preparation_allocation(**kwargs)==allocation
    assert decode_canonical_bytes(allocation.canonical_bytes(),MatrixAllocation,maximum_bytes=4*1024**2)==allocation
    planned=prepare_preparation_allocation(**kwargs,proposed_unrun=True)
    assert planned.roots==allocation.roots and planned.exposure=="PROPOSED_UNRUN"
    for root in allocation.roots:
        assert tuple(seed.purpose_id for seed in root.scientific_seeds)==tuple(sorted(MATRIX_NATIVE_PURPOSES if constructed else (*MATRIX_NATIVE_PURPOSES,"passive-probes")))
        if constructed:
            assert root.seed_for("passive-probes")==FIXED_PROBE_SEED
        else:
            probe=next(value for value in root.scientific_seeds if value.purpose_id=="passive-probes")
            assert probe.effective_seed==probe.seed>>128
    assert len(effective_seed_ids(allocation))==count*(9 if constructed else 10)+1


def test_phase_split_and_explicit_master_produce_disjoint_actual_streams():
    q=prepare_preparation_allocation(allocation_id="test.q",cohort_namespace="test.study",phase="Q",master_seed=36,constructed=True)
    e=prepare_preparation_allocation(allocation_id="test.e",cohort_namespace="test.study",phase="E",master_seed=36,constructed=True)
    assert not set(effective_seed_ids(q))&set(effective_seed_ids(e))
    assert not set(root.root_id for root in q.roots)&set(root.root_id for root in e.roots)


def test_public_matrix_seed_alias_is_rejected_even_if_census_omits_it():
    from hashlib import sha256
    allocation=prepare_preparation_allocation(allocation_id="test.proposed",cohort_namespace="test.renamed",phase="D",master_seed=234)
    root=allocation.roots[0]
    purpose=root.scientific_seeds[0].purpose_id
    exposed=int.from_bytes(sha256(f"public-exposed-matrix-input|preparation|0|{purpose}".encode()).digest()[:16],"big")
    alias=replace(root,scientific_seeds=(replace(root.scientific_seeds[0],seed=exposed),*root.scientific_seeds[1:]))
    allocation=replace(allocation,roots=(alias,*allocation.roots[1:]),exposure="PROPOSED_UNRUN")
    census=PreparationApplicabilityExposure(synthetic_identity("test.inspection"),synthetic_identity("test.receipt"),("test.old-root",),("seed.pcg64.ffffffffffffffffffffffffffffffff",))
    with pytest.raises(ValueError,match="exposed effective"):
        census.check(allocation)
    _,seeds=public_historical_exposure()
    assert f"seed.pcg64.{exposed:032x}" in seeds


@pytest.mark.parametrize("seed",(True,-1,2**128,1.5))
def test_allocator_refuses_unconsumed_master_seed_values(seed):
    with pytest.raises(ValueError,match="explicit128-bit"):
        prepare_preparation_allocation(allocation_id="test.allocation",cohort_namespace="test.study",phase="Q",master_seed=seed,constructed=True)


@pytest.mark.parametrize("access",(OutcomeAccess.EVALUATION_SEALED,OutcomeAccess.EVALUATOR_REVEAL,OutcomeAccess.EVALUATION_REVEALED))
def test_protected_output_requires_actual_authority_before_any_manifest_contact(access):
    class Plane:
        @property
        def root(self):
            raise AssertionError("protected output manifest contacted before authority")
    kind=SimpleNamespace(SCHEMA="empirical-lawhood/tests/protected-preparation")
    receipt=SimpleNamespace(output_logical_artifacts=(SimpleNamespace(payload_schema=kind.SCHEMA,outcome_access=access),))
    with pytest.raises(PermissionError,match="CURRENT_AUTHORITY"):
        _retained_output(plane=Plane(),receipt=receipt,kind=kind,result_access=None)


def test_outcome_blind_result_still_requires_actual_custody():
    # An outcome-blind label does not prove that a result was never issued.
    # Actual exposed imports are covered by the current-result custody tests.
    with pytest.raises(PermissionError, match="MISSING_ISSUED_CONTROLS"):
        authenticate_preparation_result_access(context=None,receipt=SimpleNamespace(run_id="test.exposed.unissued"),
            logical=SimpleNamespace(outcome_access=OutcomeAccess.OUTCOME_BLIND),writer=None)


def test_actual_conformance_streams_remain_exposed_after_renaming_units():
    from empirical_lawhood.adapters.methods.preparation_applicability.conformance import reject_conformance_reuse
    old=prepare_preparation_allocation(allocation_id="test.conformance.bank",cohort_namespace="test.conformance.bank",phase="D",master_seed=736)
    current=prepare_preparation_allocation(allocation_id="test.new.bank",cohort_namespace="test.new.bank",phase="D",master_seed=737,proposed_unrun=True)
    first=current.roots[0]
    exposed=old.roots[0].scientific_seeds[0]
    alias=replace(first,scientific_seeds=(exposed,*first.scientific_seeds[1:]))
    current=replace(current,roots=(alias,*current.roots[1:]))
    with pytest.raises(ValueError,match="actually exposed native-conformance"):
        reject_conformance_reuse(SimpleNamespace(allocation=current),SimpleNamespace(allocations=old.roots[:3]))


def test_operational_run_changes_with_actual_allocation_under_same_scientific_id():
    from tests.test_preparation_applicability_authoring import ordinary_stage
    from empirical_lawhood.adapters.methods.preparation_applicability.config import preparation_run_id
    stage,_=ordinary_stage("D")
    allocation=prepare_preparation_allocation(allocation_id="test.second.allocation",cohort_namespace="test.second.roots",
        phase="D",master_seed=93406)
    second=replace(stage,allocation=allocation)
    assert second.config_id==stage.config_id and second.design==stage.design
    assert preparation_run_id(second)!=preparation_run_id(stage)
