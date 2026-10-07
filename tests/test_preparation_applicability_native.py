"""Bounded actual mechanics conformance and seed/causality counterexamples."""
from dataclasses import replace
from decimal import Decimal
import numpy as np
import pytest

from empirical_lawhood.api.preparation_conformance import preparation_native_conformance
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.native import nominal_digest
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import ConstructedPreparationStream
from empirical_lawhood.adapters.methods.preparation_applicability.exposure import effective_seed_ids,PreparationApplicabilityExposure
from empirical_lawhood.adapters.simulators.preparation_applicability.native import acquire_phase
from tests.preparation_applicability_fixtures import allocation,synthetic_identity


def test_nominal_programme_and_conditioned_gaussian_are_unchanged():
    expected={"prefix":"2d5f6fc76854823ef5faee5dfe29c3de3406f7434c1958b8c71efce7557ee0f0",
        "prefix-bridge":"2f78acc6f7a9c250d4ea53eb4e5ff14c73a99a02e3ea08aa03be19e3d2ca8d13",
        "parent":"43eaedd1045f686c0547aa79912118a7fb7265ebff9ae45e749a1a43b9b219a4",
        "parent-bridge":"f02da99ba780990f098f966aa615a1927861dc9b2c456efdc8aeb189b716cd37",
        "prefix-probes":"3376f1a6e79f07627b5df9d7b8661f64f60156d91d96a8ae52fd8118e70fe2d7"}
    for purpose,digest in expected.items():
        assert nominal_digest(purpose)==digest
    from hashlib import sha256
    from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import FIXED_PROBE_SEED
    actual_probe_digest=sha256(f"cc1-applicability-v1.passive-probes\0{expected['prefix-probes']}\0q2-n4-fields12".encode()).hexdigest()
    assert actual_probe_digest==f"{FIXED_PROBE_SEED:064x}"
    stream=ConstructedPreparationStream(987,expected["prefix"],Decimal(".002"))
    actual=stream.generator().normal(size=(3,5))
    nominal=np.random.Generator(np.random.PCG64(int(expected["prefix"][:32],16)))
    fresh=np.random.Generator(np.random.PCG64(987))
    np.testing.assert_array_equal(actual,nominal.normal(size=(3,5))+.002*fresh.normal(size=(3,5)))
    ordinary=ConstructedPreparationStream(987,None,Decimal(1))
    np.testing.assert_array_equal(ordinary.generator().normal(size=10),np.random.Generator(np.random.PCG64(987)).normal(size=10))


def test_probe_high128_exposure_rejects_changed_low128_label():
    current=allocation(1)
    root=current.roots[0]
    probe=next(seed for seed in root.scientific_seeds if seed.purpose_id=="passive-probes")
    changed=replace(probe,seed=probe.seed+123)
    alternate=replace(current,roots=(replace(root,scientific_seeds=tuple(changed if seed==probe else seed for seed in root.scientific_seeds)),))
    assert effective_seed_ids(current)==effective_seed_ids(alternate)
    prior=PreparationApplicabilityExposure(synthetic_identity("test.inspection"),synthetic_identity("test.receipt"),("other.unit",),effective_seed_ids(current))
    with pytest.raises(ValueError,match="effective"):
        prior.check(alternate)


@pytest.mark.parametrize("phase",("parent","future"))
def test_continuation_refuses_before_native_contact(monkeypatch,phase):
    def forbidden(**kwargs):
        raise AssertionError("native marcher entered")
    monkeypatch.setattr("empirical_lawhood.adapters.simulators.preparation_applicability.native.march_native_intervals",forbidden)
    with pytest.raises(ValueError,match="predecessor"):
        acquire_phase(phase_name=phase,allocation=allocation(1).roots[0],refinement=1,source_sha256="a"*64)


def test_three_source_kinds_actual_two_view_short_conformance(tmp_path):
    q2=allocation(1).roots[0]
    cir1=replace(allocation(1,phase="cir").roots[0],cohort="cir1")
    constructed=allocation(1,constructed=True,phase="constructed").roots[0]
    report=preparation_native_conformance(roots=(q2,cir1,constructed),output_dir=tmp_path/"conformance",conformance_id="test.actual.native-conformance")
    assert len(report.cells)==6
    assert all(row[2:]==(16*row[1],"COMPLETE") for row in report.cells)
    assert report.ceiling=="SOFTWARE_CONFORMANCE_NOT_SCIENTIFIC_QUALIFICATION"
    assert (tmp_path/"conformance/native-conformance.canonical.json").is_file()
