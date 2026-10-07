"""Constructor geometry qualification and fresh paired valid-use comparison."""

from decimal import Decimal
from typing import Any
import numpy as np
from empirical_lawhood.adapters.methods.preparation_applicability.statistics import paired_binary_test
from .records import ConstructedPreparationReport, numbers
from empirical_lawhood.adapters.methods.preparation_applicability.exposure import effective_seed_ids
from .qualification import constructor_crossing as constructor_crossing


def report(stage: Any, measured: Any, constructor_crossings: Any) -> ConstructedPreparationReport:
    if tuple(m.root_id for m in measured) != stage.root_ids:
        raise ValueError("readout changes assigned independent units")
    identity = stage.phase, stage.root_ids, tuple(m.fingerprint() for m in measured)
    bindings = dict(lower_sha256=stage.upstream[0].artifact.sha256, design_sha256=stage.design.fingerprint(), source_sha256=stage.source.object_fingerprint, allocation_sha256=stage.allocation.fingerprint(), effective_seed_ids=effective_seed_ids(stage.allocation))
    if not all(m.complete for m in measured):
        return ConstructedPreparationReport(*identity, False, False, "UNEVALUABLE", (), (), (), (), (), None, 0, 0, None, None, (), **bindings)
    maxima = np.asarray([m.maxima for m in measured], dtype=float).reshape(-1, 3, 7)
    valid = np.asarray((maxima <= 1).all(axis=2), dtype=bool)
    support = np.asarray(maxima[:, :, 0] <= 1, dtype=bool)
    covered = np.asarray([m.covered_counts for m in measured])
    joint = np.asarray([m.joint_counts for m in measured])
    false = np.asarray([m.false_admissions for m in measured])
    effect, wins, losses, p = paired_binary_test(valid[:, 1], valid[:, 0])
    differences = (covered[:, 1] - covered[:, 0]) / 256
    rng = np.random.Generator(np.random.PCG64(stage.allocation.bootstrap_seed))
    samples = rng.integers(0, len(measured), (20000, len(measured)))
    interval = numbers(np.quantile(differences[samples].mean(axis=1), (.025, .975)))
    qualified = bool(
        stage.phase == "Q" and sum(constructor_crossings) >= 6
        and (maxima[:, :, 1] <= 1).all()
    )
    positive = effect > 0 and p <= .05 and float(interval[0]) > 0 and false[:, 1].sum() == 0
    negative = false[:, 1].sum() > 0 or (
        effect < 0 and paired_binary_test(valid[:, 0], valid[:, 1])[3] <= .05
    )
    disposition = (
        "QUALIFIED" if qualified else "CONSTRUCTOR_NOT_QUALIFIED"
    ) if stage.phase == "Q" else "SUPPORTED" if positive else "OPPOSED" if negative else "UNRESOLVED"
    return ConstructedPreparationReport(
        *identity, True, qualified, disposition,
        tuple(map(int, valid.sum(axis=0))), tuple(map(int, support.sum(axis=0))),
        tuple(map(int, covered.sum(axis=0))), tuple(map(int, joint.sum(axis=0))),
        tuple(map(int, false.sum(axis=0))), Decimal(str(effect)), wins, losses,
        Decimal(str(p)), Decimal(str(float(differences.mean()))), interval,
        constructor_crossings=tuple(constructor_crossings) if stage.phase == "Q" else (),
        numerical_maxima=numbers(maxima[:, :, 1]) if stage.phase == "Q" else (),
        **bindings,
    )
