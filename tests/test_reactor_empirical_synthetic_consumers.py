"""EXPOSED_DEVELOPMENT_NONPROMOTABLE fixture for empirical stage consumers.

The tiny synthetic rows exercise arithmetic and refusal paths. They are not
native acquisitions, a complete callback census, or qualification evidence.
"""

import json
from hashlib import sha256

import numpy as np
import pytest

from empirical_lawhood.adapters.methods.reactor_causal_response.calibration import RootOperands, calibrate, qualification_operands
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_analysis import analyze_cohort, analyze_root
from empirical_lawhood.adapters.methods.reactor_causal_response.config import ARMS, CONFIRMATION_ROOTS, ROLES, EmpiricalRecipe, arm_order
from empirical_lawhood.adapters.methods.reactor_causal_response.discovery import Rows, discover
from empirical_lawhood.adapters.methods.reactor_causal_response.experiment_records import EmpiricalDiscoveryEnvelope, EmpiricalQualificationTerminal
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation, features
from empirical_lawhood.adapters.simulators.reactor_causal_response.campaign import acquire_confirmation
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.kernel.provenance import ObjectIdentity


def _identity(name: str) -> ObjectIdentity:
    return ObjectIdentity(
        name,
        'empirical-lawhood/test/synthetic-parent',
        "1.0.0",
        sha256(name.encode()).hexdigest(),
    )


def _development_rows() -> Rows:
    roots: list[str] = []
    roles: list[str] = []
    clocks: list[float] = []
    actions: list[int] = []
    vectors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    actuator = Actuator()
    for role, count in (("fit", 16), ("nomination", 8)):
        for index in range(count):
            root = f"reactor-empirical-{role}-{index:03d}"
            # Keep nomination contexts inside the fit support boxes. Every
            # vector is computed only from an available synthetic prefix.
            temperature = 320 + (index % 8) * 0.002
            history = tuple(
                Observation(10 * k, temperature, 316.0, 0.0) for k in range(2521)
            )
            for callback in (0, 60, 2520):
                prefix = history[: callback + 1]
                for projection in actuator.project(prefix[-1], (0.0, 316.0)):
                    vector = features(prefix, 316.0, projection)
                    receiver = np.array(
                        (
                            320 + 0.02 * vector[7] + 0.01 * vector[5],
                            0.5 + 0.001 * vector[0],
                            0.0,
                        ),
                        dtype=float,
                    )
                    roots.append(root)
                    roles.append(role)
                    clocks.append(prefix[-1].time)
                    actions.append(projection.action)
                    vectors.append(vector)
                    labels.append(np.stack((receiver, receiver)))
    return Rows(
        tuple(roots),
        tuple(roles),
        np.asarray(clocks),
        np.asarray(actions),
        np.stack(vectors),
        np.stack(labels),
        np.ones(len(roots), dtype=bool),
    )


def _incomplete_root(name: str) -> RootOperands:
    return RootOperands(
        name,
        np.empty(0),
        np.empty(0, dtype=int),
        np.empty((0, 23)),
        np.empty((0, 2, 3)),
        np.empty((0, 10, 4)),
        np.empty((0, 10, 4)),
        np.empty((0, 20, 4)),
        np.empty((0, 20, 4)),
        np.empty((0, 2, 2)),
    )


def test_synthetic_empirical_consumers_keep_roles_and_refuse_missing_evidence() -> None:
    assert tuple((role, count) for role, count, _ in ROLES) == (
        ("fit", 16),
        ("nomination", 8),
        ("calibration", 32),
        ("qualification", 32),
        ("confirmation", 8),
    )
    assert sum(count for _, count, _ in ROLES) == 96
    assert len(ARMS) == 9
    assert all(set(arm_order(i)) == set(ARMS) for i in range(8))

    rows = _development_rows()
    with pytest.raises(ValueError, match="complete development root/episode census"):
        discover(rows)  # the synthetic rows cannot enter the real consumer
    result = discover(rows, require_full_census=False)
    assert len(result.fits) == len(result.calls) == len(result.nominations) == 9
    assert {call.rows for call in result.calls} == {16 * 3 * 9}
    assert sum(result.fit_mask) == 16 * 3 * 9
    assert sum(result.nomination_mask) == 8 * 3 * 9
    assert all(len(cell.roots) >= 8 for fit in result.fits for cell in fit.support)
    assert result.selected is not None
    model = result.fits[result.selected]

    calibration = calibrate(
        tuple(
            _incomplete_root(f"reactor-empirical-calibration-{i:03d}")
            for i in range(32)
        ),
        model,
    )
    assert calibration.rank == 32
    assert calibration.q is None and calibration.disposition == "CALIBRATION_UNUSABLE"
    assert all(
        score.reasons == ("INCOMPLETE_ASSIGNED_CENSUS",) for score in calibration.scores
    )
    adequacy, lower = qualification_operands(
        tuple(
            _incomplete_root(f"reactor-empirical-qualification-{i:03d}")
            for i in range(32)
        ),
        model,
        calibration.q,
    )
    assert len(adequacy) == 32 and lower == 0
    assert all(not item.adequate for item in adequacy)

    recipe = ObjectIdentity.from_record("reactor-empirical-recipe", EmpiricalRecipe())
    discovery = EmpiricalDiscoveryEnvelope(
        recipe,
        tuple(
            _identity(f"reactor-empirical-{role}-{i:03d}")
            for role, count in (("fit", 16), ("nomination", 8))
            for i in range(count)
        ),
        json.dumps({"disposition": result.disposition, "selected": result.selected}),
    )
    terminal = EmpiricalQualificationTerminal(
        recipe,
        _identity("reactor-empirical-calibration"),
        tuple(
            _identity(f"reactor-empirical-{role}-{i:03d}")
            for role in ("calibration", "qualification")
            for i in range(32)
        ),
        None,
        "CALIBRATION_UNUSABLE",
    )
    # Stop occurs before a source, controller, custody port or reader is read.
    inaccessible = object()
    confirmations = tuple(
        acquire_confirmation(
            inaccessible,
            inaccessible,
            discovery,
            terminal,
            i,
            custody=inaccessible,
            reader=inaccessible,
        )
        for i in range(8)
    )
    assert tuple(item.root for item in confirmations) == CONFIRMATION_ROOTS
    assert all(
        item.prerequisite == "CALIBRATION_UNUSABLE"
        and item.acquisition is None
        and not item.children
        and not item.owner_cpu_seconds
        and not item.telemetry
        for item in confirmations
    )
    analyses = tuple(
        analyze_root(item, terminal, discovery, inaccessible, {}, inaccessible)
        for item in confirmations
    )
    contribution = analyze_cohort(analyses, terminal)
    assert contribution.prerequisite == "CALIBRATION_UNUSABLE"
    assert contribution.prospective_evaluation is None and json.loads(contribution.contrasts_json) == []
    assert all(item.prospective_evaluation is None for item in analyses)
    matrix = json.loads(contribution.matrix_json)
    assert matrix["qualification"]["status"] == "UNENTERED"
    assert matrix["completion"]["native_benchmark_arms"] == [
        "EL",
        "REF",
        "EKF",
        "SCHEDULED_BACKOFF_ZERO",
        "SCHEDULED_BACKOFF_HALF",
    ]
