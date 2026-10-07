"""Assigned Tier 1 physical roots and native task graph before provider contact."""

from tests.finite_response_seed_fixtures import ASSIGNED_SEEDS
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

import numpy as np
import pytest

from empirical_lawhood.adapters.composition.finite_response_law.assignment import FiniteResponseLawCohortAssignment, assigned_native_seed_ids
from empirical_lawhood.adapters.composition.finite_response_law.exposure import native_seed_ids
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedCalibrationInterface
from empirical_lawhood.adapters.methods.finite_response_law.native_projection import _calibration_interface
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedCalibrationConfig, FiniteResponseLawAssignedCalibrationInvocation, FiniteResponseLawAssignedCalibrationRoot, FiniteResponseLawAssignedEvaluationConfig, FiniteResponseLawAssignedEvaluationInvocation, FiniteResponseLawAssignedEvaluationRoot
from empirical_lawhood.adapters.simulators.finite_response_law.calibration.discovery import SOURCE_CAPABILITY as CALIBRATION_SOURCE
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation.discovery import SOURCE_CAPABILITY as EVALUATION_SOURCE
from empirical_lawhood.adapters.simulators.finite_response_law.roster import native_declarations
from empirical_lawhood.adapters.simulators.finite_response_law.native_artifact import FiniteResponseLawAssignedCalibrationPairArtifact, encode_native_pair
from empirical_lawhood.adapters.simulators.finite_response_law.provider import native_result_type
from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawAssignedCalibrationCheckpoint, FiniteResponseLawAssignedCalibrationDelivery, execute_native_phase, frozen_prefix_frame
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedCalibrationTaskResult, FiniteResponseLawAssignedEvaluationTaskResult, decode_task_native
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_receiver
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _encode
from empirical_lawhood.kernel.provenance import ObjectIdentity


def _assigned(stage: str):
    science = FiniteResponseLawScienceSpec()
    namespace = f"empirical-lawhood.finite-response-law.{stage}.development"
    count, source, config_type = (
        (32, CALIBRATION_SOURCE, FiniteResponseLawAssignedCalibrationConfig)
        if stage == "calibration"
        else (64, EVALUATION_SOURCE, FiniteResponseLawAssignedEvaluationConfig)
    )
    assignment = FiniteResponseLawCohortAssignment(
        stage,
        namespace,
        science.plan_sha256,
        count,
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    scientific_seeds=ASSIGNED_SEEDS[namespace],
    )
    config = config_type(
        stage,
        science,
        ObjectIdentity.from_record(source.capability_key, source),
        None,
        (),
        namespace,
        assignment.fingerprint(),
    scientific_seeds=assignment.scientific_seeds,
    )
    return assignment, config


def test_both_assigned_cohorts_preserve_physical_roots_native_budget_and_rng() -> None:
    for stage, count, task_count, updates, root_type, invocation_type, result_type in (
        (
            "calibration",
            32,
            640,
            751104,
            FiniteResponseLawAssignedCalibrationRoot,
            FiniteResponseLawAssignedCalibrationInvocation,
            FiniteResponseLawAssignedCalibrationTaskResult,
        ),
        (
            "prospective-evaluation",
            64,
            1280,
            1502208,
            FiniteResponseLawAssignedEvaluationRoot,
            FiniteResponseLawAssignedEvaluationInvocation,
            FiniteResponseLawAssignedEvaluationTaskResult,
        ),
    ):
        assignment, config = _assigned(stage)
        assert native_result_type(config) is result_type
        assert config.assignment_sha256 == assignment.fingerprint()
        roots = config.roots
        tasks = native_invocations(config)
        declarations = native_declarations(config)
        assert len(roots) == count
        assert all(type(root) is root_type for root in roots)
        assert tuple(root.stage_unit for root in roots) == assignment.stage_units
        assert (
            tuple(root.physical_unit_id for root in roots)
            == assignment.physical_unit_ids
        )
        assert len(tasks) == len({task.task_id for task in tasks}) == task_count
        assert all(type(task) is invocation_type for task in tasks)
        assert sum(task.maximum_native_updates for task in tasks) == updates
        assert len(declarations.units) == count
        assert len(declarations.segments) == task_count
        assert len(declarations.views) == 2 * count
        assert {
            unit.physical_independent_unit_id for unit in declarations.units
        } == set(assignment.physical_unit_ids)
        assert native_seed_ids(config) == assigned_native_seed_ids(assignment)
        tasks_by_id = {task.task_id: task for task in tasks}
        for task in tasks:
            if task.predecessor_segment_id is not None:
                assert task.predecessor_segment_id in tasks_by_id
                assert tasks_by_id[task.predecessor_segment_id].root == task.root
        first_root = roots[0]
        first_tasks = [task for task in tasks if task.root == first_root]
        assert len(first_tasks) == 20
        assert [
            sum(task.phase == phase for task in first_tasks)
            for phase in ("prefix", "parent", "future")
        ] == [1, 1, 18]
        assert {task.purpose for task in first_tasks if task.phase == "future"} == {
            "future-1",
            "future-2",
        }
        assert all(len(task.task_id.encode()) <= 255 for task in first_tasks)


def test_assigned_roots_refuse_wrong_seed_index_and_phase() -> None:
    for stage, root_type, invocation_type in (
        ("calibration", FiniteResponseLawAssignedCalibrationRoot, FiniteResponseLawAssignedCalibrationInvocation),
        ("prospective-evaluation", FiniteResponseLawAssignedEvaluationRoot, FiniteResponseLawAssignedEvaluationInvocation),
    ):
        assignment, config = _assigned(stage)
        root = config.roots[0]
        source = ObjectIdentity.from_record(config.spec_id, config)
        with pytest.raises(ValueError, match="assigned .* root"):
            replace(root, seed_sha256="0" * 64)
        with pytest.raises(ValueError, match="assigned .* root"):
            replace(root, index=assignment.root_count)
        with pytest.raises(ValueError, match="assignment_sha256"):
            replace(config, assignment_sha256="not-a-digest")
        with pytest.raises(ValueError, match="assigned .* invocation"):
            invocation_type(
                source, root, "future", root.assigned_parent, None, "future-1"
            )
        with pytest.raises(ValueError, match="assigned .* invocation"):
            invocation_type(
                source,
                root,
                "parent",
                "hold" if root.assigned_parent != "hold" else "y-positive-128",
                None,
                "parent",
            )
        assert (
            root.seed_sha256
            == sha256(
                str(assignment.seed_for("prefix", 0)).encode()
            ).hexdigest()
        )


def test_assigned_calibration_native_output_preserves_handoff_and_action() -> None:
    """Run one excluded assigned root through the real marcher and typed artifacts."""
    _, config = _assigned("calibration")
    assert native_result_type(config) is FiniteResponseLawAssignedCalibrationTaskResult
    root = config.roots[0]
    tasks = tuple(task for task in native_invocations(config) if task.root == root)
    prefix_task = next(task for task in tasks if task.phase == "prefix")
    parent_task = next(task for task in tasks if task.phase == "parent")
    frame = None
    predecessor = ()
    common = None
    incoming = (None, None)
    views_by_phase = {}
    for phase, task in (("prefix", prefix_task), ("parent", parent_task)):
        values = tuple(
            execute_native_phase(
                config,
                task,
                refinement,
                incoming=incoming[refinement - 1],
                frame=frame,
            )
            for refinement in (1, 2)
        )
        assert all(
            type(value.delivery) is FiniteResponseLawAssignedCalibrationDelivery
            and type(value.checkpoint) is FiniteResponseLawAssignedCalibrationCheckpoint
            for value in values
        )
        assert all(value.delivery.disposition == "COMPLETE" for value in values)
        if phase == "prefix":
            frame = frozen_prefix_frame(values[0].checkpoint)
            assert frame is not None
        pair, payload = encode_native_pair(*values)
        assert type(pair) is FiniteResponseLawAssignedCalibrationPairArtifact
        result = FiniteResponseLawAssignedCalibrationTaskResult(
            task,
            predecessor,
            pair,
            common,
            _encode(frame.modes),
            None,
        )
        assert result.native_complete
        decoded = decode_task_native(result, payload)
        assert decoded is not None
        assert [value.checkpoint for value in decoded] == [
            value.checkpoint for value in values
        ]
        for value in values:
            interface = _calibration_interface(value, result)
            assert type(interface) is FiniteResponseLawAssignedCalibrationInterface
            assert interface.cutoff_tick == (4096 if phase == "prefix" else 4368)
            assert len(interface.values) == 24
        if phase == "prefix":
            common = ObjectIdentity.from_record(result.result_id, result)
        predecessor = (ObjectIdentity.from_record(result.result_id, result),)
        incoming = tuple(value.checkpoint for value in values)
        views_by_phase[phase] = values
    assert frame is not None and common is not None

    receivers = {}
    for word in (
        PreparedForceWord(Decimal(0), 0, 0),
        PreparedForceWord(Decimal(8), 0, 1),
    ):
        task = next(
            task
            for task in tasks
            if task.phase == "future"
            and task.purpose == "future-1"
            and task.word == word
        )
        values = tuple(
            execute_native_phase(
                config,
                task,
                refinement,
                incoming=incoming[refinement - 1],
                frame=frame,
            )
            for refinement in (1, 2)
        )
        pair, payload = encode_native_pair(*values)
        result = FiniteResponseLawAssignedCalibrationTaskResult(
            task, predecessor, pair, common, _encode(frame.modes), None
        )
        assert result.native_complete and decode_task_native(result, payload)
        for refinement, value in enumerate(values, start=1):
            assert value.delivery.accepted is True
            assert value.delivery.completed_intervals == 192 * refinement
            assert value.delivery.applied_force_kicks == 384 * refinement
            assert value.delivery.nonzero_force_intervals == (
                0 if word.sign == 0 else 64 * refinement
            )
            np.testing.assert_allclose(
                [float(v) for v in value.delivery.realized_impulse],
                [0 if word.sign == 0 else 0.512, 0],
                atol=1e-12,
                rtol=0,
            )
        receivers[word.word_id] = values

    hold, action = (
        receivers[word.word_id]
        for word in (
            PreparedForceWord(Decimal(0), 0, 0),
            PreparedForceWord(Decimal(8), 0, 1),
        )
    )
    paired = tuple(
        prepared_receiver(
            frame, applied.positions[-1, 0] - reference.positions[-1, 0]
        )
        for reference, applied in zip(hold, action, strict=True)
    )
    assert all(np.isfinite(value).all() for value in paired)
    np.testing.assert_allclose(*paired, atol=1e-4, rtol=0)
