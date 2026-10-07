"""Synthetic assigned projection types and complete physical-root denominator."""

from tests.finite_response_seed_fixtures import ASSIGNED_SEEDS
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from empirical_lawhood.adapters.composition.finite_response_law.assignment import FiniteResponseLawCohortAssignment
from empirical_lawhood.adapters.composition.finite_response_law.method_authoring import build_method_authoring
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation, FiniteResponseLawAssignedCalibrationProjectionConfig, FiniteResponseLawAssignedEvaluationNativeCompletion, FiniteResponseLawAssignedEvaluationProjectionConfig
from empirical_lawhood.adapters.methods.finite_response_law.assigned_prediction import AssignedPrediction
from empirical_lawhood.adapters.methods.finite_response_law.calibration_analysis import CalibrationScores, calibration_requests
from empirical_lawhood.adapters.methods.finite_response_law.calibration_panel import CalibrationPanel
from empirical_lawhood.adapters.methods.finite_response_law.calibration_records import FiniteResponseLawAssignedBoundaryCalibration, FiniteResponseLawBoundaryCalibration, boundary_calibration
from empirical_lawhood.adapters.methods.finite_response_law.control_delivery import FiniteResponseLawAssignedObservedNativeWord, observation_words, observe_native_word
from empirical_lawhood.adapters.methods.finite_response_law.executable_binding import METHOD_BINDING
from empirical_lawhood.adapters.methods.finite_response_law.independent_calibration import independent_requests
from empirical_lawhood.adapters.methods.finite_response_law.method_records import COEFFICIENT_SCHEMA, MANIFEST_SCHEMA, READOUT_SCHEMA, REPORT_SCHEMA, FiniteResponseLawAssignedCalibrationMethodConfig, FiniteResponseLawAssignedCalibrationReport, FiniteResponseLawCalibrationReport
from empirical_lawhood.adapters.methods.finite_response_law.native_records import FiniteResponseLawCalibrationNativeEvaluation, native_method_types
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedCalibrationConfig, FiniteResponseLawAssignedEvaluationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.calibration.discovery import SOURCE_CAPABILITY as CALIBRATION_SOURCE
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation.discovery import SOURCE_CAPABILITY as EVALUATION_SOURCE
from empirical_lawhood.adapters.simulators.finite_response_law.provider import native_result_type
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult, unentered_native_bytes
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from empirical_lawhood.kernel.action_contracts import ObservedActionOccurrence
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort


def _identity(object_id: str, record_type: type) -> ObjectIdentity:
    return ObjectIdentity(object_id, record_type.SCHEMA, record_type.VERSION, "0" * 64)


def _source(stage: str):
    science = FiniteResponseLawScienceSpec()
    namespace = f"empirical-lawhood.finite-response-law.{stage}.development"
    count, manifest, config_type = (
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
    return config_type(
        stage,
        science,
        ObjectIdentity.from_record(manifest.capability_key, manifest),
        None,
        (),
        namespace,
        assignment.fingerprint(),
    scientific_seeds=assignment.scientific_seeds,
    )


@pytest.mark.parametrize(
    ("stage", "root_count", "projection_type", "result_type"),
    (
        (
            "calibration",
            32,
            FiniteResponseLawAssignedCalibrationProjectionConfig,
            FiniteResponseLawAssignedCalibrationNativeEvaluation,
        ),
        (
            "prospective-evaluation",
            64,
            FiniteResponseLawAssignedEvaluationProjectionConfig,
            FiniteResponseLawAssignedEvaluationNativeCompletion,
        ),
    ),
)
def test_assigned_method_records_keep_the_full_root_view_denominator(
    stage: str, root_count: int, projection_type: type, result_type: type
) -> None:
    source = _source(stage)
    projection_class, evaluation_class, word_class, view_class, completion_class = (
        native_method_types(source)
    )
    assert projection_class is projection_type
    assert completion_class is result_type
    projection = projection_class(source)
    evaluation = evaluation_class(projection)
    with pytest.raises(ValueError, match="exact versioned source"):
        replace(projection, native_spec=None)
    native_result = native_result_type(source)
    tasks = native_invocations(source)
    views = []
    hold = PreparedForceWord(Decimal(0), 0, 0)
    for root in source.roots:
        local = tuple(task for task in tasks if task.root == root)
        result_ids = tuple(
            _identity(f"{task.task_id}.result", native_result) for task in local
        )
        accounting = tuple(
            (
                task.task_id,
                0,
                "NUMERICAL_FAILURE" if task.phase == "prefix" else "PREFIX_UNAVAILABLE",
            )
            for task in local
        )
        words = tuple(
            word_class(
                task,
                _identity(f"{task.task_id}.result", native_result),
                _identity(f"{replace(task, word=hold).task_id}.result", native_result),
                False,
                None,
                (None,) * 7,
            )
            for task in local
            if task.phase == "future"
        )
        observations = (
            tuple(
                sorted(
                    (
                        ObservedActionOccurrence(
                            f"{word.occurrences[0].occurrence_id}.synthetic",
                            word.occurrences[0].occurrence_id,
                            word.occurrences[0].requested,
                            None,
                            None,
                            None,
                            ("SYNTHETIC_NONENTRY",),
                        )
                        for word in observation_words(root.root_id)
                    ),
                    key=lambda value: value.expected_occurrence_id,
                )
            )
            if stage == "prospective-evaluation"
            else ()
        )
        for refinement in (1, 2):
            extras = (None, None, None)
            if stage == "prospective-evaluation":
                extras = (*extras, observations)
            views.append(
                view_class(
                    ObjectIdentity.from_record(projection.config_id, projection),
                    root,
                    refinement,
                    result_ids,
                    accounting,
                    words,
                    *extras,
                )
            )
    assert len(views) == 2 * root_count
    completion = completion_class(evaluation, tuple(views))
    assert completion.completed_native_updates == 0
    assert completion.reasons
    with pytest.raises(ValueError, match="omits an assigned physical root"):
        completion_class(evaluation, tuple(views[:-1]))


def test_assigned_evaluation_delivery_keeps_two_purposes_and_nonentry() -> None:
    source = _source("prospective-evaluation")
    root = source.roots[0]
    hold_word = PreparedForceWord(Decimal(0), 0, 0)
    tasks = tuple(
        task
        for task in native_invocations(source)
        if task.root == root and task.phase == "future" and task.word == hold_word
    )
    assert tuple(task.purpose for task in tasks) == ("future-1", "future-2")
    inputs = tuple(
        (
            FiniteResponseLawAssignedEvaluationTaskResult(
                task,
                (
                    _identity(
                        f"{task.predecessor_segment_id}.result",
                        FiniteResponseLawAssignedEvaluationTaskResult,
                    ),
                ),
                None,
                None,
                None,
                "PREFIX_UNAVAILABLE",
            ),
            unentered_native_bytes(),
        )
        for task in tasks
    )
    observed = observe_native_word(
        word=observation_words(root.root_id)[-1], refinement=1, inputs=inputs
    )
    assert type(observed) is FiniteResponseLawAssignedObservedNativeWord
    assert observed.observed.requested is not None
    assert observed.observed.accepted is None
    assert observed.observed.reason_codes == ("PREFIX_UNAVAILABLE",)
    with pytest.raises(ValueError, match="both purposes"):
        observe_native_word(
            word=observation_words(root.root_id)[-1],
            refinement=1,
            inputs=inputs[:1],
        )


def test_assigned_request_rng_uses_exact_physical_root_names() -> None:
    source = _source("calibration")
    root_ids = tuple(root.stage_unit for root in source.roots)
    request_seeds = tuple(dict(root.scientific_seeds)["calibration-request"] for root in source.roots)
    direction, requirement = calibration_requests(root_ids, request_seeds=request_seeds)
    independent_direction, independent_requirement = independent_requests(root_ids, request_seeds=request_seeds)
    np.testing.assert_array_equal(direction, independent_direction)
    np.testing.assert_array_equal(requirement, independent_requirement)
    old_direction, old_requirement = calibration_requests()
    assert not np.array_equal(direction, old_direction)
    assert not np.array_equal(requirement, old_requirement)
    with pytest.raises(ValueError, match="physical root cohort"):
        calibration_requests(root_ids[::-1])
    with pytest.raises(ValueError, match="assigned root order"):
        independent_requests(root_ids[::-1])
    shape = (32, 1, 4, 8, 2, 2)
    panel = CalibrationPanel(
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.bool_),
        np.zeros(shape, dtype=np.bool_),
        np.zeros((32, 24, 2), dtype=np.float64),
        np.zeros((32, 24, 2), dtype=np.float64),
        np.zeros((32, 1, 2), dtype=np.float64),
        np.zeros(32, dtype=np.int64),
        root_ids,
    )
    assert panel.root_ids == root_ids
    with pytest.raises(ValueError, match="assigned-root census"):
        replace(panel, root_ids=root_ids[::-1])


def test_assigned_calibration_boundary_binds_completion_and_root_ids() -> None:
    root_ids = tuple(root.stage_unit for root in _source("calibration").roots)
    scores = CalibrationScores(
        np.zeros(32, dtype=np.float64),
        np.ones(32, dtype=np.bool_),
        np.ones(32, dtype=np.bool_),
        30,
        0.0,
    )
    prediction = AssignedPrediction(
        np.zeros((32, 4, 8), dtype=np.float64),
        np.ones((32, 4, 8), dtype=np.float64),
        np.ones(32, dtype=np.bool_),
    )

    def artifact(name: str) -> ArtifactIdentity:
        return ArtifactIdentity(
            name,
            "synthetic-development",
            'empirical-lawhood/test/synthetic-artifact',
            "0" * 64,
            "application/json",
            1,
        )

    def make(native_type: type, ids: tuple[str, ...]):
        return boundary_calibration(
            scores,
            prediction,
            calibration_id="finite-response-law.calibration.lower.calibration",
            boundary="lower",
            development_manifest=artifact("synthetic-manifest"),
            frozen_coefficients=artifact("synthetic-coefficients"),
            native_evaluation=_identity("synthetic-native-completion", native_type),
            native_task_receipt=_identity(
                "synthetic-native-receipt", CanonicalTaskReceipt
            ),
            prediction_artifact=artifact("synthetic-prediction"),
            root_ids=ids,
        )

    assigned = make(FiniteResponseLawAssignedCalibrationNativeEvaluation, root_ids)
    assert type(assigned) is FiniteResponseLawAssignedBoundaryCalibration
    assert assigned.root_ids == root_ids
    assert assigned.q == Decimal(0)
    old = make(
        FiniteResponseLawCalibrationNativeEvaluation,
        tuple(f"calibration.r{i:03d}" for i in range(32)),
    )
    assert type(old) is FiniteResponseLawBoundaryCalibration
    with pytest.raises(ValueError, match="independent-root census"):
        make(FiniteResponseLawAssignedCalibrationNativeEvaluation, old.root_ids)


def test_assigned_calibration_config_and_report_decode_with_distinct_boundaries() -> None:
    source = _source("calibration")
    root_ids = tuple(root.stage_unit for root in source.roots)

    def artifact(name: str, schema: str) -> ArtifactIdentity:
        return ArtifactIdentity(
            name, "synthetic-development", schema, "0" * 64, "application/json", 1
        )

    receipt = artifact("synthetic-native-receipt", CanonicalTaskReceipt.SCHEMA)
    config = FiniteResponseLawAssignedCalibrationMethodConfig(
        "finite-response-law.calibration.method-config",
        source,
        artifact(
            "synthetic-native-evaluation",
            FiniteResponseLawAssignedCalibrationNativeEvaluation.SCHEMA,
        ),
        receipt,
        _identity(receipt.artifact_id, CanonicalTaskReceipt),
        artifact("synthetic-development-report", REPORT_SCHEMA),
        artifact("synthetic-coefficients", COEFFICIENT_SCHEMA),
        artifact("synthetic-development-manifest", MANIFEST_SCHEMA),
    )
    assert (
        decode_canonical_bytes(
            config.canonical_bytes(),
            FiniteResponseLawAssignedCalibrationMethodConfig,
            maximum_bytes=8 * 1024**2,
        )
        == config
    )
    with pytest.raises(ValueError, match="native/development input contracts"):
        replace(
            config,
            native_evaluation=artifact(
                "synthetic-native-evaluation", FiniteResponseLawCalibrationNativeEvaluation.SCHEMA
            ),
        )
    scores = CalibrationScores(
        np.zeros(32, dtype=np.float64),
        np.ones(32, dtype=np.bool_),
        np.ones(32, dtype=np.bool_),
        30,
        0.0,
    )
    prediction = AssignedPrediction(
        np.zeros((32, 4, 8), dtype=np.float64),
        np.ones((32, 4, 8), dtype=np.float64),
        np.ones(32, dtype=np.bool_),
    )
    prediction_artifact = artifact(
        "synthetic-prediction",
        'empirical-lawhood/methods/finite-response-law/calibration-predictions',
    )
    boundaries = tuple(
        boundary_calibration(
            scores,
            prediction,
            calibration_id=f"finite-response-law.calibration.{boundary}.calibration",
            boundary=boundary,
            development_manifest=config.development_manifest,
            frozen_coefficients=config.coefficients,
            native_evaluation=_identity(
                "synthetic-native-completion", FiniteResponseLawAssignedCalibrationNativeEvaluation
            ),
            native_task_receipt=_identity(
                "synthetic-native-receipt", CanonicalTaskReceipt
            ),
            prediction_artifact=prediction_artifact,
            root_ids=root_ids,
        )
        for boundary in ("lower", "composed", "cached", "direct")
    )
    report = FiniteResponseLawAssignedCalibrationReport(
        "finite-response-law.calibration.calibrate.result",
        ObjectIdentity.from_record(config.config_id, config),
        prediction_artifact,
        artifact("synthetic-readout", READOUT_SCHEMA),
        boundaries,
        tuple(
            (boundary, False) for boundary in ("lower", "composed", "cached", "direct")
        ),
    )
    assert (
        decode_canonical_bytes(
            report.canonical_bytes(),
            FiniteResponseLawAssignedCalibrationReport,
            maximum_bytes=8 * 1024**2,
        )
        == report
    )
    with pytest.raises(ValueError, match="four-boundary census"):
        FiniteResponseLawCalibrationReport(
            report.report_id,
            report.config,
            report.prediction_artifact,
            report.readout_artifact,
            report.boundaries,
            report.joint_opportunities,
        )
    assert METHOD_BINDING.required_issued_payload_schemas == (config.SCHEMA,)
    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.provider_factory(
        METHOD_BINDING.binding_id
    )
    registry = GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry
    with pytest.raises(ValueError, match="existing I/O ports"):
        factory.build_provider(registry=registry, records=(config,), platform_ports=())
    with pytest.raises(TypeError, match="working custody and payload ports"):
        factory.build_provider(
            registry=registry,
            records=(config,),
            platform_ports=(
                ExecutablePlatformPort("candidate-payload-publisher", object()),
                ExecutablePlatformPort("candidate-input-resolver", object()),
            ),
        )

    class _SyntheticResolver:
        def resolve(self, requirement: object, *, materialize: bool) -> None:
            raise AssertionError(
                'Synthetic law-qualification binding test must not resolve protected input'
            )

    class _SyntheticPayloadPlane:
        def publish_candidate_payload(self, **kwargs: object) -> None:
            raise AssertionError(
                'Synthetic law-qualification binding test must not publish a candidate'
            )

        def read_candidate_payload(self, receipt: object) -> None:
            raise AssertionError('Synthetic law-qualification binding test must not read a candidate')

    provider = factory.build_provider(
        registry=registry,
        records=(config,),
        platform_ports=(
            ExecutablePlatformPort(
                "candidate-payload-publisher", _SyntheticPayloadPlane()
            ),
            ExecutablePlatformPort("candidate-input-resolver", _SyntheticResolver()),
        ),
    )
    assert len(provider.runners(registry)) == 1
    assert {
        contract.payload_schema
        for contract in provider.output_semantic_contracts(registry)
    } >= {FiniteResponseLawAssignedCalibrationReport.SCHEMA}
    authored = build_method_authoring(
        config=config,
        implementation_sha256="0" * 64,
        exposure=ObjectIdentity(
            "synthetic-exposure",
            'empirical-lawhood/test/exposure',
            "1.0.0",
            "0" * 64,
        ),
        exposed_unit_ids=("synthetic-exposed-root",),
        exposed_seed_ids=("synthetic-exposed-stream",),
    )
    assert config in authored.payloads
    from empirical_lawhood.adapters.composition.finite_response_law.stage_input import inspect_consumer_providers

    owners = inspect_consumer_providers(
        authored,
        (
            ExecutablePlatformPort(
                "candidate-payload-publisher", _SyntheticPayloadPlane()
            ),
            ExecutablePlatformPort("candidate-input-resolver", _SyntheticResolver()),
        ),
    )
    assert len(owners) == 1
    assert owners[0]["native_tasks_planned"] == 0
