# SPDX-License-Identifier: MPL-2.0

"Synthetic nonpromotable source qualification/dependent refinement fixtures for the public dependent input seam."

from tests.prepared_seed_fixtures import fixture_seed_census

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

import numpy as np
import pytest

from empirical_lawhood.adapters.methods.prepared_response.calibration_records import PreparedResponseCalibrationCalibrationConfig, PreparedResponseCalibrationProjectionConfig
from empirical_lawhood.adapters.methods.prepared_response.development_fit import PreparedResponseDevelopmentFitConfig, fit_prepared_response_development_structure
from empirical_lawhood.adapters.methods.prepared_response.development_projection import PreparedResponseDevelopmentProjectionConfig, PreparedResponseDevelopmentViewProjection, _blocks
from empirical_lawhood.adapters.methods.prepared_response.development_selection import PreparedResponseDevelopmentCandidateCost, PreparedResponseDevelopmentSelectionConfig, select_prepared_response_development_nominal_library
from empirical_lawhood.adapters.methods.prepared_response.extension_bundle import CALIBRATION_CALIBRATION_CAPABILITY, CALIBRATION_POLICY_CAPABILITY, CALIBRATION_PROJECTION_CAPABILITY
from empirical_lawhood.adapters.methods.prepared_response.models import STRUCTURES
from empirical_lawhood.adapters.methods.prepared_response.policy_decision import PreparedPolicyDecisionConfig
from empirical_lawhood.adapters.methods.prepared_response.qualification import PreparedResponseSourceQualificationEvaluationConfig, evaluate_prepared_response_source_qualification
from empirical_lawhood.adapters.methods.prepared_response.qualification_records import PreparedResponseSourceQualificationProjectionConfig, PreparedResponseSourceQualificationViewObservation, PreparedResponseSourceQualificationWordObservation
from empirical_lawhood.adapters.methods.prepared_response.statistics import PreparedStatisticalSpec
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedNativeSpec, prepared_native_member, prepared_numerical_view
from empirical_lawhood.adapters.simulators.prepared_response.development_roster import prepared_response_development_crossfit_plan
from empirical_lawhood.adapters.simulators.prepared_response.extension_bundle import CALIBRATION_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import prepared_static_native_invocations
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedCommonStart, PreparedNativeHandoff
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.capabilities import CapabilityRegistry


@pytest.fixture(scope="module")
def qualification_fixture():
    # Preserve the original synthetic stream commitment across descriptive naming.
    seed = sha256(bytes((115, 121, 110, 116, 104, 101, 116, 105, 99, 32, 99, 104, 97, 114, 116, 101, 114, 32, 99, 111, 110, 102, 111, 114, 109, 97, 110, 99, 101, 59, 32, 110, 101, 118, 101, 114, 32, 110, 97, 116, 105, 118, 101, 32, 81, 32, 101, 118, 105, 100, 101, 110, 99, 101))).hexdigest()
    source = PreparedNativeSpec(
        'qualification',
        seed,
        "a" * 64,
        "b" * 64,
        ObjectIdentity(
            "synthetic.code", 'empirical-lawhood/synthetic/code', "1.0.0", "c" * 64
        ),
        prepared_native_member(),
        tuple(prepared_numerical_view(r) for r in (1, 2)),
        None,
        root_seed_census=fixture_seed_census("qualification", seed),
    )
    projection = PreparedResponseSourceQualificationProjectionConfig("synthetic.source-qualification.projection", source)
    config = PreparedResponseSourceQualificationEvaluationConfig("synthetic.source-qualification.evaluation", projection)
    source_id = ObjectIdentity.from_record(source.spec_id, source)
    projection_id = ObjectIdentity.from_record(projection.config_id, projection)
    tasks = prepared_static_native_invocations(source)
    reports = []
    directions = (
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    )
    for root in source.roots:
        identities = tuple(
            ObjectIdentity(
                f"{t.task_id}.result",
                PreparedNativeTaskResult.SCHEMA,
                "1.0.0",
                sha256(t.task_id.encode()).hexdigest(),
            )
            for t in tasks
            if t.root == root
        )
        by_id = {v.object_id: v for v in identities}
        words = []
        for parent in PARENTS:
            for word_index, word in enumerate(source.words):
                draw = 0 if word_index == 0 else (word_index - 1) % 8 + 1
                vector = np.array(directions[draw], dtype=np.float64)
                if draw:
                    vector *= 0.125 / np.linalg.norm(vector)
                if parent == "hold":
                    vector *= 0
                row = (
                    *(Decimal(str(float(v))) for v in vector),
                    *(Decimal(0) for _ in range(5)),
                )
                result_id = (
                    f"{root.root_id}.{parent}.common-response.{word.word_id}.native.result"
                )
                words.append(
                    PreparedResponseSourceQualificationWordObservation(
                        parent, word, by_id[result_id], True, Decimal(0), (row,) * 5
                    )
                )
        common = ObjectIdentity(
            f"{root.root_id}.common-start",
            'empirical-lawhood/simulators/prepared-response/prepared-common-start',
            "1.0.0",
            root.fingerprint(),
        )
        for refinement in (1, 2):
            reports.append(
                PreparedResponseSourceQualificationViewObservation(
                    projection_id,
                    source_id,
                    root,
                    refinement,
                    identities,
                    common,
                    "RESOLVED",
                    (Decimal(0), Decimal(0)),
                    ((Decimal(0), Decimal(0)),) * 5,
                    (Decimal(0),) * 5,
                    tuple(words),
                )
            )
    return config, tuple(reports)


@pytest.fixture(scope="module")
def development_reports(qualification_fixture):
    q, views = qualification_fixture
    nomination = evaluate_prepared_response_source_qualification(q, views)
    native = replace(q.projection.native_spec, stage='development', selected_amplitude=Decimal(8), root_seed_census=fixture_seed_census('development', q.projection.native_spec.seed_sha256))
    projection = PreparedResponseDevelopmentProjectionConfig(
        "synthetic.dependent-refinement.full-projection", native, nomination
    )
    config = PreparedResponseDevelopmentFitConfig(
        "synthetic.dependent-refinement.full-fit", projection, prepared_response_development_crossfit_plan(native)
    )
    tasks = prepared_static_native_invocations(native)
    reports = []
    rng = np.random.default_rng(199001)
    controls = np.array(
        (
            (0, 0),
            (-8, 0),
            (8, 0),
            (0, -8),
            (0, 8),
            (-8 / np.sqrt(2), -8 / np.sqrt(2)),
            (8 / np.sqrt(2), 8 / np.sqrt(2)),
            (-8 / np.sqrt(2), 8 / np.sqrt(2)),
            (8 / np.sqrt(2), -8 / np.sqrt(2)),
        )
    )
    for root in native.roots:
        history = rng.normal(size=(5, 2, 16, 12)) * 0.01
        transition = np.repeat(
            np.repeat(history[..., -1:, :], 21, axis=-2)[..., None, :, :], 9, axis=-3
        )
        for step in range(21):
            time = step * 0.016
            pulse = min(time, 0.064)
            displacement = history[..., -1, (1, 10)][
                ..., None, :
            ] * time + controls * pulse * (time - pulse / 2)
            transition[..., step, (0, 9)] = (
                history[..., -1, (0, 9)][..., None, :] + displacement
            )
        observed = np.full((5, 2, 9, 5, 7), 0.001)
        observed[..., :2] = (
            transition[..., (4, 8, 12, 16, 20), :][..., (0, 9)]
            - history[..., -1, (0, 9)][..., None, None, :]
        )
        observed[:, :, 0, :, 2:5] = 0
        hold = np.repeat(observed[:, :, 0:1], 4, axis=2)
        arrays = {
            "preparent_history": history[0],
            "preparent_sketch": np.full((2, 8), np.nan),
            "history": history,
            "sketch": np.full((5, 2, 8), np.nan),
            "handoff_displacement": np.zeros((5, 2, 2)),
            "parent_absolute_density_work": np.zeros((5, 2)),
            "observed": observed,
            "transitions": transition,
            "hold_replicates": hold,
            "delivery_complete": np.ones((5, 2, 12), dtype=bool),
            "force_component_error": np.zeros((5, 2, 12)),
            "fit_valid": np.ones((5, 2), dtype=bool),
        }
        sources = tuple(
            ObjectIdentity(
                f"{t.task_id}.result",
                PreparedNativeTaskResult.SCHEMA,
                "1.0.0",
                sha256(t.task_id.encode()).hexdigest(),
            )
            for t in tasks
            if t.root == root
        )
        common = ObjectIdentity(
            f"{root.root_id}.synthetic.common",
            PreparedCommonStart.SCHEMA,
            "1.0.0",
            root.fingerprint(),
        )
        handoffs = tuple(
            ObjectIdentity(
                f"{root.root_id}.synthetic.handoff.{i}",
                PreparedNativeHandoff.SCHEMA,
                "1.0.0",
                root.fingerprint(),
            )
            for i in range(10)
        )
        for refinement in (1, 2):
            index = refinement - 1
            selected = {
                name: value[index]
                if name in ("preparent_history", "preparent_sketch")
                else value[:, index]
                for name, value in arrays.items()
            }
            reports.append(
                PreparedResponseDevelopmentViewProjection(
                    projection,
                    root,
                    refinement,
                    sources,
                    common,
                    handoffs[index::2],
                    "RESOLVED",
                    _blocks(selected),
                )
            )
    return config, tuple(reports)


def _selection_config(config: PreparedResponseDevelopmentFitConfig) -> PreparedResponseDevelopmentSelectionConfig:
    canary = ObjectIdentity(
        "synthetic.prepared-development.excluded-canary",
        'empirical-lawhood/testing/fixtures/excluded-canary',
        "1.0.0",
        "e" * 64,
    )
    costs = tuple(
        PreparedResponseDevelopmentCandidateCost(
            f"synthetic.prepared-development.cost.{index:02d}.{structure}",
            structure,
            "I1" if structure == "mechanism-i1" else "I0",
            canary,
            11,
            100 + index,
            200 + index,
        )
        for index, structure in enumerate(STRUCTURES)
    )
    return PreparedResponseDevelopmentSelectionConfig("synthetic.prepared-development.selection", config, costs)


@pytest.fixture(scope="module")
def calibration_records(development_reports):
    d_config, reports = development_reports
    fits = tuple(
        fit_prepared_response_development_structure(
            d_config,
            context,
            structure,
            tuple(value for value in reports if value.root.context == context),
        )
        for context in ("assembling", "prepared")
        for structure in STRUCTURES
    )
    library = select_prepared_response_development_nominal_library(
        _selection_config(d_config), reports, fits
    )
    source = replace(
        d_config.projection.native_spec,
        stage='calibration',
        root_seed_census=fixture_seed_census('calibration', d_config.projection.native_spec.seed_sha256),
        native_implementation=ObjectIdentity.from_record(
            CALIBRATION_SOURCE_CAPABILITY.capability_key, CALIBRATION_SOURCE_CAPABILITY
        ),
    )
    policy = PreparedPolicyDecisionConfig("synthetic.fresh-response-calibration.policy", source, library)
    projection = PreparedResponseCalibrationProjectionConfig("synthetic.fresh-response-calibration.projection", source, policy)
    calibration = PreparedResponseCalibrationCalibrationConfig(
        "synthetic.fresh-response-calibration.calibration",
        projection,
        PreparedStatisticalSpec("synthetic.fresh-response-calibration.statistics"),
    )
    registry = CapabilityRegistry(
        "synthetic.fresh-response-calibration.registry",
        tuple(
            sorted(
                (
                    CALIBRATION_SOURCE_CAPABILITY,
                    CALIBRATION_POLICY_CAPABILITY,
                    CALIBRATION_PROJECTION_CAPABILITY,
                    CALIBRATION_CALIBRATION_CAPABILITY,
                ),
                key=lambda value: value.registry_id,
            )
        ),
    )
    return (source, policy, projection, calibration), registry, fits
