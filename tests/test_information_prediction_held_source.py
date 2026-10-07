"One exposed information response prediction development root with a plan-matched fitted bank."

import json
import os
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.held

from empirical_lawhood.adapters.composition.prepared_response.native_authoring import PreparedResponseNativeAuthoringInput, _require_target_bank_document
from empirical_lawhood.api.native_authoring import author_matrix_response
from empirical_lawhood.adapters.methods.information_response.models import InformationResponseModelBank, features
from empirical_lawhood.adapters.methods.prepared_response.projection import project_prepared_future, project_prepared_handoff
from empirical_lawhood.adapters.simulators.information_response.contracts import InformationResponseNativeConfig, native_invocations
from empirical_lawhood.adapters.simulators.information_response.executable_binding import SOURCE_BINDING
from empirical_lawhood.adapters.simulators.information_response.extension_bundle import SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord, PreparedNativeSpec, prepared_native_member, prepared_numerical_view
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_receiver
from empirical_lawhood.adapters.simulators.prepared_response.source import bind_prepared_common_start, bind_prepared_native_handoff, execute_native_future, execute_native_parent, prepare_native_prefix
from empirical_lawhood.api.codecs import load_registered_authoring
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity

ROOT = Path(__file__).parents[1]


def _external(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} authentic held input is not supplied")
    return Path(value)


def test_shipped_information_response_source_matches_bank_features_and_native_receiver() -> None:
    authoring = load_registered_authoring(
        ROOT / "experiments/information-response/information-author.json",
        root_schemas={PreparedResponseNativeAuthoringInput.SCHEMA: PreparedResponseNativeAuthoringInput},
        maximum_bytes=16 * 1024,
    )
    held = _external("PREPARED_RESPONSE_HELD_SOURCE_ROOT")
    plan = _external("INFORMATION_RESPONSE_PREDICTION_PLAN")
    prior = _external("INFORMATION_RESPONSE_PREDICTION_PRIOR_EXPOSURE")
    bank_path = _external("INFORMATION_RESPONSE_PREDICTION_MODEL_BANK")
    selection = author_matrix_response(
        authoring,
        repo_root=ROOT,
        source_root=held,
        plan=plan,
        design_packet=plan,
        prior_exposure=prior,
        model_bank=bank_path,
    )
    assert selection["native_source_binding_selected"] == SOURCE_BINDING.binding_id
    assert selection["native_source_provider_built"] is True
    assert selection["native_source_runner_selected"] == 'InformationResponseSourceTask'
    assert selection["native_contact"] is False
    assert selection["independent_units"] == 64
    assert selection["model_bank_role"] == "EXPOSED_OUTCOME_FITTED_DEVELOPMENT"
    assert selection["prospective_issue_eligible"] is False
    assert selection["campaign_issued"] is False
    assert selection["native_tasks_executed"] == 0

    bank = decode_canonical_bytes(
        _require_target_bank_document(
            json.loads(bank_path.read_bytes()), InformationResponseModelBank.SCHEMA
        ),
        InformationResponseModelBank,
        maximum_bytes=16 * 1024**2,
    )
    assert bank.plan_sha256 == selection["native_plan_sha256"]
    recipe = PreparedNativeSpec(
        'prospective-evaluation',
        authoring.source_seed_sha256,
        bank.plan_sha256,
        sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY),
        prepared_native_member(),
        tuple(prepared_numerical_view(r) for r in (1, 2)),
        Decimal(16),
    )
    config = InformationResponseNativeConfig(recipe, bank)
    assert len(config.words) == 4
    assert len(native_invocations(config)) == 2944
    root = config.roots[0]  # public, exposed target development root
    prefixes = tuple(
        prepare_native_prefix(recipe, root, refinement=r) for r in (1, 2)
    )
    assert all(p.checkpoint is not None for p in prefixes)
    common = bind_prepared_common_start(
        prefixes[0].checkpoint, prefixes[1].checkpoint
    )
    assert common.frame is not None and common.frame.cutoff_tick == root.landmark
    parent_name = "hold"
    handoffs = tuple(
        bind_prepared_native_handoff(
            execute_native_parent(recipe, common, parent=parent_name, refinement=r)
        )
        for r in (1, 2)
    )
    measured = tuple(
        project_prepared_handoff(common, handoff, instrument_tier="I1")
        for handoff in handoffs
    )
    assert all(
        handoff.checkpoint.history_ticks[-1] == root.handoff
        and any(
            tick > common.frame.cutoff_tick for tick in handoff.checkpoint.history_ticks
        )
        for handoff in handoffs
    )
    history = np.stack([row.history for row in measured])
    sketch = np.stack([row.sketch for row in measured])
    assert history.shape == (2, 16, 12) and sketch.shape == (2, 8)
    assert {
        arm: features(history, sketch, arm).shape[1] for arm in ("snapshot", "history", "snapshot-with-mechanism", "history-with-mechanism")
    } == {
        "snapshot": 8,
        "history": 128,
        "snapshot-with-mechanism": 16,
        "history-with-mechanism": 136,
    }
    context = next(c for c in bank.contexts if c.recipe.context == root.context)
    predictions = context.predict(history, sketch, (0, 0))
    assert predictions.shape == (2, 7, 5, 2, 2)
    assert np.isfinite(predictions).all()

    word = PreparedForceWord(Decimal(16), 0, 1)
    assert word in config.words
    receiver = np.empty((2, 2))
    for index, handoff in enumerate(handoffs):
        hold = execute_native_future(
            recipe,
            common,
            handoff,
            parent=parent_name,
            word=recipe.words[0],
            purpose='common-response',
        )
        action = execute_native_future(
            recipe,
            common,
            handoff,
            parent=parent_name,
            word=word,
            purpose='common-response',
        )
        assert hold.checkpoint is not None and action.checkpoint is not None
        assert action.delivery.accepted and action.delivery.word == word
        assert action.delivery.nonzero_force_intervals == 64 * (index + 1)
        assert action.delivery.applied_force_kicks == 640 * (index + 1)
        assert tuple(hold.delivery.realized_impulse) == (0, 0)
        np.testing.assert_allclose(
            tuple(map(float, action.delivery.realized_impulse)),
            (16 * 64 * 0.001, 0.0),
            rtol=0,
            atol=1e-12,
        )
        reduced = project_prepared_future(common, handoff, action, matched_hold=hold)
        assert reduced.outputs.shape == (5, 7)
        assert reduced.known[:, :4].all()
        receiver[index] = prepared_receiver(
            common.frame, action.positions[-1, 0] - hold.positions[-1, 0]
        )
        assert np.isfinite(receiver[index]).all()
    after_future = tuple(
        project_prepared_handoff(common, handoff, instrument_tier="I1")
        for handoff in handoffs
    )
    for before, after in zip(measured, after_future, strict=True):
        np.testing.assert_array_equal(after.history, before.history)
        np.testing.assert_array_equal(after.sketch, before.sketch)
    np.testing.assert_array_equal(context.predict(history, sketch, (0, 0)), predictions)
    assert np.max(np.abs(receiver[0] - receiver[1])) < 0.01
