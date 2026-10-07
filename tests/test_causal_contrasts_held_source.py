"One exposed causal response prediction root under its bank-matching earlier design plan."

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
from empirical_lawhood.adapters.methods.causal_response.models import CausalResponseModelBank, central_gain
from empirical_lawhood.adapters.methods.prepared_response.projection import project_prepared_future, project_prepared_handoff
from empirical_lawhood.adapters.simulators.causal_response.contracts import CausalResponseNativeConfig, native_invocations
from empirical_lawhood.adapters.simulators.causal_response.executable_binding import SOURCE_BINDING
from empirical_lawhood.adapters.simulators.causal_response.extension_bundle import SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedNativeSpec, prepared_native_member, prepared_numerical_view
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


def test_shipped_causal_response_source_has_bank_prediction_and_parent_zero_native_contrast() -> (
    None
):
    authoring = load_registered_authoring(
        ROOT / "experiments/causal-response/causal-author.json",
        root_schemas={PreparedResponseNativeAuthoringInput.SCHEMA: PreparedResponseNativeAuthoringInput},
        maximum_bytes=16 * 1024,
    )
    held = _external("PREPARED_RESPONSE_HELD_SOURCE_ROOT")
    earlier_plan = _external("CAUSAL_RESPONSE_PREDICTION_BANK_MATCHING_PLAN")
    design = _external("CAUSAL_RESPONSE_PREDICTION_DESIGN_PACKET")
    prior = _external("CAUSAL_RESPONSE_PREDICTION_PRIOR_EXPOSURE")
    bank_path = _external("CAUSAL_RESPONSE_PREDICTION_MODEL_BANK")
    selection = author_matrix_response(
        authoring,
        repo_root=ROOT,
        source_root=held,
        plan=earlier_plan,
        design_packet=design,
        prior_exposure=prior,
        model_bank=bank_path,
    )
    assert selection["native_source_binding_selected"] == SOURCE_BINDING.binding_id
    assert selection["native_source_provider_built"] is True
    assert selection["native_source_runner_selected"] == 'CausalResponseSourceTask'
    assert selection["native_contact"] is False
    assert selection["independent_units"] == 64
    assert selection["model_bank_role"] == "EXPOSED_OUTCOME_FITTED_DEVELOPMENT"
    assert selection["prospective_issue_eligible"] is False
    assert selection["campaign_issued"] is False
    assert selection["native_tasks_executed"] == 0
    assert (
        selection["native_plan_sha256"] == sha256(earlier_plan.read_bytes()).hexdigest()
    )

    bank = decode_canonical_bytes(
        _require_target_bank_document(
            json.loads(bank_path.read_bytes()), CausalResponseModelBank.SCHEMA
        ),
        CausalResponseModelBank,
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
    config = CausalResponseNativeConfig(recipe, bank)
    assert (
        len(recipe.words) == 9
    )  # fitted source qualification chart, including HOLD and all four directions
    assert len(config.words) == 4  # causal response prediction native assignment: ± first two directions
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
    context = next(c for c in bank.contexts if c.recipe.context == root.context)
    model = context.predictor()
    observed = np.empty((2, 2, 4, 5, 2))  # parent, view, assigned word, time, receiver
    predicted = np.empty((2, 2, 5, 2, 2))  # parent, view, time, receiver, direction
    retained_histories = []
    for parent_index, parent_name in enumerate(("hold", "y-positive-128")):
        handoffs = tuple(
            bind_prepared_native_handoff(
                execute_native_parent(
                    recipe, common, parent=parent_name, refinement=r
                )
            )
            for r in (1, 2)
        )
        tier = "I1" if model.structure == "mechanism-i1" else "I0"
        handoff_rows = tuple(
            project_prepared_handoff(common, handoff, instrument_tier=tier)
            for handoff in handoffs
        )
        assert all(
            handoff.checkpoint.history_ticks[-1] == root.handoff
            and any(
                tick > common.frame.cutoff_tick
                for tick in handoff.checkpoint.history_ticks
            )
            for handoff in handoffs
        )
        history = np.stack([row.history for row in handoff_rows])
        retained_histories.append(history)
        sketch = (
            np.stack([row.sketch for row in handoff_rows]) if tier == "I1" else None
        )
        predicted[parent_index] = central_gain(model.predict(history, sketch))
        assert np.isfinite(predicted[parent_index]).all()
        for view_index, handoff in enumerate(handoffs):
            innovations = set()
            for word_index, word in enumerate(config.words):
                future = execute_native_future(
                    recipe,
                    common,
                    handoff,
                    parent=parent_name,
                    word=word,
                    purpose='common-response',
                )
                assert future.checkpoint is not None and future.delivery.accepted
                assert future.delivery.word == word
                assert future.delivery.nonzero_force_intervals == 64 * (view_index + 1)
                assert future.delivery.applied_force_kicks == 640 * (view_index + 1)
                expected = np.zeros(2)
                expected[word.direction_index] = (
                    float(word.magnitude) * word.sign * 0.064
                )
                np.testing.assert_allclose(
                    tuple(map(float, future.delivery.realized_impulse)),
                    expected,
                    rtol=0,
                    atol=1e-12,
                )
                reduced = project_prepared_future(
                    common, handoff, future, matched_hold=None
                )
                assert reduced.outputs.shape == (5, 7)
                assert reduced.known[:, :2].all()
                observed[parent_index, view_index, word_index] = reduced.outputs[:, :2]
                innovations.add(future.delivery.innovation_sha256)
            assert len(innovations) == 1  # matched stochastic future across words
        for before, handoff in zip(handoff_rows, handoffs, strict=True):
            after = project_prepared_handoff(common, handoff, instrument_tier=tier)
            np.testing.assert_array_equal(after.history, before.history)
            if tier == "I1":
                np.testing.assert_array_equal(after.sketch, before.sketch)
        np.testing.assert_array_equal(
            central_gain(model.predict(history, sketch)), predicted[parent_index]
        )
    assert not np.array_equal(retained_histories[0], retained_histories[1])
    # The fitted bank predicts the original nine-word chart. Its central-gain
    # formula consumes only the four words actually assigned to causal response prediction here.
    gain = np.stack(
        (
            (observed[:, :, 1] - observed[:, :, 0]) / 2,
            (observed[:, :, 3] - observed[:, :, 2]) / 2,
        ),
        axis=-1,
    )
    assert gain.shape == (2, 2, 5, 2, 2)
    contrast = gain[1] - gain[0]
    assert np.isfinite(contrast).all()
    assert np.any(np.abs(contrast) > 0)
    assert np.max(np.abs(contrast[0, (2, 4)] - contrast[1, (2, 4)])) < 0.01
    assert np.isfinite(predicted[1] - predicted[0]).all()
