"One exposed source qualification development root under the shipped selected native source."

import os
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.held

from empirical_lawhood.adapters.composition.prepared_response.native_authoring import PreparedResponseNativeAuthoringInput
from empirical_lawhood.api.native_authoring import author_matrix_response
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedForceWord, PreparedNativeSpec, prepared_native_member, prepared_numerical_view
from empirical_lawhood.adapters.simulators.prepared_response.extension_bundle import SOURCE_QUALIFICATION_SOURCE_CAPABILITY
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_receiver, select_prepared_ports
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import prepared_static_native_invocations
from empirical_lawhood.adapters.simulators.prepared_response.source import bind_prepared_common_start, bind_prepared_native_handoff, execute_native_future, execute_native_parent, prepare_native_prefix
from empirical_lawhood.api.codecs import load_registered_authoring
from empirical_lawhood.kernel.provenance import ObjectIdentity

ROOT = Path(__file__).parents[1]


def _external(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} authentic held input is not supplied")
    return Path(value)


def test_shipped_q_selected_source_keeps_five_parents_and_seventeen_words() -> None:
    config = load_registered_authoring(
        ROOT / "experiments/prepared-response/prepared-source-qualification-author.json",
        root_schemas={PreparedResponseNativeAuthoringInput.SCHEMA: PreparedResponseNativeAuthoringInput},
        maximum_bytes=16 * 1024,
    )
    held = _external("PREPARED_RESPONSE_HELD_SOURCE_ROOT")
    plan = _external("PREPARED_RESPONSE_SOURCE_QUALIFICATION_PLAN")
    design = _external("PREPARED_RESPONSE_SOURCE_QUALIFICATION_DESIGN_PACKET")
    prior = _external("PREPARED_RESPONSE_SOURCE_QUALIFICATION_PRIOR_EXPOSURE")
    selection = author_matrix_response(
        config,
        repo_root=ROOT,
        source_root=held,
        plan=plan,
        design_packet=design,
        prior_exposure=prior,
        model_bank=None,
    )
    assert selection["native_source_binding_selected"] == "binding.prepared-response.source-qualification.source"
    assert selection["native_source_provider_built"] is True
    assert selection["native_source_runner_selected"] == 'PreparedStaticSourceTask'
    assert selection["native_contact"] is False
    assert selection["independent_units"] == 32
    assert selection["prior_custody_authenticated"] is False
    assert selection["prospective_issue_eligible"] is False
    assert selection["campaign_issued"] is False
    assert selection["native_tasks_executed"] == 0
    spec = PreparedNativeSpec(
        'qualification',
        config.source_seed_sha256,
        selection["native_plan_sha256"],
        sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        ObjectIdentity.from_record(
            SOURCE_QUALIFICATION_SOURCE_CAPABILITY.capability_key, SOURCE_QUALIFICATION_SOURCE_CAPABILITY
        ),
        prepared_native_member(),
        tuple(prepared_numerical_view(refinement) for refinement in (1, 2)),
        None,
    )
    assert spec.stage == 'qualification' and spec.selected_amplitude is None
    assert spec.seed_sha256 == config.source_seed_sha256
    assert spec.implementation_plan_sha256 == sha256(plan.read_bytes()).hexdigest()
    assert len(spec.words) == 17
    assert spec.words[0] == PreparedForceWord(Decimal(0), 0, 0)
    assert {
        (word.magnitude, word.direction_index, word.sign) for word in spec.words[1:]
    } == {
        (amplitude, direction, sign)
        for amplitude in (Decimal(8), Decimal(16))
        for direction in range(4)
        for sign in (-1, 1)
    }
    root = spec.roots[0]  # public target development seed, exposed by this check
    invocations = prepared_static_native_invocations(spec, root=root)
    assert len(invocations) == 1 + len(PARENTS) * (1 + len(spec.words))

    prefixes = tuple(
        prepare_native_prefix(spec, root, refinement=refinement)
        for refinement in (1, 2)
    )
    assert all(item.checkpoint is not None for item in prefixes)
    common = bind_prepared_common_start(
        prefixes[0].checkpoint, prefixes[1].checkpoint
    )
    assert common.frame is not None
    assert common.frame.cutoff_tick == root.landmark
    from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode

    causal = _decode(prefixes[0].checkpoint.history_positions_base64, (31, 2, 3, 4, 4))[
        -16:, 0
    ]
    # All observations after the sixteen-sample pre-parent cutoff are poison.
    # The actual common-start frame was already frozen from this prefix.
    poisoned = np.concatenate((causal, np.full_like(causal, np.nan)))
    independent_frame = select_prepared_ports(
        ticks=prefixes[0].checkpoint.history_ticks[-16:],
        observations=poisoned[:16],
        cutoff_tick=root.landmark,
    )
    assert independent_frame is not None
    np.testing.assert_array_equal(common.frame.modes, independent_frame.modes)
    hold_word = spec.words[0]
    action_word = PreparedForceWord(Decimal(8), 0, 1)
    receivers = np.empty((len(PARENTS), 2, 2))
    for parent_index, parent_name in enumerate(PARENTS):
        for view_index, refinement in enumerate((1, 2)):
            parent = execute_native_parent(
                spec, common, parent=parent_name, refinement=refinement
            )
            assert parent.delivery.accepted is True
            assert parent.delivery.phase == "parent"
            assert parent.delivery.parent == parent_name
            handoff = bind_prepared_native_handoff(parent)
            hold = execute_native_future(
                spec,
                common,
                handoff,
                parent=parent_name,
                word=hold_word,
                purpose='common-response',
            )
            action = execute_native_future(
                spec,
                common,
                handoff,
                parent=parent_name,
                word=action_word,
                purpose='common-response',
            )
            assert hold.checkpoint is not None and action.checkpoint is not None
            assert hold.delivery.phase == action.delivery.phase == "future"
            assert hold.delivery.word == hold_word
            assert action.delivery.word == action_word
            assert hold.delivery.accepted and action.delivery.accepted
            assert hold.delivery.nonzero_force_intervals == 0
            assert action.delivery.nonzero_force_intervals == 64 * refinement
            assert action.delivery.applied_force_kicks == 640 * refinement
            assert tuple(hold.delivery.realized_impulse) == (0, 0)
            np.testing.assert_allclose(
                tuple(map(float, action.delivery.realized_impulse)),
                (8 * 64 * 0.001, 0.0),
                rtol=0,
                atol=1e-12,
            )
            receiver = prepared_receiver(
                common.frame, action.positions[-1, 0] - hold.positions[-1, 0]
            )
            assert receiver.shape == (2,) and np.isfinite(receiver).all()
            receivers[parent_index, view_index] = receiver
    assert np.max(np.abs(receivers[:, 0] - receivers[:, 1])) < 0.01
    assert len({tuple(row[0]) for row in receivers}) > 1
