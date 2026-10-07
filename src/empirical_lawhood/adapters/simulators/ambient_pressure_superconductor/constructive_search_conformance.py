'Outcome-blind constructive search constructive-search conformance on the six frozen worlds'

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .material_source_design_contracts import ExplorationDesignFreeze
from .material_fixed_scientific_inputs import fixed_material_exploration_input
from .material_source_design_design import build_exploration_design, build_material_roster, truth_world_documents
from .material_source_design_methods import ExplorationCandidate, ExplorationHistory, ExplorationNomination, nominate_exploration_wave
from .material_control_contracts import MaterialControlTruthWorldConformance, MaterialControlTruthWorldResult


def _candidate(
    action_id: str,
    *,
    family_id: str,
    parent_action_id: str | None,
    scalar_tc: str,
    margin: str,
    unresolved: bool = False,
    sign: bool = False,
    boundary: bool = False,
    available: bool = True,
    prechecks: bool = True,
    irrecoverable: bool = False,
    out_of_support: bool = False,
) -> ExplorationCandidate:
    return ExplorationCandidate(
        action_id=action_id,
        family_id=family_id,
        parent_action_id=parent_action_id,
        gate_margin_vector=(
            ("gate.reachability", Decimal(margin)),
            ("gate.stability", Decimal(margin)),
        ),
        unresolved_chart_overlap=unresolved,
        sign_change_bracket=sign,
        phase_boundary_bracket=boundary,
        common_margin_uncertainty=Decimal("0.2") if unresolved else Decimal("0.01"),
        scalar_predicted_tc_K=Decimal(scalar_tc),
        action_cost_units=1,
        compute_cost_units=2,
        discontinuous_bridge=action_id == "action.target",
        family_restart=False,
        available=available,
        hard_prechecks_pass=prechecks,
        irrecoverable=irrecoverable,
        explicitly_out_of_support=out_of_support,
    )


def _world_candidates(name: str) -> tuple[ExplorationCandidate, ...]:
    empty = name == "complete-empty"
    mismatch = name == "requested-realized-mismatch"
    disconnected = name == "disconnected-target"
    return tuple(
        sorted(
            (
                _candidate(
                    "action.a",
                    family_id="family.f1",
                    parent_action_id=None,
                    scalar_tc="250",
                    margin="1",
                    unresolved=True,
                    prechecks=not (empty or mismatch),
                ),
                _candidate(
                    "action.b",
                    family_id="family.f1",
                    parent_action_id=None,
                    scalar_tc="500",
                    margin="0",
                    boundary=True,
                    available=not disconnected,
                    prechecks=not (empty or mismatch),
                ),
                _candidate(
                    "action.c",
                    family_id="family.f2",
                    parent_action_id="action.a",
                    scalar_tc="300",
                    margin="2",
                    sign=True,
                    available=not disconnected,
                    prechecks=not (empty or mismatch),
                ),
                _candidate(
                    "action.target",
                    family_id="family.f2",
                    parent_action_id="action.c",
                    scalar_tc="350",
                    margin="3",
                    available=not (disconnected or empty or mismatch),
                    prechecks=not (empty or mismatch),
                ),
            ),
            key=lambda value: value.action_id,
        )
    )


def _run_policy(
    *, policy_id: str, candidates: tuple[ExplorationCandidate, ...], design: ExplorationDesignFreeze
) -> tuple[tuple[ExplorationNomination, ...], ExplorationHistory]:
    history = ExplorationHistory(
        policy_id=policy_id,
        observed_action_ids=(),
        observed_family_ids=(),
        charged_action_units=0,
        charged_compute_units=0,
        family_restart_count=0,
        deterministic_seed='ambient-pressure-superconductor-constructive-search-fixed-order',
    )
    by_id = {value.action_id: value for value in candidates}
    nominations: list[ExplorationNomination] = []
    for wave in range(1, design.wave_count + 1):
        nomination = nominate_exploration_wave(
            policy_id=policy_id,
            candidates=candidates,
            history=history,
            actions_per_wave=design.actions_per_policy_wave,
            action_budget=design.policy_action_budget,
            compute_budget=design.policy_compute_budget_units,
            family_restart_quota=design.family_restart_quota,
            bridge_quota=design.bridge_action_quota_per_wave,
            wave_index=wave,
            scientific_input=fixed_material_exploration_input(
                candidates=candidates, history=history, wave_index=wave,
                budget_operands=(design.actions_per_policy_wave, design.policy_action_budget,
                                 design.policy_compute_budget_units, design.family_restart_quota,
                                 design.bridge_action_quota_per_wave),
            ),
        )
        nominations.append(nomination)
        selected = tuple(by_id[value] for value in nomination.selected_action_ids)
        history = ExplorationHistory(
            policy_id=policy_id,
            observed_action_ids=tuple(
                sorted(set(history.observed_action_ids) | set(nomination.selected_action_ids))
            ),
            observed_family_ids=tuple(
                sorted(set(history.observed_family_ids) | {value.family_id for value in selected})
            ),
            charged_action_units=history.charged_action_units + nomination.action_units_charged,
            charged_compute_units=history.charged_compute_units + nomination.compute_units_charged,
            family_restart_count=history.family_restart_count
            + sum(value.family_restart for value in selected),
            deterministic_seed=history.deterministic_seed,
        )
        if nomination.hold:
            break
    return tuple(nominations), history


def _matches_world(
    name: str,
    *,
    nominations: dict[str, tuple[ExplorationNomination, ...]],
    histories: dict[str, ExplorationHistory],
) -> bool:
    response = histories["policy.response-guided"].observed_action_ids
    scalar = histories["policy.scalar-predicted-tc"].observed_action_ids
    first_response = nominations["policy.response-guided"][0]
    if name == "narrow-corridor":
        return {"action.a", "action.c"}.issubset(response)
    if name == "scalar-deceptive-ridge":
        return "action.b" in scalar and "action.b" not in response
    if name == "fidelity-reversal":
        return "action.c" in response and "action.b" in scalar
    if name == "disconnected-target":
        return "action.target" not in response and nominations["policy.response-guided"][-1].hold
    if name in {"complete-empty", "requested-realized-mismatch"}:
        return first_response.hold and all(values[0].hold for values in nominations.values())
    raise ValueError('unknown frozen constructive search truth world')


def _policy_matches(name: str, *, histories: dict[str, ExplorationHistory]) -> dict[str, bool]:
    expected = (
        ()
        if name in {"complete-empty", "requested-realized-mismatch"}
        else (("action.a",) if name == "disconnected-target" else None)
    )
    response_expected = expected if expected is not None else ("action.a", "action.c")
    comparator_expected = expected if expected is not None else ("action.a", "action.b", "action.c")
    return {
        "policy.response-guided": (
            histories["policy.response-guided"].observed_action_ids == response_expected
        ),
        "policy.scalar-predicted-tc": (
            histories["policy.scalar-predicted-tc"].observed_action_ids == comparator_expected
        ),
        "policy.stratified-random": (
            histories["policy.stratified-random"].observed_action_ids == comparator_expected
            and histories["policy.stratified-random"].deterministic_seed
            == 'ambient-pressure-superconductor-constructive-search-fixed-order'
        ),
    }


def run_constructive_search_conformance() -> tuple[MaterialControlTruthWorldConformance, tuple[MaterialControlTruthWorldResult, ...]]:
    design = build_exploration_design(build_material_roster())
    results: list[MaterialControlTruthWorldResult] = []
    policy_world_passes: dict[str, list[bool]] = {
        value.policy_id: [] for value in design.policy_specs
    }
    policy_ids = tuple(value.policy_id for value in design.policy_specs)
    for lock in design.truth_worlds:
        name = lock.world_id.removeprefix("world.")
        public, truth = truth_world_documents(name)
        public_sha = sha256(canonical_json_bytes(public)).hexdigest()
        truth_sha = sha256(canonical_json_bytes(truth)).hexdigest()
        if public_sha != lock.public_graph_sha256 or truth_sha != lock.privileged_truth_sha256:
            raise ValueError('constructive search truth-world bytes differ from the material source design freeze')
        candidates = _world_candidates(name)
        nominations: dict[str, tuple[ExplorationNomination, ...]] = {}
        histories: dict[str, ExplorationHistory] = {}
        for policy_id in policy_ids:
            values, history = _run_policy(
                policy_id=policy_id,
                candidates=candidates,
                design=design,
            )
            nominations[policy_id] = values
            histories[policy_id] = history
        accounting = all(
            history.charged_action_units <= design.policy_action_budget
            and history.charged_compute_units <= design.policy_compute_budget_units
            and history.observed_action_ids
            == tuple(
                sorted(
                    {
                        action_id
                        for nomination in nominations[policy_id]
                        for action_id in nomination.selected_action_ids
                    }
                )
            )
            for policy_id, history in histories.items()
        )
        hold_logic = all(
            nomination.hold == (not nomination.selected_action_ids)
            for values in nominations.values()
            for nomination in values
        )
        matched_science = _matches_world(
            name,
            nominations=nominations,
            histories=histories,
        )
        policy_matches = _policy_matches(name, histories=histories)
        for policy_id, passed in policy_matches.items():
            policy_world_passes[policy_id].append(passed)
        matched_science = matched_science and all(policy_matches.values())
        observed = (
            lock.expected_disposition_id if matched_science else 'disposition.constructive-search-world-mismatch'
        )
        results.append(
            MaterialControlTruthWorldResult(
                result_id=f'result.material-control-constructive-search-{name}',
                world_id=lock.world_id,
                public_graph_sha256=public_sha,
                privileged_truth_sha256=truth_sha,
                expected_disposition_id=lock.expected_disposition_id,
                observed_disposition_id=observed,
                policy_nomination_sha256s=tuple(
                    sorted(
                        (
                            policy_id,
                            sha256(canonical_json_bytes(nominations[policy_id])).hexdigest(),
                        )
                        for policy_id in policy_ids
                    )
                ),
                policy_history_separated=(
                    {value.policy_id for value in histories.values()} == set(policy_ids)
                ),
                accounting_passed=accounting,
                hold_logic_passed=hold_logic,
                matched_expected=(
                    observed == lock.expected_disposition_id and accounting and hold_logic
                ),
                target_contact_count=0,
                reason_codes=(
                    'reason.constructive-search-frozen-world-matched'
                    if matched_science
                    else 'reason.constructive-search-frozen-world-mismatch',
                ),
            )
        )
    ordered = tuple(sorted(results, key=lambda value: value.world_id))
    all_matched = all(value.matched_expected for value in ordered)
    conformance = MaterialControlTruthWorldConformance(
        result_id='result.ambient-pressure-superconductor-material-control-constructive-search-exploration-conformance',
        exploration_design_sha256=design.fingerprint(),
        world_result_ids=tuple(value.result_id for value in ordered),
        required_world_count=len(design.truth_worlds),
        matched_world_count=sum(value.matched_expected for value in ordered),
        response_policy_pass=all(policy_world_passes["policy.response-guided"]),
        scalar_policy_pass=all(policy_world_passes["policy.scalar-predicted-tc"]),
        random_policy_pass=all(policy_world_passes["policy.stratified-random"]),
        accounting_pass=all(value.accounting_passed for value in ordered),
        hold_logic_pass=all(value.hold_logic_passed for value in ordered),
        target_contact_count=0,
        disposition_id=(
            'constructive-search.exploration-conformance-pass' if all_matched else 'constructive-search.exploration-conformance-fail'
        ),
        reason_codes=(
            'reason.constructive-search-six-world-intersection-pass'
            if all_matched
            else 'reason.constructive-search-six-world-intersection-fail',
        ),
    )
    return conformance, ordered


__all__ = ['run_constructive_search_conformance']
