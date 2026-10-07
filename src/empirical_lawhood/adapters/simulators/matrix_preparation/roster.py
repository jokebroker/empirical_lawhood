"""Exact root, acquisition, numerical-view and expected action declarations.

The native development/evaluation split is an INTERNAL model-validation split. Every original
stochastic root is exposed development in the enclosing candidate and source config;
these nested labels cannot create a fresh prospective confirmation population.
"""

from dataclasses import dataclass, replace
from decimal import Decimal

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference
from empirical_lawhood.planning.native_source import NativeProjectionGroup
from empirical_lawhood.planning.response_experiment import ResponseAcquisitionGroup, ResponseAcquisitionView, ResponsePreparationUnit, ResponseDataSplitRole
from empirical_lawhood.planning.prospective_config import ActionWordRecordedSemanticCompatibility, ProspectiveActionSemanticRole, ProspectiveSequentialActionWordChart
from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_actions import response_geometry_development_action_word
from empirical_lawhood.adapters.simulators.six_matrix_response.response_assay import ResponseGeometryNativeForcePulse

from .contracts import DEVELOPMENT, PreparationSourceConfig, preparation_actions


@dataclass(frozen=True)
class PreparationNativeDeclarations:
    units: tuple[ResponsePreparationUnit, ...]
    acquisitions: tuple[ResponseAcquisitionGroup, ...]
    views: tuple[ResponseAcquisitionView, ...]
    projections: tuple[NativeProjectionGroup, ...]
    chart: ProspectiveSequentialActionWordChart


def preparation_native_declarations(
    source: PreparationSourceConfig, clock_evaluator: ExecutableReference
) -> PreparationNativeDeclarations:
    """Use the existing native action/clock contract, with explicit numerical values."""
    words = {}
    for root in source.roots:
        for action in preparation_actions(root):
            key = (root.context, action.sign, action.magnitude)
            if key in words:
                continue
            label = {-1: "neg", 0: "hold", 1: "pos"}[action.sign]
            amplitude = str(action.magnitude).replace(".", "p")
            stem = f"{DEVELOPMENT}.expected.{root.context}.a{amplitude}.{label}"
            word = response_geometry_development_action_word(
                ResponseGeometryNativeForcePulse(stem, "short-pulse-response", action.sign, root.handoff),
                denominator_id=f"{DEVELOPMENT}.quantity.denominator",
                history_id=f"{DEVELOPMENT}.quantity.history",
                receiver_id=f"{DEVELOPMENT}.quantity.receiver",
                horizon_id=f"{DEVELOPMENT}.horizon",
                clock_evaluator=clock_evaluator,
            )
            occurrences = []
            for occurrence in word.occurrences:
                channel = replace(
                    occurrence.channel,
                    support_contract_id=f"{DEVELOPMENT}.finite-force-source-support",
                )
                occurrences.append(
                    replace(
                        occurrence,
                        channel=channel,
                        requested=replace(
                            occurrence.requested,
                            value=occurrence.requested.value * action.magnitude / Decimal(8),
                        ),
                        accepted=replace(
                            occurrence.accepted,
                            value=occurrence.accepted.value * action.magnitude / Decimal(8),
                        ),
                        applied=replace(
                            occurrence.applied,
                            value=occurrence.applied.value * action.magnitude / Decimal(8),
                        ),
                        realized=replace(
                            occurrence.realized,
                            value=occurrence.realized.value * action.magnitude / Decimal(8),
                        ),
                    )
                )
            words[key] = replace(word, occurrences=tuple(occurrences))
    chart_words = tuple(sorted(words.values(), key=lambda word: word.word_id))
    compatibility = tuple(
        sorted(
            (
                ActionWordRecordedSemanticCompatibility(
                    f"compatibility.{word.word_id}",
                    ObjectIdentity.from_record(word.word_id, word),
                    "matrix-preparation-native-finite-force-chart",
                    ProspectiveActionSemanticRole.NATIVE_HOLD
                    if key[1] == 0
                    else ProspectiveActionSemanticRole.ACTIVE,
                    False,
                    False,
                    False,
                )
                for key, word in words.items()
            ),
            key=lambda row: row.compatibility_id,
        )
    )
    first = chart_words[0]
    chart = ProspectiveSequentialActionWordChart(
        f"{DEVELOPMENT}.expected-force-chart",
        chart_words,
        compatibility,
        first.denominator_id,
        first.retained_history_id,
        first.receiver_id,
        first.horizon_id,
        first.support_status,
        first.reason_codes,
        True,
        False,
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    units: list[ResponsePreparationUnit] = []
    groups: list[ResponseAcquisitionGroup] = []
    views: list[ResponseAcquisitionView] = []
    projections: list[NativeProjectionGroup] = []
    for prefix in source.prefixes:
        root, unit = prefix.root, prefix.root.physical_unit_id
        instance = f"{unit}.instance"
        units.append(
            ResponsePreparationUnit(
                unit,
                f"{DEVELOPMENT}.coordinate.{root.context}.t{root.landmark}",
                instance,
                prefix.artifact.sha256,
                f"{unit}.native-rng",
                ResponseDataSplitRole.EVALUATION
                if root.development_role == "screen"
                else ResponseDataSplitRole.DEVELOPMENT,
            )
        )
        per_view: dict[int, list[str]] = {1: [], 2: []}
        for action in preparation_actions(root):
            group = f"group.{action.action_id}"
            pair = tuple(f"view.{action.action_id}.r{r}" for r in (1, 2))
            groups.append(ResponseAcquisitionGroup(group, unit, instance, pair))
            for refinement, view_id in zip((1, 2), pair, strict=True):
                views.append(
                    ResponseAcquisitionView(
                        view_id,
                        group,
                        unit,
                        f"r{refinement}",
                        words[(root.context, action.sign, action.magnitude)].word_id,
                    )
                )
                per_view[refinement].append(view_id)
        projections.extend(
            NativeProjectionGroup(f"{root.root_id}.project.r{r}", tuple(sorted(per_view[r])))
            for r in (1, 2)
        )
    return PreparationNativeDeclarations(
        tuple(sorted(units, key=lambda row: row.physical_independent_unit_id)),
        tuple(sorted(groups, key=lambda row: row.acquisition_group_id)),
        tuple(sorted(views, key=lambda row: row.view_id)),
        tuple(sorted(projections, key=lambda row: row.projection_task_id)),
        chart,
    )
