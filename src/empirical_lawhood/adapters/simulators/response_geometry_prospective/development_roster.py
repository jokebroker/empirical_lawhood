"""Prospective development unit, acquisition, view and expected pulse identities."""

from dataclasses import dataclass
from hashlib import sha256

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.native_source import NativeProjectionGroup
from empirical_lawhood.planning.response_experiment import ResponseAcquisitionGroup, ResponseAcquisitionView, ResponsePreparationUnit, ResponseDataSplitRole
from empirical_lawhood.planning.prospective_config import ActionWordRecordedSemanticCompatibility, ProspectiveActionSemanticRole, ProspectiveSequentialActionWordChart
from empirical_lawhood.adapters.simulators.six_matrix_response.response_assay import ResponseGeometryNativeForcePulse
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_PANEL_ID, ResponseGeometryDevelopmentNativeConfig, ResponseGeometryDevelopmentNativeSegment
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS

from .development_actions import response_geometry_development_action_word


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentNativeDeclarations:
    units: tuple[ResponsePreparationUnit, ...]
    acquisitions: tuple[ResponseAcquisitionGroup, ...]
    views: tuple[ResponseAcquisitionView, ...]
    projections: tuple[NativeProjectionGroup, ...]
    chart: ProspectiveSequentialActionWordChart


def response_geometry_development_native_declarations(
    source: ResponseGeometryDevelopmentNativeConfig, clock_evaluator: ExecutableReference
) -> ResponseGeometryDevelopmentNativeDeclarations:
    """Words describe expected delivery; each native receipt binds its own pulse."""
    prefix = DEVELOPMENT_PANEL_ID
    words = {}
    for root in source.roots:
        for sign in (-1, 0, 1):
            label = {-1: "neg", 0: "zero", 1: "pos"}[sign]
            key = (root.context, root.invocation_offset, sign)
            words[key] = response_geometry_development_action_word(
                ResponseGeometryNativeForcePulse(
                    f"{prefix}.expected.{root.context}.offset{root.invocation_offset}.{label}",
                    "short-pulse-response",
                    sign,
                    root.landmark_tick + root.invocation_offset,
                ),
                denominator_id=f"{prefix}.quantity.denominator",
                history_id=f"{prefix}.quantity.history",
                receiver_id=f"{prefix}.quantity.receiver",
                horizon_id=f"{prefix}.horizon",
                clock_evaluator=clock_evaluator,
            )
    chart_words = tuple(sorted(words.values(), key=lambda word: word.word_id))
    compatibility = tuple(
        sorted(
            (
                ActionWordRecordedSemanticCompatibility(
                    f"compatibility.{word.word_id}",
                    ObjectIdentity.from_record(word.word_id, word),
                    "response-geometry-assay.short-pulse-response-finite-native-force-assay",
                    ProspectiveActionSemanticRole.NATIVE_HOLD
                    if sign == 0
                    else ProspectiveActionSemanticRole.ACTIVE,
                    False,
                    False,
                    False,
                )
                for (_, _, sign), word in words.items()
            ),
            key=lambda record: record.compatibility_id,
        )
    )
    first = chart_words[0]
    chart = ProspectiveSequentialActionWordChart(
        f"{prefix}.expected-short-pulse-response-chart",
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
    acquisitions: list[ResponseAcquisitionGroup] = []
    views: list[ResponseAcquisitionView] = []
    projections: list[NativeProjectionGroup] = []
    for root in source.roots:
        unit_id = source.physical_unit_id(root)
        instance_id = f"{unit_id}.instance"
        units.append(
            ResponsePreparationUnit(
                unit_id,
                f"{prefix}.coordinate.{root.context}.t{root.landmark_tick}",
                instance_id,
                sha256(
                    canonical_json_bytes((source.fingerprint(), root.fingerprint()))
                ).hexdigest(),
                f"{unit_id}.native-rng",
                ResponseDataSplitRole.DEVELOPMENT if root.index < 32 else ResponseDataSplitRole.EVALUATION,
            )
        )
        by_refinement: dict[int, list[str]] = {1: [], 2: []}
        for parent in PARENTS:
            for sign in (-1, 0, 1):
                segment = ResponseGeometryDevelopmentNativeSegment(root, "inner", parent, sign)
                group_id = f"group.{segment.task_id}"
                pair = tuple(f"view.{segment.task_id}.r{r}" for r in (1, 2))
                acquisitions.append(ResponseAcquisitionGroup(group_id, unit_id, instance_id, pair))
                for refinement, view_id in zip((1, 2), pair, strict=True):
                    views.append(
                        ResponseAcquisitionView(
                            view_id,
                            group_id,
                            unit_id,
                            f"r{refinement}",
                            words[(root.context, root.invocation_offset, sign)].word_id,
                        )
                    )
                    by_refinement[refinement].append(view_id)
        projections.extend(
            NativeProjectionGroup(f"{root.root_id}.project.r{r}", tuple(sorted(by_refinement[r])))
            for r in (1, 2)
        )
    return ResponseGeometryDevelopmentNativeDeclarations(
        tuple(sorted(units, key=lambda u: u.physical_independent_unit_id)),
        tuple(sorted(acquisitions, key=lambda g: g.acquisition_group_id)),
        tuple(sorted(views, key=lambda v: v.view_id)),
        tuple(sorted(projections, key=lambda g: g.projection_task_id)),
        chart,
    )
