"Outcome-blind dependent refinement units, native futures, views and whole-root cross-fit roles."

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.action_contracts import ActionChannelBinding, ActionDeliveryStage, ActionOccurrence, ActionOccurrenceGroup, ActionOccurrenceOrder, ActionStageEvent, ActionStageQuantityBinding, OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.kernel.time import (
    ClockCoordinate,
    ClockTransport,
    ClockTransportAvailability,
    ClockTransportKind,
    ClockTransportMonotonicity,
    CoordinateOrigin,
)
from empirical_lawhood.planning.native_source import NativeCrossfitFold, NativeCrossfitPartition, NativeNestedCrossfitPlan, NativeNestedCrossfitPopulation, NativeProjectionGroup
from empirical_lawhood.planning.response_experiment import ResponseAcquisitionGroup, ResponseAcquisitionView, ResponsePreparationUnit, ResponseDataSplitRole
from empirical_lawhood.planning.prospective_config import ActionWordRecordedSemanticCompatibility, ProspectiveActionSemanticRole, ProspectiveSequentialActionWordChart

from .contracts import CAMPAIGN, CONTEXTS, PreparedForceWord, PreparedNativeSpec
from .native_tasks import prepared_static_native_invocations


DEVELOPMENT_NAMESPACE = "prepared-response.dependent-refinement"
DEVELOPMENT_CLOCK = f"{DEVELOPMENT_NAMESPACE}.episode-clock"
DEVELOPMENT_CLOCK_FRAME = f"{DEVELOPMENT_NAMESPACE}.handoff-relative-episode"
DEVELOPMENT_ACTION_FRAME = f"{DEVELOPMENT_NAMESPACE}.frozen-preparent-two-port-frame"
DEVELOPMENT_ACTION_UNIT = "native-hs-force"
DEVELOPMENT_DENOMINATOR = f"{DEVELOPMENT_NAMESPACE}.quantity.denominator"
DEVELOPMENT_HISTORY = f"{DEVELOPMENT_NAMESPACE}.quantity.history"
DEVELOPMENT_RECEIVER = f"{DEVELOPMENT_NAMESPACE}.quantity.two-port-receiver"
DEVELOPMENT_HORIZON = f"{DEVELOPMENT_NAMESPACE}.horizon.320"


def prepared_response_development_clock_evaluator() -> ExecutableReference:
    "Frozen identity evaluator for dependent refinement's shared reference-tick coordinate."

    payload = canonical_json_bytes(
        {
            "clock_id": DEVELOPMENT_CLOCK,
            "frame_id": DEVELOPMENT_CLOCK_FRAME,
            "native_unit": "reference-tick",
            "map": "IDENTITY",
        }
    )
    return ExecutableReference(
        f"{DEVELOPMENT_NAMESPACE}.clock-identity",
        f"{DEVELOPMENT_NAMESPACE}.clock-identity",
        "1.0.0",
        f"{DEVELOPMENT_NAMESPACE}.clock-identity-evaluator",
        ArtifactIdentity(
            f"{DEVELOPMENT_NAMESPACE}.clock-identity-payload",
            "clock-evaluator",
            'empirical-lawhood/simulators/prepared-response/clock-identity',
            sha256(payload).hexdigest(),
            "application/json",
            len(payload),
        ),
        SafePayloadFormat.CANONICAL_JSON,
        'empirical-lawhood/kernel/clock-coordinate',
        'empirical-lawhood/kernel/clock-coordinate',
        True,
    )


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentNativeDeclarations:
    "The dependent refinement census before any native source contact."

    units: tuple[ResponsePreparationUnit, ...]
    acquisitions: tuple[ResponseAcquisitionGroup, ...]
    views: tuple[ResponseAcquisitionView, ...]
    projections: tuple[NativeProjectionGroup, ...]
    chart: ProspectiveSequentialActionWordChart
    crossfit: NativeNestedCrossfitPlan


def prepared_response_development_action_word(
    force_word: PreparedForceWord, clock_evaluator: ExecutableReference
) -> OccurrenceActionWord:
    """Encode one two-port pulse as a scalar magnitude plus a native direction ID."""
    stem = force_word.word_id
    direction = f"{CAMPAIGN}.two-port-direction.d{force_word.direction_index}"
    origin = CoordinateOrigin.EPISODE_RELATIVE
    coordinate = ClockCoordinate(DEVELOPMENT_CLOCK, Decimal(0), "reference-tick", DEVELOPMENT_CLOCK_FRAME, origin)
    transport = ClockTransport(
        f"{stem}.clock",
        DEVELOPMENT_CLOCK,
        DEVELOPMENT_CLOCK,
        "reference-tick",
        "reference-tick",
        DEVELOPMENT_CLOCK_FRAME,
        DEVELOPMENT_CLOCK_FRAME,
        origin,
        origin,
        ClockTransportKind.IDENTITY,
        ClockTransportAvailability.AVAILABLE,
        Decimal(1),
        Decimal(0),
        Decimal(0),
        Decimal(0),
        Decimal(320),
        ClockTransportMonotonicity.STRICTLY_INCREASING,
        clock_evaluator,
        (),
    )
    channel = ActionChannelBinding(
        f"{stem}.channel",
        f"{DEVELOPMENT_NAMESPACE}.two-port-force",
        f"{DEVELOPMENT_NAMESPACE}.two-port-force-command",
        tuple(
            ActionStageQuantityBinding(
                stage,
                f"{DEVELOPMENT_NAMESPACE}.two-port-force.{stage.value.lower()}",
                DEVELOPMENT_CLOCK,
                "reference-tick",
                DEVELOPMENT_CLOCK_FRAME,
                origin,
            )
            for stage in ActionDeliveryStage
        ),
        DEVELOPMENT_ACTION_UNIT,
        DEVELOPMENT_ACTION_FRAME,
        direction,
        f"{DEVELOPMENT_NAMESPACE}.finite-force-source-support",
        f"{DEVELOPMENT_NAMESPACE}.both-kicks-completed-interval-delivery",
    )
    value = force_word.magnitude * force_word.sign
    events = tuple(
        ActionStageEvent(
            stage.stage,
            stage.quantity_id,
            value,
            DEVELOPMENT_ACTION_UNIT,
            DEVELOPMENT_ACTION_FRAME,
            direction,
            coordinate,
        )
        for stage in channel.stages
    )
    occurrence_id = f"{stem}.interval.0"
    occurrence = ActionOccurrence(
        occurrence_id,
        channel,
        events[0],
        events[1],
        events[2],
        events[3],
        transport,
        transport,
        transport,
        Decimal(64),
        "reference-tick",
    )
    return OccurrenceActionWord(
        stem,
        ActionWordMode.SEQUENTIAL,
        (occurrence,),
        (
            ActionOccurrenceGroup(
                f"{stem}.group.0", (ActionOccurrenceOrder(occurrence_id, transport),)
            ),
        ),
        DEVELOPMENT_CLOCK,
        "reference-tick",
        DEVELOPMENT_CLOCK_FRAME,
        origin,
        DEVELOPMENT_DENOMINATOR,
        DEVELOPMENT_HISTORY,
        DEVELOPMENT_RECEIVER,
        DEVELOPMENT_HORIZON,
        (f"{stem}.support.prefix.0", f"{stem}.support.prefix.1"),
        ActionWordSupportStatus.UNEVALUABLE,
        ("SUPPORT_EXPECTATION_REQUIRES_EVIDENCE",),
    )


def _partition(partition_id: str, unit_ids: tuple[str, ...]) -> NativeCrossfitPartition:
    """The exact source-index modulo-four partition used by the finite fitter."""
    folds = tuple(
        NativeCrossfitFold(
            f"{partition_id}.fold{fold}",
            tuple(unit for rank, unit in enumerate(unit_ids) if rank % 4 != fold),
            tuple(unit for rank, unit in enumerate(unit_ids) if rank % 4 == fold),
        )
        for fold in range(4)
    )
    return NativeCrossfitPartition(partition_id, unit_ids, folds)


def prepared_response_development_crossfit_plan(spec: PreparedNativeSpec) -> NativeNestedCrossfitPlan:
    if spec.stage != 'development':
        raise ValueError("prepared dependent refinement cross-fit plan cannot reinterpret another stage")
    populations = []
    for context in CONTEXTS:
        unit_ids = tuple(
            root.physical_unit_id for root in spec.roots if root.context == context
        )
        outer = _partition(f"{DEVELOPMENT_NAMESPACE}.crossfit.{context}.outer", unit_ids)
        inner = tuple(
            _partition(fold.fold_id, fold.training_unit_ids) for fold in outer.folds
        )
        final = _partition(f"{DEVELOPMENT_NAMESPACE}.crossfit.{context}.final-selection", unit_ids)
        populations.append(
            NativeNestedCrossfitPopulation(
                f"{DEVELOPMENT_NAMESPACE}.crossfit.{context}", outer, inner, final, unit_ids
            )
        )
    qualification = tuple(
        f"{CAMPAIGN}.calibration.{context}.r{index:03d}.qualification-study"
        for context in CONTEXTS
        for index in range(96)
    )
    return NativeNestedCrossfitPlan(
        f"{DEVELOPMENT_NAMESPACE}.crossfit",
        tuple(populations),
        f"{DEVELOPMENT_NAMESPACE}.fresh-conditional-policy-study",
        tuple(sorted(qualification)),
    )


def prepared_response_development_native_declarations(
    spec: PreparedNativeSpec, clock_evaluator: ExecutableReference
) -> PreparedResponseDevelopmentNativeDeclarations:
    """Declare 128 roots, 7,680 futures, 15,360 views and 256 projections."""
    if spec.stage != 'development':
        raise ValueError("prepared dependent refinement declarations cannot reinterpret another stage")
    words = tuple(
        sorted(
            (prepared_response_development_action_word(word, clock_evaluator) for word in spec.words),
            key=lambda word: word.word_id,
        )
    )
    word_by_id = {word.word_id: word for word in words}
    compatibility = tuple(
        ActionWordRecordedSemanticCompatibility(
            f"compatibility.{word.word_id}",
            ObjectIdentity.from_record(word.word_id, word),
            "prepared-response-native-finite-two-port-force-chart",
            ProspectiveActionSemanticRole.NATIVE_HOLD
            if word_by_id[word.word_id].occurrences[0].requested.value == 0
            else ProspectiveActionSemanticRole.ACTIVE,
            False,
            False,
            False,
        )
        for word in words
    )
    first = words[0]
    chart = ProspectiveSequentialActionWordChart(
        f"{DEVELOPMENT_NAMESPACE}.expected-force-chart",
        words,
        tuple(sorted(compatibility, key=lambda row: row.compatibility_id)),
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
    futures = tuple(
        task
        for task in prepared_static_native_invocations(spec)
        if task.phase == "future"
    )
    for root in spec.roots:
        unit_id = root.physical_unit_id
        instance_id = f"{unit_id}.instance"
        units.append(
            ResponsePreparationUnit(
                unit_id,
                f"{CAMPAIGN}.coordinate.{root.context}.t{root.landmark}",
                instance_id,
                sha256(canonical_json_bytes((spec.fingerprint(), root.fingerprint()))).hexdigest(),
                f"{unit_id}.native-rng",
                ResponseDataSplitRole.DEVELOPMENT,
            )
        )
        per_refinement: dict[int, list[str]] = {1: [], 2: []}
        for task in (value for value in futures if value.root == root):
            assert task.word is not None
            group_id = f"group.{task.task_id}"
            pair = tuple(f"view.{task.task_id}.r{r}" for r in (1, 2))
            acquisitions.append(ResponseAcquisitionGroup(group_id, unit_id, instance_id, pair))
            for refinement, view_id in zip((1, 2), pair, strict=True):
                views.append(
                    ResponseAcquisitionView(
                        view_id,
                        group_id,
                        unit_id,
                        f"r{refinement}",
                        task.word.word_id,
                    )
                )
                per_refinement[refinement].append(view_id)
        projections.extend(
            NativeProjectionGroup(
                f"{root.root_id}.project.r{refinement}",
                tuple(sorted(per_refinement[refinement])),
            )
            for refinement in (1, 2)
        )
    return PreparedResponseDevelopmentNativeDeclarations(
        tuple(sorted(units, key=lambda row: row.physical_independent_unit_id)),
        tuple(sorted(acquisitions, key=lambda row: row.acquisition_group_id)),
        tuple(sorted(views, key=lambda row: row.view_id)),
        tuple(sorted(projections, key=lambda row: row.projection_task_id)),
        chart,
        prepared_response_development_crossfit_plan(spec),
    )
