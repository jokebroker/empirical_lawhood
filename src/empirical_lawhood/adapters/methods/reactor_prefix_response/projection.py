"""Receipt-bound native panel to the existing evidence projector.

Only materialization identities and observed delivery operands are bound here.
The roster, units, split, receiver, action chart and method settings come from
the pre-acquisition design. Failed branches retain rows with no compact values.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.methods.evidence_projection import (
    IdentificationEvidenceProjector,
    IdentificationProjectionPlan,
    ObservationValuePartition,
)
from empirical_lawhood.adapters.methods.evidence_projection_extensions import build_identification_projection_extension
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import BRANCHES, ReactorPrefixEpisode, ReactorPrefixPanel, native_task_id
from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage, ActionStageEvent, OccurrenceActionWord
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.planning.identification_evidence import (
    ClaimUnitBinding,
    ExternalEvidencePayload,
    IdentificationEvidenceDomain,
    IdentificationEvidenceManifest,
    IdentificationEvidenceProjection,
    IdentificationManifestObservation,
    NestedCoordinateKind,
    NestedEvidenceCoordinate,
    ObservationActionDeliveryBinding,
    ObservationDisposition,
    QualificationScopeSpec,
)
from empirical_lawhood.planning.identification_evidence_extensions import ActionOccurrenceBinding, AuthorityAxis, ComputabilityEffectCoordinate, ComputabilityEffectEntry, ComputabilityEffectStatus, EffectReasonCode, EpisodeTerminalDisposition, EpisodeTerminalReasonCode, IdentificationEvidenceProjectionExtension, NumericalValidityAxis, ObservationValidityAxis, PhysicalSinkAxis, ScientificTerminalAxis
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt

from .design import CHART, CLOCK, CUTOFF, METHOD, PREFIX, RECEIVER, SUPPORT, UNIT, ReactorScienceDesign, reactor_system, unit_ids
from .words import action_words


@dataclass(frozen=True, slots=True)
class ReactorStageEvidence(CanonicalRecord):
    """An observed stage and its native clock, anchored in the published panel."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/reactor-stage-evidence'
    PANEL_SCHEMA: ClassVar[str] = ReactorPrefixPanel.SCHEMA
    receipt_id: str
    clock_binding_id: str
    panel: ObjectIdentity
    episode_index: int
    decision_index: int
    event: ActionStageEvent

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.clock_binding_id, field_name="clock_binding_id")
        if (
            self.panel.object_schema != self.PANEL_SCHEMA
            or type(self.episode_index) is not int
            or not 0 <= self.episode_index < 20
            or type(self.decision_index) is not int
            or self.decision_index not in (0, 1)
            or self.event.coordinate.coordinate != 10 * self.decision_index
        ):
            raise ValueError(
                "reactor stage evidence changes its native panel or decision clock"
            )


@dataclass(frozen=True, slots=True)
class ReactorNativeCustody(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/reactor-native-custody'
    receipts: tuple[CanonicalTaskReceipt, ...]

    def __post_init__(self) -> None:
        if (
            len(self.receipts) != len(BRANCHES)
            or tuple(sorted(r.task_id for r in self.receipts))
            != tuple(native_task_id(*b) for b in BRANCHES)
            or len({r.receipt_id for r in self.receipts}) != len(BRANCHES)
            or len({r.run_id for r in self.receipts}) != 1
            or any(
                r.operational_status is not OperationalStatus.SUCCEEDED
                for r in self.receipts
            )
        ):
            raise ValueError(
                "reactor custody requires the exact twenty committed native branches"
            )


@dataclass(frozen=True, slots=True)
class ReactorProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/reactor-projection'
    design: ReactorScienceDesign
    manifest: IdentificationEvidenceManifest
    projection: IdentificationEvidenceProjection
    extension: IdentificationEvidenceProjectionExtension | None
    stage_evidence: tuple[ReactorStageEvidence, ...]
    native_custody: ReactorNativeCustody | None = None


def _delivery_matches(episode: ReactorPrefixEpisode, word: OccurrenceActionWord) -> bool:
    if episode.failure_code is not None or len(episode.deliveries) != 2:
        return False
    for index, occurrence in enumerate(word.occurrences):
        delivery = episode.deliveries[index // 2]
        feed = occurrence.channel.controller_quantity_id == "reactor-feed"
        values = (
            delivery.command.feed_kg_s if feed else delivery.command.jacket_k,
            delivery.accepted_feed_kg_s if feed else delivery.accepted_jacket_k,
            delivery.applied_feed_kg_s if feed else delivery.applied_jacket_k,
        )
        expected = (occurrence.requested, occurrence.accepted, occurrence.applied)
        if any(
            event.value != value for event, value in zip(expected, values, strict=True)
        ):
            return False
        if any(
            (e.feed_kg_s if feed else e.jacket_k) != occurrence.realized.value
            for e in delivery.exposures
        ):
            return False
        if delivery.command.time_s != occurrence.requested.coordinate.coordinate:
            return False
    return True


class _Values:
    def __init__(self, rows: dict[str, tuple[NamedDecimal, ...]]) -> None:
        self.rows = rows

    def read_compact_values(
        self,
        manifest: IdentificationEvidenceManifest,
        observation: IdentificationManifestObservation,
        required_value_ids: tuple[str, ...],
    ) -> tuple[NamedDecimal, ...]:
        del manifest
        values = self.rows[observation.observation_id]
        if values and tuple(v.value_id for v in values) != required_value_ids:
            raise ValueError("reactor compact values differ from the frozen partition")
        return values


def project_panel(
    *,
    design: ReactorScienceDesign,
    panel: ReactorPrefixPanel,
    artifact: ArtifactManifest | None = None,
    unit_inputs: tuple[tuple[ReactorPrefixPanel, ArtifactManifest], ...] = (),
    native_custody: ReactorNativeCustody | None = None,
    source_qualification: ObjectIdentity,
    runtime_qualification: ObjectIdentity,
    observer_qualification: ObjectIdentity,
) -> ReactorProjection:
    """Consume an already authenticated published artifact; never acquire a source.

    The composition must authenticate the manifest against the dependency
    receipt. Matching bytes alone are not a substitute for that boundary.
    """
    from .assigned.records import ReactorAssignedProjection, ReactorAssignedScienceDesign, ReactorAssignedStageEvidence

    assigned = type(design) is ReactorAssignedScienceDesign
    stage_type = ReactorAssignedStageEvidence if assigned else ReactorStageEvidence
    projection_type = ReactorAssignedProjection if assigned else ReactorProjection
    if panel.config != design.native or panel.branch is not None:
        raise ValueError("reactor projection requires the complete frozen cohort")
    sources: tuple[tuple[ReactorPrefixPanel, ArtifactManifest], ...]
    if artifact is None:
        if (
            not unit_inputs
            or tuple(e for p, _ in unit_inputs for e in p.episodes) != panel.episodes
        ):
            raise ValueError(
                "reactor unit publications do not cover the complete assigned cohort"
            )
        if any(p.branch is None for p, _ in unit_inputs) or len(unit_inputs) != 20:
            raise ValueError(
                "reactor production requires twenty published branch panels"
            )
        sources = unit_inputs
    else:
        if unit_inputs:
            raise ValueError(
                "reactor projection cannot mix aggregate and unit publications"
            )
        sources = ((panel, artifact),)
    payloads = []
    origins = {}
    for native_panel, published in sources:
        if (
            native_panel.config != design.native
            or published.logical.payload_schema != panel.SCHEMA
            or published.logical.content_sha256 != native_panel.fingerprint()
            or published.materialization.size_bytes
            != len(native_panel.canonical_bytes())
            or published.publication is None
            or published.logical.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError(
                "reactor panel publication or frozen native design differs"
            )
        logical, materialization, publication = (
            published.logical,
            published.materialization,
            published.publication,
        )
        payload = ExternalEvidencePayload(
            logical.logical_artifact_id,
            ArtifactIdentity(
                logical.logical_artifact_id,
                "native-reactor-prefix",
                logical.payload_schema,
                logical.content_sha256,
                logical.media_type,
                materialization.size_bytes,
            ),
            materialization.storage_root_id,
            materialization.relative_path,
            ObjectIdentity.from_record(publication.publication_batch_id, publication),
            ObjectIdentity.from_record(
                materialization.materialization_id, materialization
            ),
        )
        payloads.append(payload)
        identity = ObjectIdentity(
            logical.logical_artifact_id,
            panel.SCHEMA,
            panel.VERSION,
            native_panel.fingerprint(),
        )
        for local_index, episode in enumerate(native_panel.episodes):
            key = (episode.scenario_id, episode.view_id, episode.arm_id)
            if key in origins:
                raise ValueError("reactor publication duplicates a native branch")
            origins[key] = (payload, identity, local_index)
    system = reactor_system(design)
    words = action_words(design)
    by_arm = {w.word_id.rsplit(".", 1)[1]: w for w in words}
    coordinates = tuple(
        NestedEvidenceCoordinate(
            f"coordinate.{u}", NestedCoordinateKind.PREPARATION, None
        )
        for u in unit_ids(design)
    )
    scope = QualificationScopeSpec(
        f"{PREFIX}.scope",
        f"{PREFIX}.claim",
        "five-assigned-scenario-seeds" if assigned else "five-public-scenario-seeds",
        UNIT,
        unit_ids(design),
        tuple(c.coordinate_id for c in coordinates),
        UNIT,
        UNIT,
        system.system_id,
        EvidenceCeiling.LOCAL_LAW,
    )
    binding = ClaimUnitBinding(
        f"{PREFIX}.claim-unit",
        scope.claim_id,
        ObjectIdentity.from_record(scope.scope_id, scope),
        unit_ids(design),
        UNIT,
    )
    observations, stages, actions, terminals = [], [], [], []
    values = {}
    for episode in panel.episodes:
        payload, panel_identity, index = origins[
            (episode.scenario_id, episode.view_id, episode.arm_id)
        ]
        suffix = f"{episode.scenario_id}.{episode.view_id}.{episode.arm_id}".replace(
            "_", "-"
        )
        observation_id, episode_id = f"observation.{suffix}", f"episode.{suffix}"
        projected_id = f"projected.{observation_id}"
        unit = f"unit.{episode.scenario_id.replace('_', '-')}"
        word = by_arm[episode.arm_id]
        word_identity = ObjectIdentity.from_record(word.word_id, word)
        valid = _delivery_matches(episode, word)
        event_ids: dict[ActionDeliveryStage, list[str]] = {
            s: [] for s in ActionDeliveryStage
        }
        if valid:
            for occurrence_index, occurrence in enumerate(word.occurrences):
                receipt_ids, clock_ids = [], []
                for event in (
                    occurrence.requested,
                    occurrence.accepted,
                    occurrence.applied,
                    occurrence.realized,
                ):
                    stage_key = (
                        f"{suffix}.{occurrence_index}.{event.stage.value.lower()}"
                    )
                    stage = stage_type(
                        f"receipt.{stage_key}",
                        f"clock.{stage_key}",
                        panel_identity,
                        index,
                        occurrence_index // 2,
                        event,
                    )
                    stages.append(stage)
                    receipt_ids.append(stage.receipt_id)
                    clock_ids.append(stage.clock_binding_id)
                    event_ids[event.stage].append(
                        f"event.{occurrence.occurrence_id}.{event.stage.value.lower()}"
                    )
                actions.append(
                    ActionOccurrenceBinding(
                        f"action-binding.{suffix}.{occurrence_index}",
                        episode_id,
                        projected_id,
                        word_identity,
                        occurrence,
                        (
                            receipt_ids[0],
                            receipt_ids[1],
                            receipt_ids[2],
                            receipt_ids[3],
                        ),
                        (clock_ids[0], clock_ids[1], clock_ids[2], clock_ids[3]),
                    )
                )
        delivery_binding = (
            ObservationActionDeliveryBinding(
                f"delivery.{suffix}",
                word_identity,
                tuple(sorted(o.occurrence_id for o in word.occurrences)),
                *(tuple(sorted(event_ids[s])) for s in ActionDeliveryStage),
            )
            if valid
            else None
        )
        observations.append(
            IdentificationManifestObservation(
                observation_id,
                unit,
                (f"coordinate.{unit}",),
                SUPPORT,
                CHART,
                "calibration"
                if episode.scenario_id in design.native.calibration_scenarios
                else "held-out",
                "primary",
                "reactor-fixed-denominator",
                "reactor-finite-candidate",
                f"reactor-{episode.view_id}",
                RECEIVER,
                CLOCK,
                "reactor-native-temperature",
                payload.payload_id,
                f"value/episodes/{index}",
                delivery_binding,
                ObservationDisposition.COMPLETE
                if valid
                else ObservationDisposition.TECHNICAL_INVALID,
                () if valid else ("REACTOR_PREFIX_INCOMPLETE_OR_DELIVERY_MISMATCH",),
            )
        )
        temperature = (
            episode.deliveries[-1].next_measurement.t_reactor_k if valid else None
        )
        values[observation_id] = (
            ()
            if temperature is None
            else tuple(
                sorted(
                    (
                        NamedDecimal(
                            "reactor-initial-temperature", Decimal("318.4"), "K"
                        ),
                        NamedDecimal("reactor-initial-jacket", Decimal(316), "K"),
                        NamedDecimal("reactor-feed", Decimal(0), "kg/s"),
                        NamedDecimal(
                            "reactor-jacket-command",
                            episode.deliveries[0].command.jacket_k,
                            "K",
                        ),
                        NamedDecimal(RECEIVER, temperature, "K"),
                    ),
                    key=lambda v: v.value_id,
                )
            )
        )
        terminals.append(
            EpisodeTerminalDisposition(
                episode_id,
                projected_id,
                unit,
                PhysicalSinkAxis.NONE,
                ObservationValidityAxis.VALID
                if valid
                else ObservationValidityAxis.INVALID,
                NumericalValidityAxis.VALID
                if valid
                else NumericalValidityAxis.NOT_EVALUATED,
                AuthorityAxis.AUTHORIZED,
                ScientificTerminalAxis.OBSERVED
                if valid
                else ScientificTerminalAxis.UNEVALUABLE,
                (),
                (NamedDecimal(f"score.{suffix}", temperature, "K"),)
                if temperature is not None
                else (),
                (
                    EpisodeTerminalReasonCode.EPISODE_OBSERVED
                    if valid
                    else EpisodeTerminalReasonCode.OBSERVATION_INVALID,
                ),
            )
        )
    manifest = IdentificationEvidenceManifest(
        f"{PREFIX}.manifest",
        ObjectIdentity.from_record(system.system_id, system),
        system.relation,
        source_qualification,
        runtime_qualification,
        observer_qualification,
        system.world.world_id,
        IdentificationEvidenceDomain.EVALUATOR_REVEAL,
        (METHOD,),
        CUTOFF,
        tuple(sorted(payloads, key=lambda p: p.payload_id)),
        coordinates,
        words,
        tuple(sorted(observations, key=lambda o: o.observation_id)),
        (scope,),
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    design_identity = ObjectIdentity.from_record(design.config_id, design)
    plan = IdentificationProjectionPlan(
        f"{PREFIX}.projection-plan",
        ObjectIdentity.from_record(manifest.manifest_id, manifest),
        METHOD,
        observer_qualification,
        design_identity,
        runtime_qualification,
        binding,
        ObservationValuePartition(
            f"{PREFIX}.partition",
            ("reactor-initial-temperature",),
            ("reactor-initial-jacket",),
            ("reactor-feed", "reactor-jacket-command"),
            (RECEIVER,),
            (),
        ),
    )
    projection = IdentificationEvidenceProjector().project(
        system, manifest, plan, _Values(values)
    )
    effects = tuple(
        ComputabilityEffectCoordinate(
            f"effect-coordinate.{v}",
            "jacket-prefix-temperature-contrast",
            "reactor-fixed-denominator",
            v,
        )
        for v in ("reactor-native", "reactor-refined")
    )
    # The frozen historical companion requires complete observed stages for every
    # episode. Preserve an explicit extraction stop when those operands do not
    # exist; never manufacture realized stages merely to enter the method.
    if len(actions) != 4 * len(panel.episodes):
        return projection_type(
            design, manifest, projection, None, tuple(stages), native_custody
        )
    extension = build_identification_projection_extension(
        extension_id=f"{PREFIX}.projection-extension",
        manifest=manifest,
        projection=projection,
        action_words=words,
        action_occurrences=tuple(actions),
        terminal_dispositions=tuple(terminals),
        declared_effect_coordinates=effects,
        computability_entries=tuple(
            ComputabilityEffectEntry(
                f"effect-entry.{e.numerical_view_id}",
                e,
                ComputabilityEffectStatus.REPRESENTED,
                (),
                EffectReasonCode.EFFECT_REPRESENTED,
            )
            for e in effects
        ),
    )
    return projection_type(
        design, manifest, projection, extension, tuple(stages), native_custody
    )
