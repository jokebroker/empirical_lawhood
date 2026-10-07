"""Actual prepared-owner delivery, with one bounded causal second decision."""

from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Callable, ClassVar

import numpy as np

from empirical_lawhood.adapters.control.finite_campaign import finite_control_services
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import FIRST, ZERO, Request
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.law_terminal import ClassicalLaws
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_lock import ClassicalFrozenRoot, ClassicalFrozenStage, freeze_stage
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_plan import ReactorStagedPulseResponseProspectivePlan
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.records import ClassicalContext, ClassicalNativeWindow, ClassicalPreparation, ClassicalPrivate
from empirical_lawhood.adapters.methods.reactor_causal_response.controller import EmpiricalNonattempt
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_lock import UndeliverableSession
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode, TapeController, acquire_episode, acquire_with_progress
from empirical_lawhood.adapters.simulators.reactor_regime_response.committed_branch import WatchedPreparedTape
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorDelivery, ReactorFinalDelivery
from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_callback import FiniteCommittedBranchOwner
from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_words import overlay_feed_tape
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerTickReceipt, TickDisposition
from .acquisition import donors, native_window
from .words import ClassicalPulseDeliveryPort


@dataclass(frozen=True, slots=True)
class ClassicalNativeStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-staged-pulse-response/classical-native-stage'
    frozen: CanonicalRecordArchive
    tick: DeliveryControllerTickReceipt | None
    deliveries: tuple[ReactorDelivery, ...]

    @lru_cache(maxsize=32)
    def unpack(self) -> ClassicalFrozenStage:
        # Measurement, sealing and reveal consume the same immutable child.
        # Key by the whole stage, including its tick and actual deliveries, so
        # another owner/receipt still traverses the validation below.
        value = decode_canonical_bytes(
            self.frozen.unpack(), ClassicalFrozenStage, maximum_bytes=self.frozen.decoded_bytes
        )
        if (
            self.frozen.subject != ObjectIdentity.from_record(value.record_id, value)
            or (self.tick is not None and self.tick.commitment != value.commitment)
            or (self.deliveries and (self.tick is None or not value.admitted))
        ):
            raise ValueError("actual native stage substitutes its frozen owner")
        return value


@dataclass(frozen=True, slots=True)
class ClassicalNativeUse(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-staged-pulse-response/classical-native-use'
    policy: str
    request: Request
    stages: tuple[ClassicalNativeStage, ...]
    windows: tuple[ClassicalNativeWindow, ...]
    native_calls: int
    failures: tuple[str, ...]
    known_unsafe: bool

    def __post_init__(self) -> None:
        if (
            self.native_calls not in (0, 1, 2)
            or not 1 <= len(self.stages) <= 2
            or any(
                (s.unpack().policy, s.unpack().request) != (self.policy, self.request)
                for s in self.stages
            )
            or (not self.native_calls and (self.windows or any(s.deliveries for s in self.stages)))
        ):
            raise ValueError("native use changes its actual decision/view census")
        if len(self.stages) == 2 and (
            self.stages[0].tick is None
            or self.stages[1].unpack().causal.predecessor
            != ObjectIdentity.from_record(self.stages[0].tick.tick_id, self.stages[0].tick)
        ):
            raise ValueError("second native stage substitutes its actual predecessor")


@dataclass(frozen=True, slots=True)
class ClassicalNativeRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-staged-pulse-response/classical-native-root'
    root: str
    frozen: ObjectIdentity
    scenario: ObjectIdentity
    uses: tuple[ClassicalNativeUse, ...]
    references: tuple[ClassicalNativeWindow, ...]
    reference_calls: int

    def __post_init__(self) -> None:
        if (
            self.frozen.object_schema != ClassicalFrozenRoot.SCHEMA
            or self.scenario.object_id != self.root
            or self.reference_calls not in range(5)
            or len({(u.policy, u.request.request_id) for u in self.uses}) != len(self.uses)
            or any(s.unpack().causal.root != self.root for u in self.uses for s in u.stages)
        ):
            raise ValueError(
                "prospective native result changes its assigned root or reference budget"
            )

    @property
    def native_calls(self) -> int:
        return self.reference_calls + sum(u.native_calls for u in self.uses)

    @property
    def record_id(self) -> str:
        return f"{self.root}.native-observation"


@dataclass
class RestoredStage:
    frozen: ClassicalFrozenStage
    delivery: ClassicalPulseDeliveryPort
    services: Any

    def deliver(self, publisher: Any) -> DeliveryControllerTickReceipt:
        tick: DeliveryControllerTickReceipt = self.services.deliver_prepared_commitment(
            self.frozen.compiled, self.frozen.commitment
        )
        publisher.publish_record(tick.tick_id, tick)
        return tick


def restore(stage: ClassicalFrozenStage) -> RestoredStage:
    assert stage.compiled is not None and stage.projection is not None and stage.lock is not None
    evaluation = stage.lock.evaluation_for(stage.compiled)
    delivery = ClassicalPulseDeliveryPort(
        stage.compiled.implementation(ImplementationRole.DELIVERY),
        UndeliverableSession(),
        stage.projection,
    )
    return RestoredStage(
        stage, delivery, finite_control_services(stage.compiled.study, evaluation, delivery)
    )


class DecisionTape(WatchedPreparedTape):
    def __init__(self, tape: np.ndarray, owner: 'ClassicalSequenceOwner') -> None:
        super().__init__(tape)
        self.owner = owner

    def step(self, t_s: float, y: dict[str, float], dt_s: float) -> tuple[float, float]:
        self.latest = (t_s, y["t_reactor_k"], y["t_jacket_k"], y["dosed_kg"])
        self.owner.observe(self, int(t_s / 10))
        return TapeController.step(self, t_s, y, dt_s)


@dataclass
class ClassicalSequenceOwner:
    initial: ClassicalFrozenStage
    donor: NativeEpisode
    publisher: Any
    prepare_second: (
        Callable[[ClassicalContext, DeliveryControllerTickReceipt], ClassicalFrozenStage] | None
    )
    active: FiniteCommittedBranchOwner
    frozen: list[ClassicalFrozenStage] = field(default_factory=list)
    owners: list[FiniteCommittedBranchOwner] = field(default_factory=list)
    observations: list[tuple[float, ...]] = field(default_factory=list)
    requests: list[tuple[float, ...]] = field(default_factory=list)
    stages: list[tuple[float, ...]] = field(default_factory=list)
    exposure: list[tuple[tuple[float, ...], ...]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.active.failed

    def observe(self, tape: DecisionTape, k: int) -> None:
        if tape.latest is None or k != len(self.observations):
            raise ValueError("prospective callback is repeated or missing")
        self.observations.append(tape.latest)
        first_k = self.initial.causal.callback
        assert first_k is not None
        if self.prepare_second is None or k != first_k + 12:
            return
        first_tick = self.owners[0].tick
        if (
            first_tick is None
            or first_tick.disposition is not TickDisposition.ACTION_DELIVERED
            or not first_tick.delivery_trace.exact
        ):
            raise EmpiricalNonattempt("FIRST_OWNER_NOT_DELIVERED")
        values = {
            name: np.asarray(getattr(self, name), dtype=np.float64)
            for name in ("observations", "requests", "stages", "exposure")
        }
        causal = ClassicalContext(
            self.initial.causal.root,
            "induced",
            k,
            LocalArrayPayload.pack(values),
            (),
            first_k,
            ObjectIdentity.from_record(first_tick.tick_id, first_tick),
            f"{self.initial.policy.lower()}.{self.initial.request.request_id.lower()}",
        )
        second = self.prepare_second(causal, first_tick)
        self.frozen.append(second)
        if second.compiled is None:
            raise EmpiricalNonattempt("SECOND_OWNER_UNAVAILABLE")
        restored = restore(second)
        assert second.projection is not None
        tape.tape = overlay_feed_tape(
            tape.tape, k, second.projection.pulse.rates(), float(second.projection.fixed_jacket_K)
        )
        self.active = FiniteCommittedBranchOwner(
            causal.root,
            f"{causal.root}.{causal.owner_scope}.second",
            k,
            values["observations"],
            values["stages"],
            tape.tape,
            restored,
            self.publisher,
            checked_callbacks=k,
        )
        self.owners.append(self.active)
        if not second.admitted:
            raise EmpiricalNonattempt("OWNED_SECOND_NONATTEMPT")

    def advance(
        self, controller: Any, previous: tuple[float, float], session: Any
    ) -> ReactorDelivery | ReactorFinalDelivery | None:
        if len(self.frozen) > len(self.owners):
            return None
        stage = self.frozen[-1]
        if not stage.admitted:
            if self.active.tick is not None:
                raise ValueError("second refusal was delivered twice")
            self.active.tick = self.active.prepared.deliver(self.publisher)
            if self.active.tick.disposition is not TickDisposition.NONATTEMPT:
                raise ValueError("second refusal delivered a fallback action")
            return None
        raw = self.active.advance(controller, previous, session)
        if raw is not None:
            self.requests.append((float(raw.command.feed_kg_s), float(raw.command.jacket_k)))
            self.stages.append(
                tuple(
                    float(v)
                    for v in (
                        raw.accepted_feed_kg_s,
                        raw.accepted_jacket_k,
                        raw.applied_feed_kg_s,
                        raw.applied_jacket_k,
                    )
                )
            )
            self.exposure.append(
                tuple(
                    tuple(float(v) for v in (e.time_s, e.duration_s, e.feed_kg_s, e.jacket_k))
                    for e in raw.exposures
                )
            )
        return raw


def acquire_prospective(
    *,
    source: ReactorBatchSource,
    preparation: ClassicalPreparation,
    private: ClassicalPrivate,
    frozen: ClassicalFrozenRoot,
    plan: ReactorStagedPulseResponseProspectivePlan,
    laws: ClassicalLaws,
    publisher: Any,
    reader: Any,
    control: Any,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> ClassicalNativeRoot:
    if (
        frozen.preparation != ObjectIdentity.from_record(preparation.record_id, preparation)
        or frozen.plan != ObjectIdentity.from_record(plan.record_id, plan)
        or source.fingerprint() != preparation.source_sha256
        or plan.laws != ObjectIdentity.from_record(laws.record_id, laws)
    ):
        raise ValueError("native continuation lacks its exact frozen parents")
    original = donors(preparation, private)
    stages = frozen.unpack()
    uses, references = [], []
    completed = reference_calls = 0
    for initial in stages:
        if not initial.admitted:
            tick = None if initial.compiled is None else restore(initial).deliver(publisher)
            actual = ClassicalNativeStage(
                CanonicalRecordArchive.pack(initial.record_id, initial), tick, ()
            )
            uses.append(
                ClassicalNativeUse(
                    initial.policy, initial.request, (actual,), (), 0, initial.reasons, False
                )
            )
            continue
        assert (
            initial.projection is not None
            and initial.causal.callback is not None
            and len(original) == 2
        )
        k = initial.causal.callback
        tape = overlay_feed_tape(
            original[0].requests,
            k,
            initial.projection.pulse.rates(initial.projection.guard_s),
            float(initial.projection.fixed_jacket_K),
        )

        def second(
            causal: ClassicalContext, first_tick: DeliveryControllerTickReceipt
        ) -> ClassicalFrozenStage:
            artifact = publisher.publish_record(causal.record_id, causal)
            selected = freeze_stage(
                laws=laws,
                plan=plan,
                causal=causal,
                policy=initial.policy,
                request=initial.request,
                kind="joint",
                source=source,
                publisher=publisher,
                reader=reader,
                control=control,
                checkpoint=ObjectIdentity.from_record(causal.record_id, causal),
                causal_artifact=artifact,
                causal_receipt=first_tick,
            )
            publisher.publish_record(selected.record_id, selected)
            return selected

        active = FiniteCommittedBranchOwner(
            preparation.root,
            f"{preparation.root}.{initial.policy.lower()}.{initial.request.request_id.lower()}.first",
            k,
            original[0].observations,
            original[0].stages,
            tape,
            restore(initial),
            publisher,
        )
        owner = ClassicalSequenceOwner(
            initial,
            original[0],
            publisher,
            second if initial.stage_kind == "first" else None,
            active,
            [initial],
            [active],
        )
        controller = DecisionTape(tape, owner)
        nominal, completed = acquire_with_progress(
            source,
            private.scenario,
            f"{initial.policy.lower()}.{initial.request.request_id.lower()}.owner",
            controller,
            1.0,
            partial(acquire, owner=owner, retain_stopped_prefix=True),
            completed,
            progress,
        )
        episodes = [nominal]
        if len(nominal.requests) >= k + 12:
            refined, completed = acquire_with_progress(
                source,
                private.scenario,
                f"{initial.policy.lower()}.{initial.request.request_id.lower()}.refined",
                TapeController(nominal.requests),
                0.5,
                partial(acquire, retain_stopped_prefix=True),
                completed,
                progress,
            )
            episodes.append(refined)
        windows = []
        for view, episode in enumerate(episodes):
            if initial.stage_kind == "first":
                windows.append(
                    native_window(
                        original[view], episode, k, FIRST.rates(), nominal.requests, "actual.first"
                    )
                )
                if len(owner.frozen) == 2 and owner.frozen[1].admitted:
                    projection = owner.frozen[1].projection
                    assert projection is not None
                    windows.append(
                        native_window(
                            original[view],
                            episode,
                            k,
                            FIRST.rates() + projection.pulse.rates(),
                            nominal.requests,
                            "actual.episode",
                        )
                    )
            else:
                windows.append(
                    native_window(
                        original[view],
                        episode,
                        k,
                        initial.projection.pulse.rates(initial.projection.guard_s),
                        nominal.requests,
                        "actual.episode",
                    )
                )
        actual_stages = tuple(
            ClassicalNativeStage(
                CanonicalRecordArchive.pack(s.record_id, s),
                None if i >= len(owner.owners) else owner.owners[i].tick,
                ()
                if i >= len(owner.owners)
                else tuple(owner.owners[i].prepared.delivery.last_deliveries),
            )
            for i, s in enumerate(owner.frozen)
        )
        guard = 240 if initial.stage_kind in ("first", "baseline") else 120
        unsafe = any(
            len(e.grid) and (e.grid[: int((k * 10 + guard) / e.dt) + 1, 1] > 356.2).any()
            for e in episodes
        )
        uses.append(
            ClassicalNativeUse(
                initial.policy,
                initial.request,
                actual_stages,
                tuple(windows),
                len(episodes),
                tuple(sorted({e.failure for e in episodes if e.failure is not None})),
                bool(unsafe),
            )
        )
    # Matched references are the only additional D acquisitions. They are not
    # action audits and never enter a controller's causal inputs.
    for ca in preparation.contexts:
        if (
            ca.callback is None
            or ca.reasons
            or not any(
                u.policy == s.policy
                and u.request == s.request
                and u.native_calls
                and s.causal.context == ca.context
                for u, s in zip(uses, stages, strict=True)
            )
        ):
            continue
        k = ca.callback
        branches = (
            (("00", ZERO.rates(240)), ("a0", FIRST.rates(240)))
            if plan.block == "staged-sequence-comparison"
            else ((f"{ca.context}.{ZERO.word_id}", ZERO.rates()),)
        )
        for name, rates in branches:
            tape = overlay_feed_tape(
                original[0].requests, k, rates, float(original[0].stages[k - 1, 3])
            )
            for view, donor in enumerate(original):
                episode, completed = acquire_with_progress(
                    source,
                    private.scenario,
                    name,
                    TapeController(tape),
                    0.5 if view else 1.0,
                    partial(acquire, retain_stopped_prefix=True),
                    completed,
                    progress,
                )
                references.append(native_window(donor, episode, k, rates, tape, name))
                reference_calls += 1
    return ClassicalNativeRoot(
        preparation.root,
        ObjectIdentity.from_record(frozen.record_id, frozen),
        ObjectIdentity.from_record(preparation.root, private.scenario),
        tuple(uses),
        tuple(references),
        reference_calls,
    )
