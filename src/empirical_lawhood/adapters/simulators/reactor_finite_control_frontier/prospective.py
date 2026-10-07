"""Reconstruct frozen owners, deliver native macros, then measure the audit chart."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar

from empirical_lawhood.adapters.control.finite_campaign import finite_control_services
from empirical_lawhood.adapters.control.composition import ControllerStudyComposition
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_lock import FrontierFrozenRoot, FrontierFrozenUse, UndeliverableSession
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_plan import ReactorFiniteControlFrontierProspectivePlan
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierPreparation, FrontierPrivate, FrontierAssay
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.selection import FrontierUseRequest
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode, TapeController, acquire_episode
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import acquire_with_progress
from empirical_lawhood.adapters.simulators.reactor_regime_response.committed_branch import WatchedPreparedTape
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorDelivery
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerTickReceipt
from .acquisition import acquire_assay, donor_episodes, native_window, scenario_for
from .delivery import FrontierPulseDeliveryPort, FrontierCommittedBranchOwner
from .words import pulse_tape


@dataclass(frozen=True, slots=True)
class FrontierNativeUse(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-finite-control-frontier/frontier-native-use'
    request: FrontierUseRequest
    arrays: LocalArrayPayload
    tick: DeliveryControllerTickReceipt | None
    native_deliveries: tuple[ReactorDelivery, ...]
    native_calls: int
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.native_calls not in (0, 1, 2)
            or len(self.native_deliveries) > 12
            or (
                self.native_calls == 0
                and (
                    self.tick is not None
                    or self.native_deliveries
                    or self.arrays.unpack()
                    or not self.failures
                )
            )
        ):
            raise ValueError("native use invented an acquisition or complete macro")


@dataclass(frozen=True, slots=True)
class FrontierNativeRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-finite-control-frontier/frontier-native-root'
    root: str
    frozen: ObjectIdentity
    uses: tuple[FrontierNativeUse, ...]
    assay: FrontierAssay

    def __post_init__(self) -> None:
        if (
            self.frozen.object_schema != FrontierFrozenRoot.SCHEMA
            or self.assay.root != self.root
            or self.native_calls > 128
        ):
            raise ValueError("prospective root changed its frozen parent or native budget")

    @property
    def native_calls(self) -> int:
        return self.assay.native_calls + sum(u.native_calls for u in self.uses)

    @property
    def record_id(self) -> str:
        return f"{self.root}.frontier-native-root"


@dataclass
class RestoredRequest:
    frozen: FrontierFrozenUse
    delivery: FrontierPulseDeliveryPort
    services: ControllerStudyComposition

    def deliver(self, publisher: Any) -> DeliveryControllerTickReceipt:
        assert self.frozen.compiled is not None and self.frozen.commitment is not None
        tick = self.services.deliver_prepared_commitment(
            self.frozen.compiled, self.frozen.commitment
        )
        publisher.publish_record(tick.tick_id, tick)
        return tick


def acquire_prospective(
    *,
    source: ReactorBatchSource,
    preparation: FrontierPreparation,
    private: FrontierPrivate,
    frozen: FrontierFrozenRoot,
    plan: ReactorFiniteControlFrontierProspectivePlan,
    publisher: Any,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> FrontierNativeRoot:
    if (
        frozen.preparation != ObjectIdentity.from_record(preparation.record_id, preparation)
        or frozen.plan != ObjectIdentity.from_record(plan.record_id, plan)
        or source.fingerprint() != preparation.source_sha256
    ):
        raise ValueError("native macro lacks the exact complete precontact lock")
    donors = donor_episodes(preparation, private)
    scenario = scenario_for(preparation.root)
    uses = []
    completed = 0
    for use in frozen.uses:
        if use.projection is None:
            uses.append(
                FrontierNativeUse(
                    use.request, LocalArrayPayload.pack({}), None, (), 0, use.reasons
                )
            )
            continue
        assert use.compiled is not None
        ca = next(c for c in preparation.contexts if c.context == use.request.context)
        assert ca.callback is not None and len(donors) == 2
        assert use.lock is not None
        evaluation = use.lock.evaluation_for(use.compiled)
        if (
            use.lock.parent.decision != use.commitment
            or evaluation.root.root_id != preparation.root
            or evaluation.policy.policy_id != f"el.{use.request.request_id}"
        ):
            raise ValueError("restored use changed its authenticated prepared design")
        delivery = FrontierPulseDeliveryPort(
            use.compiled.implementation(ImplementationRole.DELIVERY),
            UndeliverableSession(),
            use.projection,
        )
        restored = RestoredRequest(
            use, delivery, finite_control_services(use.compiled.study, evaluation, delivery)
        )
        tape = pulse_tape(
            donors[0].requests,
            ca.callback,
            use.projection.pulse,
            float(use.projection.fixed_jacket_K),
        )
        owner = FrontierCommittedBranchOwner(
            preparation.root,
            f"{preparation.root}.{use.request.request_id}",
            ca.callback,
            donors[0].observations,
            donors[0].stages,
            tape,
            restored,
            publisher,
        )
        nominal, completed = acquire_with_progress(
            source,
            scenario,
            f"{use.request.request_id}.owner",
            WatchedPreparedTape(tape),
            1.0,
            partial(acquire, owner=owner),
            completed,
            progress,
        )
        calls, failures, data = 1, [], {}
        for field, value in native_window(donors[0], nominal, ca.callback, tape).items():
            data[f"v0_{field}"] = value
        if nominal.complete and owner.delivered and not owner.failed:
            refined, completed = acquire_with_progress(
                source,
                scenario,
                f"{use.request.request_id}.refined",
                TapeController(tape),
                0.5,
                acquire,
                completed,
                progress,
            )
            calls += 1
            for field, value in native_window(donors[1], refined, ca.callback, tape).items():
                data[f"v1_{field}"] = value
            if not refined.complete:
                failures.append(refined.failure or "INCOMPLETE_REFINED_REPLAY")
        else:
            failures.append(nominal.failure or "INCOMPLETE_OWNER_DELIVERY")
        uses.append(
            FrontierNativeUse(
                use.request,
                LocalArrayPayload.pack(data),
                owner.tick,
                tuple(delivery.last_deliveries),
                calls,
                tuple(failures),
            )
        )
    base = completed
    assay = acquire_assay(
        source,
        preparation,
        private,
        frozen.prediction_seal,
        acquire=acquire,
        progress=None if progress is None else lambda value: progress(base + value),
    )
    return FrontierNativeRoot(
        preparation.root, ObjectIdentity.from_record(frozen.record_id, frozen), tuple(uses), assay
    )
