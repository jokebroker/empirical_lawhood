"Cumulative-elapsed campaign budget for deadline-free execution envelopes.\n\n``ExecutionResourceEnvelopeSpec`` is deliberately elapsed-free: its transitions\ndo not read a total time target, and it refuses\n``elapsed_time_has_no_control_effect=False``. This companion contract records\ncumulative active campaign time across launch and recovery. It refuses further\nphysical reservations when the central budget, which is the ceiling minus its\nreserve, cannot cover a projected task. A scheduler consults the companion\ncontract before reserving a native cell and brackets each active interval with\n:meth:`open_interval`/:meth:`close_interval`. Per-worker progress liveness uses\nthe separate ``ProgressLivenessContract``.\n\nThe budget is bound to one campaign anchor. Successive conditional-stage\nenvelopes attach to the same durable ledger before use, so a new stage or an\noperator restart cannot reset elapsed accounting. Recovery reloads the ledger\nand reconciles an open interval from observed evidence.\n"

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec

ELAPSED_INTERVAL_KINDS = ("LAUNCH", "RECOVERY")


def _active_seconds(opened_at_utc: str, closed_at_utc: str) -> Decimal:
    """Exact non-negative active seconds between two strict UTC timestamps."""

    opened = parse_utc_timestamp(opened_at_utc, field_name="opened_at_utc")
    closed = parse_utc_timestamp(closed_at_utc, field_name="closed_at_utc")
    delta = (closed - opened).total_seconds()
    if delta < 0:
        raise ValueError("active interval closes before it opened")
    return Decimal(str(delta))


@dataclass(frozen=True, slots=True)
class CampaignElapsedBudgetSpec(CanonicalRecord):
    """Cumulative-active-time ceiling and reserve for one conditional campaign."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/campaign-elapsed-budget-spec'

    budget_id: str
    campaign_anchor: ObjectIdentity
    cumulative_elapsed_ceiling_seconds: int
    reserve_fraction: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.budget_id, field_name="budget_id")
        if not self.campaign_anchor.object_schema:
            raise ValueError("elapsed budget requires an exact campaign anchor")
        if type(self.cumulative_elapsed_ceiling_seconds) is not int or (
            self.cumulative_elapsed_ceiling_seconds <= 0
        ):
            raise ValueError("cumulative elapsed ceiling must be a positive integer of seconds")
        validate_decimal(self.reserve_fraction, field_name="reserve_fraction", minimum=Decimal(0))
        if self.reserve_fraction >= 1:
            raise ValueError("reserve fraction must be below one so central work remains")

    @property
    def central_ceiling_seconds(self) -> Decimal:
        """Seconds available to central scientific work plus its ordinary pipeline."""

        return Decimal(self.cumulative_elapsed_ceiling_seconds) * (
            Decimal(1) - self.reserve_fraction
        )

    @property
    def hard_ceiling_seconds(self) -> Decimal:
        """Absolute campaign boundary, including the protected reserve."""

        return Decimal(self.cumulative_elapsed_ceiling_seconds)


@dataclass(frozen=True, slots=True)
class CampaignElapsedEnvelopeBinding(CanonicalRecord):
    "One conditional stage's exact envelope, chained into the campaign ledger."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/campaign-elapsed-envelope-binding'

    binding_id: str
    budget: ObjectIdentity
    stage_id: str
    governed_envelope: ObjectIdentity
    predecessor_binding: ObjectIdentity | None
    bound_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.stage_id, field_name="stage_id")
        if self.budget.object_schema != CampaignElapsedBudgetSpec.SCHEMA:
            raise ValueError("elapsed envelope binding references a non-budget identity")
        if self.governed_envelope.object_schema != ExecutionResourceEnvelopeSpec.SCHEMA:
            raise ValueError("elapsed envelope binding requires an exact resource envelope")
        if self.predecessor_binding is not None and (
            self.predecessor_binding.object_schema != self.SCHEMA
        ):
            raise ValueError("elapsed envelope predecessor has an incompatible schema")
        parse_utc_timestamp(self.bound_at_utc, field_name="bound_at_utc")


@dataclass(frozen=True, slots=True)
class CampaignElapsedInterval(CanonicalRecord):
    """One closed active-campaign interval; recovery and launch both accrue time."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/campaign-elapsed-interval'

    interval_id: str
    envelope_binding: ObjectIdentity
    kind: str
    opened_at_utc: str
    closed_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.interval_id, field_name="interval_id")
        if self.envelope_binding.object_schema != CampaignElapsedEnvelopeBinding.SCHEMA:
            raise ValueError("campaign interval requires its exact stage-envelope binding")
        if self.kind not in ELAPSED_INTERVAL_KINDS:
            raise ValueError("campaign elapsed interval kind is outside its fixed vocabulary")
        # Validates ordering and non-negativity as a side effect.
        _active_seconds(self.opened_at_utc, self.closed_at_utc)

    @property
    def active_seconds(self) -> Decimal:
        return _active_seconds(self.opened_at_utc, self.closed_at_utc)


@dataclass(frozen=True, slots=True)
class CampaignElapsedAdmission(CanonicalRecord):
    """Outcome of one reservation admission check against the central ceiling."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/campaign-elapsed-admission'

    admission_id: str
    budget: ObjectIdentity
    envelope_binding: ObjectIdentity
    evaluated_at_utc: str
    used_active_seconds: Decimal
    projected_active_seconds: Decimal
    central_ceiling_seconds: Decimal
    admitted: bool
    reason_code: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.admission_id, field_name="admission_id")
        if self.budget.object_schema != CampaignElapsedBudgetSpec.SCHEMA:
            raise ValueError("elapsed admission references a non-budget identity")
        if self.envelope_binding.object_schema != CampaignElapsedEnvelopeBinding.SCHEMA:
            raise ValueError("elapsed admission references a non-envelope binding")
        parse_utc_timestamp(self.evaluated_at_utc, field_name="evaluated_at_utc")
        for name in ("used_active_seconds", "projected_active_seconds", "central_ceiling_seconds"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if type(self.admitted) is not bool:
            raise ValueError("elapsed admission verdict must be a bool")
        if self.admitted:
            if self.reason_code is not None:
                raise ValueError("an admitted reservation carries no refusal reason")
        else:
            validate_nonempty(self.reason_code or "", field_name="reason_code")

    @property
    def remaining_central_seconds(self) -> Decimal:
        return self.central_ceiling_seconds - self.used_active_seconds


@dataclass(frozen=True, slots=True)
class CampaignElapsedTaskProjection(CanonicalRecord):
    """Canary-derived elapsed reservation for one native task."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/campaign-elapsed-task-projection'

    task_id: str
    projected_active_seconds: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        validate_decimal(
            self.projected_active_seconds,
            field_name="projected_active_seconds",
            minimum=Decimal("0.000001"),
        )


@dataclass(frozen=True, slots=True)
class CampaignElapsedReservationPlan(CanonicalRecord):
    """Exact native-task roster governed by one conditional-stage envelope."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/campaign-elapsed-reservation-plan'

    plan_id: str
    budget: ObjectIdentity
    stage_id: str
    envelope_binding_id: str
    governed_envelope: ObjectIdentity
    projection_basis: ObjectIdentity
    task_projections: tuple[CampaignElapsedTaskProjection, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        validate_stable_id(self.stage_id, field_name="stage_id")
        validate_stable_id(self.envelope_binding_id, field_name="envelope_binding_id")
        if self.budget.object_schema != CampaignElapsedBudgetSpec.SCHEMA:
            raise ValueError("elapsed reservation plan references a non-budget identity")
        if self.governed_envelope.object_schema != ExecutionResourceEnvelopeSpec.SCHEMA:
            raise ValueError("elapsed reservation plan requires an exact resource envelope")
        task_ids = tuple(value.task_id for value in self.task_projections)
        if not task_ids or task_ids != tuple(sorted(set(task_ids))):
            raise ValueError("elapsed reservation task projections must be sorted and unique")

    def projected_seconds(self, task_id: str) -> Decimal:
        matches = tuple(
            value.projected_active_seconds
            for value in self.task_projections
            if value.task_id == task_id
        )
        if len(matches) != 1:
            raise KeyError(task_id)
        return matches[0]

    def validate_task_census(self, envelope: ExecutionResourceEnvelopeSpec) -> None:
        if self.governed_envelope != ObjectIdentity.from_record(
            envelope.envelope_spec_id, envelope
        ):
            raise ValueError("elapsed reservation plan binds another resource envelope")
        native_ids = tuple(
            sorted(value.task_id for value in envelope.task_cells if value.native_simulator_launch)
        )
        if tuple(value.task_id for value in self.task_projections) != native_ids:
            raise ValueError("elapsed reservation plan changes the exact native task census")


@dataclass(frozen=True, slots=True)
class CampaignElapsedLedger(CanonicalRecord):
    """Durable cumulative-active-time prefix with at most one open interval."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/campaign-elapsed-ledger'

    ledger_id: str
    budget: ObjectIdentity
    envelope_bindings: tuple[CampaignElapsedEnvelopeBinding, ...]
    closed_intervals: tuple[CampaignElapsedInterval, ...]
    open_interval_id: str | None
    open_interval_kind: str | None
    open_interval_started_utc: str | None
    open_envelope_binding: ObjectIdentity | None
    exhausted: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        if self.budget.object_schema != CampaignElapsedBudgetSpec.SCHEMA:
            raise ValueError("elapsed ledger references a non-budget identity")
        binding_ids = tuple(value.binding_id for value in self.envelope_bindings)
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("elapsed ledger repeats a stage-envelope binding")
        previous_binding: CampaignElapsedEnvelopeBinding | None = None
        for binding in self.envelope_bindings:
            expected_predecessor = (
                None
                if previous_binding is None
                else ObjectIdentity.from_record(previous_binding.binding_id, previous_binding)
            )
            if binding.budget != self.budget or binding.predecessor_binding != expected_predecessor:
                raise ValueError("elapsed stage-envelope chain changes its budget or predecessor")
            if previous_binding is not None and parse_utc_timestamp(
                binding.bound_at_utc, field_name="bound_at_utc"
            ) < parse_utc_timestamp(previous_binding.bound_at_utc, field_name="prior_bound_at_utc"):
                raise ValueError("elapsed stage-envelope bindings run backwards in campaign time")
            previous_binding = binding
        binding_identities = {
            ObjectIdentity.from_record(binding.binding_id, binding)
            for binding in self.envelope_bindings
        }
        interval_ids = tuple(value.interval_id for value in self.closed_intervals)
        if len(set(interval_ids)) != len(interval_ids):
            raise ValueError("closed_intervals must have unique interval identities")
        previous_close: datetime | None = None
        for interval in self.closed_intervals:
            if interval.envelope_binding not in binding_identities:
                raise ValueError("closed interval uses an unbound stage envelope")
            opened = parse_utc_timestamp(interval.opened_at_utc, field_name="opened_at_utc")
            if previous_close is not None and opened < previous_close:
                raise ValueError("closed active intervals overlap in campaign time")
            previous_close = parse_utc_timestamp(interval.closed_at_utc, field_name="closed_at_utc")
        open_fields = (
            self.open_interval_id,
            self.open_interval_kind,
            self.open_interval_started_utc,
            self.open_envelope_binding,
        )
        if (None in open_fields) != (all(value is None for value in open_fields)):
            raise ValueError("an open interval requires its id, kind and start together")
        if self.open_interval_id is not None:
            validate_stable_id(self.open_interval_id, field_name="open_interval_id")
            if self.open_interval_kind not in ELAPSED_INTERVAL_KINDS:
                raise ValueError("open interval kind is outside its fixed vocabulary")
            assert self.open_interval_started_utc is not None
            assert self.open_envelope_binding is not None
            if (
                not self.envelope_bindings
                or self.open_envelope_binding
                != ObjectIdentity.from_record(
                    self.envelope_bindings[-1].binding_id, self.envelope_bindings[-1]
                )
            ):
                raise ValueError("open interval does not use the current stage envelope")
            started = parse_utc_timestamp(
                self.open_interval_started_utc, field_name="open_interval_started_utc"
            )
            if previous_close is not None and started < previous_close:
                raise ValueError("open active interval overlaps a closed interval")
            if self.open_interval_id in {value.interval_id for value in self.closed_intervals}:
                raise ValueError("open interval reuses a closed interval identity")
        if type(self.exhausted) is not bool:
            raise ValueError("ledger exhaustion flag must be a bool")

    @property
    def accumulated_active_seconds(self) -> Decimal:
        return sum(
            (interval.active_seconds for interval in self.closed_intervals),
            Decimal(0),
        )

    def active_seconds_at(self, at_utc: str) -> Decimal:
        """Cumulative active seconds including any interval still open at ``at_utc``."""

        used = self.accumulated_active_seconds
        if self.open_interval_started_utc is not None:
            used += _active_seconds(self.open_interval_started_utc, at_utc)
        return used

    def hard_remaining_seconds_at(
        self,
        budget: CampaignElapsedBudgetSpec,
        *,
        at_utc: str,
    ) -> Decimal:
        """Active seconds remaining before the absolute campaign boundary."""

        if self.budget != ObjectIdentity.from_record(budget.budget_id, budget):
            raise ValueError("hard-boundary query uses another campaign budget")
        return max(budget.hard_ceiling_seconds - self.active_seconds_at(at_utc), Decimal(0))


class CampaignElapsedBudgetMachine:
    """Pure transitions over the cumulative-active-time ledger."""

    @staticmethod
    def initial(*, ledger_id: str, budget: CampaignElapsedBudgetSpec) -> CampaignElapsedLedger:
        return CampaignElapsedLedger(
            ledger_id=ledger_id,
            budget=ObjectIdentity.from_record(budget.budget_id, budget),
            envelope_bindings=(),
            closed_intervals=(),
            open_interval_id=None,
            open_interval_kind=None,
            open_interval_started_utc=None,
            open_envelope_binding=None,
            exhausted=False,
        )

    @staticmethod
    def bind_envelope(
        ledger: CampaignElapsedLedger,
        budget: CampaignElapsedBudgetSpec,
        *,
        binding_id: str,
        stage_id: str,
        governed_envelope: ObjectIdentity,
        at_utc: str,
    ) -> CampaignElapsedLedger:
        """Attach the next conditional stage without resetting campaign time."""

        if ObjectIdentity.from_record(budget.budget_id, budget) != ledger.budget:
            raise ValueError("stage binding budget differs from the durable campaign ledger")
        if ledger.exhausted or ledger.open_interval_id is not None:
            raise ValueError("cannot bind a stage while the budget is exhausted or active")
        if governed_envelope in {value.governed_envelope for value in ledger.envelope_bindings}:
            raise ValueError("campaign stage envelope is already bound")
        predecessor = (
            None
            if not ledger.envelope_bindings
            else ObjectIdentity.from_record(
                ledger.envelope_bindings[-1].binding_id, ledger.envelope_bindings[-1]
            )
        )
        binding = CampaignElapsedEnvelopeBinding(
            binding_id=binding_id,
            budget=ledger.budget,
            stage_id=stage_id,
            governed_envelope=governed_envelope,
            predecessor_binding=predecessor,
            bound_at_utc=at_utc,
        )
        return replace(ledger, envelope_bindings=(*ledger.envelope_bindings, binding))

    @staticmethod
    def open_interval(
        ledger: CampaignElapsedLedger,
        *,
        envelope_binding: CampaignElapsedEnvelopeBinding,
        interval_id: str,
        kind: str,
        at_utc: str,
    ) -> CampaignElapsedLedger:
        if ledger.exhausted:
            raise ValueError("cannot open an active interval after the budget is exhausted")
        if ledger.open_interval_id is not None:
            raise ValueError("an active interval is already open")
        if not ledger.envelope_bindings or envelope_binding != ledger.envelope_bindings[-1]:
            raise ValueError("active interval must use the current bound stage envelope")
        validate_stable_id(interval_id, field_name="interval_id")
        if kind not in ELAPSED_INTERVAL_KINDS:
            raise ValueError("active interval kind is outside its fixed vocabulary")
        parse_utc_timestamp(at_utc, field_name="at_utc")
        return replace(
            ledger,
            open_interval_id=interval_id,
            open_interval_kind=kind,
            open_interval_started_utc=at_utc,
            open_envelope_binding=ObjectIdentity.from_record(
                envelope_binding.binding_id, envelope_binding
            ),
        )

    @staticmethod
    def close_interval(ledger: CampaignElapsedLedger, *, at_utc: str) -> CampaignElapsedLedger:
        if ledger.open_interval_id is None:
            raise ValueError("no active interval is open")
        assert ledger.open_interval_kind is not None
        assert ledger.open_interval_started_utc is not None
        assert ledger.open_envelope_binding is not None
        interval = CampaignElapsedInterval(
            interval_id=ledger.open_interval_id,
            envelope_binding=ledger.open_envelope_binding,
            kind=ledger.open_interval_kind,
            opened_at_utc=ledger.open_interval_started_utc,
            closed_at_utc=at_utc,
        )
        return replace(
            ledger,
            closed_intervals=(*ledger.closed_intervals, interval),
            open_interval_id=None,
            open_interval_kind=None,
            open_interval_started_utc=None,
            open_envelope_binding=None,
        )

    @classmethod
    def reconcile_open_interval(
        cls, ledger: CampaignElapsedLedger, *, observed_close_utc: str
    ) -> CampaignElapsedLedger:
        """On recovery, close any interval left open by a crashed operator.

        Absent an open interval this is the identity, so recovery that already
        observed a clean close does not double-count active time.
        """

        if ledger.open_interval_id is None:
            parse_utc_timestamp(observed_close_utc, field_name="observed_close_utc")
            return ledger
        return cls.close_interval(ledger, at_utc=observed_close_utc)

    @staticmethod
    def admit_reservation(
        ledger: CampaignElapsedLedger,
        budget: CampaignElapsedBudgetSpec,
        *,
        envelope_binding: CampaignElapsedEnvelopeBinding,
        admission_id: str,
        at_utc: str,
        projected_active_seconds: Decimal,
    ) -> CampaignElapsedAdmission:
        if ObjectIdentity.from_record(budget.budget_id, budget) != ledger.budget:
            raise ValueError("admission budget differs from the ledger's governed budget")
        if not ledger.envelope_bindings or envelope_binding != ledger.envelope_bindings[-1]:
            raise ValueError("admission uses a stage envelope outside the durable campaign chain")
        validate_decimal(
            projected_active_seconds,
            field_name="projected_active_seconds",
            minimum=Decimal(0),
        )
        used = ledger.active_seconds_at(at_utc)
        central = budget.central_ceiling_seconds
        admitted = not ledger.exhausted and used + projected_active_seconds <= central
        reason: str | None = None
        if not admitted:
            reason = (
                "CAMPAIGN_ELAPSED_BUDGET_EXHAUSTED"
                if ledger.exhausted
                else "CAMPAIGN_ELAPSED_BUDGET_INSUFFICIENT_FOR_PROJECTED_TASK"
            )
        return CampaignElapsedAdmission(
            admission_id=admission_id,
            budget=ledger.budget,
            envelope_binding=ObjectIdentity.from_record(
                envelope_binding.binding_id, envelope_binding
            ),
            evaluated_at_utc=at_utc,
            used_active_seconds=used,
            projected_active_seconds=projected_active_seconds,
            central_ceiling_seconds=central,
            admitted=admitted,
            reason_code=reason,
        )

    @classmethod
    def stop_at_boundary(
        cls, ledger: CampaignElapsedLedger, *, at_utc: str
    ) -> CampaignElapsedLedger:
        """Close any in-flight interval and mark the budget exhausted.

        The caller pairs this with the envelope's ``UNKNOWN_COMPLETION`` disposition
        for work still running; it never fabricates a completed receipt.
        """

        closed = (
            ledger if ledger.open_interval_id is None else cls.close_interval(ledger, at_utc=at_utc)
        )
        return replace(closed, exhausted=True)


class DurableCampaignElapsedBudgetStore(Protocol):
    """Externally durable compare-and-append authority for the elapsed ledger."""

    def create(self, ledger: CampaignElapsedLedger) -> None: ...

    def load(self, ledger_id: str) -> CampaignElapsedLedger: ...

    def compare_and_append(
        self,
        *,
        expected_ledger_sha256: str,
        updated: CampaignElapsedLedger,
    ) -> None: ...


class DurableCampaignElapsedBudgetCoordinator:
    """Persist every campaign-time transition before acknowledging it."""

    def __init__(
        self,
        store: DurableCampaignElapsedBudgetStore,
        *,
        budget: CampaignElapsedBudgetSpec,
        ledger_id: str,
    ) -> None:
        validate_stable_id(ledger_id, field_name="ledger_id")
        self._store = store
        self.budget = budget
        self.ledger_id = ledger_id

    @property
    def budget_identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.budget.budget_id, self.budget)

    def initialize(self) -> CampaignElapsedLedger:
        ledger = CampaignElapsedBudgetMachine.initial(
            ledger_id=self.ledger_id,
            budget=self.budget,
        )
        self._store.create(ledger)
        return ledger

    def ensure_initialized(self) -> CampaignElapsedLedger:
        try:
            current = self.current()
        except (KeyError, FileNotFoundError):
            return self.initialize()
        return current

    def current(self) -> CampaignElapsedLedger:
        ledger = self._store.load(self.ledger_id)
        if ledger.ledger_id != self.ledger_id or ledger.budget != self.budget_identity:
            raise ValueError("durable elapsed-budget store returned a substitution")
        return ledger

    def _persist(
        self,
        current: CampaignElapsedLedger,
        updated: CampaignElapsedLedger,
    ) -> CampaignElapsedLedger:
        self._store.compare_and_append(
            expected_ledger_sha256=current.fingerprint(),
            updated=updated,
        )
        return updated

    def bind_envelope(
        self,
        *,
        binding_id: str,
        stage_id: str,
        governed_envelope: ObjectIdentity,
        at_utc: str,
    ) -> CampaignElapsedLedger:
        current = self.ensure_initialized()
        matches = tuple(
            value
            for value in current.envelope_bindings
            if value.binding_id == binding_id or value.stage_id == stage_id
        )
        if matches:
            if (
                len(matches) != 1
                or matches[0] != current.envelope_bindings[-1]
                or matches[0].binding_id != binding_id
                or matches[0].stage_id != stage_id
                or matches[0].governed_envelope != governed_envelope
            ):
                raise ValueError("durable elapsed stage binding conflicts with its campaign prefix")
            return current
        updated = CampaignElapsedBudgetMachine.bind_envelope(
            current,
            self.budget,
            binding_id=binding_id,
            stage_id=stage_id,
            governed_envelope=governed_envelope,
            at_utc=at_utc,
        )
        return self._persist(current, updated)

    def open_interval(
        self,
        *,
        interval_id: str,
        kind: str,
        at_utc: str,
    ) -> CampaignElapsedLedger:
        current = self.current()
        if current.open_interval_id is not None:
            if current.open_interval_id != interval_id or current.open_interval_kind != kind:
                raise ValueError("another durable elapsed interval is already active")
            return current
        if not current.envelope_bindings:
            raise ValueError("durable elapsed interval precedes its stage-envelope binding")
        updated = CampaignElapsedBudgetMachine.open_interval(
            current,
            envelope_binding=current.envelope_bindings[-1],
            interval_id=interval_id,
            kind=kind,
            at_utc=at_utc,
        )
        return self._persist(current, updated)

    def close_interval(self, *, at_utc: str) -> CampaignElapsedLedger:
        current = self.current()
        updated = CampaignElapsedBudgetMachine.close_interval(current, at_utc=at_utc)
        return self._persist(current, updated)

    def reconcile_open_interval(self, *, observed_close_utc: str) -> CampaignElapsedLedger:
        current = self.current()
        updated = CampaignElapsedBudgetMachine.reconcile_open_interval(
            current,
            observed_close_utc=observed_close_utc,
        )
        return current if updated == current else self._persist(current, updated)

    def admit_reservation(
        self,
        *,
        admission_id: str,
        at_utc: str,
        projected_active_seconds: Decimal,
    ) -> CampaignElapsedAdmission:
        current = self.current()
        if not current.envelope_bindings:
            raise ValueError("elapsed admission precedes its stage-envelope binding")
        return CampaignElapsedBudgetMachine.admit_reservation(
            current,
            self.budget,
            envelope_binding=current.envelope_bindings[-1],
            admission_id=admission_id,
            at_utc=at_utc,
            projected_active_seconds=projected_active_seconds,
        )

    def hard_remaining_seconds(self, *, at_utc: str) -> Decimal:
        return self.current().hard_remaining_seconds_at(self.budget, at_utc=at_utc)

    def stop_at_boundary(self, *, at_utc: str) -> CampaignElapsedLedger:
        current = self.current()
        if current.exhausted:
            return current
        updated = CampaignElapsedBudgetMachine.stop_at_boundary(current, at_utc=at_utc)
        return self._persist(current, updated)


def response_rc_study_elapsed_budget(
    campaign_anchor: ObjectIdentity,
    *,
    budget_id: str = "response-controller-study.campaign-elapsed-budget",
) -> CampaignElapsedBudgetSpec:
    """The plan's 24 cumulative elapsed hours with a 20% reserve (19.2h central)."""

    return CampaignElapsedBudgetSpec(
        budget_id=budget_id,
        campaign_anchor=campaign_anchor,
        cumulative_elapsed_ceiling_seconds=24 * 60 * 60,
        reserve_fraction=Decimal("0.20"),
    )
