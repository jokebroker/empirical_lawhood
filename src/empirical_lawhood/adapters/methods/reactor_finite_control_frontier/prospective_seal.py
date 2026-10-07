"""Native receipt custody and separate prepared-future closure, without verdicts."""

from dataclasses import dataclass
from typing import ClassVar
from empirical_lawhood.adapters.control.prepared_forecast import seal_forecast_request
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.prospective import FrontierNativeRoot
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore, PreparedFutureDisposition, SealedPreparedForecastPolicyBundle, SealedPreparedFutureLocator
from .prospective_lock import FrontierFrozenRoot
from .prospective_measure import FrontierMeasuredProspective, measure_prospective
from .prospective_plan import ReactorFiniteControlFrontierProspectivePlan
from .records import FrontierPreparation
from .selection import FrontierUseRequest


@dataclass(frozen=True, slots=True)
class FrontierSealedRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-sealed-root'
    root: str
    native: ObjectIdentity
    plan: ObjectIdentity
    source_receipt: ObjectIdentity
    measured: FrontierMeasuredProspective
    bundle_archives: tuple[tuple[FrontierUseRequest, CanonicalRecordArchive], ...]

    def bundle_for(self, request: FrontierUseRequest) -> SealedPreparedForecastPolicyBundle:
        archive = next(a for r, a in self.bundle_archives if r == request)
        if archive.subject.object_schema != SealedPreparedForecastPolicyBundle.SCHEMA:
            raise ValueError("sealed owner archive changed its schema")
        bundle = decode_canonical_bytes(
            archive.unpack(),
            SealedPreparedForecastPolicyBundle,
            maximum_bytes=archive.decoded_bytes,
        )
        if (
            archive.subject != ObjectIdentity.from_record(bundle.bundle_id, bundle)
            or bundle.design.root_id != self.root
            or bundle.design.policy_id != f"el.{request.request_id}"
        ):
            raise ValueError("sealed owner archive substituted its exact root/request")
        return bundle

    @property
    def record_id(self) -> str:
        return f"{self.root}.frontier-sealed-root"


def seal_native_root(
    *,
    native: FrontierNativeRoot,
    frozen: FrontierFrozenRoot,
    plan: ReactorFiniteControlFrontierProspectivePlan,
    preparation: FrontierPreparation,
    receipt: CanonicalTaskReceipt,
    artifact: ArtifactIdentity,
    store: DurablePreparedExecutionEventStore,
    occurred_at_utc: str,
) -> FrontierSealedRoot:
    if (
        receipt.task_id != f"frontier.action.{native.root}"
        or artifact.sha256 != native.fingerprint()
        or artifact.payload_schema != native.SCHEMA
        or not any(
            o.logical_artifact_id == artifact.artifact_id and o.content_sha256 == artifact.sha256
            for o in receipt.output_logical_artifacts
        )
        or frozen.plan != ObjectIdentity.from_record(plan.record_id, plan)
    ):
        raise ValueError("Controller use seal lacks its exact issued native receipt/plan")
    uses = tuple(frozen.uses)
    measured = measure_prospective(preparation, frozen, native, uses)
    receipt_id = ObjectIdentity.from_record(receipt.receipt_id, receipt)
    bundles = []
    for use, measurement in zip(uses, measured.uses, strict=True):
        if use.lock is None:
            continue
        assert use.projection is not None and use.compiled is not None
        evaluation = use.lock.evaluation_for(use.compiled)
        audit = (
            next(
                o
                for o in measured.chart.observations
                if o.coordinate == measurement.observation.coordinate
            )
            if measurement.observation is not None
            else None
        )
        locators = []
        for slot in evaluation.futures:
            if slot.root_id != native.root:
                continue
            row = (
                measurement.observation if slot.role is PreparedFutureRole.COMMITTED_TASK else audit
            )
            complete = row is not None and row.evaluable
            if slot.role is PreparedFutureRole.COMMITTED_TASK:
                complete &= measurement.owner_delivered
            else:
                complete &= row is not None and row.delivery_valid
            for view in evaluation.numerical_view_ids:
                disposition = (
                    PreparedFutureDisposition.COMPLETED
                    if complete
                    else PreparedFutureDisposition.INVALID_OBSERVATION
                )
                locators.append(
                    SealedPreparedFutureLocator(
                        f"{slot.slot_id}.{view}.locator",
                        slot.slot_id,
                        view,
                        disposition,
                        receipt_id,
                        artifact,
                        () if complete else ("MISSING_OR_INVALID_NATIVE_DELIVERY",),
                        OutcomeAccess.EVALUATION_SEALED,
                    )
                )
        bundle = seal_forecast_request(
            lock=use.lock,
            locators=tuple(sorted(locators, key=lambda loc: loc.locator_id)),
            store=store,
            occurred_at_utc=occurred_at_utc,
        )
        bundles.append((use.request, CanonicalRecordArchive.pack(bundle.bundle_id, bundle)))
    return FrontierSealedRoot(
        native.root,
        ObjectIdentity.from_record(native.record_id, native),
        ObjectIdentity.from_record(plan.record_id, plan),
        receipt_id,
        measured,
        tuple(bundles),
    )
