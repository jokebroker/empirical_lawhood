"""Receipt-authenticated native measurement and prepared-future sealing."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.control.prepared_forecast import seal_forecast_request
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.prospective import ClassicalNativeRoot
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureRole
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore, PreparedFutureDisposition, SealedPreparedForecastPolicyBundle, SealedPreparedFutureLocator
from .prospective_lock import ClassicalFrozenRoot
from .prospective_measure import ClassicalMeasuredProspective, measure_prospective
from .prospective_plan import ReactorStagedPulseResponseProspectivePlan
from .records import ClassicalPreparation


@dataclass(frozen=True, slots=True)
class ClassicalSealedRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-sealed-root'
    root: str
    native: ObjectIdentity
    native_artifact: ArtifactIdentity
    plan: ObjectIdentity
    source_receipt: ObjectIdentity
    measured: ClassicalMeasuredProspective
    bundles: tuple[tuple[ObjectIdentity, CanonicalRecordArchive], ...]

    @property
    def record_id(self) -> str:
        return f"{self.root}.sealed-outcomes"

    def bundle_for(self, frozen: ObjectIdentity) -> SealedPreparedForecastPolicyBundle:
        archive = next(a for identity, a in self.bundles if identity == frozen)
        value = decode_canonical_bytes(
            archive.unpack(),
            SealedPreparedForecastPolicyBundle,
            maximum_bytes=archive.decoded_bytes,
        )
        if (
            archive.subject != ObjectIdentity.from_record(value.bundle_id, value)
            or value.design.root_id != self.root
        ):
            raise ValueError("sealed archive substitutes its actual prepared future")
        return value


def seal_native_root(
    *,
    native: ClassicalNativeRoot,
    frozen: ClassicalFrozenRoot,
    plan: ReactorStagedPulseResponseProspectivePlan,
    preparation: ClassicalPreparation,
    receipt: CanonicalTaskReceipt,
    artifact: ArtifactIdentity,
    store: DurablePreparedExecutionEventStore,
    occurred_at_utc: str,
) -> ClassicalSealedRoot:
    if (
        receipt.task_id != f"classical.action.{native.root}"
        or artifact.sha256 != native.fingerprint()
        or artifact.payload_schema != native.SCHEMA
        or not any(
            o.logical_artifact_id == artifact.artifact_id
            and o.content_sha256 == artifact.sha256
            and o.payload_schema == artifact.payload_schema
            for o in receipt.output_logical_artifacts
        )
        or native.frozen != ObjectIdentity.from_record(frozen.record_id, frozen)
        or frozen.plan != ObjectIdentity.from_record(plan.record_id, plan)
        or tuple(u.stages[0].frozen for u in native.uses) != frozen.stages
    ):
        raise ValueError("Controller use sealing lacks its exact issued native receipt and frozen census")
    measured = measure_prospective(preparation, native)
    receipt_id = ObjectIdentity.from_record(receipt.receipt_id, receipt)
    bundles = []
    for use in native.uses:
        for stage in use.stages:
            selected = stage.unpack()
            if selected.lock is None:
                continue
            assert selected.compiled is not None
            evaluation = selected.lock.evaluation_for(selected.compiled)
            row = next(m for m in measured.stages if m.frozen == stage.frozen.subject)
            locators = []
            for slot in evaluation.futures:
                reference = slot.role is PreparedFutureRole.MATCHED_HOLD
                reserved = selected.admitted or reference
                complete = (
                    row.reference_known and row.reference_valid
                    if reference
                    else row.observation is not None
                    and row.observation.evaluable
                    and row.reference_known
                )
                disposition = (
                    PreparedFutureDisposition.NONATTEMPT
                    if not reserved
                    else PreparedFutureDisposition.COMPLETED
                    if complete
                    else PreparedFutureDisposition.INVALID_OBSERVATION
                )
                for view in evaluation.numerical_view_ids:
                    locators.append(
                        SealedPreparedFutureLocator(
                            f"{slot.slot_id}.{view}.locator",
                            slot.slot_id,
                            view,
                            disposition,
                            receipt_id if reserved else None,
                            artifact if reserved else None,
                            ()
                            if complete
                            else ("OWNED_NONATTEMPT",)
                            if not reserved
                            else ("MISSING_MANDATORY_NATIVE_MEASUREMENT",),
                            OutcomeAccess.EVALUATION_SEALED,
                        )
                    )
            bundle = seal_forecast_request(
                lock=selected.lock,
                locators=tuple(sorted(locators, key=lambda x: x.locator_id)),
                store=store,
                occurred_at_utc=occurred_at_utc,
            )
            bundles.append(
                (stage.frozen.subject, CanonicalRecordArchive.pack(bundle.bundle_id, bundle))
            )
    return ClassicalSealedRoot(
        native.root,
        ObjectIdentity.from_record(native.record_id, native),
        artifact,
        ObjectIdentity.from_record(plan.record_id, plan),
        receipt_id,
        measured,
        tuple(bundles),
    )
