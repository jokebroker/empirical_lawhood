"""Reactor binding to the shared prepared forecast commitment sequence."""

from empirical_lawhood.adapters.control.prepared_forecast import PreparedForecastLock, freeze_forecast_request
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.planning.nested_controller_evaluation import CoupledRealizationControllerEvaluationPlan
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore
from .control_owner import PreparedClassicalRequest
from .records import ClassicalCausal



def freeze_prepared_request(
    *,
    root: str,
    request_index: int,
    prepared: PreparedClassicalRequest,
    causal: ClassicalCausal,
    causal_receipt: CanonicalTaskReceipt,
    causal_artifact: ArtifactIdentity,
    issued_study: ObjectIdentity,
    evaluation: CoupledRealizationControllerEvaluationPlan,
    store: DurablePreparedExecutionEventStore,
    occurred_at_utc: str,
) -> PreparedForecastLock:
    if (
        request_index != 0
        or causal.root != root
        or causal_receipt.task_id != f"classical.prepare.{root}"
    ):
        raise ValueError("classical request changed its root, policy or preparation receipt")
    return freeze_forecast_request(
        prefix_id=f"{root}.request-0.prepared",
        policy_id="reactor-classical-request-0",
        root=root,
        compiled=prepared.compiled,
        commitment=prepared.commitment,
        task=prepared.task,
        audit_forecasts=prepared.audit_forecasts,
        checkpoint=ObjectIdentity.from_record(f"{root}.causal-preparation", causal),
        causal_receipt=causal_receipt,
        causal_artifact=causal_artifact,
        issued_study=issued_study,
        evaluation=evaluation,
        store=store,
        occurred_at_utc=occurred_at_utc,
    )
