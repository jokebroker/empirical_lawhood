"Finite response-law evaluation's causal method-task products and exact retained input bindings."

from dataclasses import dataclass
from decimal import Decimal as D
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationConfig, FiniteResponseLawAssignedEvaluationInvocation, FiniteResponseLawAssignedEvaluationRoot
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_contracts import FiniteResponseLawEvaluationConfig, FiniteResponseLawEvaluationInvocation, FiniteResponseLawEvaluationRoot
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult, FiniteResponseLawEvaluationTaskResult
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.models import ModelSetSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.controller_evaluation_nested import PreparedDesignBindingReceipt, PreparedForecastParentCommitment, PreparedInstanceBindingReceipt, PreparedParentReturn

from .consumer import FiniteResponseLawConsumerRequest, evaluation_requests
from .control_prediction import FiniteResponseLawControlPredictionTable
from .assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation, FiniteResponseLawAssignedEvaluationInterface
from .evaluation_native_records import FiniteResponseLawEvaluationInterface
from .law_binding import feature_quantities, parent_quantity
from .method_records import FiniteResponseLawAssignedQualificationReport, FiniteResponseLawQualificationReport
from .native_records import FiniteResponseLawCalibrationNativeEvaluation
from .science import PARENTS


@dataclass(frozen=True, slots=True)
class FiniteResponseLawControlConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-control-config'
    source: FiniteResponseLawEvaluationConfig | FiniteResponseLawAssignedEvaluationConfig
    qualification: ArtifactIdentity
    qualification_receipt: ArtifactIdentity
    expected_qualification_receipt: ObjectIdentity
    calibration_native: ArtifactIdentity
    calibration_native_receipt: ArtifactIdentity
    expected_calibration_native_receipt: ObjectIdentity
    primary_model_set: ModelSetSpec

    def __post_init__(self) -> None:
        assigned = type(self.source) is FiniteResponseLawAssignedEvaluationConfig
        if (
            type(self.source) not in (FiniteResponseLawEvaluationConfig, FiniteResponseLawAssignedEvaluationConfig)
            or self.primary_model_set.model_set_id != "finite-response-law.composed.prospective-control.models"
            or self.qualification.payload_schema
            != (FiniteResponseLawAssignedQualificationReport if assigned else FiniteResponseLawQualificationReport).SCHEMA
            or self.calibration_native.payload_schema
            != (FiniteResponseLawAssignedCalibrationNativeEvaluation if assigned else FiniteResponseLawCalibrationNativeEvaluation).SCHEMA
            or any(
                a.payload_schema != CanonicalTaskReceipt.SCHEMA
                for a in (self.qualification_receipt, self.calibration_native_receipt)
            )
            or any(
                identity.object_schema != CanonicalTaskReceipt.SCHEMA
                or identity.object_fingerprint != artifact.sha256
                for artifact, identity in (
                    (self.qualification_receipt, self.expected_qualification_receipt),
                    (self.calibration_native_receipt, self.expected_calibration_native_receipt),
                )
            )
            or len({a.artifact_id for a in self.inputs}) != 4
            or any(a.extensions or not 0 < a.size_bytes <= 8 * 1024**2 for a in self.inputs)
        ):
            raise ValueError("Finite response-law evaluation control changes its source or exact prior qualification custody")

    @property
    def config_id(self) -> str:
        if type(self.source) is FiniteResponseLawAssignedEvaluationConfig:
            tag = sha256(self.source.cohort_namespace.encode()).hexdigest()[:16]
            return f"finite-response-law.prospective-evaluation.assigned-{tag}.control-config"
        return "finite-response-law.prospective-evaluation.control-config"

    @property
    def inputs(self) -> tuple[ArtifactIdentity, ...]:
        return tuple(
            sorted(
                (
                    self.qualification,
                    self.qualification_receipt,
                    self.calibration_native,
                    self.calibration_native_receipt,
                ),
                key=lambda a: a.artifact_id,
            )
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawUnavailableControlBoundary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-unavailable-control-boundary'
    boundary: str
    qualification: LawQualificationResult

    def __post_init__(self) -> None:
        if (
            self.boundary not in ("cached", "direct")
            or self.qualification.scientific_status is ScientificStatus.SUPPORTED
        ):
            raise ValueError("Only an actual unavailable comparator can lack a finite response-law evaluation consumer")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRootForecast(CanonicalRecord):
    """Only prefix observations, frozen prior evidence and predictions may enter."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-root-forecast'
    root: FiniteResponseLawEvaluationRoot | FiniteResponseLawAssignedEvaluationRoot
    config: ObjectIdentity
    qualification: ObjectIdentity
    prefix: FiniteResponseLawEvaluationTaskResult | FiniteResponseLawAssignedEvaluationTaskResult
    primary_interface: FiniteResponseLawEvaluationInterface | FiniteResponseLawAssignedEvaluationInterface | None
    tables: tuple[FiniteResponseLawControlPredictionTable, ...]
    unavailable: tuple[FiniteResponseLawUnavailableControlBoundary, ...]

    def __post_init__(self) -> None:
        assigned = type(self.root) is FiniteResponseLawAssignedEvaluationRoot
        if (
            self.config.object_schema != FiniteResponseLawControlConfig.SCHEMA
            or self.qualification.object_schema
            != (FiniteResponseLawAssignedQualificationReport if assigned else FiniteResponseLawQualificationReport).SCHEMA
            or type(self.prefix)
            is not (FiniteResponseLawAssignedEvaluationTaskResult if assigned else FiniteResponseLawEvaluationTaskResult)
            or self.primary_interface is not None
            and type(self.primary_interface)
            is not (FiniteResponseLawAssignedEvaluationInterface if assigned else FiniteResponseLawEvaluationInterface)
            or self.prefix.invocation.root != self.root
            or self.prefix.invocation.phase != "prefix"
            or self.primary_interface is not None
            and (
                self.primary_interface.cutoff_tick != 4096
                or self.primary_interface.checkpoint.object_id
                != f"{self.prefix.invocation.task_id}.r1.checkpoint"
            )
            or tuple(t.boundary for t in self.tables)
            != tuple(sorted(t.boundary for t in self.tables))
            or tuple(u.boundary for u in self.unavailable)
            != tuple(sorted(u.boundary for u in self.unavailable))
            or sorted([t.boundary for t in self.tables] + [u.boundary for u in self.unavailable])
            != ["cached", "composed", "direct"]
            or any(
                t.root_id != self.root.stage_unit
                or t.qualified_report != self.qualification
                or t.prefix_source != ObjectIdentity.from_record(self.prefix.result_id, self.prefix)
                for t in self.tables
            )
        ):
            raise ValueError("Finite response-law evaluation forecast loses its complete boundary census or causal prefix")
        for table in self.tables:
            expected = {parent_quantity().quantity_id: D(PARENTS.index(self.root.assigned_parent))}
            if table.boundary != "cached" and self.primary_interface is not None:
                expected.update(
                    {
                        q.quantity_id: value
                        for q, value in zip(
                            feature_quantities(table.boundary),
                            self.primary_interface.values,
                            strict=True,
                        )
                    }
                )
            if any(
                len(r.input_values) != len(expected)
                or {v.quantity_id: v.value for v in r.input_values} != expected
                for r in table.requests
            ):
                raise ValueError(
                    "Finite response-law evaluation forecast uses inputs other than this primary causal prefix and parent"
                )

    @property
    def forecast_id(self) -> str:
        return f"{self.root.stage_unit}.full-forecast"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRootRequestReveal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-root-request-reveal'
    root: FiniteResponseLawEvaluationRoot | FiniteResponseLawAssignedEvaluationRoot
    forecast: ObjectIdentity
    forecast_artifact: ArtifactIdentity
    forecast_receipt_id: str
    requests: tuple[FiniteResponseLawConsumerRequest, FiniteResponseLawConsumerRequest]

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_stable_id

        validate_stable_id(self.forecast_receipt_id, field_name="forecast_receipt_id")
        if (
            self.forecast.object_schema != FiniteResponseLawRootForecast.SCHEMA
            or self.forecast.object_id != f"{self.root.stage_unit}.full-forecast"
            or self.forecast_artifact.payload_schema != FiniteResponseLawRootForecast.SCHEMA
            or self.forecast_artifact.sha256 != self.forecast.object_fingerprint
            or self.requests != evaluation_requests(self.root)
        ):
            raise ValueError("Finite response-law evaluation requests must follow the full forecast and exact independent RNG")

    @property
    def reveal_id(self) -> str:
        return f"{self.root.stage_unit}.requests"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRootControlLock(CanonicalRecord):
    """References to durable generic locks; not a replacement commitment owner."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-root-control-lock'
    forecast: FiniteResponseLawRootForecast
    requests: FiniteResponseLawRootRequestReveal
    designs: tuple[PreparedDesignBindingReceipt, ...]
    parents: tuple[PreparedForecastParentCommitment, ...]
    compiled_artifacts: tuple[ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            sorted(f"{t.boundary}.consumer-{c}" for t in self.forecast.tables for c in (0, 1))
        )
        if (
            self.requests.root != self.forecast.root
            or self.requests.forecast
            != ObjectIdentity.from_record(self.forecast.forecast_id, self.forecast)
            or tuple(d.policy_id for d in self.designs) != expected
            or tuple(p.policy_id for p in self.parents) != expected
            or len(self.compiled_artifacts) != len(expected)
        ):
            raise ValueError(
                "Finite response-law evaluation locks lose a qualified method/consumer or replace the shared request"
            )
        for design, parent, compiled in zip(
            self.designs, self.parents, self.compiled_artifacts, strict=True
        ):
            request = self.requests.requests[int(design.policy_id[-1])]
            invocation_type = (
                FiniteResponseLawAssignedEvaluationInvocation
                if type(self.forecast.root) is FiniteResponseLawAssignedEvaluationRoot
                else FiniteResponseLawEvaluationInvocation
            )
            native_parent = invocation_type(
                self.forecast.prefix.invocation.source,
                self.forecast.root,
                "parent",
                self.forecast.root.assigned_parent,
                None,
                "parent",
            )
            if (
                design.root_id != self.forecast.root.stage_unit
                or parent.root_id != design.root_id
                or design.parent_commitment
                != ObjectIdentity.from_record(parent.commitment_id, parent)
                or parent.task.target_reveal
                != ObjectIdentity.from_record(request.request_id, request)
                or parent.parent_action
                != ObjectIdentity.from_record(native_parent.task_id, native_parent)
                or parent.evaluation_plan != design.evaluation_plan.identity
                or design.common_checkpoint
                != ObjectIdentity.from_record(self.forecast.prefix.result_id, self.forecast.prefix)
                or parent.common_checkpoint != design.common_checkpoint
                or compiled.payload_schema != parent.decision.compiled_study.object_schema
                or compiled.sha256 != parent.decision.compiled_study.object_fingerprint
            ):
                raise ValueError("Finite response-law evaluation lock substitutes its root, parent, request or common prefix")

    @property
    def lock_id(self) -> str:
        return f"{self.forecast.root.stage_unit}.all-consumer-locks"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRootParentJoin(CanonicalRecord):
    """Actual parent joined to early decisions; no revised prediction/selection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-root-parent-join'
    lock: FiniteResponseLawRootControlLock
    parent: FiniteResponseLawEvaluationTaskResult | FiniteResponseLawAssignedEvaluationTaskResult
    parent_receipt: ObjectIdentity
    parent_artifact: ArtifactIdentity
    handoff: FiniteResponseLawEvaluationInterface | FiniteResponseLawAssignedEvaluationInterface | None
    parent_work: tuple[D | None, D | None]
    returns: tuple[PreparedParentReturn, ...]
    instances: tuple[PreparedInstanceBindingReceipt | None, ...]

    def __post_init__(self) -> None:
        assigned = type(self.lock.forecast.root) is FiniteResponseLawAssignedEvaluationRoot
        if (
            type(self.parent)
            is not (FiniteResponseLawAssignedEvaluationTaskResult if assigned else FiniteResponseLawEvaluationTaskResult)
            or self.handoff is not None
            and type(self.handoff)
            is not (FiniteResponseLawAssignedEvaluationInterface if assigned else FiniteResponseLawEvaluationInterface)
            or self.parent.invocation.root != self.lock.forecast.root
            or self.parent.invocation.source != self.lock.forecast.prefix.invocation.source
            or self.parent.invocation.phase != "parent"
            or self.parent.predecessors
            != (
                ObjectIdentity.from_record(
                    self.lock.forecast.prefix.result_id, self.lock.forecast.prefix
                ),
            )
            or self.parent_receipt.object_schema != CanonicalTaskReceipt.SCHEMA
            or self.parent_artifact.payload_schema != self.parent.SCHEMA
            or self.parent_artifact.sha256 != self.parent.fingerprint()
            or len(self.returns) != len(self.lock.designs)
            or len(self.instances) != len(self.returns)
            or any(w is not None and (not w.is_finite() or w < 0) for w in self.parent_work)
            or self.handoff is not None
            and (
                self.handoff.cutoff_tick != 4368
                or self.handoff.checkpoint.object_id
                != f"{self.parent.invocation.task_id}.r1.checkpoint"
            )
        ):
            raise ValueError("Finite response-law evaluation parent join substitutes actual parent, handoff or census")
        for design, returned, instance in zip(
            self.lock.designs, self.returns, self.instances, strict=True
        ):
            if (
                returned.design_binding != ObjectIdentity.from_record(design.binding_id, design)
                or returned.common_checkpoint != design.common_checkpoint
                or returned.parent_commitment != design.parent_commitment
                or returned.source_receipt != self.parent_receipt
                or returned.source_artifact != self.parent_artifact
                or returned.public_handoff
                != (
                    None
                    if self.handoff is None
                    else ObjectIdentity.from_record(
                        f"{self.lock.forecast.root.stage_unit}.actual-handoff", self.handoff
                    )
                )
                or (instance is None) != (self.handoff is None)
                or instance is not None
                and (
                    instance.design_binding != returned.design_binding
                    or instance.parent_return
                    != ObjectIdentity.from_record(returned.return_id, returned)
                    or instance.public_handoff != returned.public_handoff
                )
            ):
                raise ValueError("Finite response-law evaluation parent join changes the committed consumer or actual return")

    @property
    def join_id(self) -> str:
        return f"{self.lock.forecast.root.stage_unit}.parent-join"

    @property
    def use_allowed(self) -> bool:
        return self.handoff is not None and all(w is not None and w <= 32 for w in self.parent_work)
