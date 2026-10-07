"Seal the complete finite response-law evaluation native census before recovery and separate reveal."

from typing import cast

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import TaskContext, TaskRunner, RunnerResult
from empirical_lawhood.runtime.providers import ExternalInputPayload
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import CapabilityOutputSemanticContract
from empirical_lawhood.adapters.methods.prepared_response.qualification_provider import _group
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import decode_port, output_result, semantic_contracts, config_payloads
from .control_ports import FiniteResponseLawControlRuntimePort
from .control_records import FiniteResponseLawRootParentJoin
from .control_closeout import FiniteResponseLawRootSealedControl, seal_root
from .control_provider import input_artifact
from .assigned_native_records import FiniteResponseLawAssignedEvaluationCompletionConfig, FiniteResponseLawAssignedEvaluationNativeCompletion, FiniteResponseLawAssignedEvaluationViewObservation
from .native_provider import EVALUATOR_TASK_ID, FiniteResponseLawNativeMethodProvider, _read_config, projection_stage


class FiniteResponseLawEvaluationCompletionTask:
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: FiniteResponseLawAssignedEvaluationCompletionConfig,
        runtime: FiniteResponseLawControlRuntimePort,
    ):
        self.manifest, self.config = manifest, config
        self.runtime = runtime

    def _seal(self, context: TaskContext) -> RunnerResult:
        roots = {
            f"{r.stage_unit}.seal-control": r for r in self.config.projection.native_spec.roots
        }
        root = roots.get(context.task_id)
        if root is None or {r.task_id for r in context.dependency_receipts} != {
            f"{root.stage_unit}.join-parent",
            *(f"{root.root_id}.flh-project.r{v}" for v in (1, 2)),
        }:
            raise ValueError("Finite response-law evaluation sealed root changes its exact parent/view dependency census")
        views, artifacts, receipts = [], [], []
        join = None
        for binding in context.dependency_receipts:
            if binding.task_id.endswith(".join-parent"):
                group = _group(
                    context,
                    binding.task_id,
                    binding.output_materialization_ids,
                    (FiniteResponseLawRootParentJoin.SCHEMA,),
                )
                join = decode_port(
                    group[FiniteResponseLawRootParentJoin.SCHEMA], FiniteResponseLawRootParentJoin, maximum=16 * 1024**2
                )
                continue
            group = _group(
                context,
                binding.task_id,
                binding.output_materialization_ids,
                (FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
            )
            port = group[FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA]
            view = decode_port(port, FiniteResponseLawAssignedEvaluationViewObservation)
            if decode_port(
                group[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
            ) != projection_stage(view):
                raise ValueError("Finite response-law evaluation sealed view changes its projection stage")
            receipt = self.runtime.dependency_receipt(context.run_id, binding)
            views.append(view)
            artifacts.append(input_artifact(port, view.canonical_bytes()))
            receipts.append(ObjectIdentity.from_record(receipt.receipt_id, receipt))
        assert join is not None
        ordered = sorted(
            zip(views, artifacts, receipts, strict=True), key=lambda row: row[0].refinement
        )
        result = seal_root(
            join=join,
            views=tuple(r[0] for r in ordered),
            artifacts=tuple(r[1] for r in ordered),
            receipts=tuple(r[2] for r in ordered),
            store=self.runtime.prepared_store,
            now=self.runtime.now,
        )
        return output_result(
            context, {result.SCHEMA: result.canonical_bytes()}, ("complete-sealed-consumer-census",)
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            _read_config(context, self.config, self.manifest, None)
            if context.task_id != EVALUATOR_TASK_ID:
                return self._seal(context)
            reports, sealed_indices = [], []
            for receipt in context.dependency_receipts:
                if receipt.task_id.endswith(".seal-control"):
                    group = _group(
                        context,
                        receipt.task_id,
                        receipt.output_materialization_ids,
                        (FiniteResponseLawRootSealedControl.SCHEMA,),
                    )
                    sealed = decode_port(
                        group[FiniteResponseLawRootSealedControl.SCHEMA],
                        FiniteResponseLawRootSealedControl,
                        maximum=16 * 1024**2,
                    )
                    sealed_indices.append(sealed.join.lock.forecast.root.index)
                    continue
                group = _group(
                    context,
                    receipt.task_id,
                    receipt.output_materialization_ids,
                    (FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
                )
                report = decode_port(
                    group[FiniteResponseLawAssignedEvaluationViewObservation.SCHEMA], FiniteResponseLawAssignedEvaluationViewObservation
                )
                if report.report_id != receipt.task_id or decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
                ) != projection_stage(report):
                    raise ValueError("Finite response-law evaluation sealed completion substitutes an actual view receipt")
                reports.append(report)
            if tuple(sorted(sealed_indices)) != tuple(range(64)):
                raise ValueError("Finite response-law evaluation terminal lacks the complete persisted consumer census")
            result = FiniteResponseLawAssignedEvaluationNativeCompletion(
                self.config, tuple(sorted(reports, key=lambda r: r.report_id))
            )
            return output_result(
                context,
                {result.SCHEMA: result.canonical_bytes()},
                ("complete-64-root-sealed-census", "no-scientific-adjudication-or-reveal"),
            )
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawEvaluationCompletionProvider(FiniteResponseLawNativeMethodProvider):
    def bind_control_runtime(self, runtime: FiniteResponseLawControlRuntimePort) -> None:
        if hasattr(self, "runtime"):
            raise ValueError("Finite response-law evaluation sealed provider runtime is already bound")
        self.runtime = runtime

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if (
            plan.registry_sha256 != self.registry_sha256
            or source_records
            or type(self.config) is not FiniteResponseLawAssignedEvaluationCompletionConfig
        ):
            raise ValueError("Finite response-law evaluation sealed provider changes its exact registry/configuration")
        expected = {
            EVALUATOR_TASK_ID,
            *(f"{r.stage_unit}.seal-control" for r in self.config.projection.native_spec.roots),
        }
        return config_payloads(plan, self.manifest, self.config, self.config.config_id, expected)

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if (
            registry != self.registry
            or source_records
            or type(self.config) is not FiniteResponseLawAssignedEvaluationCompletionConfig
            or not hasattr(self, "runtime")
        ):
            raise ValueError("Finite response-law evaluation sealed completion changes its installed configuration")
        return (
            cast(
                TaskRunner, FiniteResponseLawEvaluationCompletionTask(self.manifest, self.config, self.runtime)
            ),
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("Finite response-law evaluation sealed completion changes its registry")
        return semantic_contracts(
            self.manifest, (FiniteResponseLawAssignedEvaluationNativeCompletion, FiniteResponseLawRootSealedControl), None
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("Finite response-law evaluation sealed completion changes its registry")
        return None
