"""Declared native preparation, finite assays and actual prospective delivery."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import Any
import platform
import numpy as np

from empirical_lawhood.adapters.composition.phase_inputs import config_input, dependency, result, upstream
from empirical_lawhood.adapters.composition.record_provider import RecordCampaignProvider
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import ClassicalStage, retained_key
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.discovery import ClassicalNomination
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.law_terminal import ClassicalLaws
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_lock import ClassicalFrozenRoot
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_plan import ReactorStagedPulseResponseProspectivePlan
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.phase_records import ClassicalFirstScreen, ClassicalInducedBundle
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prediction import ClassicalPredictionSeal, ClassicalInducedPrediction, induced_prediction
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.records import ClassicalPreparation, ClassicalPrivate, ClassicalContext
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierAssay, FrontierPreparation, FrontierPrivate
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.provider import source_input
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskProgressEmitter
from empirical_lawhood.runtime.providers import ExternalInputPayload
from .acquisition import acquire_preparation, reuse_preparation, acquire_assay
from .config import ClassicalNativeConfig
from .prospective import acquire_prospective
from .extension_bundle import CAPABILITY, OUTPUT_RECORDS


@dataclass(frozen=True)
class ClassicalNativeRunner:
    config: ClassicalNativeConfig
    stage: ClassicalStage
    custody: DependencyCustodyReader
    limits: EmpiricalResourceGuard
    control: Any
    reader: CandidatePayloadReader
    manifest = CAPABILITY

    def execute(self, context: TaskContext) -> RunnerResult:
        return self.execute_with_progress(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter | None
    ) -> RunnerResult:
        with self.limits.task(context.task_id):
            config_input(context, self.config)
            source = source_input(context).batch
            if (
                source.fingerprint() != self.config.source_sha256
                or platform.python_version() != "3.11.14"
                or np.__version__ != "2.4.6"
            ):
                raise ValueError(
                    "native source or numerical environment differs from its denominator"
                )
            kind, root = context.task_id.split(".", 2)[1:]
            if root not in self.stage.root_ids:
                raise ValueError("native task is outside its separate stage issue")
            progress = None if emitter is None else lambda n: emitter.advance(D(n))
            nomination = (
                None
                if self.stage.role == "NOMINATION"
                else upstream(context, self.stage, "nomination", ClassicalNomination)
            )
            if kind == "prepare":
                if self.stage.role == "NOMINATION":
                    old = upstream(
                        context,
                        self.stage,
                        retained_key(root, "preparation"),
                        FrontierPreparation,
                    )
                    private = upstream(
                        context, self.stage, retained_key(root, "private"), FrontierPrivate
                    )
                    identity = next(
                        u.artifact
                        for u in self.stage.upstream
                        if u.key == retained_key(root, "private")
                    )
                    records = reuse_preparation(
                        self.config,
                        self.stage.block,
                        old,
                        private,
                        private_identity=ObjectIdentity(
                            identity.artifact_id,
                            identity.payload_schema,
                            private.VERSION,
                            identity.sha256,
                        ),
                    )
                else:
                    assert nomination is not None
                    entered = nomination.entered
                    if self.stage.role == "PROSPECTIVE":
                        entered &= bool(
                            upstream(context, self.stage, "laws", ClassicalLaws).qualified
                        )
                    records = acquire_preparation(
                        self.config,
                        source,
                        self.stage.block,
                        root,
                        released=entered,
                        progress=progress,
                    )
                return result(context, records)
            prep = dependency(
                context, self.custody, ClassicalPreparation, f"classical.prepare.{root}"
            )[0]
            private_new = dependency(
                context, self.custody, ClassicalPrivate, f"classical.prepare.{root}"
            )[0]
            publisher = self.control.open_control_store(root)
            if kind == "assay":
                seal = dependency(
                    context,
                    self.custody,
                    ClassicalPredictionSeal,
                    f"classical.predictions.{root}",
                )[0]
                induced: list[ClassicalInducedPrediction] = []

                def publish(causal: ClassicalContext) -> ObjectIdentity:
                    saved = induced_prediction(
                        source, causal, self.config.prepared_domain, seal, nomination
                    )
                    publisher.publish_record(saved.record_id, saved)
                    induced.append(saved)
                    return ObjectIdentity.from_record(saved.record_id, saved)

                if (
                    self.stage.stage == "staged-sequence-comparison-NOMINATION"
                    and not dependency(
                        context, self.custody, ClassicalFirstScreen, "classical.first-screen"
                    )[0].entered
                ):
                    from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.records import ClassicalAssay

                    assay = ClassicalAssay(
                        root,
                        ObjectIdentity.from_record(prep.record_id, prep),
                        ObjectIdentity.from_record(seal.record_id, seal),
                        (),
                        (),
                        0,
                        ("FIRST_RELATION_PREREQUISITE_NONENTRY",),
                    )
                else:
                    assay = acquire_assay(
                        source,
                        self.stage,
                        prep,
                        private_new,
                        ObjectIdentity.from_record(seal.record_id, seal),
                        old_assay=upstream(
                            context, self.stage, retained_key(root, "assay"), FrontierAssay
                        )
                        if self.stage.stage == "expanded-menu-comparison-NOMINATION"
                        else None,
                        on_induced=publish if self.stage.block == "staged-sequence-comparison" else None,
                        progress=progress,
                    )
                return result(
                    context,
                    (assay, ClassicalInducedBundle(root, tuple(induced)))
                    if self.stage.block == "staged-sequence-comparison"
                    else (assay,),
                )
            if kind != "action" or self.stage.role != "PROSPECTIVE":
                raise ValueError("unregistered classical staged-pulse native operation")
            acquired = acquire_prospective(
                source=source,
                preparation=prep,
                private=private_new,
                frozen=dependency(
                    context, self.custody, ClassicalFrozenRoot, f"classical.freeze.{root}"
                )[0],
                plan=dependency(context, self.custody, ReactorStagedPulseResponseProspectivePlan, "classical.prospective-evaluation-plan")[0],
                laws=upstream(context, self.stage, "laws", ClassicalLaws),
                publisher=publisher,
                reader=self.reader,
                control=self.control,
                progress=progress,
            )
            return result(context, (acquired,))


class ClassicalNativeProvider(RecordCampaignProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        runner: ClassicalNativeRunner,
        inputs: tuple[ExternalInputPayload, ...],
    ) -> None:
        super().__init__(
            registry,
            CAPABILITY,
            runner,
            runner.config,
            runner.config.config_id,
            inputs,
            OUTPUT_RECORDS,
            adjudication=False,
        )
