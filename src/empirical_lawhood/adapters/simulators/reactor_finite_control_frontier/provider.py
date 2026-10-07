"""Native tasks on the installed acquisition, resource and receipt seams."""

from empirical_lawhood.adapters.composition.record_provider import RecordCampaignProvider
from dataclasses import dataclass
from decimal import Decimal as D
from typing import Any
import platform
import numpy as np
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.phase import FrontierPhase
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.provider_common import config_input, dependency, result, upstream
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierPreparation, FrontierPrivate, FrontierRetainedRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.selection import FrontierPredictionSeal
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_plan import ReactorFiniteControlFrontierProspectivePlan
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_lock import FrontierFrozenRoot
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import TaskContext, RunnerResult, TaskProgressEmitter
from empirical_lawhood.runtime.providers import (
    ExternalInputPayload,
)
from .acquisition import acquire_preparation, acquire_assay
from .prospective import acquire_prospective
from .config import FrontierNativeConfig
from .extension_bundle import CAPABILITY, OUTPUT_RECORDS


def source_input(context: TaskContext) -> EmpiricalStudySource:
    ports = tuple(
        p for p in context.input_ports if p.payload_schema == EmpiricalStudySource.SCHEMA
    )
    if len(ports) != 1:
        raise ValueError("frontier task requires its exact pinned source configuration")
    return decode_canonical_bytes(
        ports[0].read(), EmpiricalStudySource, maximum_bytes=4 * 1024**2
    )


@dataclass(frozen=True)
class FrontierNativeRunner:
    config: FrontierNativeConfig
    phase: FrontierPhase
    custody: DependencyCustodyReader
    limits: EmpiricalResourceGuard
    control: Any
    manifest = CAPABILITY

    def execute(self, context: TaskContext) -> RunnerResult:
        return self.execute_with_progress(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter | None
    ) -> RunnerResult:
        with self.limits.task(context.task_id):
            config_input(context, self.config)
            source = source_input(context)
            if (
                source.batch.fingerprint() != self.config.source_sha256
                or platform.python_version() != "3.11.14"
                or np.__version__ != "2.4.6"
            ):
                raise ValueError("native source or numerical environment differs")
            kind, root = context.task_id.split(".", 2)[1:]
            if root not in self.phase.roots:
                raise ValueError("source task is outside this separate phase issue")
            progress = None if emitter is None else lambda value: emitter.advance(D(value))
            if kind == "prepare":
                if self.phase.phase == "B":
                    retained = upstream(context, self.phase, root, FrontierRetainedRoot)
                    if (
                        retained.preparation.root != root
                        or retained.preparation.source_sha256 != source.batch.fingerprint()
                    ):
                        raise ValueError("retained preparation changed physical root/source")
                    records = (retained.preparation, retained.private)
                else:
                    records = acquire_preparation(
                        self.config, source.batch, root, released=True, progress=progress
                    )
                return result(context, records)
            prep = dependency(
                context, self.custody, FrontierPreparation, f"frontier.prepare.{root}"
            )[0]
            private = dependency(
                context, self.custody, FrontierPrivate, f"frontier.prepare.{root}"
            )[0]
            if kind == "assay" and self.phase.phase in ("B", "C"):
                seal = dependency(
                    context, self.custody, FrontierPredictionSeal, f"frontier.predictions.{root}"
                )[0]
                from empirical_lawhood.kernel.provenance import ObjectIdentity

                return result(
                    context,
                    (
                        acquire_assay(
                            source.batch,
                            prep,
                            private,
                            ObjectIdentity.from_record(seal.record_id, seal),
                            progress=progress,
                        ),
                    ),
                )
            if kind != "action" or self.phase.phase != "D":
                raise ValueError("unregistered frontier native operation")
            frozen = dependency(
                context, self.custody, FrontierFrozenRoot, f"frontier.freeze.{root}"
            )[0]
            plan = dependency(context, self.custody, ReactorFiniteControlFrontierProspectivePlan, "frontier.prospective-evaluation-plan")[0]
            acquired = acquire_prospective(
                source=source.batch,
                preparation=prep,
                private=private,
                frozen=frozen,
                plan=plan,
                publisher=self.control.open_control_store(root),
                progress=progress,
            )
            return result(context, (acquired,))


class FrontierNativeProvider(RecordCampaignProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        runner: FrontierNativeRunner,
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
