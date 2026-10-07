"""Registered source tasks with separately receipted causal prerequisites."""

from typing import Any
from collections.abc import Callable


from dataclasses import dataclass
from decimal import Decimal
from empirical_lawhood.adapters.composition.preparation_applicability.inputs import PreparationApplicabilityRecordProvider
from empirical_lawhood.adapters.composition.phase_inputs import config_input, dependency
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .records import PreparationApplicabilityNativePhase
from empirical_lawhood.runtime.task_records import canonical_task_result
from empirical_lawhood.runtime.execution import TaskContext, TaskProgressEmitter
from empirical_lawhood.adapters.methods.preparation_applicability.config import (
    PreparationApplicabilityNativeConfig,
    PreparationApplicabilitySource,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.runtime.execution import WorkerInputKind
from empirical_lawhood.adapters.methods.preparation_applicability.records import PreparationApplicabilitySelection
from empirical_lawhood.adapters.methods.preparation_applicability.seals import (
    PreparationApplicabilityParents,
    PreparationApplicabilityLowerSeal,
)
from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from .records import PreparationApplicabilityPrefix, PreparationApplicabilityPanel
from .native import acquire_phase, acquire_prefix


def native_progress_sink(
    emitter: TaskProgressEmitter | None,
) -> Callable[[int], None] | None:
    """Coalesce repeated observations without changing realized work counts."""
    if emitter is None:
        return None
    last = 0

    def advance(count: int) -> None:
        nonlocal last
        if type(count) is not int or count < last:
            raise ValueError("native work counter regressed or changed type")
        if count > last:
            emitter.advance(Decimal(count))
            last = count

    return advance


@dataclass(frozen=True)
class PreparationApplicabilityNativeRunner:
    config: PreparationApplicabilityNativeConfig
    custody: Any
    limits: Any

    @property
    def manifest(self) -> Any:
        from .extension_bundle import CAPABILITY

        return CAPABILITY

    def execute(self, context: TaskContext) -> Any:
        return self.execute_with_progress(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter | None
    ) -> Any:
        with self.limits.task(context.task_id):
            config_input(context, self.config)
            import platform
            import numpy as np

            ports = tuple(
                p
                for p in context.input_ports
                if p.kind is WorkerInputKind.EXTERNAL
                and p.payload_schema == PreparationApplicabilitySource.SCHEMA
            )
            if len(ports) != 1:
                raise ValueError("native task lacks its exact source denominator")
            source = decode_canonical_bytes(
                ports[0].read(), PreparationApplicabilitySource, maximum_bytes=65536
            )
            if (
                source.fingerprint() != self.config.stage.source.object_fingerprint
                or source.design != self.config.stage.design
                or platform.python_version() != source.python_version
                or np.__version__ != source.numpy_version
            ):
                raise ValueError("native numerical environment differs from frozen source")
            kind, root = context.task_id.split(".", 2)[1:]
            stage = self.config.stage
            index = stage.root_ids.index(root)
            progress = native_progress_sink(emitter)

            def dep(record: Any, producer: Any) -> Any:
                return dependency(context, self.custody, record, producer)[0]

            value: CanonicalRecord
            if kind == "prefix":
                value = acquire_prefix(stage.allocation.roots[index], source.fingerprint(), progress)
            else:
                prefix = dep(PreparationApplicabilityPrefix, f"ap.prefix.{root}")
                selection = dep(PreparationApplicabilitySelection, f"ap.select.{root}")
                if selection.root_id != root or selection.prefix_sha256 != prefix.fingerprint():
                    raise ValueError("native intervention substitutes pre-parent seal")
                frame = (
                    None
                    if prefix.frame_base64 is None
                    else PreparedPortFrame(4096, _decode(prefix.frame_base64, (2, 3, 4, 4)))
                )
                if kind == "parents":
                    phases: list[PreparationApplicabilityNativePhase] = []
                    if frame is not None and selection.selected_index is not None:
                        for s in range(3):
                            for v in (1, 2):
                                offset = sum(p.completed_intervals for p in phases)
                                phases.append(
                                    acquire_phase(
                                        phase_name="parent",
                                        allocation=stage.allocation.roots[index],
                                        refinement=v,
                                        source_sha256=source.fingerprint(),
                                        incoming=prefix.phases[v - 1],
                                        schedule_index=s,
                                        frame=frame,
                                        progress=None
                                        if progress is None
                                        else lambda n: progress(offset + n),
                                    )
                                )
                    value = PreparationApplicabilityParents(
                        root, prefix.fingerprint(), selection.fingerprint(), tuple(phases)
                    )
                elif kind == "assay":
                    parents = dep(PreparationApplicabilityParents, f"ap.parents.{root}")
                    sealed = dep(PreparationApplicabilityLowerSeal, f"ap.lower.{root}")
                    if (
                        sealed.parents_sha256 != parents.fingerprint()
                        or parents.selection_sha256 != selection.fingerprint()
                    ):
                        raise ValueError("future assay bypasses post-handoff lower-use seal")
                    phases = list(parents.phases)
                    offset = 0
                    if sealed.complete:
                        for parent in parents.phases:
                            if parent.disposition != "COMPLETE":
                                continue
                            for f in (0, 1):
                                for w in range(9):
                                    p = acquire_phase(
                                        phase_name="future",
                                        allocation=stage.allocation.roots[index],
                                        refinement=parent.refinement,
                                        source_sha256=source.fingerprint(),
                                        incoming=parent,
                                        schedule_index=parent.schedule_index,
                                        future_index=f,
                                        word_index=w,
                                        frame=frame,
                                        progress=None
                                        if progress is None
                                        else lambda n: progress(offset + n),
                                    )
                                    phases.append(p)
                                    offset += p.completed_intervals
                    value = PreparationApplicabilityPanel(
                        root,
                        prefix.fingerprint(),
                        selection.fingerprint(),
                        tuple(phases),
                        sealed.fingerprint(),
                    )
                else:
                    raise ValueError("unregistered native task")
            return canonical_task_result(
                context, (value,), check_id="applicability-native-cutoff-and-realized-census"
            )


class PreparationApplicabilityNativeProvider(PreparationApplicabilityRecordProvider):
    def __init__(self, registry: Any, runner: Any, inputs: Any) -> None:
        from .extension_bundle import CAPABILITY, OUTPUT_RECORDS

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
