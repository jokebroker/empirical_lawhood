"""Registered method tasks; future evidence never constructs causal seals."""

from typing import Any


from dataclasses import dataclass
import numpy as np

from empirical_lawhood.adapters.composition.phase_inputs import config_input, dependency, upstream
from empirical_lawhood.adapters.composition.preparation_applicability.inputs import PreparationApplicabilityRecordProvider
from empirical_lawhood.runtime.task_records import canonical_task_result
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord, AdjudicationEvaluability
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.status import ScientificStatus, AdmissionStatus
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.methods.finite_response_law.original_f import OriginalFiniteResponseLaw
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import (
    ConstructedPreparationPrefix,
    ConstructedPreparationPanel,
)
from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_instruments import (
    preparation_policy_compact_interface,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from .config import ConstructedPreparationStage
from .records import ConstructedPreparationSelection, ConstructedPreparationMeasuredRoot, numbers, DELTA
from .records import ConstructedPreparationLowerSeal
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import ConstructedPreparationParents
from .readout import report, constructor_crossing
from .measurement import lower_choices, measure_root, requests
from .records import ConstructedPreparationReport
from .qualification import require_qualified_constructor


def response_admission_status(status: ScientificStatus) -> AdmissionStatus:
    """A response test makes no admission finding; missing data stays unevaluable."""
    return (
        AdmissionStatus.UNEVALUABLE
        if status is ScientificStatus.UNEVALUABLE
        else AdmissionStatus.NOT_EVALUATED
    )


@dataclass(frozen=True)
class ConstructedPreparationMethodRunner:
    stage: ConstructedPreparationStage
    custody: Any
    limits: Any

    @property
    def manifest(self) -> Any:
        from .extension_bundle import CAPABILITY

        return CAPABILITY

    def execute(self, context: TaskContext) -> RunnerResult:
        with self.limits.task(context.task_id):
            config_input(context, self.stage)
            return canonical_task_result(
                context,
                self._execute(context),
                check_id="applicability-exact-causal-seals-and-custody",
            )

    def _execute(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        def dep(kind: Any, task: Any) -> Any:
            return dependency(context, self.custody, kind, task)[0]

        task = context.task_id
        kind = task.split(".")[1]
        if kind in ("select", "lower", "measure"):
            root = task.split(".", 2)[2]
            index = self.stage.root_ids.index(root)
            prefix = dep(ConstructedPreparationPrefix, f"ap.prefix.{root}")
            lower = (
                upstream(context, self.stage, "lower", OriginalFiniteResponseLaw)
                if kind != "select"
                else None
            )
            if kind == "select":
                if self.stage.phase == "E":
                    qualification = upstream(context, self.stage, "qualification", ConstructedPreparationReport)
                    require_qualified_constructor(self.stage, qualification)
                complete = prefix.frame_base64 is not None
                return (ConstructedPreparationSelection(
                    root, prefix.fingerprint(), None,
                    1 if complete else None, numbers((0, 1, 0)) if complete else (),
                ),)
            parents = dep(ConstructedPreparationParents, f"ap.parents.{root}") if kind == "lower" else None
            assert lower is not None and lower.q is not None
            if kind == "lower":
                assert parents is not None
                complete = (
                    prefix.frame_base64 is not None
                    and len(parents.phases) == 6
                    and all(p.disposition == "COMPLETE" for p in parents.phases)
                )
                if not complete:
                    return (
                        ConstructedPreparationLowerSeal(
                            root,
                            parents.fingerprint(),
                            lower.fingerprint(),
                            False,
                            (),
                            (),
                            (),
                            (),
                            (),
                            (),
                        ),
                    )
                frame = PreparedPortFrame(4096, _decode(prefix.frame_base64, (2, 3, 4, 4)))
                # Primary handoff only, with actual causal source join.
                z = []
                for s in range(3):
                    p = next(
                        p for p in parents.phases if p.schedule_index == s and p.refinement == 1
                    )
                    if p.incoming_sha256 != prefix.phases[0].fingerprint():
                        raise ValueError("lower seal substitutes its observed handoff")
                    z.append(
                        preparation_policy_compact_interface(
                            frame=frame,
                            ticks=p.history_ticks,
                            positions=_decode(p.history_positions_base64, (31, 2, 3, 4, 4)),
                            momenta=_decode(p.history_momenta_base64, (31, 2, 3, 4, 4)),
                        ).values
                    )
                pred = lower.predict(np.asarray(z))
                width = float(lower.q) * pred.sigma + np.asarray(DELTA) / 8
                choices: tuple[int, ...] = ()
                directions: tuple[int, ...] = ()
                requirements: tuple[Any, ...] = ()
                if self.stage.phase in ("Q", "E"):
                    direction, requirement = requests(self.stage.allocation.roots[index].seed_for("requests"))
                    choices = tuple(
                        map(
                            int,
                            lower_choices(
                                pred.mean, width, pred.supported, direction, requirement
                            ).ravel(),
                        )
                    )
                    directions, requirements = (
                        tuple(map(int, direction.ravel())),
                        numbers(requirement),
                    )
                return (
                    ConstructedPreparationLowerSeal(
                        root,
                        parents.fingerprint(),
                        lower.fingerprint(),
                        True,
                        numbers(pred.mean),
                        numbers(width),
                        tuple(map(bool, pred.supported)),
                        choices,
                        directions,
                        requirements,
                    ),
                )
            panel = dep(ConstructedPreparationPanel, f"ap.assay.{root}")
            sealed = dep(ConstructedPreparationLowerSeal, f"ap.lower.{root}")
            if (
                panel.lower_seal_sha256 != sealed.fingerprint()
                or sealed.lower_sha256 != lower.fingerprint()
            ):
                raise ValueError("measurement substitutes its preceding lower-use seal")
            # The actual response cannot alter the preceding choices/payload.
            measured = measure_root(prefix, panel, lower, self.stage.phase, self.stage.allocation.roots[index].seed_for("requests"), sealed)
            if measured.complete and not sealed.complete:
                raise ValueError("measurement bypassed a missing pre-future lower seal")
            return (measured,)
        if kind == "report":
            measurements = tuple(dep(ConstructedPreparationMeasuredRoot, f"ap.measure.{r}") for r in self.stage.root_ids)
            lower = upstream(context, self.stage, "lower", OriginalFiniteResponseLaw)
            crossings = tuple(
                constructor_crossing(m, lower) for m in measurements
            )
            return (report(self.stage, measurements, crossings),)
        if kind != "adjudication" or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("unregistered method task or missing reveal authority")
        a = context.scientific_adjudication_context
        if a is None:
            raise ValueError("terminal lacks its issued scientific relation")
        summary = dep(ConstructedPreparationReport, "ap.report")
        status = (
            ScientificStatus.SUPPORTED if summary.disposition in ("QUALIFIED", "SUPPORTED")
            else ScientificStatus.UNEVALUABLE if not summary.complete
            else ScientificStatus.NOT_SUPPORTED
        )
        reason = f"BOUNDARY_{self.stage.phase}_{summary.disposition}"
        return (
            ScientificAdjudicationRecord(
                f"{context.run_id}.adjudication",
                context.run_id,
                task,
                a.execution_plan,
                context.input_materialization_ids,
                tuple(
                    sorted(
                        p.logical_artifact_id
                        for p in context.output_ports
                        if p.logical_artifact_id is not None
                    )
                ),
                context.dependency_receipt_ids,
                a.evidence_world_id,
                a.evidence_world_kind,
                a.relation,
                a.independent_unit_id,
                a.information_cutoffs,
                a.visibility_ceiling,
                a.outcome_access,
                AdjudicationEvaluability.UNEVALUABLE
                if status is ScientificStatus.UNEVALUABLE
                else AdjudicationEvaluability.EVALUABLE,
                status,
                response_admission_status(status),
                (reason,),
            ),
        )


class ConstructedPreparationMethodProvider(PreparationApplicabilityRecordProvider):
    def __init__(self, registry: Any, runner: Any, inputs: Any) -> None:
        from .extension_bundle import CAPABILITY, OUTPUT_RECORDS

        super().__init__(
            registry,
            CAPABILITY,
            runner,
            runner.stage,
            runner.stage.config_id,
            inputs,
            OUTPUT_RECORDS,
            adjudication=True,
        )
