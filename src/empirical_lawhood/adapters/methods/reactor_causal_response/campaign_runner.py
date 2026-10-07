"Authenticated method dispatch for root controller use, contribution and native scoring."

from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider import DependencyCustodyReader
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.execution import TaskContext, WorkerInputKind
from empirical_lawhood.adapters.simulators.reactor_causal_response.campaign import ControlCustodyPort
from .campaign_records import EmpiricalStudySource, TimedReactorConfirmationEnvelope, EmpiricalRootAnalysis
from .experiment_records import EmpiricalDiscoveryEnvelope, EmpiricalQualificationTerminal
from .campaign_analysis import analyze_root, analyze_cohort
from .native_benchmark import NativeVerifierPort, native_benchmark
from .config import EmpiricalRecipe, CONFIRMATION_ROOTS, NATIVE_BENCHMARK_ARMS

ROOT_TASKS = tuple(f"empirical.prospective-evaluation.{root}" for root in CONFIRMATION_ROOTS)
NATIVE_BENCHMARK_TASKS = tuple(f"empirical.native.{arm.lower()}" for arm in NATIVE_BENCHMARK_ARMS)
CAMPAIGN_TASKS = (*ROOT_TASKS, "empirical.contribution", *NATIVE_BENCHMARK_TASKS)


def run_campaign_method(
    context: TaskContext,
    recipe: EmpiricalRecipe,
    custody: DependencyCustodyReader,
    control: ControlCustodyPort,
    verifier: NativeVerifierPort,
) -> CanonicalRecord:
    from .provider import validate_dependency

    if (
        context.task_id not in CAMPAIGN_TASKS
        or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        or context.config.content_sha256 != recipe.fingerprint()
    ):
        raise ValueError("campaign task/config/outcome-access differs")
    types_ = {
        t.SCHEMA: t
        for t in (
            EmpiricalRecipe,
            EmpiricalStudySource,
            TimedReactorConfirmationEnvelope,
            EmpiricalRootAnalysis,
            EmpiricalDiscoveryEnvelope,
            EmpiricalQualificationTerminal,
        )
    }
    records = []
    for port in context.input_ports:
        if port.payload_schema not in types_:
            raise ValueError("undeclared campaign input schema")
        record_type = types_[port.payload_schema]
        record = decode_canonical_bytes(port.read(), record_type, maximum_bytes=256 * 1024**2)
        if isinstance(record, (EmpiricalRecipe, EmpiricalStudySource)):
            if port.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
                raise ValueError("campaign code/configuration must be predeclared")
        else:
            if port.kind is not WorkerInputKind.DEPENDENCY:
                raise ValueError("campaign scientific operands require dependency custody")
            manifest, receipt = custody.read_dependency(context, port.binding)
            validate_dependency(port.binding, manifest, receipt)
            if (
                receipt.run_id != context.run_id
                or manifest.logical.content_sha256 != record.fingerprint()
            ):
                raise ValueError("campaign dependency run/bytes differ")
        records.append(record)
    configs = [r for r in records if isinstance(r, EmpiricalRecipe)]
    qs = [r for r in records if isinstance(r, EmpiricalQualificationTerminal)]
    if configs != [recipe] or len(qs) != 1:
        raise ValueError("campaign recipe/qualification census differs")
    q = qs[0]
    if context.task_id == "empirical.contribution":
        roots = tuple(r for r in records if isinstance(r, EmpiricalRootAnalysis))
        if len(records) != len(CONFIRMATION_ROOTS) + 2 or len(roots) != len(CONFIRMATION_ROOTS):
            raise ValueError("campaign analysis loses assigned roots")
        return analyze_cohort(roots, q)
    sources = [r for r in records if isinstance(r, EmpiricalStudySource)]
    discoveries = [r for r in records if isinstance(r, EmpiricalDiscoveryEnvelope)]
    if len(sources) != 1 or len(discoveries) != 1:
        raise ValueError("campaign source/nominee census differs")
    source, discovery = sources[0], discoveries[0]
    if context.task_id in ROOT_TASKS:
        raw_roots = [r for r in records if isinstance(r, TimedReactorConfirmationEnvelope)]
        if (
            len(records) != 5
            or len(raw_roots) != 1
            or context.task_id != f"empirical.prospective-evaluation.{raw_roots[0].root}"
        ):
            raise ValueError("Controller-use input substitutes its independent root")
        return analyze_root(
            raw_roots[0],
            q,
            discovery,
            source.comparators,
            json.loads(source.batch.plant_params),
            control,
        )
    if len(records) != 4:
        raise ValueError("native verifier dependency census differs")
    arm = next(
        a for a in NATIVE_BENCHMARK_ARMS if context.task_id == f"empirical.native.{a.lower()}"
    )
    return native_benchmark(arm, q, discovery, source, verifier)
