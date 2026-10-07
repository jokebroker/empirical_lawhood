"Persist finite response-law evaluation's six pre-parent decisions using the existing coordinator."

from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal as D

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.planning.nested_controller_evaluation import CommonStartControllerEvaluationPlan, prepared_interface_evaluation_slice
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore, PreparedDesignBindingReceipt, PreparedExecutionEventKind, PreparedForecastParentCommitment, ProspectiveEvaluationBindingCoordinator
from empirical_lawhood.runtime.controller_runtime import RuntimeObservation
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_contracts import FiniteResponseLawEvaluationInvocation
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationRoot, FiniteResponseLawAssignedEvaluationInvocation
from .consumer import FiniteResponseLawNativeUnitReadoutMap, SPEC
from .control_causal import prior_causal_bindings
from .control_corpus import control_corpus
from .control_delivery import FiniteResponseLawNativeReceiptDelivery
from .control_operational import pre_parent_operational_inputs
from .control_plan import FiniteResponseLawControlLawContext, clock
from .control_study import control_study
from .control_records import FiniteResponseLawRootForecast, FiniteResponseLawRootRequestReveal, FiniteResponseLawRootControlLock
from .control_services import control_services, implementation_payloads
from .method_records import FiniteResponseLawQualificationReport
from .native_records import FiniteResponseLawCalibrationNativeEvaluation


def execution_prefix_id(root_id: str, policy_id: str) -> str:
    return f"{root_id}.{policy_id}.execution"


def freeze_root_choices(
    *,
    forecast: FiniteResponseLawRootForecast,
    requests: FiniteResponseLawRootRequestReveal,
    report: FiniteResponseLawQualificationReport,
    native: FiniteResponseLawCalibrationNativeEvaluation,
    contexts: tuple[FiniteResponseLawControlLawContext, ...],
    readout_map: FiniteResponseLawNativeUnitReadoutMap,
    evaluation_plan: CommonStartControllerEvaluationPlan,
    artifacts: tuple[ArtifactIdentity, ...],
    qualification_artifact: ArtifactIdentity,
    authority: StudyOperationAuthority,
    issued_study: ObjectIdentity,
    prerequisite_authority: ObjectIdentity | None,
    grantee_id: str,
    compiler_release_id: str,
    store: DurablePreparedExecutionEventStore,
    now: Callable[[], str],
) -> FiniteResponseLawRootControlLock:
    """Commit all qualified consumers before reserving any parent effects.

    Inputs have already passed worker custody and authority replay. No source,
    scheduler, storage implementation, fitting or selection algorithm lives here.
    An interrupted partial freeze cannot publish the all-consumer output guard.
    """
    if (
        requests.forecast != ObjectIdentity.from_record(forecast.forecast_id, forecast)
        or requests.root != forecast.root
        or qualification_artifact.sha256 != report.fingerprint()
        or qualification_artifact.payload_schema != report.SCHEMA
        or requests.forecast_artifact not in artifacts
        or qualification_artifact not in artifacts
        or len({a.artifact_id for a in artifacts}) != len(artifacts)
    ):
        raise ValueError("Finite response-law evaluation choices lose their exact persisted forecast, request or qualification")
    # Nested tables/requests retain typed identities; only actual published
    # containing artifacts are cited as byte custody.
    required_records = (requests, readout_map, authority, native)
    if any(
        not any(a.sha256 == r.fingerprint() and a.payload_schema == r.SCHEMA for a in artifacts)
        for r in required_records
    ):
        raise ValueError(
            "Finite response-law evaluation choices lack actual request, map, authority or retained-native custody"
        )
    implementations = tuple(b for b, _ in implementation_payloads())
    roles = {b.role: b for b in implementations}
    if any(b.reference.payload not in artifacts for b in implementations):
        raise ValueError("Finite response-law evaluation choices lack published installed controller configurations")
    by_qualification = {
        ObjectIdentity.from_record(c.qualification.result_id, c.qualification): c for c in contexts
    }
    if len(by_qualification) != len(contexts) or len(contexts) != len(forecast.tables):
        raise ValueError("Finite response-law evaluation choices change the qualified-method context census")
    coordinator = ProspectiveEvaluationBindingCoordinator(prepared_store=store)
    invocation_type = (
        FiniteResponseLawAssignedEvaluationInvocation
        if type(forecast.root) is FiniteResponseLawAssignedEvaluationRoot
        else FiniteResponseLawEvaluationInvocation
    )
    parent = invocation_type(
        forecast.prefix.invocation.source,
        forecast.root,
        "parent",
        forecast.root.assigned_parent,
        None,
        "parent",
    )
    common = ObjectIdentity.from_record(forecast.prefix.result_id, forecast.prefix)
    frozen = []
    compiled_artifacts = []
    for table in forecast.tables:
        qualification = report.qualifications[
            tuple(b.boundary for b in report.calibration.boundaries).index(table.boundary)
        ]
        context = by_qualification[
            ObjectIdentity.from_record(qualification.result_id, qualification)
        ]
        law = context.qualification.response_law
        assert law is not None
        # Implementation configuration belongs to programme.implementations.
        # Repeating all seven role artifacts in every scientific operand adds
        # no evidence and needlessly multiplies the compiled record size.
        implementation_artifacts = {b.reference.payload for b in implementations}
        native_artifact = next(
            a
            for a in artifacts
            if a.payload_schema == native.SCHEMA and a.sha256 == native.fingerprint()
        )
        boundary_artifacts = tuple(
            sorted(
                {
                    *[
                        a
                        for a in artifacts
                        if a not in implementation_artifacts and a != native_artifact
                    ],
                    law.evaluator.payload,
                },
                key=lambda a: a.artifact_id,
            )
        )
        evidence = (
            EvidenceLink(
                f"{table.table_id}.pre-parent-custody",
                EvidenceRelation.DERIVED_FROM,
                ObjectIdentity.from_record(forecast.forecast_id, forecast),
                ObjectIdentity.from_record(context.plan.plan_id, context.plan),
                tuple(a.artifact_id for a in boundary_artifacts),
                context.system.world.world_id,
                context.plan.information_cutoff.cutoff_id,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                (VisibilityCeiling.PROSPECTIVE,),
                'Frozen qualified law; actual primary prefix, independent request and execution authority.',
            ),
        )
        raw = pre_parent_operational_inputs(
            context=context,
            forecast=forecast,
            table=table,
            execution_authority=authority,
            issued_study=issued_study,
            prerequisite_authority=prerequisite_authority,
            grantee_id=grantee_id,
            at_utc=now(),
            artifacts=boundary_artifacts,
            evidence_links=evidence,
        )
        bindings = prior_causal_bindings(
            context=context,
            report=report,
            native=native,
            evidence_links=(
                replace(
                    evidence[0],
                    link_id=f"{table.table_id}.prior-causal-custody",
                    source=ObjectIdentity.from_record(native.evaluation_id, native),
                    artifact_ids=tuple(
                        sorted((native_artifact.artifact_id, qualification_artifact.artifact_id))
                    ),
                    reason='Retained law-qualification causal witnesses; no prospective evaluation outcome.',
                ),
            ),
        )
        for request in requests.requests:
            policy_id = f"{table.boundary}.consumer-{request.consumer}"
            prefix_id = execution_prefix_id(forecast.root.stage_unit, policy_id)
            corpus = control_corpus(
                context=context,
                report=report,
                table=table,
                consumer=request,
                readout_map=readout_map,
                calibration_artifact=qualification_artifact,
                source_artifacts=boundary_artifacts,
                evidence_links=evidence,
                operational_inputs=raw,
            )
            observation = RuntimeObservation(
                f"{request.request_id}.{table.boundary}.observation",
                table.root_id,
                tuple(
                    NamedDecimal(v.quantity_id, v.value, v.native_unit)
                    for v in table.requests[0].input_values
                ),
                clock(4096),
                boundary_artifacts,
                OutcomeAccess.OUTCOME_BLIND,
            )
            programme = control_study(
                system=context.system,
                corpus=corpus,
                action_bindings=bindings,
                implementations=implementations,
                observation=observation,
                frozen_recipe=ObjectIdentity.from_record("flh-science", SPEC),
                compiler_release_id=compiler_release_id,
                decision_budget_seconds=D(60),
                prospective_evaluation=ObjectIdentity.from_record(
                    evaluation_plan.evaluation_plan_id, evaluation_plan
                ),
            )
            route = control_services(
                programme,
                evaluation_plan,
                FiniteResponseLawNativeReceiptDelivery(roles[ImplementationRole.DELIVERY]),
            )
            compiled = route.compile(programme)
            decision = route.prepare_commitment(
                compiled, observation, commitment_coordinate=observation.coordinate
            )
            task = corpus.reachability_receipts[0].request.task
            locked = PreparedForecastParentCommitment(
                f"{prefix_id}.parent-lock",
                ObjectIdentity.from_record(evaluation_plan.evaluation_plan_id, evaluation_plan),
                forecast.root.stage_unit,
                policy_id,
                common,
                ObjectIdentity.from_record(parent.task_id, parent),
                decision,
                task,
                now(),
            )
            design = PreparedDesignBindingReceipt(
                f"{prefix_id}.design",
                issued_study,
                prepared_interface_evaluation_slice(
                    evaluation_plan, root_id=forecast.root.stage_unit, policy_id=policy_id
                ),
                forecast.root.stage_unit,
                policy_id,
                common,
                ObjectIdentity.from_record(locked.commitment_id, locked),
                roles[ImplementationRole.OUTCOME_EVALUATOR],
                now(),
            )
            coordinator.freeze_prepared_forecast_design(
                prefix_id=prefix_id,
                binding=design,
                plan=evaluation_plan,
                forecast_parent=locked,
                compiled=compiled,
            )
            compiled_artifacts.append(
                store.publish_prepared_record(
                    prefix_id=prefix_id, object_id=compiled.compiled_study_id, record=compiled
                )
            )
            frozen.append((design, locked))
    # Validate the complete census before enabling a single physical parent.
    result = FiniteResponseLawRootControlLock(
        forecast,
        requests,
        tuple(d for d, _ in frozen),
        tuple(p for _, p in frozen),
        tuple(compiled_artifacts),
    )
    for design, locked in frozen:
        prefix_id = execution_prefix_id(design.root_id, design.policy_id)
        artifact = store.publish_prepared_record(
            prefix_id=prefix_id, object_id=locked.commitment_id, record=locked
        )
        coordinator.record_prepared_boundary(
            prefix_id=prefix_id,
            design=design,
            kind=PreparedExecutionEventKind.PARENT_RESERVED,
            subject=design.parent_commitment,
            artifact=artifact,
            occurred_at_utc=now(),
        )
    return result
