"""Actual current publication custody for disposable software tests only.

The proposer identities, authority records and scientific responses are fictional.
These records can neither issue a real study nor qualify a real cohort.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
import json
import numpy as np
from hashlib import sha256

from empirical_lawhood.adapters.composition.finite_response_law.rerun_input import FiniteResponseLawCurrentPublication
from empirical_lawhood.infrastructure.study_issue import ExternalIssuedStudyPublisher, ExternalStudyOperationAuthorityStore
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.study_issue import AccountableHumanIdentity, HumanProposerAttestation, ImplementationSourceClosure, StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.runtime.candidate_compiler import ExecutableStudyCompilationReport
from empirical_lawhood.runtime.study_issue import prepare_study_issue, prepare_executable_study_issue, build_study_approval_proposal, MAX_COMPOSED_PROGRAMME_CONTROL_BYTES
from empirical_lawhood.kernel.evidence import VisibilityCeiling
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.runtime.artifacts import ArtifactWriteRequest, ArtifactStreamWriteRequest, ArtifactProfile, CanonicalTaskReceipt, ReceiptCheck
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore, decode_artifact_manifest
from empirical_lawhood.adapters.composition.finite_response_law.rerun_input import FiniteResponseLawCurrentParent


@dataclass(frozen=True)
class CurrentIssueFixture:
    manifest: object
    publication: FiniteResponseLawCurrentPublication
    package: object
    control_sizes: tuple[tuple[str, int], ...]
    approval_trust_path: object
    execution_plan: object


def _approve_current_publication(plane, published, system, *, label, trust_directory):
    """Obtain structurally real approvals from explicitly synthetic test signers."""
    from empirical_lawhood.kernel.authority import AuthorityAction, SourceAccessClass
    from empirical_lawhood.planning.approval import ApprovalCheckerRegistry, AttestationResult, issue_approval_gate_attestation
    from empirical_lawhood.api.composition import compose_operator_approval_runtime
    from tests.approval_support import FixedDecisionClock, deterministic_approval_signer
    from tests.runtime_platform.test_study_approval import _checker_registrations

    at_utc = "2026-10-07T00:00:00Z"
    registrations = _checker_registrations(system.authority_policy.required_gate_ids)
    checker_registry = ApprovalCheckerRegistry(f"synthetic.{label}.checkers", registrations)
    trust_directory.mkdir(mode=0o700)
    trust_path = trust_directory / "checkers.json"
    trust_path.write_bytes(checker_registry.canonical_bytes())
    trust_path.chmod(0o600)
    stores = compose_operator_approval_runtime(artifact_plane=plane, checker_trust_path=trust_path,
        clock=FixedDecisionClock(clock_id=f"synthetic.{label}.clock", value=at_utc))
    service = stores.service
    proposal = build_study_approval_proposal(issued_study=published.manifest, publication_receipt=published.publication_receipt)
    frozen = service.freeze_study_proposal(proposal=proposal, system=system,
        requested_scope_id=system.authority_policy.scope_ids[0], implementation_commit=published.manifest.source_closure.implementation_commit,
        authority_action=AuthorityAction.SIMULATION_EXECUTION, source_access=SourceAccessClass.NONE)
    authorization_id = f"synthetic.{label}.approval"
    identities = []
    for registration in registrations:
        attestation = issue_approval_gate_attestation(registration=registration,
            signer=deterministic_approval_signer(registration.gate_id),
            attestation_id=f"synthetic.{label}.{registration.gate_id}.attestation", authorization_id=authorization_id,
            subject=frozen, result=AttestationResult.PASSED, checked_at_utc=at_utc, issued_at_utc=at_utc,
            information_cutoff=proposal.decision_cutoff)
        identities.append(stores.attestation_store.persist(attestation))
    approval = service.authorize(policy=system.authority_policy, frozen_proposal=frozen,
        attestations=tuple(identities),
        authorization_id=authorization_id, approver_id=system.authority_policy.delegate_id)
    return stores.study_proposal_store.load(frozen.object_id), approval, service, trust_path


def publish_current_issue(plane, handoff, *, label):
    """Publish the actual compiled full-size fixture via the existing issue owners."""
    directory = handoff.directory
    if hasattr(handoff, "composition"):
        composition = handoff.composition
        authoring = composition.authoring
        context = composition.bundle.context
        payloads = composition.payloads
        registrations = composition.decoder_registrations
    else:
        bundle = handoff.bundle
        authoring = bundle.authoring
        context = bundle.standard_context
        payloads = bundle.payloads
        registrations = bundle.decoder_registrations
    report = decode_canonical_bytes((directory / "candidate-report.json").read_bytes(),
        ExecutableStudyCompilationReport, maximum_bytes=MAX_COMPOSED_PROGRAMME_CONTROL_BYTES)
    closure = decode_canonical_bytes((directory / "source-closure.json").read_bytes(),
        ImplementationSourceClosure, maximum_bytes=65536)
    candidate, base_candidate = report.candidate, report.base_report.candidate
    assert candidate is not None and base_candidate is not None
    base = base_candidate.base_candidate
    human = AccountableHumanIdentity(f"synthetic.{label}.human", "synthetic-software-fixture", "synthetic-tests", "f" * 64)
    issuer = ObjectIdentity.from_record(human.human_id, human)
    at_utc = "2026-10-07T00:00:00Z"
    store = ExternalStudyOperationAuthorityStore(plane)
    publisher = ExternalIssuedStudyPublisher(artifact_plane=plane, authority_store=store, grantee_id="operator.issue-service")

    def attest(value, raw_hash, suffix):
        return HumanProposerAttestation(f"synthetic.{label}.{suffix}.attestation",
            ObjectIdentity.from_record(value.candidate_id, value), issuer, human.human_id,
            base.design_origin.origin_id, base.design_input_ids, base.source_qualification_receipts,
            raw_hash, base.semantic_config_sha256, True, True, False, False, at_utc, OutcomeAccess.OUTCOME_BLIND)

    def grant(kind, subject, suffix, prerequisite=None):
        custody = kind is StudyAuthorityKind.CUSTODY_PUBLICATION
        execution = kind is StudyAuthorityKind.EXPERIMENT_EXECUTION
        authority = StudyOperationAuthority(f"synthetic.{label}.{suffix}.authority", kind, subject, prerequisite,
            issuer, "operator.issue-service" if custody else "operator.execution-service" if execution else "operator.evaluator-service",
            f"synthetic.{label}.{suffix}.scope", plane.root.contract.storage_root_id if custody else None,
            "issued-programmes" if custody else None, False, custody, execution, False, not custody and not execution,
            at_utc, None, OutcomeAccess.OUTCOME_BLIND if custody else OutcomeAccess.EVALUATION_SEALED if execution else OutcomeAccess.EVALUATOR_REVEAL)
        store.persist(authority)
        return authority

    base_raw = (directory / "base-authoring.json").read_bytes()
    base_custody = grant(StudyAuthorityKind.CUSTODY_PUBLICATION, ObjectIdentity.from_record(base_candidate.candidate_id, base_candidate), "base-custody")
    preparation = prepare_study_issue(authoring_package=authoring.base, raw_authoring_package_bytes=base_raw,
        current_compilation=report.base_report, candidate=base_candidate, registry=context.base.registry,
        materialization_qualifications=context.base.qualifications,
        source_closure=closure, observed_source_closure=closure,
        proposer_attestation=attest(base_candidate, sha256(b"application/json\0" + base_raw).hexdigest(), "base"),
        custody_authority=base_custody, storage_root_id=plane.root.contract.storage_root_id,
        grantee_id="operator.issue-service", at_utc=at_utc)
    base_receipt = publisher.publish(preparation, at_utc=at_utc)
    extension_custody = grant(StudyAuthorityKind.CUSTODY_PUBLICATION, ObjectIdentity.from_record(candidate.candidate_id, candidate), "extension-custody")
    extended = prepare_executable_study_issue(base_manifest=preparation.manifest, base_publication_receipt=base_receipt,
        current_compilation=report, candidate=candidate, extension_payload_bytes=tuple(record.canonical_bytes() for record in payloads),
        decoder_registrations=registrations, extension_materializations=report.extension_materializations,
        extension_proposer_attestation=attest(candidate, report.authoring_materialization.raw_materialization_sha256, "extension"),
        extension_custody_authority=extension_custody, storage_root_id=plane.root.contract.storage_root_id,
        grantee_id="operator.issue-service", at_utc=at_utc)
    publisher.publish_standard(extended, at_utc=at_utc)
    publication = publisher.load_executable_study(extended.manifest.issue_id)
    replayed = publication.manifest
    assert replayed == extended.manifest
    issued = ObjectIdentity.from_record(replayed.issue_id, replayed)
    base_published = publisher.load_study(preparation.manifest.issue_id)
    trust_directory = directory / "synthetic-approval-trust"
    trust_directory.mkdir(mode=0o700)
    base_frozen, base_approval, _, _ = _approve_current_publication(plane, base_published, base.system,
        label=f"{label}.base", trust_directory=trust_directory / "base")
    base_execution = grant(StudyAuthorityKind.EXPERIMENT_EXECUTION,
        ObjectIdentity.from_record(preparation.manifest.issue_id, preparation.manifest), "base-execution",
        ObjectIdentity.from_record(base_approval.authorization_id, base_approval))
    from empirical_lawhood.api.models import assemble_issued_study_package, assemble_experiment_package
    run_id = handoff.projection.source_plan.object_id
    base_package = assemble_issued_study_package(issued_study=preparation.manifest, publication_receipt=base_receipt,
        frozen_proposal=base_frozen, scientific_approval=base_approval, execution_authority=base_execution,
        run_plan_id=run_id, grantee_id="operator.execution-service", at_utc=at_utc,
        model_set=None if not hasattr(handoff, "packet") or handoff.packet.control is None else handoff.packet.control.primary_model_set)
    frozen, approval, approval_service, trust_path = _approve_current_publication(plane, publication, base.system,
        label=label, trust_directory=trust_directory / "extensions")
    execution = grant(StudyAuthorityKind.EXPERIMENT_EXECUTION, issued, "execution",
        ObjectIdentity.from_record(approval.authorization_id, approval))
    execution_identity = ObjectIdentity.from_record(execution.authority_id, execution)
    reveal = grant(StudyAuthorityKind.OUTCOME_REVEAL, issued, "reveal", execution_identity)
    package = assemble_experiment_package(base=base_package, issued_study=replayed,
        publication_receipt=publication.publication_receipt, frozen_proposal=frozen, scientific_approval=approval,
        execution_authority=execution, execution_resource_envelope_spec=handoff.resources,
        predevelopment_jit_signature_census=None, jit_graph_signature_manifest=None,
        run_plan_id=run_id, grantee_id="operator.execution-service", at_utc=at_utc)
    from empirical_lawhood.runtime.compiler import compile_run_plan, lower_run_plan
    plan = compile_run_plan(run_plan_id=run_id, campaign=base.campaign, system=base.system,
        experiment=package.experiment, frozen_proposal=frozen, authorization=approval,
        template=base.protocol, registry=context.base.registry, implementation_commit=closure.implementation_commit,
        approval_service=approval_service,
        model_set=None if not hasattr(handoff, "packet") or handoff.packet.control is None else handoff.packet.control.primary_model_set,
        scientific_graph=base.scientific_graph, candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        issued_extension_set=replayed.issued_extensions, execution_resource_envelope_spec=handoff.resources)
    projection = lower_run_plan(plan, context.base.registry)
    sizes = []
    for name, value in (("campaign-package", package), ("run-plan", plan), ("execution-plan", projection)):
        raw = value.canonical_bytes()
        sizes.append((name, len(raw)))
        print(f"synthetic_{label}_{name}_bytes", len(raw), flush=True)
        plane.write_stream(ArtifactStreamWriteRequest(logical_artifact_id=f"{name}.{run_id}",
            relative_path=f"runs/{run_id}/plans/{name}.json", payload_schema=value.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON, media_type="application/json",
            publication_scope_id=f"synthetic.{label}.plans", publication_scope_relative_root=f"runs/{run_id}/plans",
            chunks=(raw[index:index + 1024**2] for index in range(0, len(raw), 1024**2)),
            maximum_bytes=len(raw), maximum_chunk_bytes=min(len(raw), 1024**2),
            expected_size_bytes=len(raw), expected_physical_sha256=sha256(raw).hexdigest(),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            parent_visibility_ceilings=(), outcome_access=OutcomeAccess.OUTCOME_BLIND, minimum_free_bytes=1))
    return CurrentIssueFixture(replayed, FiniteResponseLawCurrentPublication(run_id, issued,
        execution_identity, ObjectIdentity.from_record(reveal.authority_id, reveal)), package, tuple(sizes), trust_path, projection)


def publish_native_parent(plane, handoff, issue, *, label, native=None):
    """Retain a complete fictional native completion in the actual issued output census."""
    from tests.finite_response_custody_fixtures import synthetic_native
    native = synthetic_native(handoff.packet.source) if native is None else native
    receipt, manifests, primary = publish_current_step_output(plane, issue, label=label, record=native)
    run_id = issue.publication.run_id
    parent = FiniteResponseLawCurrentParent(f"synthetic.{label}.parent", issue.publication.issued_study,
        issue.publication.execution_authority, issue.publication.reveal_authority, handoff.packet.source,
        manifests, ((run_id, receipt.task_id, receipt.receipt_id),))
    native_artifact = ArtifactIdentity(primary.logical.logical_artifact_id, "synthetic-current-calibration",
        native.SCHEMA, native.fingerprint(), primary.logical.media_type, len(native.canonical_bytes()))
    receipt_artifact = ArtifactIdentity(receipt.receipt_id, "synthetic-current-calibration", receipt.SCHEMA,
        receipt.fingerprint(), "application/json", len(receipt.canonical_bytes()))
    return parent, native, receipt, native_artifact, receipt_artifact


def publish_current_step_output(plane, issue, *, label, record, task_id=None, input_materialization_ids=()):
    """Retain one typed fictional result and its full issued sibling output census."""
    task = next(task for task in issue.execution_plan.tasks
        if (task_id is None or task.task_id == task_id)
        and any(output.payload_schema == record.SCHEMA for output in task.outputs))
    run_id = issue.publication.run_id
    requests = []
    for output in task.outputs:
        raw = record.canonical_bytes() if output.payload_schema == record.SCHEMA else canonical_json_bytes({
            "schema": output.payload_schema, "version": "1.0.0", "value": {
                "synthetic_nonexecuting_fixture": label, "scientific_evidence": False}})
        requests.append(ArtifactWriteRequest(logical_artifact_id=output.logical_artifact_id,
            relative_path=output.relative_path,
            payload_schema=output.payload_schema, profile=output.profile, media_type=output.media_type,
            publication_scope_id=f"synthetic.{label}.native-outputs.{output.output_id}", publication_scope_relative_root=f"runs/{run_id}/outputs",
            payload=raw, visibility_ceiling=output.visibility_ceiling,
            parent_visibility_ceilings=(), outcome_access=output.outcome_access, minimum_free_bytes=1))
    outputs = tuple(plane.write(request) for request in requests)
    ordered = tuple(sorted(outputs, key=lambda result: result.logical.logical_artifact_id))
    attempt = f"{run_id}.{task.task_id}.attempt-001"
    receipt = CanonicalTaskReceipt(f"receipt.{attempt}", run_id, task.task_id, attempt,
        issue.manifest.source_closure.implementation_commit, input_materialization_ids, tuple(result.materialization for result in ordered),
        tuple(result.logical for result in ordered), (ReceiptCheck("fictional-software-output-census", True, ()),),
        OperationalStatus.SUCCEEDED, ())
    ExternalTaskReceiptStore(plane, minimum_free_bytes=1).commit(receipt,
        visibility_ceiling=VisibilityCeiling.most_restrictive(*(output.logical.visibility_ceiling for output in outputs)),
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL)
    manifests = tuple(sorted((decode_artifact_manifest(plane.root.resolve(result.manifest_materialization.relative_path, for_write=False).read_bytes())
        for result in outputs), key=lambda manifest: manifest.materialization.materialization_id))
    primary = next(result for result in outputs if result.logical.payload_schema == record.SCHEMA)
    return receipt, manifests, primary


def nominated_synthetic_native(source, nomination, directory):
    """Fictional exact-model calibration data for public application plumbing.

    No source is acquired. The exposed observations are deliberately generated
    from the fixed nomination so this fixture cannot test scientific accuracy.
    """
    from tests.finite_response_custody_fixtures import synthetic_native
    from empirical_lawhood.adapters.methods.finite_response_law.frozen_package import frozen_development
    from empirical_lawhood.adapters.methods.finite_response_law.assigned_prediction import before_parent
    from empirical_lawhood.adapters.methods.finite_response_law.science import PARENTS

    points, widths = frozen_development(report_bytes=(directory / "development-report.json").read_bytes(),
        coefficient_bytes=(directory / "coefficients.transport.json").read_bytes(),
        report_identity=nomination.development_report, coefficient_identity=nomination.coefficients)
    native = synthetic_native(source)
    prefix = np.tile(points.n0.center[None, :, None], (32, 1, 2))
    parents = np.asarray([PARENTS.index(root.assigned_parent) for root in source.roots], dtype=np.int64)
    predicted = before_parent(points, {key: widths[key] for key in ("composed", "cached", "direct")}, prefix, parents)
    views = []
    for view in native.views:
        index = view.root.index
        words = []
        for observation in view.words:
            word = observation.invocation.word
            if word.sign:
                pair = (2 if word.magnitude == 16 else 0) + word.direction_index
                means = predicted.composed.mean[index, pair]
                preservation = means[2:5] if word.sign > 0 else means[5:8]
                values = tuple(Decimal(str(value)) for value in (*tuple(word.sign * means[:2]), *preservation, 0.0, 0.0))
                observation = replace(observation, outputs=values)
            words.append(observation)
        views.append(replace(view, words=tuple(words),
            prefix_interface=replace(view.prefix_interface, values=tuple(Decimal(str(value)) for value in points.n0.center)),
            handoff_interface=replace(view.handoff_interface, values=tuple(Decimal(str(value)) for value in predicted.predicted_interface[index]))))
    return replace(native, views=tuple(views))


def qualify_nominated_fixture(plane, config, native, native_receipt, directory):
    """Use the actual sole qualification service on explicitly fictional observations."""
    from empirical_lawhood.adapters.methods.finite_response_law.calibration_operands import calibration_operands
    from empirical_lawhood.adapters.methods.finite_response_law.method_records import FiniteResponseLawAssignedCalibrationReport, READOUT_SCHEMA, CALIBRATE
    from empirical_lawhood.adapters.methods.finite_response_law.terminal import qualify_calibration
    from empirical_lawhood.adapters.methods.finite_response_law.executable_binding import METHOD_CAPABILITY
    from empirical_lawhood.adapters.composition.finite_response_law.design import qualification_system
    from empirical_lawhood.infrastructure.candidate_payloads import ExternalCandidatePayloadPlane
    from empirical_lawhood.runtime.execution import DependencyReceiptBinding

    report_bytes = (directory / "development-report.json").read_bytes()
    coefficients = (directory / "coefficients.transport.json").read_bytes()
    operands = calibration_operands(native_bytes=native.canonical_bytes(), native_artifact=config.native_evaluation,
        native_receipt=native_receipt, expected_native_receipt=config.expected_native_receipt, source=config.native_source,
        development_report_bytes=report_bytes, coefficient_bytes=coefficients, development_report=config.development_report,
        coefficients=config.coefficients, development_manifest=config.development_manifest)
    readout_bytes = json.dumps(operands.readout, sort_keys=True).encode()
    readout = ArtifactIdentity("synthetic.calibration.readout", "fictional-software-calibration", READOUT_SCHEMA,
        sha256(readout_bytes).hexdigest(), "application/json", len(readout_bytes))
    report = FiniteResponseLawAssignedCalibrationReport(f"{CALIBRATE}.result", ObjectIdentity.from_record(config.config_id, config),
        operands.prediction_artifact, readout, operands.boundaries,
        tuple((boundary, bool(operands.readout["usability"][boundary]["joint_decision_opportunity"])) for boundary in ("lower", "composed", "cached", "direct")))
    result = qualify_calibration(config=config, report=report, operands=operands, development_report_bytes=report_bytes,
        coefficient_bytes=coefficients, system=qualification_system(config.native_source), manifest=METHOD_CAPABILITY,
        payload_plane=ExternalCandidatePayloadPlane(plane, "synthetic/nominated-laws", "synthetic.nominated-laws",
            VisibilityCeiling.PROSPECTIVE, OutcomeAccess.EVALUATOR_REVEAL, (), 1),
        dependency=DependencyReceiptBinding("synthetic.calibration.receipt", CALIBRATE, ("synthetic.calibration", "synthetic.predictions")),
        calibration_materialization_id="synthetic.calibration", prediction_materialization_id="synthetic.predictions")
    assert result.eligible_for_prospective_evaluation
    return result
