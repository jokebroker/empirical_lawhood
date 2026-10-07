# SPDX-License-Identifier: MPL-2.0
"""Synthetic complete calibration data; no source integration or real authority.

The real reducers and sole qualification service calculate these fictional
responses. Published contexts and grants belong only to disposable test stores.
They cannot qualify a real cohort or authorize scientific issue.
"""

from dataclasses import dataclass, replace
from decimal import Decimal as D
from hashlib import sha256
from io import BytesIO
import json
from typing import ClassVar
import numpy as np
from tests.test_finite_response_assigned_method_records import _source
from empirical_lawhood.adapters.methods.finite_response_law import method_records as mr
from empirical_lawhood.adapters.methods.finite_response_law.native_records import native_method_types
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedCalibrationInterface
from empirical_lawhood.adapters.methods.finite_response_law.calibration_operands import calibration_operands
from empirical_lawhood.adapters.methods.finite_response_law.terminal import qualify_calibration
from empirical_lawhood.adapters.methods.finite_response_law.executable_binding import METHOD_CAPABILITY
from empirical_lawhood.adapters.composition.finite_response_law.design import qualification_system
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.provider import native_result_type
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT
from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawAssignedCalibrationCheckpoint
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from empirical_lawhood.infrastructure.candidate_payloads import (
    ExternalCandidatePayloadPlane,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import (
    ArtifactProfile,
    ArtifactWriteRequest,
    CanonicalTaskReceipt,
)
from empirical_lawhood.runtime.execution import DependencyReceiptBinding


@dataclass(frozen=True)
class SyntheticSourceInventory(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/synthetic-source-inventory'
    root_ids: tuple[str, ...]
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"


def publish_runtime(
    plane,
    source,
    *,
    context_id="synthetic.finite-runtime",
    source_sha256=None,
    wrong_subject=False,
):
    from empirical_lawhood.adapters.composition.finite_response_law.consumer_ports import FiniteResponseLawRuntimeContext
    from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore
    from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority

    def identity(name):
        return ObjectIdentity(
            name, 'empirical-lawhood/test/synthetic', "1.0.0", "a" * 64
        )

    issued, approval = identity("synthetic.issue"), identity("synthetic.approval")
    grant = StudyOperationAuthority(
        f"{context_id}.execution",
        StudyAuthorityKind.EXPERIMENT_EXECUTION,
        identity("synthetic.other-issue") if wrong_subject else issued,
        approval,
        identity("synthetic.owner"),
        "operator.execution-service",
        "synthetic.scope",
        None,
        None,
        False,
        False,
        True,
        False,
        False,
        "2026-09-30T00:00:00Z",
        None,
        OutcomeAccess.EVALUATION_SEALED,
    )
    ExternalStudyOperationAuthorityStore(plane).persist(grant)
    context = FiniteResponseLawRuntimeContext(
        context_id,
        "synthetic-run",
        source.fingerprint() if source_sha256 is None else source_sha256,
        issued,
        approval,
        ObjectIdentity.from_record(grant.authority_id, grant),
        grant.grantee_id,
        "synthetic.compiler",
    )
    plane.write(
        ArtifactWriteRequest(
            logical_artifact_id=context.context_id,
            relative_path=f"operator/finite-response-law-contexts/{context.context_id}.json",
            payload_schema=context.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/json",
            publication_scope_id="synthetic.context",
            publication_scope_relative_root="operator/finite-response-law-contexts",
            payload=context.canonical_bytes(),
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            parent_visibility_ceilings=(),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            minimum_free_bytes=1,
        )
    )
    return ObjectIdentity.from_record(context.context_id, context)


def artifact(name, schema, raw, media="application/json"):
    return ArtifactIdentity(
        name, "synthetic-development", schema, sha256(raw).hexdigest(), media, len(raw)
    )


def synthetic_native(source=None):
    source = _source("calibration") if source is None else source
    projection_type, evaluation_type, word_type, view_type, completion_type = (
        native_method_types(source)
    )
    projection = projection_type(source)
    projection_id = ObjectIdentity.from_record(projection.config_id, projection)
    result_type = native_result_type(source)

    def identity(name, kind):
        return ObjectIdentity(name, kind.SCHEMA, kind.VERSION, "0" * 64)

    views = []
    tasks = native_invocations(source)
    hold = PreparedForceWord(D(0), 0, 0)
    for root in source.roots:
        local = tuple(t for t in tasks if t.root == root)
        words = []
        for task in local:
            if task.phase != "future":
                continue
            outputs = [D(0)] * 7
            if task.word.sign:
                outputs[task.word.direction_index] = task.word.sign * (
                    D(".08") if task.word.magnitude == 8 else D(".14")
                )
            words.append(
                word_type(
                    task,
                    identity(task.task_id + ".result", result_type),
                    identity(replace(task, word=hold).task_id + ".result", result_type),
                    True,
                    D(0),
                    tuple(outputs),
                )
            )
        for refinement in (1, 2):
            interfaces = []
            for phase, cutoff in (("prefix", 4096), ("parent", 4368)):
                task = next(t for t in local if t.phase == phase)
                interfaces.append(
                    FiniteResponseLawAssignedCalibrationInterface(
                        identity(
                            f"{task.task_id}.r{refinement}.checkpoint",
                            FiniteResponseLawAssignedCalibrationCheckpoint,
                        ),
                        REFERENCE_INSTRUMENT.identity,
                        cutoff,
                        (D(0),) * 24,
                    )
                )
            views.append(
                view_type(
                    projection_id,
                    root,
                    refinement,
                    tuple(identity(t.task_id + ".result", result_type) for t in local),
                    tuple(
                        (
                            t.task_id,
                            (t.clocks[1] - t.clocks[0]) * refinement,
                            "COMPLETE",
                        )
                        for t in local
                    ),
                    tuple(words),
                    *interfaces,
                    D(0),
                )
            )
    return completion_type(evaluation_type(projection), tuple(views))


def development():
    recipe = {"dimension": 24, "family": "SNAPSHOT_REFERENCE_SKETCH_AND_RATE", "gamma": 1.0, "ridge": 10.0}
    report = json.dumps(
        {
            "recipe": recipe,
            "direct_recipe": recipe,
            "training_roots": list(range(48)),
            "held_roots": [],
            "normalizer_prefix_root_ids": list(range(48)),
            "normalizer_handoff_rows": [[r, p] for r in range(48) for p in range(5)],
        }
    ).encode()
    target = np.zeros((4, 8), dtype=np.float64)
    target[0, 0] = target[1, 1] = 0.08
    target[2, 0] = target[3, 1] = 0.14
    lower = np.zeros((25, 32), dtype=np.float64)
    lower[-1] = target.ravel()
    direct = np.zeros((5, 25, 32), dtype=np.float64)
    direct[:, -1] = target.ravel()
    arrays = {
        "point_n0_center": np.zeros(24),
        "point_n0_scale": np.ones(24),
        "point_nh_center": np.zeros(24),
        "point_nh_scale": np.ones(24),
        "point_lower": lower,
        "point_lower_mean": target.ravel(),
        "point_upper": np.zeros((5, 25, 24)),
        "point_cached": np.tile(target.ravel(), (5, 1)),
        "point_direct": direct,
    }
    for boundary, shape in (
        ("lower", (25, 1)),
        ("composed", (5, 25, 1)),
        ("cached", (5,)),
        ("direct", (5, 25, 1)),
    ):
        arrays.update(
            {
                f"{boundary}_base_scale": np.ones((4, 8)),
                f"{boundary}_log_multiplier": np.zeros(shape),
                f"{boundary}_log_multiplier_target": np.zeros((48, 5)),
                f"{boundary}_log_multiplier_eligible": np.ones(48, dtype=np.bool_),
                f"{boundary}_base_scale_root_counts": np.ones((4, 8), dtype=np.int64),
            }
        )
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return report, stream.getvalue()


def qualified_parent(plane):
    native = synthetic_native()
    native_raw = native.canonical_bytes()
    native_artifact = artifact("synthetic.native", native.SCHEMA, native_raw)
    written = plane.write(
        ArtifactWriteRequest(
            logical_artifact_id=native_artifact.artifact_id,
            relative_path="synthetic/native.json",
            payload_schema=native.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/json",
            publication_scope_id="synthetic.native",
            publication_scope_relative_root="synthetic",
            payload=native_raw,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            parent_visibility_ceilings=(),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            minimum_free_bytes=1,
        )
    )
    receipt = CanonicalTaskReceipt(
        "synthetic.native.receipt",
        "synthetic-run",
        "finite-response-law.native-evaluate",
        "attempt-001",
        "0" * 40,
        (),
        (written.materialization,),
        (written.logical,),
        (),
        OperationalStatus.SUCCEEDED,
        (),
    )
    receipt_artifact = artifact(
        receipt.receipt_id, receipt.SCHEMA, receipt.canonical_bytes()
    )
    receipt_id = ObjectIdentity.from_record(receipt.receipt_id, receipt)
    report_raw, coeff_raw = development()
    report_artifact = artifact(
        "synthetic.development.report", mr.REPORT_SCHEMA, report_raw
    )
    coeff_artifact = artifact(
        "synthetic.development.coefficients",
        mr.COEFFICIENT_SCHEMA,
        coeff_raw,
        "application/x-npz",
    )
    manifest = artifact("synthetic.development.manifest", mr.MANIFEST_SCHEMA, b"{}")
    config = mr.FiniteResponseLawAssignedCalibrationMethodConfig(
        "finite-response-law.calibration.method-config",
        native.config.projection.native_spec,
        native_artifact,
        receipt_artifact,
        receipt_id,
        report_artifact,
        coeff_artifact,
        manifest,
    )
    operands = calibration_operands(
        native_bytes=native_raw,
        native_artifact=native_artifact,
        native_receipt=receipt,
        expected_native_receipt=receipt_id,
        source=config.native_source,
        development_report_bytes=report_raw,
        coefficient_bytes=coeff_raw,
        development_report=report_artifact,
        coefficients=coeff_artifact,
        development_manifest=manifest,
    )
    readout_raw = json.dumps(operands.readout, sort_keys=True).encode()
    report = mr.FiniteResponseLawAssignedCalibrationReport(
        mr.CALIBRATE + ".result",
        ObjectIdentity.from_record(config.config_id, config),
        operands.prediction_artifact,
        artifact("synthetic.calibration.readout", mr.READOUT_SCHEMA, readout_raw),
        operands.boundaries,
        tuple(
            (b, bool(operands.readout["usability"][b]["joint_decision_opportunity"]))
            for b in ("lower", "composed", "cached", "direct")
        ),
    )
    payloads = ExternalCandidatePayloadPlane(
        plane,
        "synthetic/laws",
        "synthetic.laws",
        VisibilityCeiling.PROSPECTIVE,
        OutcomeAccess.EVALUATOR_REVEAL,
        (),
        1,
    )
    result = qualify_calibration(
        config=config,
        report=report,
        operands=operands,
        development_report_bytes=report_raw,
        coefficient_bytes=coeff_raw,
        system=qualification_system(config.native_source),
        manifest=METHOD_CAPABILITY,
        payload_plane=payloads,
        dependency=DependencyReceiptBinding(
            "synthetic.calibration.receipt",
            mr.CALIBRATE,
            ("synthetic.calibration", "synthetic.predictions"),
        ),
        calibration_materialization_id="synthetic.calibration",
        prediction_materialization_id="synthetic.predictions",
    )
    assert result.eligible_for_prospective_evaluation
    return result, native, receipt, native_artifact, receipt_artifact


def retained_continuation(source):
    """Declare the original 64 prefixes; raw fixture members cannot execute.

    The public input checker authenticates inventories and constructs ports.
    Native workers must decode their own task/pair records if execution is ever
    attempted; these explicitly fictional bytes are not native measurements.
    """
    from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_retention import FiniteResponseLawEvaluationRetention
    from empirical_lawhood.adapters.simulators.finite_response_law.native_artifact import NATIVE_PAIR_SCHEMA
    from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult
    from empirical_lawhood.planning.source_qualification import ProspectiveRetainedPredecessor
    from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
    from empirical_lawhood.kernel.serialization import (
        CanonicalRecord,
        canonical_json_bytes,
    )
    from dataclasses import dataclass
    from typing import ClassVar

    @dataclass(frozen=True)
    class DonorCloseout(CanonicalRecord):
        SCHEMA: ClassVar[str] = 'empirical-lawhood/document/json'
        role: str
        original_cohort: str

    closeout = DonorCloseout("SYNTHETIC_NONPROMOTABLE", source.fingerprint())
    closeout_artifact = artifact(
        "synthetic.donor-closeout", closeout.SCHEMA, closeout.canonical_bytes()
    )
    members = []
    prefixes = []
    for invocation in native_invocations(source):
        if invocation.phase != "prefix":
            continue
        stem = invocation.task_id
        outputs = []
        for suffix, schema, media in (
            ("result", FiniteResponseLawAssignedEvaluationTaskResult.SCHEMA, "application/json"),
            ("stage", LinkedCampaignStageEnvelope.SCHEMA, "application/json"),
            ("native", NATIVE_PAIR_SCHEMA, "application/json"),
            ("receipt", CanonicalTaskReceipt.SCHEMA, "application/json"),
        ):
            name = f"{stem}.{suffix}"
            raw = canonical_json_bytes(
                {
                    "schema": schema,
                    "version": "1.0.0",
                    "value": {"synthetic_nonexecuting_fixture": name},
                }
            )
            identity = artifact(name, schema, raw, media)
            members.append((identity, raw))
            outputs.append(identity)
        prefixes.append(
            ProspectiveRetainedPredecessor(
                stem,
                invocation.root.physical_unit_id,
                "finite-response-law.reference-clock",
                D(invocation.clocks[1]),
                tuple(f"{invocation.root.root_id}.flh-project.r{v}" for v in (1, 2)),
                tuple(sorted(outputs[:3], key=lambda a: a.artifact_id)),
                outputs[3],
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            )
        )
    return (
        closeout,
        FiniteResponseLawEvaluationRetention(
            source, "synthetic-donor-run", closeout_artifact, tuple(prefixes)
        ),
        tuple(members),
    )
