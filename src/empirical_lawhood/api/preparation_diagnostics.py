"""Authenticated outcome-visible Q/E diagnostics; no native acquisition.

SPDX-License-Identifier: MPL-2.0
"""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from empirical_lawhood.adapters.methods.constructed_preparation_applicability import diagnostics
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.diagnostic_records import (
    PreparationDiagnosticInput, PreparationDiagnosticPhaseInput,
    PreparationDiagnosticReport, PreparationDiagnosticRootReceipts, diagnostic_json_payload,
)
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.measurement import measure_root, requests
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.readout import report, constructor_crossing
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import (
    ConstructedPreparationSelection, ConstructedPreparationLowerSeal,
    ConstructedPreparationMeasuredRoot, ConstructedPreparationReport,
)
from empirical_lawhood.adapters.methods.preparation_applicability.measurement import reduce_panel, service, validity
from empirical_lawhood.adapters.methods.preparation_applicability.result_access import PREPARATION_PROTECTED_OUTCOMES
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import (
    ConstructedPreparationPrefix, ConstructedPreparationParents, ConstructedPreparationPanel,
)
from empirical_lawhood.api.current_result_custody import require_current_result_receipt_outputs
from empirical_lawhood.api.preparation_inputs import _retained_output
from empirical_lawhood.api.preparation_result_access import authenticate_preparation_result_access
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes, MAX_ARTIFACT_MANIFEST_BYTES
from empirical_lawhood.infrastructure.recovery import ExternalRunRecoveryStore
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore, decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes, validate_relative_locator
from empirical_lawhood.kernel.status import OperationalStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import ArtifactWriteRequest, ArtifactProfile, ArtifactLineageParent, lineage_parent_sort_key
from empirical_lawhood.runtime.recovery import RunRecoveryIndex

TASKS = ("prefix", "select", "parents", "lower", "assay", "measure")
KINDS = (ConstructedPreparationPrefix, ConstructedPreparationSelection,
         ConstructedPreparationParents, ConstructedPreparationLowerSeal,
         ConstructedPreparationPanel, ConstructedPreparationMeasuredRoot)


def current_preparation_diagnostic_source_sha256():
    """Bind actual mathematical and input/API owners, independently of labels."""
    paths = (Path(__file__), Path(diagnostics.__file__),
             Path(__file__).parents[1] / "adapters/methods/constructed_preparation_applicability/diagnostic_records.py")
    return sha256(canonical_json_bytes(tuple(sha256(read_bounded_bytes(path, maximum_bytes=1024**2)).hexdigest()
                                            for path in paths))).hexdigest()


def _terminal(*, relative_path, plane):
    recovery = ExternalRunRecoveryStore(plane)
    index = recovery.read_index(relative_path, expected_schema=RunRecoveryIndex.SCHEMA)
    terminal = recovery.read_terminal(index)
    if terminal is None or terminal.operational_status != "SUCCEEDED":
        raise ValueError("diagnostic requires an actual complete successful terminal recovery closure")
    return index, terminal


def prepare_preparation_diagnostic_phase(*, selection, result_access, recovery_index_relative_path,
                                        artifact_writer):
    """Use the exact terminal's bound receipt IDs; never infer first/latest attempts."""
    if not isinstance(artifact_writer, ExternalArtifactPlane):
        raise TypeError("diagnostic requires actual guarded external storage")
    # The index and receipt are public control metadata. Authorize before the
    # outcome-bearing recovery terminal supplies the exact selected attempts.
    index = ExternalRunRecoveryStore(artifact_writer).read_index(
        recovery_index_relative_path, expected_schema=RunRecoveryIndex.SCHEMA)
    store = ExternalTaskReceiptStore(artifact_writer)
    report_task = next((task for task in index.tasks if task.task_id == "ap.report"), None)
    if report_task is None:
        raise ValueError("diagnostic recovery index omits its report task")
    report_receipts = tuple(receipt for attempt in report_task.attempts
        if (receipt := store.read(index.run_id, report_task.task_id, attempt.attempt_id)) is not None
        and receipt.operational_status is OperationalStatus.SUCCEEDED)
    if not report_receipts:
        raise ValueError("diagnostic requires a complete stored report receipt")
    for receipt in report_receipts:
        _report_access(result_access, receipt, artifact_writer)
    index, terminal = _terminal(relative_path=recovery_index_relative_path, plane=artifact_writer)
    if index.run_id != result_access.run_id:
        raise ValueError("diagnostic recovery closure changes its selected current run")
    bindings = {attempt.task_id: attempt for attempt in terminal.attempts}
    expected = {f"ap.{task}.{root}" for root in selection.stage.root_ids for task in TASKS} | {"ap.report", "ap.adjudication"}
    if expected != set(bindings) or expected != set(terminal.completed_task_ids):
        raise ValueError("diagnostic terminal omits assigned root or scientific terminal receipts")
    roots = tuple(PreparationDiagnosticRootReceipts(root, tuple(bindings[f"ap.{task}.{root}"].receipt_id for task in TASKS))
                  for root in selection.stage.root_ids)
    value = PreparationDiagnosticPhaseInput(selection, result_access, roots,
        bindings["ap.report"].receipt_id, bindings["ap.adjudication"].receipt_id,
        recovery_index_relative_path)
    _phase_metadata(value, artifact_writer=artifact_writer)
    return value


def _report_access(context, receipt, artifact_writer):
    if (receipt is None or receipt.task_id != "ap.report"
        or receipt.operational_status is not OperationalStatus.SUCCEEDED
        or not receipt.checks or len(receipt.output_logical_artifacts) != 1
        or receipt.output_logical_artifacts[0].payload_schema != ConstructedPreparationReport.SCHEMA
        or receipt.output_logical_artifacts[0].outcome_access not in PREPARATION_PROTECTED_OUTCOMES):
        raise PermissionError("diagnostic requires its exact protected report receipt")
    return authenticate_preparation_result_access(context=context, receipt=receipt,
        logical=receipt.output_logical_artifacts[0], writer=artifact_writer, include_package=True)


def _phase_access(phase, *, artifact_writer):
    """Check selected grants using control metadata before any protected bytes."""
    if not isinstance(artifact_writer, ExternalArtifactPlane):
        raise TypeError("diagnostic requires actual guarded external storage")
    receipt = ExternalTaskReceiptStore(artifact_writer).read_by_receipt_id(
        phase.result_access.run_id, "ap.report", phase.report_receipt_id)
    return _report_access(phase.result_access, receipt, artifact_writer)


def _phase_metadata(phase, *, artifact_writer, authenticated_access=None):
    """Authenticate controls/grants once, then every exact stored output contract.

    The projection is operation-local, with authority rechecked after analysis.
    Every canonical receipt remains authoritative; no catalog cache is trusted.
    """
    if not isinstance(artifact_writer, ExternalArtifactPlane):
        raise TypeError("diagnostic requires actual guarded external storage")
    execution, package = (authenticated_access if authenticated_access is not None
                          else _phase_access(phase, artifact_writer=artifact_writer))
    index, terminal = _terminal(relative_path=phase.recovery_index_relative_path, plane=artifact_writer)
    bindings = {attempt.task_id: attempt for attempt in terminal.attempts}
    store = ExternalTaskReceiptStore(artifact_writer)
    selected = (("ap.report", phase.report_receipt_id, ConstructedPreparationReport),
                ("ap.adjudication", phase.adjudication_receipt_id, ScientificAdjudicationRecord)) + tuple(
        (f"ap.{task}.{root.root_id}", receipt_id, kind)
        for root in phase.roots for task, receipt_id, kind in zip(TASKS, root.receipt_ids, KINDS, strict=True))
    expected = {task for task, _, _ in selected}
    if (index.run_id != phase.result_access.run_id
        or expected != {task.task_id for task in index.tasks}
        or expected != set(bindings) or expected != set(terminal.completed_task_ids)):
        raise ValueError("diagnostic recovery closure changes its run or complete frozen task census")
    receipts = {}
    for task, receipt_id, kind in selected:
        receipt = store.read_by_receipt_id(phase.result_access.run_id, task, receipt_id)
        binding = bindings.get(task)
        if (receipt is None or binding is None or binding.receipt_id != receipt_id
            or binding.receipt_sha256 != receipt.fingerprint()
            or receipt.operational_status is not OperationalStatus.SUCCEEDED
            or not receipt.checks or len(receipt.output_logical_artifacts) != 1
            or receipt.output_logical_artifacts[0].payload_schema != kind.SCHEMA):
            raise ValueError("diagnostic changes a complete exact terminal task/receipt/output binding")
        receipts[task] = receipt
    index.validate_plan(execution)
    for receipt in receipts.values():
        require_current_result_receipt_outputs(receipt=receipt, logical=receipt.output_logical_artifacts[0],
            execution_plan=execution, capability_prefixes=("constructed-preparation-applicability.",))
        if any(value.storage_root_id != artifact_writer.root.contract.storage_root_id
               for value in receipt.output_materializations):
            raise PermissionError("diagnostic substitutes an output storage root")
    return execution, receipts, package


def _phase_read(phase, *, artifact_writer, bootstrap_seed, common_request, authenticated_access=None):
    execution, receipts, package = _phase_metadata(phase, artifact_writer=artifact_writer,
        authenticated_access=authenticated_access)
    stage = phase.selection.stage
    manifests = []

    def checked_access(*, receipt, logical, **kwargs):
        require_current_result_receipt_outputs(receipt=receipt, logical=logical, execution_plan=execution,
            capability_prefixes=("constructed-preparation-applicability.",))
        if receipts.get(receipt.task_id) != receipt:
            raise PermissionError("diagnostic substitutes its scoped stored receipt")

    def read(task, kind):
        value, manifest = _retained_output(plane=artifact_writer, receipt=receipts[task], kind=kind,
            result_access=phase.result_access, access_authenticator=checked_access)
        manifests.append(manifest)
        return value

    from empirical_lawhood.api.constructed_preparation_applicability import _authenticate
    _authenticate(phase.selection, artifact_writer, artifact_writer)
    lower = next(value.record for value in phase.selection.upstream_records if value.key == "lower")
    summary = read("ap.report", ConstructedPreparationReport)
    terminal = read("ap.adjudication", ScientificAdjudicationRecord)
    if (not summary.complete or summary.root_ids != stage.root_ids
        or summary.phase != stage.phase or summary.lower_sha256 != stage.upstream[0].artifact.sha256
        or summary.design_sha256 != stage.design.fingerprint()
        or summary.source_sha256 != stage.source.object_fingerprint
        or summary.allocation_sha256 != stage.allocation.fingerprint()):
        raise ValueError("diagnostic requires the entire authenticated complete assigned phase")
    expected_status = ScientificStatus.SUPPORTED if summary.disposition in ("QUALIFIED", "SUPPORTED") else ScientificStatus.NOT_SUPPORTED
    if (terminal.run_id != phase.result_access.run_id or terminal.adjudication_task_id != "ap.adjudication"
        or terminal.scientific_status != expected_status
        or terminal.reason_codes != (f"BOUNDARY_{stage.phase}_{summary.disposition}",)):
        raise ValueError("diagnostic substitutes the original adjudication")
    from empirical_lawhood.runtime.adjudication import ScientificAdjudicationContext, AdjudicationEvaluability
    from empirical_lawhood.kernel.status import AdmissionStatus
    task = next(value for value in execution.tasks if value.task_id == "ap.adjudication")
    receipt = receipts[task.task_id]
    output = task.outputs[0]
    context = ScientificAdjudicationContext(ObjectIdentity.from_record(execution.execution_plan_id, execution),
        package.system.world.world_id, package.system.world.kind,
        ObjectIdentity.from_record(package.experiment.relation.relation_id, package.experiment.relation),
        package.experiment.independent_unit_id,
        tuple(ObjectIdentity.from_record(value.cutoff_id, value) for value in package.experiment.information_cutoffs),
        output.visibility_ceiling, output.outcome_access)
    if (not terminal.matches_context(context)
        or terminal.input_materialization_ids != receipt.input_materialization_ids
        or terminal.output_logical_artifact_ids != tuple(value.logical_artifact_id for value in receipt.output_logical_artifacts)
        or terminal.required_receipt_ids != tuple(sorted(receipts[value].receipt_id for value in task.dependency_task_ids))
        or terminal.evaluability is not AdjudicationEvaluability.EVALUABLE
        or terminal.admission_status is not AdmissionStatus.NOT_EVALUATED):
        raise ValueError("diagnostic substitutes the issued terminal scientific context or receipt census")
    measured, crossings, rows, prefix_histories = [], [], [], []
    for root, allocation in zip(phase.roots, stage.allocation.roots, strict=True):
        prefix, selected, parents, sealed, panel, retained = tuple(
            read(f"ap.{task}.{root.root_id}", kind) for task, kind in zip(TASKS, KINDS, strict=True))
        if (selected.selected_index != 1 or selected.policy_sha256 is not None
            or selected.prefix_sha256 != prefix.fingerprint()
            or parents.prefix_sha256 != prefix.fingerprint()
            or parents.selection_sha256 != selected.fingerprint()
            or panel.selection_sha256 != selected.fingerprint()
            or sealed.parents_sha256 != parents.fingerprint()
            or tuple(value for value in panel.phases if value.phase == "parent") != parents.phases
            or any(value.allocation != allocation for value in (*prefix.phases, *panel.phases))):
            raise ValueError("diagnostic changes the causal source/allocation/selection/parent chain")
        actual = measure_root(prefix, panel, lower, stage.phase, allocation.seed_for("requests"), sealed)
        if actual != retained or not actual.complete:
            raise ValueError("diagnostic owned measurement replay differs or omits a root")
        operands = reduce_panel(prefix, panel)
        if operands is None:
            raise ValueError("diagnostic cannot drop an assigned incomplete root")
        z, y, work = operands
        direction, requirement = requests(allocation.seed_for("requests"))
        choices = np.asarray(sealed.choices).reshape(3, 256, 2)
        row, _ = diagnostics.diagnose_root(lower=lower, z=z, y=y, work=work,
            direction=direction, requirement=requirement, choices=choices)
        _, _, _, mean, width, support = validity(lower, z, y, work)
        common_choices, common_success = service(mean, width, support, y, work, *common_request)
        row.update(root=root.root_id, common_E000_requests=diagnostics.counts(common_success, common_choices))
        rows.append(row)
        measured.append(actual)
        crossings.append(constructor_crossing(actual, lower))
        prefix_histories.append(sha256(prefix.phases[0].history_positions_base64.encode()).hexdigest())
    if report(stage, tuple(measured), tuple(crossings)) != summary:
        raise ValueError("diagnostic full owned readout replay differs from original adjudication")
    result = diagnostics.aggregate_rows(rows, bootstrap_seed=bootstrap_seed)
    result.update(original_report=summary.to_document(), original_adjudication=terminal.to_document(), rows=rows,
        common_request_counts=[row["common_E000_requests"]["joint_pairs"] for row in rows])
    if stage.phase == "E":
        false_roots = sum(row["actual"]["false_admissions"][1] > 0 for row in rows)
        valid_roots = summary.full_valid_counts[1]
        result["probability_bounds"] = diagnostics.root_probability_bounds(assigned=32, complete=32,
            false_admission_roots=false_roots, full_valid_roots=valid_roots)
        result["secondary_probability_bounds"] = {key: diagnostics.root_probability_bounds(
            assigned=32, complete=32, false_admission_roots=false_roots,
            full_valid_roots=sum(row["mask_H_N_P"][key][1] for row in rows))
            for key in ("primary", "both_views", "view_envelope")}
    else:
        result["probability_bounds"] = {"status": "QUALIFICATION_EXCLUDED_FROM_CONFIRMATION"}
    _phase_metadata(phase, artifact_writer=artifact_writer)  # current grant validity/closure after work
    return result, tuple(manifests), tuple(prefix_histories), summary


def analyze_preparation_diagnostics(config: PreparationDiagnosticInput, *, artifact_writer):
    """Replay all40 owned reductions before publishing separately identified diagnostics."""
    if config.analysis_source_sha256 != current_preparation_diagnostic_source_sha256():
        raise ValueError("diagnostic source differs from its explicit input binding")
    access = tuple(_phase_access(phase, artifact_writer=artifact_writer)
                   for phase in (config.qualification, config.evaluation))
    common = requests(config.evaluation.selection.stage.allocation.roots[0].seed_for("requests"))
    q, q_inputs, q_histories, q_summary = _phase_read(config.qualification, artifact_writer=artifact_writer,
        bootstrap_seed=config.bootstrap_seed_q, common_request=common, authenticated_access=access[0])
    current_q = next(value for value in config.evaluation.selection.upstream_records if value.key == "qualification")
    if current_q.record != q_summary or current_q.receipt.receipt_id != config.qualification.report_receipt_id:
        raise ValueError("diagnostic E substitutes its actual Q prerequisite")
    e, e_inputs, e_histories, _ = _phase_read(config.evaluation, artifact_writer=artifact_writer,
        bootstrap_seed=config.bootstrap_seed_e, common_request=common, authenticated_access=access[1])
    if len(set((*q_histories, *e_histories))) != 40:
        raise ValueError("diagnostic independent roots share retained prefix histories")
    inputs = tuple(sorted((*q_inputs, *e_inputs), key=lambda value: value.logical.logical_artifact_id))
    identities = tuple(ObjectIdentity.from_record(value.logical.logical_artifact_id, value.logical) for value in inputs)
    record = PreparationDiagnosticReport(config.analysis_id, ObjectIdentity.from_record(config.analysis_id, config),
        identities, diagnostic_json_payload({"Q": q, "E": e}))
    return record, inputs


def publish_preparation_diagnostics(config, *, artifact_writer, relative_root):
    """Publish a separate immutable outcome-visible product; original outcomes stay unchanged."""
    validate_relative_locator(relative_root)
    artifact_writer.root.verify(for_write=True)
    path = artifact_writer.root.resolve(relative_root, for_write=True)
    if path.exists():
        raise FileExistsError("diagnostic publication requires a fresh output root")
    record, inputs = analyze_preparation_diagnostics(config, artifact_writer=artifact_writer)
    parents = tuple(sorted((ArtifactLineageParent(ObjectIdentity.from_record(value.logical.logical_artifact_id, value.logical),
        value.logical.visibility_ceiling, value.logical.outcome_access) for value in inputs), key=lineage_parent_sort_key))
    return artifact_writer.write(ArtifactWriteRequest(f"{config.analysis_id}.report", f"{relative_root}/report.json",
        record.SCHEMA, ArtifactProfile.CANONICAL_JSON, "application/json", f"{config.analysis_id}.publication",
        relative_root, record.canonical_bytes(), VisibilityCeiling.OUTCOME_VISIBLE,
        tuple(value.visibility_ceiling for value in parents), OutcomeAccess.EVALUATION_REVEALED,
        lineage_parents=parents))


def read_preparation_diagnostics(config, *, artifact_writer, relative_root):
    """Reauthenticate current receipt/grant metadata; perform no numerical replay or native call."""
    if config.analysis_source_sha256 != current_preparation_diagnostic_source_sha256():
        raise ValueError("diagnostic source binding changed")
    access = tuple(_phase_access(phase, artifact_writer=artifact_writer)
                   for phase in (config.qualification, config.evaluation))
    expected_inputs = []
    for phase, authenticated in zip((config.qualification, config.evaluation), access, strict=True):
        _, receipts, _ = _phase_metadata(phase, artifact_writer=artifact_writer, authenticated_access=authenticated)
        expected_inputs.extend(ObjectIdentity.from_record(logical.logical_artifact_id, logical)
            for receipt in receipts.values() for logical in receipt.output_logical_artifacts)
    validate_relative_locator(relative_root)
    manifest = decode_artifact_manifest(read_bounded_bytes(artifact_writer.root.resolve(
        f"{relative_root}/report.json.manifest.json", for_write=False), maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES))
    if (manifest.logical.logical_artifact_id != f"{config.analysis_id}.report"
        or manifest.logical.payload_schema != PreparationDiagnosticReport.SCHEMA
        or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
        or manifest.logical.media_type != "application/json"
        or manifest.logical.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        or manifest.logical.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
        or manifest.materialization.relative_path != f"{relative_root}/report.json"
        or manifest.materialization.size_bytes > 4 * 1024**2
        or manifest.materialization.storage_root_id != artifact_writer.root.contract.storage_root_id):
        raise ValueError("diagnostic result changes its exact immutable output contract")
    artifact_writer.verify_manifest(manifest)
    value = decode_canonical_bytes(read_bounded_bytes(artifact_writer.root.resolve(
        manifest.materialization.relative_path, for_write=False), maximum_bytes=4*1024**2),
        PreparationDiagnosticReport, maximum_bytes=4*1024**2)
    if (value.analysis_id != config.analysis_id
        or value.fingerprint() != manifest.logical.content_sha256
        or value.input_identity != ObjectIdentity.from_record(config.analysis_id, config)
        or value.input_artifacts != tuple(sorted(expected_inputs, key=lambda item: item.object_id))
        or tuple(parent.identity for parent in manifest.logical.lineage_parents) != value.input_artifacts):
        raise ValueError("diagnostic result belongs to another input")
    return {"analysis_id": value.analysis_id, "report_sha256": value.fingerprint(),
            "ceiling": value.ceiling, "native_calls": 0, "phases": json.loads(value.data_payload)}
