# SPDX-License-Identifier: MPL-2.0

"""Independent scoring and saved-input checks: both native views, all assigned 64 roots, at
least 48 contacted, five contrasts, fixed 20,000 root bootstrap draws and 99 percent
one-sided limits. This verifies supplied operands; it issues no scientific adjudication.

Callers authenticate retained inputs and supply analysis authority separately.
Historical operators and their custody remain external. No storage effects.
"""

from __future__ import annotations
from decimal import Decimal as D
from math import isfinite, sqrt
from typing import cast
import numpy as np
from empirical_lawhood.adapters.methods.reactor_regime_response.measured_panel import CONTEXTS, measured_contexts
from empirical_lawhood.adapters.methods.reactor_regime_response.model_records import RegimeFitPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.nomination_records import RegimeNominationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.qualification_pipeline import ComparisonRoot, ReactorRegimeContrastResult
from empirical_lawhood.adapters.methods.reactor_regime_response.records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.runtime.plans import ExecutionTask
from collections.abc import Mapping
from hashlib import sha256
from typing import TypeVar
from scipy.stats import beta
from .calibration_pipeline import RegimeCalibrationPackage
from .config import ROOTS, ReactorRegimeResponseDesign
from .discovery_diagnostics import RegimeDiscoveryConfirmation
from .confirmation_pipeline import selected_operand
from .law_terminal import RegimeJointLawResult
from .qualification_pipeline import RegimeQualificationPackage
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.plans import CandidateExecutionPlan, ExecutionPlan

RecordT = TypeVar("RecordT", bound=CanonicalRecord)


CONTRASTS = ("R", "H", "I_READ", "I_EXCITATION", "I_USE")
ARM_INDICES = ((0, 1), (2, 3), (6, 7), (6, 7, 4, 5), (8, 9))
CONTACT_INDICES = (0, 0, 1, 1, 2)
BASELINE = ("early", "middle", "late", "prepared_t0")


def _optional_float(value: D | None) -> float | None:
    return None if value is None else float(value)


def _same_number(actual: D | None, expected: float | None) -> bool:
    return actual == (None if expected is None else D(repr(expected)))


def _precision(
    operand: object, *, coverage: bool = False
) -> tuple[bool, float | None, float | None, float | None, float | None]:
    """Score both native views with the fixed T/cooling padding, without producer scoring."""
    from empirical_lawhood.adapters.methods.reactor_regime_response.qualification_math import SelectedRootOperand

    assert isinstance(operand, SelectedRootOperand)
    chart, forecast = operand.chart, operand.forecast
    contacted = bool(
        chart.valid
        and forecast is not None
        and chart.nominal_peak_K is not None
        and chart.refined_peak_K is not None
        and chart.nominal_cooling_K is not None
        and chart.refined_cooling_K is not None
        and all(word.supported and word.projection_valid for word in forecast)
    )
    if not contacted or forecast is None:
        return False, None, None, None, None
    assert chart.nominal_peak_K is not None and chart.refined_peak_K is not None
    assert chart.nominal_cooling_K is not None and chart.refined_cooling_K is not None
    absolute_errors = []
    cooling_errors = []
    cooling_scores = []
    for peaks, coolings in (
        (chart.nominal_peak_K, chart.nominal_cooling_K),
        (chart.refined_peak_K, chart.refined_cooling_K),
    ):
        for word, measured_peak, measured_cooling in zip(
            forecast, peaks, coolings, strict=True
        ):
            t_error = abs(word.predicted_peak_K - measured_peak)
            c_error = abs(word.predicted_cooling_K - measured_cooling)
            absolute_errors.append(t_error)
            cooling_errors.append(c_error)
            cooling_scores.append(
                max(c_error - (1e-6 if coverage else 0), 0)
                / (0.00005 + 0.5 * abs(word.predicted_cooling_K))
            )
    if not all(
        isfinite(value)
        for value in (*absolute_errors, *cooling_errors, *cooling_scores)
    ):
        return False, None, None, None, None
    return (
        True,
        max(
            max(error - (0.01 if coverage else 0), 0) / 0.25
            for error in absolute_errors
        ),
        max(cooling_scores),
        max(absolute_errors),
        max(cooling_errors),
    )


def _nominated_comparator(
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    *,
    local: bool,
) -> str | None:
    scores = {item.model_id: item for item in nomination.coefficient_scores}
    if set(scores) != {item.model_id for item in fit.coefficients}:
        raise ValueError("C saved rival roster differs from B fit")
    choices = []
    for item in fit.coefficients:
        score = scores[item.model_id]
        if (
            item.mask != "history"
            or (item.family == "local") != local
            or item.fitted is None
            or score.mean_loss_K2 is None
            or score.reasons
        ):
            continue
        frozen = item.fitted
        choices.append(
            (
                item.model_id,
                float(score.mean_loss_K2),
                len(frozen.leaves) if local else 1,
                sum(len(leaf.operator) for leaf in frozen.leaves)
                if local
                else len(frozen.operator),
                float(item.penalty),
            )
        )
    if not choices:
        return None
    best_loss = min(item[1] for item in choices)
    ties = (item for item in choices if item[1] <= 1.01 * best_loss)
    return min(ties, key=lambda item: (item[2], item[3], -item[4], item[0]))[0]


def _information_arm_ids(
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    pair_id: str,
) -> dict[str, str]:
    nominated = next(item for item in nomination.information if item.pair_id == pair_id)
    if nominated.spec_id is None:
        return {}
    matching = tuple(
        item
        for item in fit.information
        if (
            item.pair_id == pair_id
            and f"{item.family}.l{float(item.penalty):.6g}.r"
            f"{float(item.multiplier) if item.multiplier is not None else 'none'}"
            == nominated.spec_id
        )
    )
    if len(matching) != 1:
        raise ValueError(f"C {pair_id} changed its frozen information pair")
    return {
        arm: f"information.{model.fit_id}"
        for arm, model in zip(matching[0].arms, matching[0].fits, strict=True)
    }


def _comparison_from_saved(
    row: tuple[
        RegimeCausalPreparation,
        RegimePrivatePreparation,
        RegimePredictionSeal,
        RegimeAssayPanel,
    ],
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
) -> ComparisonRoot:
    """Reconstruct ten root losses from sealed forecasts and measured arrays."""
    causal, _, seal, assay = row
    charts = {item.context: item for item in measured_contexts(causal, assay)}
    rival = _nominated_comparator(fit, nomination, local=False)
    local = _nominated_comparator(fit, nomination, local=True)
    h = _information_arm_ids(fit, nomination, "S_H")
    i = _information_arm_ids(fit, nomination, "S_I")
    q = _information_arm_ids(fit, nomination, "S_Q")
    arrays = seal.predictions.unpack()
    ids = {name: index for index, name in enumerate(seal.candidate_ids)}

    def loss(model_id: str | None, input_name: str, target_name: str) -> float | None:
        target = charts[target_name]
        if (
            model_id is None
            or model_id not in ids
            or target.nominal_mass_kg is None
            or target.nominal_cooling_K is None
        ):
            return None
        model_index = ids[model_id]
        column = CONTEXTS.index(input_name)
        if not arrays["present"][model_index, column]:
            return None
        coefficient = float(arrays["forecast"][model_index, column])
        residuals = tuple(
            coefficient * mass - measured
            for mass, measured in zip(
                target.nominal_mass_kg[1:],
                target.nominal_cooling_K[1:],
                strict=True,
            )
        )
        return float(np.mean(np.square(np.asarray(residuals, dtype=np.float64))))

    def baseline(model_id: str | None) -> float | None:
        values = tuple(loss(model_id, name, name) for name in BASELINE)
        return (
            None
            if any(value is None for value in values)
            else float(
                np.mean(tuple(float(value) for value in values if value is not None))
            )
        )

    def contact(names: tuple[str, ...]) -> bool:
        return all(
            charts[name].callback is not None
            and charts[name].nominal_mass_kg is not None
            and charts[name].nominal_cooling_K is not None
            for name in names
        )

    flags = (
        all(charts[name].valid for name in BASELINE),
        all(charts[name].valid for name in BASELINE),
        all(charts[name].valid for name in ("prepared_t0", "c_q", "p_q")),
        all(charts[name].valid for name in ("prepared_t0", "c_q", "p_q")),
        charts["p_q"].valid,
    )
    losses = (
        baseline(None if rival is None else f"coefficient.{rival}".lower()),
        baseline(None if local is None else f"coefficient.{local}".lower()),
        baseline(h.get("current")),
        baseline(h.get("history")),
        loss(i.get("c_masked"), "c_q", "prepared_t0"),
        loss(i.get("c_full"), "c_q", "prepared_t0"),
        loss(i.get("p_masked"), "p_q", "prepared_t0"),
        loss(i.get("p_full"), "p_q", "prepared_t0"),
        loss(q.get("p_masked"), "p_q", "p_q"),
        loss(q.get("p_full"), "p_q", "p_q"),
    )
    return ComparisonRoot(
        causal.root,
        (
            contact(BASELINE),
            contact(("prepared_t0", "c_q", "p_q")),
            contact(("p_q",)),
            contact(("prepared_t0", "p_q")),
        ),
        flags,
        True,
        tuple(None if value is None else D(repr(value)) for value in losses),
    )


def _contrast(rows: tuple[ComparisonRoot, ...], index: int) -> dict[str, object]:
    contact_index = CONTACT_INDICES[index]
    arms = ARM_INDICES[index]
    contacted = tuple(row for row in rows if row.contacts[contact_index])
    failed = sum(
        not row.source_validity[index]
        or not row.sealed_before_labels
        or any(row.losses_K2[arm] is None for arm in arms)
        for row in contacted
    )
    valid = tuple(
        row
        for row in contacted
        if row.source_validity[index]
        and row.sealed_before_labels
        and all(row.losses_K2[arm] is not None for arm in arms)
    )
    reasons = set()
    if len(contacted) < 48:
        reasons.add("INSUFFICIENT_COMPLETE_CONTEXT_CONTACT")
    if failed:
        reasons.add("MODEL_OR_SEAL_FAILURE_ON_CONTACTED_ROOT")
    result: dict[str, object] = {
        "contrast_id": CONTRASTS[index],
        "assigned_roots": 64,
        "complete_contact_roots": len(contacted),
        "model_failure_roots": failed,
    }
    if not valid:
        reasons.add("NO_EVALUABLE_MATCHED_ROOTS")
        return {
            **result,
            "mean": None,
            "lower": None,
            "upper": None,
            "rmse": (),
            "ratio": None,
            "verdict": "UNEVALUABLE",
            "reasons": tuple(sorted(reasons)),
        }
    values = np.asarray(
        [[float(row.losses_K2[arm] or D(0)) for arm in arms] for row in valid],
        dtype=np.float64,
    )
    if index == 3:
        advantage = (
            (values[:, 0] - values[:, 1])
            - (values[:, 2] - values[:, 3])
            - 0.2 * values[:, 0]
        )
    else:
        advantage = 0.8 * values[:, 0] - values[:, 1]
    rng = np.random.default_rng(20260924)
    draws = rng.integers(0, len(advantage), size=(20000, len(advantage)))
    sampled_means = np.mean(advantage[draws], axis=1)
    losses = np.mean(values, axis=0)
    lower = float(np.quantile(sampled_means, 0.01))
    upper = float(np.quantile(sampled_means, 0.99))
    verdict = (
        "UNEVALUABLE"
        if reasons
        else "SUPPORTED_20_PERCENT_ADVANTAGE"
        if lower > 0
        else "OPPOSED_20_PERCENT_ADVANTAGE"
        if upper < 0
        else "UNRESOLVED_20_PERCENT_ADVANTAGE"
    )
    return {
        **result,
        "mean": float(np.mean(advantage)),
        "lower": lower,
        "upper": upper,
        "rmse": tuple(sqrt(float(loss)) for loss in losses),
        "ratio": float(losses[1] / losses[0]) if losses[0] > 0 else None,
        "verdict": verdict,
        "reasons": tuple(sorted(reasons)),
    }


def _compare_contrast(produced: ReactorRegimeContrastResult, checked: dict[str, object]) -> None:
    if any(
        getattr(produced, key) != checked[key]
        for key in (
            "contrast_id",
            "assigned_roots",
            "complete_contact_roots",
            "model_failure_roots",
            "verdict",
            "reasons",
        )
    ):
        raise ValueError(
            f"C {produced.contrast_id} changed its independent root census/verdict"
        )
    for field, name in (
        ("mean_advantage_K2", "mean"),
        ("lower_99_one_sided_K2", "lower"),
        ("upper_99_one_sided_K2", "upper"),
        ("arm_loss_ratio", "ratio"),
    ):
        if not _same_number(
            getattr(produced, field), cast(float | None, checked[name])
        ):
            raise ValueError(f"C {produced.contrast_id} changed independent {field}")
    expected_rmse = tuple(
        D(repr(value)) for value in cast(tuple[float, ...], checked["rmse"])
    )
    if produced.arm_rmse_K != expected_rmse:
        raise ValueError(f"C {produced.contrast_id} changed independent arm RMSE")


def _terminal_input_ids(
    task: ExecutionTask,
    dependencies: tuple[CanonicalTaskReceipt, ...],
    config: ArtifactManifest,
) -> tuple[str, ...]:
    """Select declared edges, not every output made by their producers."""
    if tuple(receipt.task_id for receipt in dependencies) != task.dependency_task_ids:
        raise ValueError("C terminal changed its dependency receipt census")
    if (
        len(task.external_inputs) != 1
        or task.external_inputs[0].logical_artifact_id
        != task.capability.config.artifact_id
        or task.external_inputs[0].expected_content_sha256
        != task.capability.config.content_sha256
        or task.external_inputs[0].expected_payload_schema
        != task.capability.config.config_schema
        or config.logical.logical_artifact_id != task.capability.config.artifact_id
        or config.logical.content_sha256 != task.capability.config.content_sha256
        or config.logical.payload_schema != task.capability.config.config_schema
    ):
        raise ValueError("C terminal changed its declared configuration")
    parents = {receipt.task_id: receipt for receipt in dependencies}
    selected = [config.materialization.materialization_id]
    for edge in task.scientific_inputs:
        if edge.producer_task_id not in parents:
            raise ValueError("C terminal lacks a declared scientific parent")
        parent = parents[edge.producer_task_id]
        outputs = [
            value
            for value in parent.output_materializations
            if value.logical_artifact_id == edge.operational_logical_artifact_id
        ]
        logical = [
            value
            for value in parent.output_logical_artifacts
            if value.logical_artifact_id == edge.operational_logical_artifact_id
            and value.payload_schema == edge.payload_schema
        ]
        if len(outputs) != 1 or len(logical) != 1:
            raise ValueError("C terminal lacks its declared scientific output")
        selected.append(outputs[0].materialization_id)
    if len(set(selected)) != len(selected):
        raise ValueError("C terminal repeats an input materialization")
    return tuple(sorted(selected))


def verify_saved_qualification(
    *,
    plan: CandidateExecutionPlan | ExecutionPlan,
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    records: Mapping[tuple[str, str], CanonicalRecord],
    matched_receipts: Mapping[str, CanonicalTaskReceipt],
    config: ArtifactManifest,
    config_payload: bytes,
    marker: bytes,
    artifact_storage_root_id: str,
    retained_tasks: tuple[ExecutionTask, ...] = (),
) -> dict[str, object]:
    """Recheck the complete original C science and terminal ancestry.

    Authenticate every supplied record's canonical bytes, declared output,
    publication marker and successful exact receipt before entry. For retained
    C, also authenticate the original 160 completed tasks, their frozen source
    and unchanged exposure ceiling against the 132-task continuation. Supply
    its successful receipts in matched_receipts; never infer those proofs from
    task labels. The config payload/manifest/marker are checked here as well.
    This library performs the original outcome-visible scoring and bootstrap;
    independent reveal/analysis authority remains a caller prerequisite. It
    neither acquires outcomes nor issues or publishes scientific adjudication.
    """
    tasks = {task.task_id: task for task in plan.tasks}
    retained_ids = {task.task_id for task in retained_tasks}
    if retained_tasks:
        if (
            len(retained_ids) != 160
            or len(tasks) != 132
            or retained_ids.intersection(tasks)
        ):
            raise ValueError("C checker mixes retained and newly scheduled tasks")
        tasks.update({task.task_id: task for task in retained_tasks})
    if len(tasks) != 292:
        raise ValueError("C checker lacks its 292-task evidence denominator")
    namespaces = {
        output.relative_path.split("/")[1]
        for task in plan.tasks
        for output in task.outputs
    }
    if len(namespaces) != 1:
        raise ValueError("C checker mixes task run namespaces")
    run_id = next(iter(namespaces))
    checked: set[str] = set()

    def read(task_id: str, record_type: type[RecordT]) -> RecordT:
        task = tasks[task_id]
        outputs = [
            output
            for output in task.outputs
            if output.payload_schema == record_type.SCHEMA
        ]
        if len(outputs) != 1:
            raise ValueError(f"C {task_id} lacks one declared {record_type.SCHEMA}")
        record = records[(task_id, record_type.SCHEMA)]
        if not isinstance(record, record_type):
            raise ValueError(f"C {task_id} changed its saved operand type")
        if (
            len(record.canonical_bytes())
            > task.capability.requested_resources.output_bytes
        ):
            raise ValueError(f"C {task_id} exceeds its compiled output budget")
        checked.add(task_id)
        return record

    calibration = read("regime.calibration", RegimeCalibrationPackage)
    qualification = read("regime.qualification", RegimeQualificationPackage)
    nomination_id = ObjectIdentity.from_record(nomination.package_id, nomination)
    if nomination.fit_package != ObjectIdentity.from_record(fit.package_id, fit):
        raise ValueError("C checker changed the authenticated frozen B fit")
    if (
        calibration.nomination != nomination_id
        or qualification.nomination != nomination_id
    ):
        raise ValueError("C checker lacks its authenticated frozen B nomination")
    if (
        calibration.fit_package != nomination.fit_package
        or qualification.fit_package != nomination.fit_package
        or qualification.calibration
        != ObjectIdentity.from_record(calibration.package_id, calibration)
    ):
        raise ValueError("C checker changed the B fit/C calibration ancestry")
    expected_calibration = tuple(
        (root for root, role, _, _ in ROOTS if role == "calibration")
    )
    expected_qualification = tuple(
        (root for root, role, _, _ in ROOTS if role == "qualification")
    )
    contacted = []
    causal_ids: dict[str, list[ObjectIdentity]] = {
        "calibration": [],
        "qualification": [],
    }
    assay_ids: dict[str, list[ObjectIdentity]] = {
        "calibration": [],
        "qualification": [],
    }
    for role, expected in (
        ("calibration", expected_calibration),
        ("qualification", expected_qualification),
    ):
        for index, name in enumerate(expected):
            causal = read(f"regime.prepare.{name}", RegimeCausalPreparation)
            private = read(f"regime.prepare.{name}", RegimePrivatePreparation)
            seal = read(f"regime.seal.{name}", RegimePredictionSeal)
            assay = read(f"regime.assay.{name}", RegimeAssayPanel)
            causal_id = ObjectIdentity.from_record(f"{name}.causal-preparation", causal)
            expected_seed = next((seed for root, _, _, seed in ROOTS if root == name))
            if (
                (causal.root, private.root, seal.root, assay.root) != (name,) * 4
                or (causal.role, private.role, seal.role, assay.role) != (role,) * 4
                or (causal.seed, private.seed, seal.seed, assay.seed)
                != (expected_seed,) * 4
                or (private.causal_preparation != causal_id)
                or (seal.causal_preparation != causal_id)
                or (seal.model_package != nomination_id)
                or (assay.causal_preparation != causal_id)
                or (
                    assay.private_preparation
                    != ObjectIdentity.from_record(
                        f"{name}.private-preparation", private
                    )
                )
                or (
                    assay.sealed_prediction
                    != ObjectIdentity.from_record(f"{name}.prediction-seal", seal)
                )
            ):
                raise ValueError(
                    f"C {name} changed its causal/prediction assay barrier"
                )
            causal_ids[role].append(causal_id)
            assay_ids[role].append(
                ObjectIdentity.from_record(f"{name}.assay-panel", assay)
            )
            operand = selected_operand(
                (causal, private, seal, assay),
                nomination,
                q_temperature=0
                if role == "calibration"
                else float(calibration.q_temperature or 0),
                q_cooling=0
                if role == "calibration"
                else float(calibration.q_cooling or 0),
            )
            score = (
                (False, None, None, None, None)
                if operand is None
                else _precision(operand, coverage=role == "qualification")
            )
            if role == "calibration":
                calibration_record = calibration.roots[index]
                if (
                    calibration_record.root != name
                    or calibration_record.contacted_valid != score[0]
                    or any(
                        (
                            not _same_number(actual, expected_score)
                            for actual, expected_score in zip(
                                (
                                    calibration_record.temperature_score,
                                    calibration_record.cooling_score,
                                    calibration_record.maximum_absolute_error_K,
                                    calibration_record.maximum_contrast_error_K,
                                ),
                                score[1:],
                                strict=True,
                            )
                        )
                    )
                ):
                    raise ValueError(
                        f"C {name} differs from independent calibration maxima"
                    )
                if score[0]:
                    contacted.append(score)
            else:
                qualification_record = qualification.roots[index]
                comparison = _comparison_from_saved(
                    (causal, private, seal, assay), fit, nomination
                )
                if qualification.comparison_roots[index] != comparison:
                    raise ValueError(
                        f"C {name} changed its ten saved comparison losses"
                    )
                if (
                    nomination.selected_route is None
                    or calibration.q_temperature is None
                    or calibration.q_cooling is None
                ):
                    expected_gate = (False, False, False, False, False)
                else:
                    covered = bool(
                        score[0]
                        and score[1] is not None
                        and (score[2] is not None)
                        and (score[1] <= float(calibration.q_temperature))
                        and (score[2] <= float(calibration.q_cooling))
                    )
                    safe = operand is not None and operand.preparation_safe
                    law = covered and calibration.precision_pass
                    expected_gate = (score[0], covered, law, safe, law and safe)
                if (
                    qualification_record.root != name
                    or (
                        qualification_record.contacted_valid,
                        qualification_record.interval_covered,
                        qualification_record.law_adequate,
                        qualification_record.preparation_safe,
                        qualification_record.joined,
                    )
                    != expected_gate
                ):
                    raise ValueError(
                        f"C {name} differs from independent two-view joint gate"
                    )
    if (
        calibration.calibration_causal != tuple(causal_ids["calibration"])
        or calibration.calibration_assays != tuple(assay_ids["calibration"])
        or qualification.qualification_causal != tuple(causal_ids["qualification"])
        or (qualification.qualification_assays != tuple(assay_ids["qualification"]))
    ):
        raise ValueError("C aggregate changed its receipt-backed root ancestry")
    if (
        tuple(
            (
                name
                for name, score in zip(
                    expected_calibration, calibration.roots, strict=True
                )
                if score.contacted_valid
            )
        )
        != calibration.contacted_roots
    ):
        raise ValueError("C calibration changed its complete-contact denominator")
    q_t = q_c = None
    if contacted:
        q_t = max((float(score[1]) for score in contacted if score[1] is not None))
        q_c = max((float(score[2]) for score in contacted if score[2] is not None))
    if not _same_number(calibration.q_temperature, q_t) or not _same_number(
        calibration.q_cooling, q_c
    ):
        raise ValueError("C calibration changed its fixed worst-root factors")
    expected_precision = bool(
        len(contacted) >= 29
        and q_t is not None
        and (q_c is not None)
        and (q_t <= 1)
        and (q_c <= 1)
        and (nomination.selected_route is not None)
    )
    if calibration.precision_pass != expected_precision:
        raise ValueError("C calibration changed its independent precision gate")
    law_successes = sum((row.law_adequate for row in qualification.roots))
    joint_successes = sum((row.joined for row in qualification.roots))
    law_lower = (
        0.0
        if law_successes == 0
        else float(beta.ppf(0.05, law_successes, 65 - law_successes))
    )
    joint_lower = (
        0.0
        if joint_successes == 0
        else float(beta.ppf(0.05, joint_successes, 65 - joint_successes))
    )
    release = calibration.precision_pass and joint_lower >= 0.9
    if (
        qualification.law_successes != law_successes
        or qualification.joint_successes != joint_successes
        or (not _same_number(qualification.law_lower_95, law_lower))
        or (not _same_number(qualification.joint_lower_95, joint_lower))
        or (qualification.law_pass != (calibration.precision_pass and law_lower >= 0.9))
        or (qualification.release_D != release)
    ):
        raise ValueError("C changed its independent all-assigned CP release")
    if (
        tuple((row.root for row in qualification.comparison_roots))
        != expected_qualification
    ):
        raise ValueError("C contrast root denominator differs")
    contrasts = tuple(
        (_contrast(qualification.comparison_roots, index) for index in range(5))
    )
    for contrast_record, checked_contrast in zip(
        qualification.contrasts, contrasts, strict=True
    ):
        _compare_contrast(contrast_record, checked_contrast)
    joint_law = read("regime.law-qualification", RegimeJointLawResult)
    discovery = read("regime.qualification", RegimeDiscoveryConfirmation)
    adjudication = read("regime.adjudication", ScientificAdjudicationRecord)
    if (
        joint_law.payload.fit_package != nomination.fit_package
        or joint_law.payload.nomination != nomination_id
        or joint_law.payload.calibration
        != ObjectIdentity.from_record(calibration.package_id, calibration)
        or (
            joint_law.payload.qualification
            != ObjectIdentity.from_record(qualification.package_id, qualification)
        )
        or (joint_law.payload.route != nomination.selected_route)
        or (joint_law.payload.q_temperature != calibration.q_temperature)
        or (joint_law.payload.q_cooling != calibration.q_cooling)
    ):
        raise ValueError("C local law result changed its selected B/C operands")
    terminal_task = tasks["regime.adjudication"]
    dependencies = tuple(
        (matched_receipts[task_id] for task_id in terminal_task.dependency_task_ids)
    )
    config_path = (
        f"runs/{run_id}/inputs/{terminal_task.capability.config.artifact_id}.bin"
    )
    publication = config.publication
    if publication is None:
        raise ValueError("C terminal configuration lacks its publication")
    design = decode_canonical_bytes(
        config_payload, ReactorRegimeResponseDesign, maximum_bytes=65536
    )
    if (
        config.materialization.storage_root_id != artifact_storage_root_id
        or config.materialization.relative_path != config_path
        or config.materialization.physical_sha256 != sha256(config_payload).hexdigest()
        or (config.materialization.size_bytes != len(config_payload))
        or (design.canonical_bytes() != config_payload)
        or (design.fingerprint() != config.logical.content_sha256)
        or (sha256(marker).hexdigest() != publication.commit_marker_sha256)
        or (len(marker) != publication.commit_marker_size_bytes)
    ):
        raise ValueError("C terminal configuration changed its saved publication")
    expected_inputs = _terminal_input_ids(terminal_task, dependencies, config)
    if (
        adjudication.run_id != run_id
        or adjudication.adjudication_task_id != terminal_task.task_id
        or adjudication.execution_plan
        != ObjectIdentity.from_record(plan.execution_plan_id, plan)
        or (adjudication.input_materialization_ids != expected_inputs)
        or (
            matched_receipts[terminal_task.task_id].input_materialization_ids
            != expected_inputs
        )
        or (
            adjudication.output_logical_artifact_ids
            != tuple((output.logical_artifact_id for output in terminal_task.outputs))
        )
        or (
            adjudication.required_receipt_ids
            != tuple(sorted((receipt.receipt_id for receipt in dependencies)))
        )
        or (
            discovery.qualification
            != ObjectIdentity.from_record(qualification.package_id, qualification)
        )
    ):
        raise ValueError("C terminal adjudication lost its exact receipt ancestry")
    if checked != set(tasks):
        raise ValueError("C checker did not authenticate every declared task output")
    return {
        "scope": "RECEIPT_BACKED_C_SAVED_OPERAND_CHECK_NOT_SCIENTIFIC_ADJUDICATION",
        "execution_plan_sha256": plan.fingerprint(),
        "receipt_backed_tasks_checked": len(checked),
        "newly_scheduled_tasks_checked": len(plan.tasks),
        "retained_tasks_checked": len(retained_ids),
        "calibration_roots": 32,
        "qualification_roots": 64,
        "calibration_contact_roots": len(contacted),
        "law_successes": law_successes,
        "joint_successes": joint_successes,
        "law_lower_95": law_lower,
        "joint_lower_95": joint_lower,
        "release_D": release,
        "contrasts": contrasts,
        "shared_helpers": (
            "typed context projection, preparation evidence and chart construction",
        ),
        "independent_operations": (
            "both-view fixed-padded absolute and cooling scores",
            "all-32 calibration maxima and all-64 law/preparation intersection",
            "ten per-root comparator losses from frozen B choices and sealed C forecasts",
            "five root-paired contrasts and 20000-draw bounds",
            "law qualification payload operand and terminal receipt ancestry",
        ),
    }
