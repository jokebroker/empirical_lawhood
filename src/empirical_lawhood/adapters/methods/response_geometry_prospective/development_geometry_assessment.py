"""Bind development's authenticated fitted models and validation observations to geometry."""

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord, require_sorted_unique_ids
from empirical_lawhood.adapters.methods.receiver_conditioned_io import (
    ControlledIOProductDisposition,
    PrefixPathwiseMargin,
)
from empirical_lawhood.adapters.methods.receiver_conditioned_io.contracts import decimal_from_float
from empirical_lawhood.adapters.methods.receiver_conditioned_io.member_construction import ControlledIOOperatorEvidence
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS

from .development_geometry import DEVELOPMENT_GEOMETRY_MAX_STAGE_BYTES, DEVELOPMENT_RECEIVER_UNIT, ResponseGeometryDevelopmentGeometryDesign, ResponseGeometryDevelopmentGeometryReference, ResponseGeometryDevelopmentGeometryWitnesses, response_geometry_development_geometry_design, response_geometry_development_geometry_steps, evaluate_response_geometry_development_geometry
from .development_models import REPRESENTATIONS, ResponseGeometryDevelopmentAffineModel, ResponseGeometryDevelopmentMeasuredView, response_geometry_development_endpoint_coefficients, endpoint_predictions, organize_response_geometry_development_views, root_series
from .development_projection import DEVELOPMENT_DATA_SCHEMA
from .development_records import DEVELOPMENT_FIT_SCHEMA, ResponseGeometryDevelopmentFitResult, read_response_geometry_development_fit, report_identities
from .development_terminal import ResponseGeometryDevelopmentQualificationConfig


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentGeometryCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-geometry-cell'
    cell_id: str
    representation: str
    parent: str
    invocation_offset: int
    design: ResponseGeometryDevelopmentGeometryDesign | None
    reference: ResponseGeometryDevelopmentGeometryReference | None
    evidence: ControlledIOOperatorEvidence | None
    result: ResponseGeometryDevelopmentGeometryWitnesses | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.representation not in REPRESENTATIONS
            or self.parent not in PARENTS
            or self.invocation_offset not in range(384, 513, 16)
        ):
            raise ValueError("development geometry cell is outside its declared roster")
        if (self.evidence is None) != (self.result is None) or (self.reference is None) != (
            self.result is None
        ):
            raise ValueError("development geometry loses its actual reference/evidence/owner result")
        if self.result is not None:
            if (
                self.design is None
                or self.result.design
                != ObjectIdentity.from_record(self.design.design_id, self.design)
                or self.reason_codes != self.result.reason_codes
            ):
                raise ValueError("development geometry result changes its actual frozen design")
        elif not self.reason_codes:
            raise ValueError("development geometry without measurements must retain its refusal")


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentGeometryReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-geometry-report'
    report_id: str
    context: str
    config: ObjectIdentity
    fit_result: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    cells: tuple[ResponseGeometryDevelopmentGeometryCell, ...]
    limitation_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.context not in ("assembling", "prepared")
            or self.config.object_schema != ResponseGeometryDevelopmentQualificationConfig.SCHEMA
            or self.fit_result.object_schema != ResponseGeometryDevelopmentFitResult.SCHEMA
        ):
            raise ValueError("development geometry report changes its method/context lineage")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        expected = {
            (r, p, t) for r in REPRESENTATIONS for p in PARENTS for t in range(384, 513, 16)
        }
        if {(c.representation, c.parent, c.invocation_offset) for c in self.cells} != expected:
            raise ValueError(
                "development geometry must retain all 225 selected-model cells, including refusals"
            )
        expected_reports = tuple(
            sorted(
                f"report.response-geometry-development.{self.context}.r{i:02d}.r{r}"
                for i in range(32, 64)
                for r in (1, 2)
            )
        )
        if tuple(r.object_id for r in self.input_reports) != expected_reports:
            raise ValueError("development geometry changes its full independent-validation roster")


def _assess_cell(
    *,
    design: ResponseGeometryDevelopmentGeometryDesign,
    model: ResponseGeometryDevelopmentAffineModel,
    fit: ResponseGeometryDevelopmentFitResult,
    pairs: tuple[tuple[ResponseGeometryDevelopmentMeasuredView, ResponseGeometryDevelopmentMeasuredView], ...],
    source_evaluator: ExecutableReference,
    input_artifacts: tuple[ArtifactIdentity, ...],
    config: ResponseGeometryDevelopmentQualificationConfig,
) -> ResponseGeometryDevelopmentGeometryCell:
    selected = tuple(
        pair for pair in pairs if pair[0].report.root.invocation_offset == design.invocation_offset
    )
    cell_id = f"development.geometry.{design.context}.{design.representation}.{design.parent}.t{design.invocation_offset}"
    design_identity = ObjectIdentity.from_record(design.design_id, design)
    initial_states, errors = [], []
    delivered, margins_known = True, True
    active_slack, hold_slack = Decimal(1), Decimal(1)
    for pair in selected:
        series = root_series(pair, model.parent, model.representation)
        if series is None:
            delivered, margins_known = False, False
            continue
        try:
            prediction, _, actual = endpoint_predictions(model, series)
            if not np.isfinite(prediction).all():
                raise FloatingPointError("development endpoint prediction unavailable")
            for view_index in range(2):
                initial_states.append(
                    model.feature_map.transform(
                        series.scalar[view_index, 1, 0],
                        series.features[view_index, 1, 0],
                        model.dimension,
                    )
                )
            errors.append(prediction - actual)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            delivered, margins_known = False, False
            continue
        for view in pair:
            packet = next(p for p in view.report.packets if p.parent == design.parent)
            if not packet.delivered:
                delivered = False
            required = (
                packet.maximum_orthogonal_x,
                packet.maximum_relative_y,
                packet.maximum_transfer_difference,
                packet.parent_absolute_density_work,
            )
            if any(v is None for v in required):
                margins_known = False
                continue
            for value, bound in zip(
                required,
                (Decimal(".125"), Decimal(".05"), Decimal(".05"), Decimal(32)),
                strict=True,
            ):
                assert value is not None
                active_slack = min(active_slack, Decimal(1) - value / bound)
            assert packet.parent_absolute_density_work is not None
            hold_slack = min(
                hold_slack, Decimal(1) - packet.parent_absolute_density_work / Decimal(32)
            )
    if not errors:
        return ResponseGeometryDevelopmentGeometryCell(
            cell_id,
            design.representation,
            design.parent,
            design.invocation_offset,
            design,
            None,
            None,
            None,
            ("DEVELOPMENT_GEOMETRY_VALIDATION_UNAVAILABLE",),
        )
    initial = np.mean(initial_states, axis=0)
    transition, _, affine, _ = response_geometry_development_endpoint_coefficients(
        model.dimension, model.drift, model.diffusion
    )
    reference = ResponseGeometryDevelopmentGeometryReference(
        design.template.expected_reference_trajectory_id,
        design_identity,
        ObjectIdentity.from_record(fit.result_id, fit),
        report_identities(tuple(view for pair in selected for view in pair)),
        tuple(decimal_from_float(v) for v in initial),
        tuple(decimal_from_float(v) for v in transition @ initial + affine),
    )
    reference_identity = ObjectIdentity.from_record(reference.reference_id, reference)
    steps = response_geometry_development_geometry_steps(model, design)
    # Repeatability is same-implementation deterministic construction, not an
    # independent numerical view or empirical agreement measurement.
    operator_bytes = b"".join(step.canonical_bytes() for step in steps)
    repeated_bytes = b"".join(step.canonical_bytes() for step in response_geometry_development_geometry_steps(model, design))
    error = np.abs(np.stack(errors))
    maximum = decimal_from_float(float(np.max(error)))
    link = EvidenceLink(
        f"{cell_id}.evidence-link",
        EvidenceRelation.DERIVED_FROM,
        reference_identity,
        design_identity,
        tuple(a.artifact_id for a in input_artifacts),
        config.system.world.world_id,
        f"development.{design.context}.validation-after-fit-freeze",
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        (VisibilityCeiling.DEVELOPMENT_ONLY,),
        "Fitted mean-HOLD deviation map and observed validation errors; one-pulse finite model diagnostic only.",
    )
    evidence = ControlledIOOperatorEvidence(
        evidence_id=f"{cell_id}.operator-evidence",
        prepared_denominator_id=design.template.prepared_denominator_id,
        denominator_member_id=design.template.denominator_member_id,
        candidate_version_id=design.template.candidate_version_id,
        qualification_view_ids=design.template.qualification_view_ids,
        support_cell_ids=design.template.support_cell_ids,
        normalization=design.template.normalization,
        operator_topology=design.template.clock_contract,
        state_basis=design.normalization.state_basis,
        input_basis=design.normalization.input_basis,
        receiver_basis=design.normalization.receiver_basis,
        steps=steps,
        reference_trajectory=reference_identity,
        reference_action_word=design.template.reference_action_word,
        action_words=design.action_words,
        retained_history_id=design.template.retained_history_id,
        horizon_id=design.template.horizon_id,
        held_out_physical_unit_ids=design.template.expected_held_out_physical_unit_ids,
        residual_norm=NamedDecimal(
            f"{cell_id}.normalized-receiver-error", maximum / Decimal(".125"), "1"
        ),
        active_prediction_error=NamedDecimal(
            f"{cell_id}.pos-error",
            decimal_from_float(float(np.max(error[..., 2]))),
            DEVELOPMENT_RECEIVER_UNIT,
        ),
        wrong_sign_prediction_error=NamedDecimal(
            f"{cell_id}.neg-error",
            decimal_from_float(float(np.max(error[..., 0]))),
            DEVELOPMENT_RECEIVER_UNIT,
        ),
        zero_input_hold_error=NamedDecimal(
            f"{cell_id}.hold-error",
            decimal_from_float(float(np.max(error[..., 1]))),
            DEVELOPMENT_RECEIVER_UNIT,
        ),
        first_operator_sha256=sha256(operator_bytes).hexdigest(),
        repeated_operator_sha256=sha256(repeated_bytes).hexdigest(),
        largest_operator_artifact_bytes=max(a.size_bytes for a in input_artifacts),
        materialized_member_bytes=0,
        whole_stage_bytes=0,
        complete_delivery=delivered and len(errors) == len(selected),
        source_evaluator=source_evaluator,
        input_artifacts=input_artifacts,
        evidence_links=(link,),
        disposition=ControlledIOProductDisposition.SUPPORTED,
        reason_codes=(),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
    )
    margins = (
        tuple(
            PrefixPathwiseMargin(
                f"{cell_id}.margin.{word_index}.{prefix_index}",
                ObjectIdentity.from_record(word.word_id, word),
                prefix_index,
                support_id,
                hold_slack if word == design.action_words[0] else active_slack,
                "1",
            )
            for word_index, word in enumerate(design.action_words)
            for prefix_index, support_id in enumerate(word.prefix_support_ids)
        )
        if margins_known
        else ()
    )
    # The construction is pure. Measure its actual canonical representation,
    # then bind those bytes before retaining the final owner result.
    result = evaluate_response_geometry_development_geometry(design=design, evidence=evidence, pathwise_margins=margins)
    member_bytes = len(result.construction.member.canonical_bytes())
    evidence = replace(evidence, materialized_member_bytes=member_bytes)
    result = evaluate_response_geometry_development_geometry(design=design, evidence=evidence, pathwise_margins=margins)
    if len(result.construction.member.canonical_bytes()) != member_bytes:
        raise ValueError("development controlled member representation measurement is unstable")
    return ResponseGeometryDevelopmentGeometryCell(
        cell_id,
        design.representation,
        design.parent,
        design.invocation_offset,
        design,
        reference,
        evidence,
        result,
        result.reason_codes,
    )


def assess_response_geometry_development_geometry(
    *,
    config: ResponseGeometryDevelopmentQualificationConfig,
    fit: ResponseGeometryDevelopmentFitResult,
    fit_payload: bytes,
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    clock_evaluator: ExecutableReference,
    source_evaluator: ExecutableReference,
    implementation_sha256: str,
    input_artifacts: tuple[ArtifactIdentity, ...],
) -> ResponseGeometryDevelopmentGeometryReport:
    """Use frozen fits once, retain every cell, and never refit on validation."""
    method_identity = ObjectIdentity.from_record(config.method.config_id, config.method)
    if fit.config != method_identity or any(
        v.report.projection_config != config.method.projection_config for v in views
    ):
        raise ValueError("development geometry changes authenticated fit/projection lineage")
    require_sorted_unique_ids(
        input_artifacts, attribute="artifact_id", field_name="input_artifacts"
    )
    artifact_sources = {(a.payload_schema, a.sha256) for a in input_artifacts}
    if (
        (DEVELOPMENT_FIT_SCHEMA, fit.data_sha256) not in artifact_sources
        or (fit.SCHEMA, fit.fingerprint()) not in artifact_sources
        or source_evaluator.payload not in input_artifacts
        or clock_evaluator.payload not in input_artifacts
    ):
        raise ValueError("development geometry lacks its authentic fit/evaluator artifacts")
    if any(
        (DEVELOPMENT_DATA_SCHEMA, v.report.data_sha256) not in artifact_sources
        or (v.report.SCHEMA, v.report.fingerprint()) not in artifact_sources
        for v in views
    ):
        raise ValueError("development geometry lacks an actual validation report/array artifact")
    pairs = organize_response_geometry_development_views(views, context=fit.context, role="validation")
    roots = {r.root_id: r for r in config.source.roots}
    if any(roots.get(v.report.root.root_id) != v.report.root for v in views):
        raise ValueError("development geometry changes its native validation root")
    groups = read_response_geometry_development_fit(fit, fit_payload)
    cells = []
    for group in groups:
        models = {m.parent: m for m in group.models}
        for parent in PARENTS:
            for offset in range(384, 513, 16):
                model = models.get(parent)
                if model is None:
                    cells.append(
                        ResponseGeometryDevelopmentGeometryCell(
                            f"development.geometry.{fit.context}.{group.representation}.{parent}.t{offset}",
                            group.representation,
                            parent,
                            offset,
                            None,
                            None,
                            None,
                            None,
                            ("DEVELOPMENT_FITTED_MODEL_UNAVAILABLE",),
                        )
                    )
                    continue
                design = response_geometry_development_geometry_design(
                    source=config.source,
                    system=config.system,
                    context=fit.context,
                    representation=group.representation,
                    parent=parent,
                    invocation_offset=offset,
                    dimension=model.dimension,
                    clock_evaluator=clock_evaluator,
                    implementation_sha256=implementation_sha256,
                )
                # Each cell carries only the artifacts it actually consumes.
                reports = {
                    v.report.fingerprint()
                    for pair in pairs.values()
                    if pair[0].report.root.invocation_offset == offset
                    for v in pair
                }
                arrays = {
                    v.report.data_sha256
                    for pair in pairs.values()
                    if pair[0].report.root.invocation_offset == offset
                    for v in pair
                }
                cell_artifacts = tuple(
                    a
                    for a in input_artifacts
                    if a.sha256 in {*reports, *arrays, fit.fingerprint(), fit.data_sha256}
                    or a in (source_evaluator.payload, clock_evaluator.payload)
                )
                cells.append(
                    _assess_cell(
                        design=design,
                        model=model,
                        fit=fit,
                        pairs=tuple(pairs.values()),
                        source_evaluator=source_evaluator,
                        input_artifacts=cell_artifacts,
                        config=config,
                    )
                )
    report = ResponseGeometryDevelopmentGeometryReport(
        f"response-geometry-development.geometry.{fit.context}",
        fit.context,
        ObjectIdentity.from_record(config.config_id, config),
        ObjectIdentity.from_record(fit.result_id, fit),
        report_identities(views),
        tuple(sorted(cells, key=lambda c: c.cell_id)),
        (
            "DEVELOPMENT_CONTROLLED_MAP_SYMMETRIC_RESIDUAL_STRUCTURALLY_ZERO_FOR_ONE_PULSE",
            "DEVELOPMENT_EXPECTED_ACTION_WORDS_DO_NOT_AUTHORIZE_DELIVERY_OR_ADMISSION",
            "DEVELOPMENT_MODEL_POWERS_ARE_NOT_REPEATED_PULSE_OR_LONGER_HORIZON_EVIDENCE",
            "DEVELOPMENT_NATIVE_GUARDS_ARE_NOT_A_QUALIFIED_HIDDEN_SINK_MODEL",
            "DEVELOPMENT_WHOLE_PATH_GUARD_LOWER_BOUND_REUSED_WITHOUT_INFLATING_N",
        ),
    )
    # Fill the whole-report byte count only after assembling every cell. Hashes
    # have fixed width; the only size change is replacing the provisional zero
    # integers. Recompute owner results with the final count before publication.
    initial_size = len(report.canonical_bytes())
    count = sum(cell.evidence is not None for cell in report.cells)
    size = initial_size
    while (updated := initial_size + count * (len(str(size)) - 1)) != size:
        size = updated
    if size > DEVELOPMENT_GEOMETRY_MAX_STAGE_BYTES:
        raise ValueError("development finite geometry report exceeds its declared complete-stage ceiling")
    final_cells = []
    for cell in report.cells:
        if cell.evidence is None:
            final_cells.append(cell)
            continue
        assert cell.design is not None and cell.result is not None
        evidence = replace(cell.evidence, whole_stage_bytes=size)
        margins = cell.result.nonnormal_audits[0].pathwise_margins
        result = evaluate_response_geometry_development_geometry(
            design=cell.design, evidence=evidence, pathwise_margins=margins
        )
        final_cells.append(
            replace(cell, evidence=evidence, result=result, reason_codes=result.reason_codes)
        )
    report = replace(report, cells=tuple(final_cells))
    if len(report.canonical_bytes()) != size:
        raise ValueError("development finite geometry report changed its measured representation size")
    return report
