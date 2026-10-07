"""Finite development model witnesses through the existing controlled-IO scientific owners.

The complete stochastic predictor remains the affine law. These operators act
on deviations about its fitted mean HOLD, in a predeclared normalized chart.
Algebraic powers diagnose the fitted model only; development observes one short-pulse-response pulse.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import CoordinateOrigin
from empirical_lawhood.adapters.methods.receiver_conditioned_io import (
    CanonicalMatrix,
    CanonicalVector,
    ControlledIOEvaluator,
    ControlledIOProductDisposition,
    ControlledIOStep,
    CoordinateBasis,
    GammaRule,
    JacobiConfig,
    JacobiDeflationRule,
    JacobiDisposition,
    JacobiGaugeConvention,
    JacobiMomentNorm,
    JacobiWitness,
    JacobiWitnessService,
    MarkovKernelFamily,
    NonnormalAudit,
    NonnormalAuditConfig,
    NonnormalAuditDisposition,
    NonnormalAuditService,
    PrefixPathwiseMargin,
    ReceiverMetricService,
    ReceiverRieszFamily,
    StateMetric,
    StateMetricConfig,
    WhiteningConvention,
)
from empirical_lawhood.adapters.methods.receiver_conditioned_io.contracts import decimal_from_float
from empirical_lawhood.adapters.methods.receiver_conditioned_io.member_construction import ControlledIOBasisNormalization, ControlledIOClockAxis, ControlledIOMemberConstructionResult, ControlledIOOperatorEvidence, ControlledIOOperatorStepTopology, ControlledIOOperatorTopology, NativeCoordinateCompatibility, StateCoordinateNormalization, construct_controlled_io_member
from empirical_lawhood.adapters.methods.receiver_conditioned_io.prospective_design import ControlledIOMemberConstructionTemplate, ControlledIOMemberConstructionTemplateBindingReceipt, bind_controlled_io_member_construction_template
from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_actions import DEVELOPMENT_CLOCK, DEVELOPMENT_EPISODE_FRAME, DEVELOPMENT_FORCE_FRAME, DEVELOPMENT_FORCE_UNIT, response_geometry_development_action_word
from empirical_lawhood.adapters.simulators.six_matrix_response.response_assay import ResponseGeometryNativeForcePulse
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import ResponseGeometryDevelopmentNativeConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS

from .development_law import DEVELOPMENT_LATENT_QUANTITIES
from .development_models import REPRESENTATIONS, ResponseGeometryDevelopmentAffineModel, FloatArray, response_geometry_development_endpoint_coefficients
from .development_projection import ResponseGeometryDevelopmentViewReport
from .development_records import DEVELOPMENT_FIT_MAXIMUM_BYTES, ResponseGeometryDevelopmentFitResult


DEVELOPMENT_GEOMETRY_IMPLEMENTATION = "response-geometry-development.finite-model-geometry"
DEVELOPMENT_GEOMETRY_MAX_CELL_BYTES = 512 * 1024
DEVELOPMENT_GEOMETRY_MAX_STAGE_BYTES = 256 * 1024**2
DEVELOPMENT_RANK_TOLERANCE = Decimal("1e-10")
DEVELOPMENT_RECEIVER_UNIT = "hilbert-schmidt-native"
DEVELOPMENT_CONTROLLED_UNIT = f"{DEVELOPMENT_RECEIVER_UNIT}/{DEVELOPMENT_FORCE_UNIT}"


def response_geometry_development_state_scales(dimension: int) -> FloatArray:
    if dimension not in (4, 8):
        raise ValueError("development geometry requires a predeclared latent dimension")
    return np.asarray((0.125, 0.390625, *((1.0,) * (dimension - 2))))


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentGeometryDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-geometry-design'

    design_id: str
    context: str
    representation: str
    parent: str
    invocation_offset: int
    source_config: ObjectIdentity
    normalization: ControlledIOBasisNormalization
    template: ControlledIOMemberConstructionTemplate
    action_words: tuple[OccurrenceActionWord, ...]
    metric_config: StateMetricConfig
    jacobi_configs: tuple[JacobiConfig, ...]
    nonnormal_config: NonnormalAuditConfig

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        if (
            self.context not in ("assembling", "prepared")
            or self.representation not in REPRESENTATIONS
            or self.parent not in PARENTS
            or self.invocation_offset not in range(384, 513, 16)
        ):
            raise ValueError("development geometry changes its native cell")
        if self.source_config.object_schema != ResponseGeometryDevelopmentNativeConfig.SCHEMA:
            raise ValueError("development geometry changes its native source schema")
        if (
            self.template.normalization
            != ObjectIdentity.from_record(self.normalization.normalization_id, self.normalization)
            or self.template.action_words
            != tuple(ObjectIdentity.from_record(w.word_id, w) for w in self.action_words)
            or self.metric_config.state_basis != self.normalization.state_basis
            or self.metric_config.receiver_basis != self.normalization.receiver_basis
            or tuple(c.maximum_order for c in self.jacobi_configs) != (2, 3, 4)
        ):
            raise ValueError("development geometry changes its frozen basis/action/witness design")


def response_geometry_development_geometry_design(
    *,
    source: ResponseGeometryDevelopmentNativeConfig,
    system: SystemSpec,
    context: str,
    representation: str,
    parent: str,
    invocation_offset: int,
    dimension: int,
    clock_evaluator: ExecutableReference,
    implementation_sha256: str,
) -> ResponseGeometryDevelopmentGeometryDesign:
    """Enumerate this design for both dimensions before fit outcomes are available."""
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    if (
        context not in ("assembling", "prepared")
        or representation not in REPRESENTATIONS
        or parent not in PARENTS
        or invocation_offset not in range(384, 513, 16)
    ):
        raise ValueError("development geometry is outside its predeclared native chart")
    scales = response_geometry_development_state_scales(dimension)
    stem = f"development.geometry.{context}.{representation}.{parent}.t{invocation_offset}.n{dimension}"
    state_ids = tuple(f"{stem}.z{i}" for i in range(dimension))
    state = CoordinateBasis(
        f"{stem}.state", state_ids, ("1",) * dimension, (f"{stem}.chart",) * dimension
    )
    inputs = CoordinateBasis(
        f"{stem}.input", ("response-geometry.x-force-command",), (DEVELOPMENT_FORCE_UNIT,), (DEVELOPMENT_FORCE_FRAME,)
    )
    receivers = CoordinateBasis(
        f"{stem}.receiver",
        system.relation.receiver_quantity_ids,
        (DEVELOPMENT_RECEIVER_UNIT,),
        (DEVELOPMENT_FORCE_FRAME,),
    )
    quantities = {q.quantity_id: q for q in system.quantities}
    schema_sha = system.fingerprint()
    state_maps = []
    for index, coordinate_id in enumerate(state_ids):
        quantity = quantities[DEVELOPMENT_LATENT_QUANTITIES[index]]
        state_maps.append(
            StateCoordinateNormalization(
                f"{stem}.state-map.{index}",
                quantity.quantity_id,
                index,
                quantity.label,
                schema_sha,
                coordinate_id,
                quantity.quantity_id,
                quantity.native_unit,
                quantity.coordinate_frame,
                NamedDecimal(
                    f"{stem}.scale.{index}", decimal_from_float(scales[index]), quantity.native_unit
                ),
                state.native_frames[index],
            )
        )

    def native_map(
        basis: CoordinateBasis, role: str
    ) -> tuple[NativeCoordinateCompatibility, ...]:
        quantity = quantities[basis.coordinate_ids[0]]
        return (
            NativeCoordinateCompatibility(
                f"{stem}.{role}-map",
                quantity.quantity_id,
                0,
                quantity.label,
                schema_sha,
                quantity.quantity_id,
                quantity.quantity_id,
                quantity.native_unit,
                quantity.coordinate_frame,
                basis.native_units[0],
                basis.native_frames[0],
                "response-geometry.native-coordinate-identity",
            ),
        )

    normalization = ControlledIOBasisNormalization(
        f"{stem}.normalization",
        state,
        inputs,
        receivers,
        tuple(state_maps),
        native_map(inputs, "input"),
        native_map(receivers, "receiver"),
        schema_sha,
        "development.delta-horizon-and-unit-latent-scales",
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    axis = ControlledIOClockAxis(
        f"{stem}.clock",
        DEVELOPMENT_CLOCK,
        "reference-tick",
        DEVELOPMENT_EPISODE_FRAME,
        CoordinateOrigin.EPISODE_RELATIVE,
    )
    units = tuple(
        r
        for r in source.roots
        if r.context == context and r.index >= 32 and r.invocation_offset == invocation_offset
    )
    if len(units) not in (3, 4):
        raise ValueError("development geometry changes the complete validation stratum")
    origin = units[0].landmark_tick + invocation_offset
    topology = ControlledIOOperatorTopology(
        f"{stem}.topology",
        axis,
        axis,
        axis,
        (
            ControlledIOOperatorStepTopology(
                f"{stem}.step.0", 0, 0, 0, Decimal(origin), Decimal(origin), Decimal(origin), False
            ),
            ControlledIOOperatorStepTopology(
                f"{stem}.step.1",
                1,
                1,
                None,
                Decimal(origin + 320),
                None,
                Decimal(origin + 320),
                True,
            ),
        ),
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    words = tuple(
        sorted(
            (
                response_geometry_development_action_word(
                    ResponseGeometryNativeForcePulse(f"{stem}.{role}", "short-pulse-response", sign, origin),
                    denominator_id=system.relation.denominator_quantity_ids[0],
                    history_id=f"development.{context}.observed-invocation-history",
                    receiver_id=receivers.basis_id,
                    horizon_id=system.relation.horizon.horizon_id,
                    clock_evaluator=clock_evaluator,
                )
                for role, sign in (("neg", -1), ("hold", 0), ("pos", 1))
            ),
            key=lambda w: w.word_id,
        )
    )
    template = ControlledIOMemberConstructionTemplate(
        template_id=f"{stem}.template",
        member_record_id=f"{stem}.member",
        prepared_denominator_id=words[0].denominator_id,
        denominator_member_id=f"development.{context}.{parent}",
        candidate_version_id=f"development.affine.{context}.{representation}.{parent}",
        qualification_view_ids=tuple(sorted(v.view_id for v in system.numerical_views)),
        support_cell_ids=(f"development.{context}.{parent}.t{invocation_offset}",),
        action_words=tuple(ObjectIdentity.from_record(w.word_id, w) for w in words),
        retained_history_id=words[0].retained_history_id,
        horizon_id=words[0].horizon_id,
        normalization=ObjectIdentity.from_record(normalization.normalization_id, normalization),
        operator_topology=topology,
        expected_held_out_physical_unit_ids=tuple(
            sorted(source.physical_unit_id(r) for r in units)
        ),
        expected_reference_trajectory_id=f"{stem}.mean-hold",
        expected_reference_trajectory_schema=ResponseGeometryDevelopmentGeometryReference.SCHEMA,
        reference_action_word=ObjectIdentity.from_record(words[0].word_id, words[0]),
        clock_contract=ObjectIdentity.from_record(topology.topology_id, topology),
        required_step_count=2,
        state_dimension=dimension,
        input_dimension=1,
        receiver_dimension=1,
        maximum_residual_norm=Decimal(1),
        maximum_held_out_prediction_error=Decimal(".125"),
        maximum_spectral_radius=Decimal(4),
        maximum_condition_number=Decimal("1e8"),
        minimum_controllability_rank=1,
        minimum_observability_rank=1,
        gamma_rule=GammaRule.ZERO,
        stability_rule_id="development.finite-endpoint-radius-below-four",
        rank_tolerance=DEVELOPMENT_RANK_TOLERANCE,
        maximum_zero_input_hold_error=Decimal(".125"),
        maximum_state_dimension=8,
        maximum_input_dimension=1,
        maximum_receiver_dimension=1,
        maximum_steps=2,
        maximum_operator_artifact_bytes=DEVELOPMENT_FIT_MAXIMUM_BYTES,
        maximum_materialized_member_bytes=DEVELOPMENT_GEOMETRY_MAX_CELL_BYTES,
        maximum_whole_stage_bytes=DEVELOPMENT_GEOMETRY_MAX_STAGE_BYTES,
        binder_implementation_id=DEVELOPMENT_GEOMETRY_IMPLEMENTATION,
        binder_implementation_sha256=implementation_sha256,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    metric = StateMetricConfig(
        f"{stem}.metric-config",
        state,
        receivers,
        Decimal("1e-12"),
        Decimal("1e8"),
        DEVELOPMENT_RANK_TOLERANCE,
        Decimal("1e-12"),
        WhiteningConvention.SYMMETRIC_POSITIVE_SQUARE_ROOT_CANONICAL_QR,
    )
    jacobi = tuple(
        JacobiConfig(
            f"{stem}.jacobi-config.{order}",
            state,
            inputs,
            receivers,
            order,
            DEVELOPMENT_RANK_TOLERANCE,
            Decimal("1e-9"),
            Decimal("1e-9"),
            Decimal("1e-12"),
            JacobiGaugeConvention.LARGEST_MAGNITUDE_ENTRY_POSITIVE,
            JacobiDeflationRule.SVD_RANK_BELOW_FROZEN_TOLERANCE,
            JacobiMomentNorm.FROBENIUS,
            topology.steps[0].step_id,
            DEVELOPMENT_CLOCK,
        )
        for order in (2, 3, 4)
    )
    nonnormal = NonnormalAuditConfig(
        f"{stem}.nonnormal-config",
        Decimal(4),
        Decimal(1),
        Decimal(".015625"),
        Decimal(0),
        Decimal(0),
        DEVELOPMENT_CONTROLLED_UNIT,
        DEVELOPMENT_CONTROLLED_UNIT,
        "1",
    )
    return ResponseGeometryDevelopmentGeometryDesign(
        stem,
        context,
        representation,
        parent,
        invocation_offset,
        ObjectIdentity.from_record(source.config_id, source),
        normalization,
        template,
        words,
        metric,
        jacobi,
        nonnormal,
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentGeometryReference(CanonicalRecord):
    """Actual mean fitted HOLD at the two slots, including initial state and drift."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-geometry-reference'
    reference_id: str
    design: ObjectIdentity
    fit_result: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    mean_initial: tuple[Decimal, ...]
    mean_endpoint: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.reference_id, field_name="reference_id")
        if (
            self.design.object_schema != ResponseGeometryDevelopmentGeometryDesign.SCHEMA
            or self.fit_result.object_schema != ResponseGeometryDevelopmentFitResult.SCHEMA
            or not self.input_reports
            or any(r.object_schema != ResponseGeometryDevelopmentViewReport.SCHEMA for r in self.input_reports)
            or tuple(r.object_id for r in self.input_reports)
            != tuple(sorted({r.object_id for r in self.input_reports}))
        ):
            raise ValueError("development mean HOLD requires its design and actual input lineage")
        if len(self.mean_initial) not in (4, 8) or len(self.mean_endpoint) != len(
            self.mean_initial
        ):
            raise ValueError("development mean HOLD changes its latent dimension")
        if any(not v.is_finite() for v in (*self.mean_initial, *self.mean_endpoint)):
            raise ValueError("development mean HOLD must retain finite fitted coordinates")


def response_geometry_development_geometry_steps(
    model: ResponseGeometryDevelopmentAffineModel, design: ResponseGeometryDevelopmentGeometryDesign
) -> tuple[ControlledIOStep, ...]:
    """Deviation dynamics; the separate reference retains the complete mean HOLD."""
    scale = response_geometry_development_state_scales(model.dimension)
    normalization, topology = design.normalization, design.template.operator_topology
    if model.dimension != normalization.state_basis.dimension or (
        model.context,
        model.representation,
        model.parent,
    ) != (design.context, design.representation, design.parent):
        raise ValueError("development fitted operator changes its predeclared cell or dimension")
    transition, pulse, _, _ = response_geometry_development_endpoint_coefficients(
        model.dimension, model.drift, model.diffusion
    )
    transition = transition * scale[None, :] / scale[:, None]
    pulse = (pulse / scale)[:, None]
    receiver = np.zeros((1, model.dimension))
    receiver[0, 0] = scale[0]
    state_ids = normalization.state_basis.coordinate_ids
    input_ids, receiver_ids = (
        normalization.input_basis.coordinate_ids,
        normalization.receiver_basis.coordinate_ids,
    )
    output = []
    for index, slot in enumerate(topology.steps):

        def matrix(
            label: str, rows: tuple[str, ...], columns: tuple[str, ...], values: FloatArray
        ) -> CanonicalMatrix:
            return CanonicalMatrix.from_array(
                matrix_id=f"{slot.step_id}.{label}",
                row_coordinate_ids=rows,
                column_coordinate_ids=columns,
                values=values,
            )

        output.append(
            ControlledIOStep(
                slot.step_id,
                index,
                matrix(
                    "transition",
                    state_ids,
                    state_ids,
                    transition if index == 0 else np.eye(model.dimension),
                ),
                matrix(
                    "pulse", state_ids, input_ids, pulse if index == 0 else np.zeros_like(pulse)
                ),
                matrix("receiver", receiver_ids, state_ids, receiver),
                CanonicalVector.zeros(
                    vector_id=f"{slot.step_id}.state-affine", coordinate_ids=state_ids
                ),
                CanonicalVector.zeros(
                    vector_id=f"{slot.step_id}.receiver-affine", coordinate_ids=receiver_ids
                ),
                GammaRule.ZERO,
                CanonicalVector.zeros(vector_id=f"{slot.step_id}.gamma", coordinate_ids=state_ids),
                DEVELOPMENT_CLOCK,
                DEVELOPMENT_CLOCK,
                DEVELOPMENT_CLOCK,
            )
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentFiniteModelRanks(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-finite-model-ranks'
    pulse_image_rank: int
    reachable_rank: int
    observable_rank: int
    reachable_observable_quotient_rank: int
    reachable_singular_values: tuple[Decimal, ...]
    observable_singular_values: tuple[Decimal, ...]
    quotient_singular_values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if (
            self.pulse_image_rank not in (0, 1)
            or not 0 <= self.reachable_rank <= 8
            or not 0 <= self.observable_rank <= 8
            or not 0
            <= self.reachable_observable_quotient_rank
            <= min(self.reachable_rank, self.observable_rank)
            <= 8
            or any(
                v < 0 or not v.is_finite()
                for values in (
                    self.reachable_singular_values,
                    self.observable_singular_values,
                    self.quotient_singular_values,
                )
                for v in values
            )
        ):
            raise ValueError("development finite model rank diagnostics are inconsistent")


def response_geometry_development_model_ranks(steps: tuple[ControlledIOStep, ...]) -> ResponseGeometryDevelopmentFiniteModelRanks:
    """Rank R/(R intersect N) by observing an orthonormal reachable basis.

    Powers through n-1 define this finite fitted-model calculation only. The
    observed pulse image has one input column regardless of these model ranks.
    """
    if len(steps) != 2:
        raise ValueError("development rank calculation requires the one-pulse endpoint topology")
    transition, pulse, receiver = (
        steps[0].state_transition.as_array(),
        steps[0].realized_input_map.as_array(),
        steps[1].receiver_map.as_array(),
    )
    n = len(transition)
    if n not in (4, 8) or pulse.shape != (n, 1) or receiver.shape != (1, n):
        raise ValueError("development rank operands change their declared dimensions")
    power, reachable, observable = np.eye(n), [], []
    for _ in range(n):
        reachable.append(power @ pulse)
        observable.append(receiver @ power)
        power = transition @ power
    r, o = np.hstack(reachable), np.vstack(observable)
    if not np.isfinite(r).all() or not np.isfinite(o).all():
        raise FloatingPointError("development finite model rank powers are nonfinite")

    def rank(singular: FloatArray) -> int:
        return (
            int(np.sum(singular > float(DEVELOPMENT_RANK_TOLERANCE) * singular[0]))
            if singular.size and singular[0] > 0
            else 0
        )

    basis, rs, _ = np.linalg.svd(r, full_matrices=False)
    os = np.asarray(np.linalg.svd(o, compute_uv=False), dtype=np.float64)
    rr = rank(rs)
    qs = np.linalg.svd(o @ basis[:, :rr], compute_uv=False) if rr else np.empty(0)
    return ResponseGeometryDevelopmentFiniteModelRanks(
        int(np.linalg.norm(pulse) > 0),
        rr,
        rank(os),
        int(np.sum(qs > float(DEVELOPMENT_RANK_TOLERANCE) * os[0])) if os.size else 0,
        tuple(decimal_from_float(v) for v in rs),
        tuple(decimal_from_float(v) for v in os),
        tuple(decimal_from_float(v) for v in qs),
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentGeometryWitnesses(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-geometry-witnesses'
    design: ObjectIdentity
    binding: ControlledIOMemberConstructionTemplateBindingReceipt
    construction: ControlledIOMemberConstructionResult
    markov: MarkovKernelFamily
    metric: StateMetric
    riesz: ReceiverRieszFamily
    witnesses: tuple[JacobiWitness, ...]
    nonnormal_audits: tuple[NonnormalAudit, ...]
    selected_order: int | None
    ranks: ResponseGeometryDevelopmentFiniteModelRanks | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.witnesses) != 3 or len(self.nonnormal_audits) != 3:
            raise ValueError("development geometry must retain all three predeclared witness attempts")
        member = ObjectIdentity.from_record(
            self.construction.member.member_record_id, self.construction.member
        )
        if (
            self.binding.construction_config != self.construction.receipt.config
            or self.markov.controlled_io_member != member
            or any(w.controlled_io_member != member for w in self.witnesses)
            or any(a.controlled_io_member != member for a in self.nonnormal_audits)
        ):
            raise ValueError("development geometry loses its exact constructed member lineage")
        qualified_orders = tuple(
            order
            for order, witness, audit in zip(
                (2, 3, 4), self.witnesses, self.nonnormal_audits, strict=True
            )
            if witness.disposition is JacobiDisposition.SUPPORTED
            and audit.disposition is NonnormalAuditDisposition.FAVORABLE
        )
        if self.selected_order != (qualified_orders[0] if qualified_orders else None):
            raise ValueError("development geometry must select the least qualifying predeclared order")
        if self.design.object_schema != ResponseGeometryDevelopmentGeometryDesign.SCHEMA:
            raise ValueError("development geometry must bind its own design schema")

    @property
    def applicable(self) -> bool:
        return not self.reason_codes


def evaluate_response_geometry_development_geometry(
    *,
    design: ResponseGeometryDevelopmentGeometryDesign,
    evidence: ControlledIOOperatorEvidence,
    pathwise_margins: tuple[PrefixPathwiseMargin, ...],
) -> ResponseGeometryDevelopmentGeometryWitnesses:
    """Consume authentic method evidence; never supply a caller-chosen final law."""
    template, normalization = design.template, design.normalization
    config, binding = bind_controlled_io_member_construction_template(
        binding_id=f"{design.design_id}.binding",
        template=template,
        reference_trajectory=evidence.reference_trajectory,
        binder_implementation_id=DEVELOPMENT_GEOMETRY_IMPLEMENTATION,
        binder_implementation_sha256=template.binder_implementation_sha256,
    )
    construction = construct_controlled_io_member(
        evidence=evidence,
        normalization=normalization,
        config=config,
        implementation_sha256=template.binder_implementation_sha256,
    )
    member = construction.member
    markov = ControlledIOEvaluator().evaluate(member, member.qualification_config)
    service, state_ids = ReceiverMetricService(), normalization.state_basis.coordinate_ids
    metric = service.qualify_metric(
        metric_id=f"{design.design_id}.metric",
        state_basis=normalization.state_basis,
        matrix=CanonicalMatrix.from_array(
            matrix_id=f"{design.design_id}.metric-matrix",
            row_coordinate_ids=state_ids,
            column_coordinate_ids=state_ids,
            values=np.eye(len(state_ids)),
        ),
        config=design.metric_config,
        evidence_link_ids=tuple(link.link_id for link in evidence.evidence_links),
    )
    covector = np.zeros((len(state_ids), 1))
    covector[0, 0] = 0.125
    riesz = service.riesz(
        family_id=f"{design.design_id}.riesz",
        metric=metric,
        receiver_coordinate_ids=normalization.receiver_basis.coordinate_ids,
        receiver_covectors=CanonicalMatrix.from_array(
            matrix_id=f"{design.design_id}.receiver-covectors",
            row_coordinate_ids=state_ids,
            column_coordinate_ids=normalization.receiver_basis.coordinate_ids,
            values=covector,
        ),
        config=design.metric_config,
    )
    witnesses, audits = [], []
    for jacobi_config in design.jacobi_configs:
        witness = JacobiWitnessService().construct(
            witness_id=f"{design.design_id}.jacobi.{jacobi_config.maximum_order}",
            member=member,
            metric=metric,
            riesz=riesz,
            config=jacobi_config,
        )
        audit = NonnormalAuditService().audit(
            audit_id=f"{design.design_id}.nonnormal.{jacobi_config.maximum_order}",
            member=member,
            markov=markov,
            metric=metric,
            witness=witness,
            sink_map=None,
            pathwise_margins=pathwise_margins,
            config=design.nonnormal_config,
        )
        witnesses.append(witness)
        audits.append(audit)
    reasons = set(member.reason_codes)
    if member.disposition is not ControlledIOProductDisposition.SUPPORTED:
        reasons.add("DEVELOPMENT_CONTROLLED_MEMBER_UNQUALIFIED")
    if all(w.disposition is not JacobiDisposition.SUPPORTED for w in witnesses):
        reasons.add("DEVELOPMENT_FINITE_JACOBI_UNQUALIFIED")
    available_orders = tuple(
        order
        for order, witness, audit in zip((2, 3, 4), witnesses, audits, strict=True)
        if witness.disposition is JacobiDisposition.SUPPORTED
        and audit.disposition is NonnormalAuditDisposition.FAVORABLE
    )
    if not available_orders:
        reasons.add("DEVELOPMENT_NONNORMAL_OR_PATHWISE_GUARD_UNQUALIFIED")
    ranks = None
    if evidence.steps:
        try:
            ranks = response_geometry_development_model_ranks(evidence.steps)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            reasons.add("DEVELOPMENT_FINITE_MODEL_RANK_UNAVAILABLE")
    # A missing hidden-sink operator is not measured zero leakage. Applicability
    # here is only the finite scalar model plus explicitly supplied native guards;
    # neither this property nor the owner records grant action admission.
    return ResponseGeometryDevelopmentGeometryWitnesses(
        ObjectIdentity.from_record(design.design_id, design),
        binding,
        construction,
        markov,
        metric,
        riesz,
        tuple(witnesses),
        tuple(audits),
        available_orders[0] if available_orders else None,
        ranks,
        tuple(sorted(reasons)),
    )
