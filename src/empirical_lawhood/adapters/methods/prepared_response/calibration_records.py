"Protected fresh calibration projection and simultaneous calibration products."

from __future__ import annotations

import base64
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedNativeSpec, PreparedRoot
from empirical_lawhood.adapters.simulators.prepared_response.policy_native import CALIBRATION_POLICIES, PreparedResponseCalibrationNativeTaskResult, decode_prepared_response_calibration_task_native, prepared_response_calibration_native_invocations
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedCommonStart, PreparedNativeHandoff, bind_prepared_native_handoff
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .development_policy import POLICIES
from .development_selection import PreparedResponseDevelopmentNominalLibrary
from .policy_decision import PreparedParentDecision, PreparedPolicyDecisionConfig
from .projection import project_prepared_future, project_prepared_handoff
from .qualification_projection import _force_error
from .statistics import PreparedStatisticalSpec, conformal_order_index


Array = npt.NDArray[np.float64]
_PROJECTION_SHAPES = dict(
    sorted(
        {
            "delivery_complete": (9,),
            "force_component_error": (9,),
            "handoff_displacement": (2,),
            "history": (16, 12),
            "known": (9, 5, 7),
            "observed": (9, 5, 7),
            "parent_absolute_density_work": (1,),
            "sketch": (8,),
        }.items()
    )
)
_BOOL = frozenset(("delivery_complete", "known"))
_ENCODING = "BASE64_LITTLE_ENDIAN_FLOAT64_OR_UINT8_BOOL_C_ORDER_FIXED_SHAPES_NAN_UNKNOWN"


def _freeze(value: Array) -> Array:
    array = np.asarray(value, dtype="<f8")
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _blocks(arrays: dict[str, Array]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            name,
            base64.b64encode(
                np.asarray(value, dtype="u1" if name in _BOOL else "<f8").tobytes(order="C")
            ).decode("ascii"),
        )
        for name, value in sorted(arrays.items())
    )


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-calibration-projection-config'
    config_id: str
    native_spec: PreparedNativeSpec
    policy_config: PreparedPolicyDecisionConfig
    force_component_tolerance: Decimal = Decimal("0.000000000001")
    policy_ids: tuple[str, ...] = POLICIES

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        charter = self.development_library.config.fit.projection.source_qualification.selected_charter
        if (
            self.native_spec.stage != 'calibration'
            or self.policy_config.native_spec != self.native_spec
            or charter is None
            or self.native_spec.selected_amplitude != charter.amplitude
            or self.development_library.policy_library is None
            or self.development_library.selected_structure is None
            or self.force_component_tolerance != Decimal("0.000000000001")
            or self.policy_ids != CALIBRATION_POLICIES
        ):
            raise ValueError("prepared fresh calibration projection changes its dependent refinement/source qualification/policy freeze")

    @property
    def development_library(self) -> PreparedResponseDevelopmentNominalLibrary:
        return self.policy_config.development_library


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationViewProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-calibration-view-projection'
    report_id: str
    config: ObjectIdentity
    root: PreparedRoot
    policy_id: str
    parent_decision: ObjectIdentity
    realized_parent: str | None
    source_results: tuple[ObjectIdentity, ...]
    common_start: ObjectIdentity | None
    handoff: ObjectIdentity | None
    refinement: int
    disposition: str
    blocks: tuple[tuple[str, str], ...]
    encoding: str = _ENCODING

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        if (
            self.config.object_schema != PreparedResponseCalibrationProjectionConfig.SCHEMA
            or self.root.stage != 'calibration'
            or self.policy_id not in CALIBRATION_POLICIES
            or self.parent_decision.object_schema
            != 'empirical-lawhood/methods/prepared-response/prepared-parent-decision'
            or self.realized_parent is not None
            and self.realized_parent not in PARENTS
            or any(
                value.object_schema != PreparedResponseCalibrationNativeTaskResult.SCHEMA
                for value in self.source_results
            )
            or len(self.source_results) != 11
            or len({value.object_id for value in self.source_results}) != 11
            or self.common_start is not None
            and self.common_start.object_schema != PreparedCommonStart.SCHEMA
            or self.handoff is not None
            and self.handoff.object_schema != PreparedNativeHandoff.SCHEMA
            or self.refinement not in (1, 2)
            or self.disposition not in (
                "COMPLETE",
                "PREFIX_UNAVAILABLE",
                "PORT_FRAME_UNRESOLVED",
                "POLICY_NONATTEMPT",
                "HANDOFF_UNAVAILABLE",
                "INCOMPLETE_NATIVE_CHART",
            )
            or tuple(name for name, _ in self.blocks) != tuple(_PROJECTION_SHAPES)
            or self.encoding != _ENCODING
            or self.report_id
            != f"{self.root.root_id}.policy.{self.policy_id}.project.r{self.refinement}"
        ):
            raise ValueError("prepared fresh calibration view projection changes its source/policy contract")
        arrays = self.arrays()
        if self.disposition == "COMPLETE":
            if (
                self.realized_parent is None
                or self.common_start is None
                or self.handoff is None
                or not np.isfinite(arrays["history"]).all()
                or not np.isfinite(arrays["parent_absolute_density_work"]).all()
                or not (
                    (arrays["parent_absolute_density_work"] >= 0)
                    & (arrays["parent_absolute_density_work"] <= 32)
                ).all()
                or not arrays["delivery_complete"].all()
                or not arrays["known"].all()
                or not np.isfinite(arrays["observed"]).all()
            ):
                raise ValueError("complete fresh calibration projection lacks its full chart")
        elif self.realized_parent is None and self.handoff is not None:
            raise ValueError("unentered fresh calibration policy cannot fabricate a handoff")

    def arrays(self) -> dict[str, Array]:
        result = {}
        for name, encoded in self.blocks:
            shape = _PROJECTION_SHAPES[name]
            size = int(np.prod(shape)) * (1 if name in _BOOL else 8)
            if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
                raise ValueError("prepared fresh calibration projection block changes its encoded size")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) != size or base64.b64encode(raw).decode("ascii") != encoded:
                raise ValueError("prepared fresh calibration projection block is not canonical base64")
            value = np.frombuffer(raw, dtype="u1" if name in _BOOL else "<f8").reshape(shape)
            if name in _BOOL:
                if np.any(value > 1):
                    raise ValueError("prepared fresh calibration projection Boolean block is invalid")
                value = np.frombuffer(value.tobytes(), dtype=np.bool_).reshape(shape)
            elif np.isinf(value).any():
                raise ValueError("prepared fresh calibration projection requires finite or NaN operands")
            result[name] = value
        return result


def project_prepared_response_calibration_view(
    config: PreparedResponseCalibrationProjectionConfig,
    root: PreparedRoot,
    policy_id: str,
    refinement: int,
    decision: PreparedParentDecision,
    inputs: tuple[tuple[PreparedResponseCalibrationNativeTaskResult, bytes], ...],
) -> PreparedResponseCalibrationViewProjection:
    if (
        policy_id not in config.policy_ids
        or type(refinement) is not int
        or refinement not in (1, 2)
        or decision.root != root
        or decision.policy_id != policy_id
        or decision.decision_id != f"{root.root_id}.policy.{policy_id}.decision.result"
        or decision.config != ObjectIdentity.from_record(
            config.policy_config.config_id, config.policy_config
        )
    ):
        raise ValueError("prepared fresh calibration projection changes its frozen parent decision or view")
    context_policy = next(
        value for value in config.policy_config.policy_library.contexts
        if value.context == root.context
    )
    if decision.context_policy != ObjectIdentity.from_record(context_policy.policy_id, context_policy):
        raise ValueError("prepared fresh calibration decision changes its dependent refinement-frozen context policy")
    invocations = tuple(
        value
        for value in prepared_response_calibration_native_invocations(config.native_spec, root=root)
        if value.phase == "prefix" or value.policy_id == policy_id
    )
    expected = tuple(value.task_id for value in invocations)
    by_id = {result.invocation.task_id: (result, payload) for result, payload in inputs}
    if tuple(sorted(by_id)) != expected or len(inputs) != len(expected):
        raise ValueError("prepared fresh calibration projection requires prefix, policy parent and nine futures")
    ordered = tuple(by_id[value][0] for value in expected)
    source_results = tuple(ObjectIdentity.from_record(value.result_id, value) for value in ordered)
    decision_identity = ObjectIdentity.from_record(decision.decision_id, decision)
    identities = dict(zip(expected, source_results, strict=True))
    identities[f"{root.root_id}.policy.{policy_id}.decision"] = decision_identity
    prefix = by_id[next(value.task_id for value in invocations if value.phase == "prefix")][0]
    parent = by_id[next(value.task_id for value in invocations if value.phase == "parent")]
    common = prefix.common_start
    common_identity = (
        None if common is None else ObjectIdentity.from_record(common.common_start_id, common)
    )
    if decision.common_start != common_identity:
        raise ValueError("prepared fresh calibration decision changes its single frozen common start")
    realized_parent = (
        None if common is None or common.frame is None else decision.selected_parent
    )
    # Authenticate the complete root/policy chain before decoding any native bytes.
    for invocation in invocations:
        result = by_id[invocation.task_id][0]
        if (
            result.invocation != invocation
            or result.predecessors != tuple(
                identities[key] for key in invocation.dependency_task_ids
            )
            or result.common_start != common
            or result.realized_parent != (
                None if invocation.phase == "prefix" else realized_parent
            )
        ):
            raise ValueError("prepared fresh calibration source identities do not form the frozen lineage")
        if result.native_pair is not None:
            views = result.native_pair.views
            if invocation.phase == "prefix":
                if common is not None and tuple(view.checkpoint for view in views) != tuple(
                    ObjectIdentity.from_record(value.checkpoint_id, value)
                    for value in common.checkpoints
                ):
                    raise ValueError("prepared fresh calibration common start changes its native prefix checkpoints")
            else:
                incoming = (
                    tuple(ObjectIdentity.from_record(value.checkpoint_id, value)
                          for value in common.checkpoints)
                    if invocation.phase == "parent" and common is not None
                    else tuple(view.checkpoint for view in parent[0].native_pair.views)
                    if parent[0].native_pair is not None
                    else ()
                )
                if (
                    any(view.delivery.common_start != common_identity for view in views)
                    or tuple(view.delivery.incoming_checkpoint for view in views) != incoming
                ):
                    raise ValueError("prepared fresh calibration native delivery changes its incoming checkpoint lineage")
    data = {
        task_id: decode_prepared_response_calibration_task_native(*by_id[task_id]) for task_id in expected
    }
    parent_data = data[parent[0].invocation.task_id]
    arrays = {
        name: np.zeros(shape, dtype=bool) if name in _BOOL else np.full(shape, np.nan)
        for name, shape in _PROJECTION_SHAPES.items()
    }
    disposition = "COMPLETE"
    handoff_identity = None
    if common is None:
        disposition = "PREFIX_UNAVAILABLE"
    elif common.frame is None:
        disposition = "PORT_FRAME_UNRESOLVED"
    elif parent[0].realized_parent is None:
        disposition = "POLICY_NONATTEMPT"
    elif parent_data is None or parent_data[refinement - 1].checkpoint is None:
        disposition = "HANDOFF_UNAVAILABLE"
    else:
        native_parent = parent_data[refinement - 1]
        handoff = bind_prepared_native_handoff(native_parent)
        handoff_identity = ObjectIdentity.from_record(handoff.handoff_id, handoff)
        observed_handoff = project_prepared_handoff(
            common,
            handoff,
            instrument_tier=(
                "I1" if config.development_library.selected_structure == "mechanism-i1" else "I0"
            ),
        )
        arrays["history"] = observed_handoff.history
        if observed_handoff.sketch is not None:
            arrays["sketch"] = observed_handoff.sketch
        arrays["handoff_displacement"] = observed_handoff.preparent_displacement
        arrays["parent_absolute_density_work"][0] = observed_handoff.parent_absolute_density_work
        by_word = {value.word: value for value in invocations if value.phase == "future"}
        futures = tuple(by_word[word] for word in config.native_spec.words)
        hold_task = next(value for value in futures if value.word is not None and value.word.sign == 0)
        hold_pair = data[hold_task.task_id]
        hold = None if hold_pair is None else hold_pair[refinement - 1]
        for index, invocation in enumerate(futures):
            pair = data[invocation.task_id]
            if pair is None:
                continue
            future = pair[refinement - 1]
            projected = project_prepared_future(
                common,
                handoff,
                future,
                matched_hold=hold,
            )
            arrays["observed"][index] = projected.outputs
            arrays["known"][index] = projected.known
            arrays["delivery_complete"][index] = projected.delivery_complete
            error = _force_error(future)
            arrays["force_component_error"][index] = np.nan if error is None else float(error)
        complete = (
            np.isfinite(arrays["parent_absolute_density_work"]).all()
            and (
                (arrays["parent_absolute_density_work"] >= 0)
                & (arrays["parent_absolute_density_work"] <= 32)
            ).all()
            and arrays["delivery_complete"].all()
            and arrays["known"].all()
            and np.isfinite(arrays["observed"]).all()
            and np.isfinite(arrays["force_component_error"]).all()
            and (
                arrays["force_component_error"] <= float(config.force_component_tolerance)
            ).all()
        )
        if not complete:
            disposition = "INCOMPLETE_NATIVE_CHART"
    return PreparedResponseCalibrationViewProjection(
        f"{root.root_id}.policy.{policy_id}.project.r{refinement}",
        ObjectIdentity.from_record(config.config_id, config),
        root,
        policy_id,
        decision_identity,
        parent[0].realized_parent,
        source_results,
        None if common is None else ObjectIdentity.from_record(common.common_start_id, common),
        handoff_identity,
        refinement,
        disposition,
        _blocks(arrays),
    )


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationCalibrationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-calibration-calibration-config'
    config_id: str
    projection: PreparedResponseCalibrationProjectionConfig
    statistics: PreparedStatisticalSpec
    simultaneous_scope: str = "ROOT_MAX_POLICY_VIEW_WORD_READOUT_OUTPUT"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.statistics.calibration_roots_per_context != 96
            or self.simultaneous_scope != "ROOT_MAX_POLICY_VIEW_WORD_READOUT_OUTPUT"
        ):
            raise ValueError("prepared fresh calibration calibration changes its protected simultaneous scope")


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationContextCalibration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-calibration-context-calibration'
    calibration_id: str
    context: str
    coefficient_product: ObjectIdentity
    projection_reports: tuple[ObjectIdentity, ...]
    assigned_roots: int
    finite_scores: int
    order_index: int
    calibrated_multiplier: Decimal | None
    score_bytes_base64: str
    disposition: str

    def __post_init__(self) -> None:
        validate_stable_id(self.calibration_id, field_name="calibration_id")
        if (
            self.context not in ("assembling", "prepared")
            or self.coefficient_product.object_schema
            != 'empirical-lawhood/prepared-response/prepared-bilinear-predictor-coefficients'
            or len(self.projection_reports) != 96 * len(CALIBRATION_POLICIES) * 2
            or any(
                value.object_schema != PreparedResponseCalibrationViewProjection.SCHEMA
                for value in self.projection_reports
            )
            or self.assigned_roots != 96
            or not 0 <= self.finite_scores <= 96
            or self.order_index != conformal_order_index(96)
            or self.disposition not in ("CALIBRATED", "NOT_CALIBRATABLE")
        ):
            raise ValueError("prepared fresh calibration context calibration changes its complete census")
        scores = self.scores()
        expected = float(np.sort(scores)[self.order_index - 1])
        if self.disposition == "CALIBRATED":
            if (
                self.calibrated_multiplier is None
                or not self.calibrated_multiplier.is_finite()
                or float(self.calibrated_multiplier) != expected
            ):
                raise ValueError("prepared fresh calibration multiplier differs from the frozen order statistic")
        elif self.calibrated_multiplier is not None or np.isfinite(expected):
            raise ValueError("uncalibratable fresh calibration result fabricates a finite prediction multiplier")

    def scores(self) -> Array:
        raw = base64.b64decode(self.score_bytes_base64, validate=True)
        if len(raw) != 96 * 8 or base64.b64encode(raw).decode("ascii") != self.score_bytes_base64:
            raise ValueError("prepared fresh calibration scores change their exact float64 byte census")
        values = np.frombuffer(raw, dtype="<f8")
        if np.isnan(values).any() or np.any(values < 0):
            raise ValueError("prepared fresh calibration scores must be nonnegative or explicit infinity")
        if int(np.isfinite(values).sum()) != self.finite_scores:
            raise ValueError("prepared fresh calibration finite-score census differs")
        return values


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationCalibratedLibrary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-calibration-calibrated-library'
    library_id: str
    config: PreparedResponseCalibrationCalibrationConfig
    contexts: tuple[PreparedResponseCalibrationContextCalibration, ...]
    disposition: str
    grants_law_or_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.library_id, field_name="library_id")
        calibrated = all(value.disposition == "CALIBRATED" for value in self.contexts)
        if (
            tuple(value.context for value in self.contexts) != ("assembling", "prepared")
            or self.disposition
            != ("CALIBRATED_FOR_C_R" if calibrated else "NOT_CALIBRATABLE")
            or self.grants_law_or_authority
        ):
            raise ValueError("prepared fresh calibration library changes its calibration or authority status")


def calibrate_prepared_response_calibration(
    config: PreparedResponseCalibrationCalibrationConfig,
    reports: tuple[PreparedResponseCalibrationViewProjection, ...],
) -> PreparedResponseCalibrationCalibratedLibrary:
    library = config.projection.development_library
    projection_identity = ObjectIdentity.from_record(config.projection.config_id, config.projection)
    if any(value.config != projection_identity for value in reports):
        raise ValueError("prepared fresh calibration calibration received a different projection configuration")
    values = []
    for context_name, coefficients in zip(
        ("assembling", "prepared"), library.selected_coefficients, strict=True
    ):
        roots = tuple(
            root
            for root in config.projection.native_spec.roots
            if root.context == context_name
        )
        expected = tuple(
            (root, policy, refinement)
            for root in roots
            for policy in CALIBRATION_POLICIES
            for refinement in (1, 2)
        )
        selected = tuple(value for value in reports if value.root.context == context_name)
        if tuple((value.root, value.policy_id, value.refinement) for value in selected) != expected:
            raise ValueError("prepared fresh calibration calibration requires every root/policy/view report")
        arrays = [value.arrays() for value in selected]
        history = np.stack([value["history"] for value in arrays]).reshape(96, 4, 2, 16, 12)
        sketch = np.stack([value["sketch"] for value in arrays]).reshape(96, 4, 2, 8)
        observed = np.stack([value["observed"] for value in arrays]).reshape(
            96, 4, 2, 9, 5, 7
        )
        complete = np.asarray(
            [value.disposition == "COMPLETE" for value in selected], dtype=bool
        ).reshape(96, 4, 2)
        model, scale = coefficients.predictors()
        flat_history = history.reshape(-1, 16, 12)
        flat_sketch = sketch.reshape(-1, 8)
        predicted = np.full(observed.shape, np.nan)
        widths = np.full(observed.shape, np.nan)
        valid = complete.reshape(-1) & np.isfinite(flat_history).all(axis=(1, 2))
        if coefficients.structure == "mechanism-i1":
            valid &= np.isfinite(flat_sketch).all(axis=1)
        if np.any(valid):
            predicted.reshape(-1, 9, 5, 7)[valid] = model.predict(
                flat_history[valid],
                flat_sketch[valid] if coefficients.structure == "mechanism-i1" else None,
            )
            widths.reshape(-1, 9, 5, 7)[valid] = scale.predict(
                flat_history[valid],
                flat_sketch[valid] if coefficients.structure == "mechanism-i1" else None,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = abs(observed - predicted) / widths
        residual[~np.isfinite(residual)] = np.inf
        scores = np.max(residual, axis=(1, 2, 3, 4, 5))
        order = conformal_order_index(96)
        quantile = float(np.sort(scores)[order - 1])
        identities = tuple(
            ObjectIdentity.from_record(value.report_id, value) for value in selected
        )
        values.append(
            PreparedResponseCalibrationContextCalibration(
                f"prepared-response.fresh-response-calibration.calibration.{context_name}",
                context_name,
                ObjectIdentity.from_record(
                    f"prepared-response.dependent-refinement.coefficients.{context_name}", coefficients
                ),
                identities,
                96,
                int(np.isfinite(scores).sum()),
                order,
                Decimal(format(quantile, ".17g")) if np.isfinite(quantile) else None,
                base64.b64encode(np.asarray(scores, dtype="<f8").tobytes()).decode("ascii"),
                "CALIBRATED" if np.isfinite(quantile) else "NOT_CALIBRATABLE",
            )
        )
    contexts = tuple(values)
    return PreparedResponseCalibrationCalibratedLibrary(
        "prepared-response.fresh-response-calibration.calibrated-library",
        config,
        contexts,
        "CALIBRATED_FOR_C_R"
        if all(value.disposition == "CALIBRATED" for value in contexts)
        else "NOT_CALIBRATABLE",
    )
