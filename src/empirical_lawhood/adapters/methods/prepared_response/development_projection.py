"""dependent refinement-only scalar training products from the complete authenticated native census.

Portable scalar blocks contain no native matrices or checkpoints. Root seeds
remain development lineage and are never part of a deployable predictor.
"""

import base64
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedNativeSpec, PreparedRoot
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_mechanism_sketch, prepared_observation_history
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedCommonStart, PreparedNativeHandoff, bind_prepared_native_handoff
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import prepared_static_native_invocations
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import state_from_checkpoint
from .models import PreparedTrainingPanel, validate_prepared_training_scalar_labels
from .native_projection import authenticate_prepared_root
from .projection import project_prepared_response_development_transitions, project_prepared_future, project_prepared_handoff
from .qualification import PreparedResponseSourceQualificationEvaluation
from .qualification_projection import _force_error


MAXIMUM_DEVELOPMENT_VIEW_PROJECTION_BYTES = 768 * 1024
ScalarArray = npt.NDArray[np.float64] | npt.NDArray[np.bool_]
_VIEW_SHAPES = dict(
    sorted(
        {
            "preparent_history": (16, 12),
            "preparent_sketch": (8,),
            "history": (5, 16, 12),
            "sketch": (5, 8),
            "handoff_displacement": (5, 2),
            "parent_absolute_density_work": (5,),
            "observed": (5, 9, 5, 7),
            "transitions": (5, 9, 21, 12),
            "hold_replicates": (5, 4, 5, 7),
            "delivery_complete": (5, 12),
            "force_component_error": (5, 12),
            "fit_valid": (5,),
        }.items()
    )
)
_BOOL = frozenset(("delivery_complete", "fit_valid"))
_ENCODING = "BASE64_LITTLE_ENDIAN_FLOAT64_OR_UINT8_BOOL_C_ORDER_FIXED_SHAPES_NAN_UNKNOWN"


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-projection-config'
    config_id: str
    native_spec: PreparedNativeSpec
    source_qualification: PreparedResponseSourceQualificationEvaluation
    feature_rule: str = "CAUSAL_I0_AND_SEPARATE_I1_PREPARENT_AND_HANDOFF"
    missing_rule: str = "RETAIN_ALL_ASSIGNED_ROOTS_AND_PATHS_NAN_UNKNOWN_NO_IMPUTATION"
    force_component_tolerance: Decimal = Decimal("0.000000000001")

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.native_spec.stage != 'development'
            or self.source_qualification.selected_charter is None
            or self.native_spec.selected_amplitude != self.source_qualification.selected_charter.amplitude
            or self.feature_rule != "CAUSAL_I0_AND_SEPARATE_I1_PREPARENT_AND_HANDOFF"
            or self.missing_rule != "RETAIN_ALL_ASSIGNED_ROOTS_AND_PATHS_NAN_UNKNOWN_NO_IMPUTATION"
            or type(self.force_component_tolerance) is not Decimal
            or self.force_component_tolerance != Decimal("0.000000000001")
        ):
            raise ValueError(
                "prepared dependent refinement projection changes its source qualification nomination or fixed scalar contract"
            )


def _blocks(arrays: dict[str, ScalarArray]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            name,
            base64.b64encode(
                np.asarray(arrays[name], dtype="u1" if name in _BOOL else "<f8").tobytes(order="C")
            ).decode("ascii"),
        )
        for name in _VIEW_SHAPES
    )


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentViewProjection(CanonicalRecord):
    "One root/numerical-view dependent refinement product; the paired source acquisition is shared."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-development-view-projection'
    config: PreparedResponseDevelopmentProjectionConfig
    root: PreparedRoot
    refinement: int
    source_results: tuple[ObjectIdentity, ...]
    common_start: ObjectIdentity | None
    handoffs: tuple[ObjectIdentity | None, ...]
    mode_disposition: str
    blocks: tuple[tuple[str, str], ...]
    encoding: str = _ENCODING

    def __post_init__(self) -> None:
        expected = tuple(
            f"{task.task_id}.result"
            for task in prepared_static_native_invocations(self.config.native_spec, root=self.root)
        )
        if (
            self.root not in self.config.native_spec.roots
            or self.refinement not in (1, 2)
            or len(expected) != 66
            or tuple(value.object_id for value in self.source_results) != expected
            or any(
                value.object_schema != PreparedNativeTaskResult.SCHEMA
                for value in self.source_results
            )
            or len(self.handoffs) != 5
            or any(
                value is not None and value.object_schema != PreparedNativeHandoff.SCHEMA
                for value in self.handoffs
            )
            or self.mode_disposition
            not in ("PREFIX_UNAVAILABLE", "NONATTEMPT_UNRESOLVED_PORT_FRAME", "RESOLVED")
            or (self.common_start is None) != (self.mode_disposition == "PREFIX_UNAVAILABLE")
            or self.common_start is not None
            and self.common_start.object_schema != PreparedCommonStart.SCHEMA
            or self.encoding != _ENCODING
            or tuple(name for name, _ in self.blocks) != tuple(_VIEW_SHAPES)
        ):
            raise ValueError("prepared dependent refinement view product changes its source or root/view census")
        arrays = self.arrays()
        if self.mode_disposition != "RESOLVED" and (
            any(value is not None for value in self.handoffs)
            or any(
                np.any(value) if name in _BOOL else not np.isnan(value).all()
                for name, value in arrays.items()
            )
        ):
            raise ValueError("an unavailable dependent refinement view cannot expose measured scalar inputs")
        for parent, handoff in enumerate(self.handoffs):
            if handoff is None and (
                arrays["fit_valid"][parent]
                or arrays["delivery_complete"][parent].any()
                or not np.isnan(arrays["history"][parent]).all()
            ):
                raise ValueError("dependent refinement view measurements require their authenticated handoff")
        expected_valid = (
            np.isfinite(arrays["history"]).all(axis=(-1, -2))
            & np.isfinite(arrays["observed"]).all(axis=(-1, -2, -3))
            & np.isfinite(arrays["transitions"]).all(axis=(-1, -2, -3))
            & arrays["delivery_complete"][..., :9].all(axis=-1)
            & (
                arrays["force_component_error"][..., :9]
                <= float(self.config.force_component_tolerance)
            ).all(axis=-1)
        )
        if not np.array_equal(arrays["fit_valid"], expected_valid):
            raise ValueError("dependent refinement view validity must derive from its complete delivered chart")
        validate_prepared_training_scalar_labels(
            np.asarray(arrays["history"], dtype=np.float64),
            np.asarray(arrays["observed"], dtype=np.float64),
            np.asarray(arrays["transitions"], dtype=np.float64),
            np.asarray(arrays["fit_valid"], dtype=np.bool_),
        )
        if len(self.canonical_bytes()) > MAXIMUM_DEVELOPMENT_VIEW_PROJECTION_BYTES:
            raise ValueError("prepared dependent refinement view product exceeds its bounded portable size")

    @property
    def report_id(self) -> str:
        return f"{self.root.root_id}.project.r{self.refinement}"

    def arrays(self) -> dict[str, ScalarArray]:
        result = {}
        for name, encoded in self.blocks:
            shape = _VIEW_SHAPES[name]
            size = int(np.prod(shape)) * (1 if name in _BOOL else 8)
            if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
                raise ValueError("dependent refinement view scalar block changes its exact encoded byte count")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) != size or base64.b64encode(raw).decode("ascii") != encoded:
                raise ValueError("dependent refinement view scalar block changes its canonical encoding")
            value = np.frombuffer(raw, dtype="u1" if name in _BOOL else "<f8").reshape(shape)
            if name in _BOOL:
                if np.any(value > 1):
                    raise ValueError("dependent refinement view Boolean block has a non-Boolean byte")
                value = np.frombuffer(value.tobytes(), dtype=np.bool_).reshape(shape)
            elif np.isinf(value).any():
                raise ValueError("dependent refinement view scalar operands must be finite or explicit NaN unknown")
            result[name] = value
        return result


def project_prepared_response_development_view(
    config: PreparedResponseDevelopmentProjectionConfig,
    root: PreparedRoot,
    refinement: int,
    inputs: tuple[tuple[PreparedNativeTaskResult, bytes], ...],
) -> PreparedResponseDevelopmentViewProjection:
    """Authenticate the complete paired census, then reduce only the requested view."""
    if refinement not in (1, 2):
        raise ValueError("prepared dependent refinement projection has exactly two numerical views")
    authenticated = authenticate_prepared_root(config.native_spec, root, inputs)
    common, data = authenticated.common, authenticated.data
    arrays = {
        name: np.zeros(shape, dtype=bool) if name in _BOOL else np.full(shape, np.nan)
        for name, shape in _VIEW_SHAPES.items()
    }
    handoffs: list[ObjectIdentity | None] = [None] * len(PARENTS)
    v = refinement - 1
    words = config.native_spec.words
    assignments = tuple(('common-response', word) for word in words) + tuple(
        (purpose, words[0]) for purpose in ('independent-response-1', 'independent-response-2', 'independent-response-3')
    )
    if common is not None and common.frame is not None:
        checkpoint = common.checkpoints[v]
        positions = _decode(checkpoint.history_positions_base64, (31, 2, 3, 4, 4))
        momenta = _decode(checkpoint.history_momenta_base64, (31, 2, 3, 4, 4))
        state, _ = state_from_checkpoint(checkpoint.native)
        arrays["preparent_history"] = prepared_observation_history(
            frame=common.frame,
            ticks=checkpoint.history_ticks,
            positions=positions,
            momenta=momenta,
            cutoff_tick=root.landmark,
        )
        # A failed sketch does not invalidate the separately measured I0 channel.
        try:
            arrays["preparent_sketch"] = prepared_mechanism_sketch(state, common.frame)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            pass
        for p, parent in enumerate(PARENTS):
            pair = data[f"{root.root_id}.{parent}.parent.native"]
            if pair is None:
                continue
            native_parent = pair[v]
            arrays["parent_absolute_density_work"][p] = float(
                native_parent.delivery.parent_absolute_density_work
            )
            if native_parent.checkpoint is None:
                continue
            handoff = bind_prepared_native_handoff(native_parent)
            handoffs[p] = ObjectIdentity.from_record(handoff.handoff_id, handoff)
            observed = project_prepared_handoff(common, handoff, instrument_tier="I0")
            arrays["history"][p] = observed.history
            arrays["handoff_displacement"][p] = observed.preparent_displacement
            arrays["parent_absolute_density_work"][p] = observed.parent_absolute_density_work
            state, _ = state_from_checkpoint(handoff.checkpoint.native)
            try:
                arrays["sketch"][p] = prepared_mechanism_sketch(state, common.frame)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                pass
            hold_pair = data[f"{root.root_id}.{parent}.common-response.{words[0].word_id}.native"]
            hold = None if hold_pair is None else hold_pair[v]
            for w, (purpose, word) in enumerate(assignments):
                pair = data[f"{root.root_id}.{parent}.{purpose}.{word.word_id}.native"]
                if pair is None:
                    continue
                future = pair[v]
                observation = project_prepared_future(
                    common, handoff, future, matched_hold=hold if w < 9 else future
                )
                arrays["delivery_complete"][p, w] = observation.delivery_complete
                error = _force_error(future)
                arrays["force_component_error"][p, w] = np.nan if error is None else float(error)
                if w < 9:
                    arrays["observed"][p, w] = observation.outputs
                    arrays["transitions"][p, w] = project_prepared_response_development_transitions(
                        common, handoff, future
                    )
                if w == 0 or w >= 9:
                    arrays["hold_replicates"][p, 0 if w == 0 else w - 8] = observation.outputs
        arrays["fit_valid"] = np.asarray(
            np.isfinite(arrays["history"]).all(axis=(-1, -2))
            & np.isfinite(arrays["observed"]).all(axis=(-1, -2, -3))
            & np.isfinite(arrays["transitions"]).all(axis=(-1, -2, -3))
            & arrays["delivery_complete"][..., :9].all(axis=-1)
            & (
                arrays["force_component_error"][..., :9] <= float(config.force_component_tolerance)
            ).all(axis=-1),
            dtype=np.bool_,
        )
    return PreparedResponseDevelopmentViewProjection(
        config,
        root,
        refinement,
        authenticated.identities,
        None if common is None else ObjectIdentity.from_record(common.common_start_id, common),
        tuple(handoffs),
        "PREFIX_UNAVAILABLE" if common is None else common.mode_disposition,
        _blocks(arrays),
    )


def collect_prepared_response_development_view_panel(
    config: PreparedResponseDevelopmentProjectionConfig,
    context: str,
    reports: tuple[PreparedResponseDevelopmentViewProjection, ...],
) -> PreparedTrainingPanel:
    """Join both declared views for each root without treating views as replication."""
    roots = tuple(root for root in config.native_spec.roots if root.context == context)
    expected = tuple((root, refinement) for root in roots for refinement in (1, 2))
    if (
        len(roots) != 64
        or tuple((report.root, report.refinement) for report in reports) != expected
        or any(report.config != config for report in reports)
    ):
        raise ValueError("dependent refinement panel requires both view products for all 64 assigned roots")
    arrays = tuple(report.arrays() for report in reports)

    def paired(name: str) -> ScalarArray:
        value = np.stack(
            tuple(
                np.stack(
                    (arrays[2 * index][name], arrays[2 * index + 1][name]), axis=1
                )
                for index in range(len(roots))
            )
        )
        return cast(
            ScalarArray,
            np.asarray(value, dtype=np.bool_ if name in _BOOL else np.float64),
        )

    return PreparedTrainingPanel(
        roots,
        np.asarray(paired("history"), dtype=np.float64),
        np.asarray(paired("sketch"), dtype=np.float64),
        np.asarray(paired("observed"), dtype=np.float64),
        np.asarray(paired("transitions"), dtype=np.float64),
        np.asarray(paired("fit_valid"), dtype=np.bool_),
    )


def decode_prepared_response_development_view_projection(payload: bytes) -> PreparedResponseDevelopmentViewProjection:
    return decode_canonical_bytes(
        payload,
        PreparedResponseDevelopmentViewProjection,
        maximum_bytes=MAXIMUM_DEVELOPMENT_VIEW_PROJECTION_BYTES,
    )
