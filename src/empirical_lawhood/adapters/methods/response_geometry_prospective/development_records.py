"""development task payloads at existing artifact ports, preserving fit/calibration cutoffs."""

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import io
import json
from typing import Any, ClassVar, cast

import h5py  # type: ignore[import-untyped]
import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.adapters.methods.causal_access_tournament import StandardizedRidge
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_PANEL_ID
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_group, response_hdf5_read_text, response_hdf5_text, response_hdf5_writer

from .development_assessment import ResponseGeometryDevelopmentCalibration, ResponseGeometryDevelopmentSupportFit
from .development_models import REPRESENTATIONS, ResponseGeometryDevelopmentAffineModel, ResponseGeometryDevelopmentCandidateScore, ResponseGeometryDevelopmentFeatureMap, ResponseGeometryDevelopmentMeasuredView, ResponseGeometryDevelopmentModelGroup, FloatArray, Representation
from .development_projection import ResponseGeometryDevelopmentProjectionConfig, ResponseGeometryDevelopmentViewReport


DEVELOPMENT_FIT_SCHEMA = 'empirical-lawhood/methods/response-geometry-prospective/development-fitted-models-hdf5'

DEVELOPMENT_FIT_METADATA = {
    "empirical_lawhood_payload_schema": DEVELOPMENT_FIT_SCHEMA,
    "empirical_lawhood_units": '{"drift":"latent-per-native-time","diffusion":"latent-squared-per-native-time","errors":"native-X-HS-displacement"}',
    "empirical_lawhood_frames": '{"features":"frozen-training-normalization-and-basis","state":"mode-position,mode-momentum,latent"}',
    "empirical_lawhood_clocks": '{"model":"native-Langevin-time","training":"root-indices-0-through-15"}',
    "empirical_lawhood_keys": '["context","representation","parent","training-roots"]',
}
DEVELOPMENT_FIT_MAXIMUM_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-method-config'
    config_id: str
    projection_config: ObjectIdentity
    design_packet_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.design_packet_sha256, field_name="design_packet_sha256")
        if self.projection_config.object_schema != ResponseGeometryDevelopmentProjectionConfig.SCHEMA:
            raise ValueError("development methods require their exact projection config")


def _inputs(
    context: str, config: ObjectIdentity, reports: tuple[ObjectIdentity, ...], indices: range
) -> None:
    expected = tuple(
        sorted(f"report.{DEVELOPMENT_PANEL_ID}.{context}.r{i:02d}.r{r}" for i in indices for r in (1, 2))
    )
    if context not in ("assembling", "prepared") or config.object_schema != ResponseGeometryDevelopmentMethodConfig.SCHEMA:
        raise ValueError("development method result changes its context/config")
    if tuple(r.object_id for r in reports) != expected or any(
        r.object_schema != ResponseGeometryDevelopmentViewReport.SCHEMA for r in reports
    ):
        raise ValueError("development method result changes its exact independent-root input role")


def report_identities(views: tuple[ResponseGeometryDevelopmentMeasuredView, ...]) -> tuple[ObjectIdentity, ...]:
    return tuple(
        sorted(
            (ObjectIdentity.from_record(v.report.report_id, v.report) for v in views),
            key=lambda v: v.object_id,
        )
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentFitResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-fit-result'
    context: str
    config: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    available_representations: tuple[str, ...]
    data_sha256: str

    def __post_init__(self) -> None:
        _inputs(self.context, self.config, self.input_reports, range(16))
        if self.available_representations != tuple(
            r for r in REPRESENTATIONS if r in self.available_representations
        ):
            raise ValueError("development fit changes its available representation roster")
        validate_sha256(self.data_sha256, field_name="data_sha256")

    @property
    def result_id(self) -> str:
        return f"{DEVELOPMENT_PANEL_ID}.fit.{self.context}"


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentCalibrationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-calibration-result'
    config: ObjectIdentity
    fit_result: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    calibration: ResponseGeometryDevelopmentCalibration

    def __post_init__(self) -> None:
        _inputs(self.calibration.context, self.config, self.input_reports, range(16, 32))
        if (
            self.fit_result.object_id != f"{DEVELOPMENT_PANEL_ID}.fit.{self.calibration.context}"
            or self.fit_result.object_schema != ResponseGeometryDevelopmentFitResult.SCHEMA
        ):
            raise ValueError("development calibration requires the preceding frozen fit result")

    @property
    def result_id(self) -> str:
        return f"{DEVELOPMENT_PANEL_ID}.calibrate.{self.calibration.context}"


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentSupportRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-support-row'
    representation: str
    parent: str
    roots: tuple[int, ...]
    labels: tuple[int | None, ...]
    predictor_json: str | None
    training_prevalence: Decimal | None
    reason: str | None

    def __post_init__(self) -> None:
        if (
            self.representation not in REPRESENTATIONS
            or self.parent not in PARENTS
            or self.roots != tuple(range(16))
        ):
            raise ValueError("development support record changes its chart/root role")
        if len(self.labels) != 16 or any(
            v is not None and (type(v) is not int or v not in (0, 1)) for v in self.labels
        ):
            raise ValueError("development support labels must retain binary or unavailable root outcomes")
        if self.predictor_json is not None:
            if len(self.predictor_json) > 16384:
                raise ValueError("development support predictor exceeds its bounded serialization")
            predictor = StandardizedRidge.from_mapping(json.loads(self.predictor_json))
            if (
                predictor.input_dimension not in (11, 19)
                or predictor.ridge_alpha != 1.0
                or json.dumps(
                    predictor.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                != self.predictor_json
            ):
                raise ValueError("development support predictor changes its fixed linear family")
        if self.training_prevalence is not None and (
            not self.training_prevalence.is_finite() or not 0 <= self.training_prevalence <= 1
        ):
            raise ValueError("development support prevalence must be finite and bounded")
        if (self.predictor_json is None) != (self.training_prevalence is None) or (
            self.predictor_json is not None and sum(v is not None for v in self.labels) < 12
        ):
            raise ValueError("development support predictor lacks its complete training label denominator")

    @classmethod
    def from_fit(cls, fitted: ResponseGeometryDevelopmentSupportFit) -> 'ResponseGeometryDevelopmentSupportRow':
        return cls(
            fitted.representation,
            fitted.parent,
            fitted.roots,
            fitted.labels,
            None
            if fitted.predictor is None
            else json.dumps(
                fitted.predictor.to_mapping(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            None
            if fitted.training_prevalence is None
            else Decimal(str(fitted.training_prevalence)),
            fitted.reason,
        )

    def as_fit(self, context: str) -> ResponseGeometryDevelopmentSupportFit:
        return ResponseGeometryDevelopmentSupportFit(
            context,
            cast(Representation, self.representation),
            self.parent,
            self.roots,
            self.labels,
            None
            if self.predictor_json is None
            else StandardizedRidge.from_mapping(json.loads(self.predictor_json)),
            None if self.training_prevalence is None else float(self.training_prevalence),
            self.reason,
        )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentSupportResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-support-result'
    context: str
    config: ObjectIdentity
    fit_result: ObjectIdentity
    calibration_result: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    calibration_input_reports: tuple[ObjectIdentity, ...]
    outer_calibrations: tuple[ResponseGeometryDevelopmentCalibration | None, ...]
    models: tuple[ResponseGeometryDevelopmentSupportRow, ...]

    def __post_init__(self) -> None:
        _inputs(self.context, self.config, self.input_reports, range(16))
        _inputs(self.context, self.config, self.calibration_input_reports, range(16, 32))
        if len(self.outer_calibrations) != 4 or any(
            c is not None and c.context != self.context for c in self.outer_calibrations
        ):
            raise ValueError("development support must retain the four outer-fold calibrations")
        if (
            self.fit_result.object_schema != ResponseGeometryDevelopmentFitResult.SCHEMA
            or self.fit_result.object_id != f"{DEVELOPMENT_PANEL_ID}.fit.{self.context}"
            or self.calibration_result.object_schema != ResponseGeometryDevelopmentCalibrationResult.SCHEMA
            or self.calibration_result.object_id != f"{DEVELOPMENT_PANEL_ID}.calibrate.{self.context}"
        ):
            raise ValueError("development support fit changes preceding fit/calibration lineage")
        if tuple((m.representation, m.parent) for m in self.models) != tuple(
            (r, p) for r in REPRESENTATIONS for p in PARENTS
        ):
            raise ValueError("development support fit must retain all 25 fitted or refused charts")

    @property
    def result_id(self) -> str:
        return f"{DEVELOPMENT_PANEL_ID}.support.{self.context}"


def write_response_geometry_development_fit(
    config: ResponseGeometryDevelopmentMethodConfig,
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    groups: tuple[ResponseGeometryDevelopmentModelGroup, ...],
) -> tuple[ResponseGeometryDevelopmentFitResult, bytes]:
    if (
        tuple(g.representation for g in groups) != REPRESENTATIONS
        or len({g.context for g in groups}) != 1
        or any(v.report.projection_config != config.projection_config for v in views)
    ):
        raise ValueError("development fit publication changes its model/config roster")
    context = groups[0].context
    output = io.BytesIO()
    with response_hdf5_writer(output) as artifact:
        for name, value in {
            "schema": DEVELOPMENT_FIT_SCHEMA,
            "context": context,
            "method_config_sha256": config.fingerprint(),
            **DEVELOPMENT_FIT_METADATA,
        }.items():
            response_hdf5_text(artifact, name, value)
        for group in groups:
            target = response_hdf5_group(artifact, group.representation)
            metadata = dict(
                roots=group.roots,
                selected_index=group.selected_index,
                reason=group.reason,
                candidates=[
                    dict(
                        dimension=s.dimension,
                        ridge=s.ridge,
                        split=s.split_at_force_off,
                        reason=s.reason,
                    )
                    for s in group.scores
                ],
                parents=[m.parent for m in group.models],
            )
            response_hdf5_text(
                target,
                "metadata",
                json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False),
            )
            errors = (
                np.stack(
                    [
                        np.full((len(group.roots), 5, 2, 3), np.nan)
                        if s.errors is None
                        else s.errors
                        for s in group.scores
                    ]
                )
                if group.scores
                else np.empty((0, len(group.roots), 5, 2, 3))
            )
            target.create_dataset("cv_errors", data=errors, track_times=False)
            for model in group.models:
                child = response_hdf5_group(target, model.parent)
                feature = model.feature_map
                arrays = dict(
                    center=feature.center,
                    scale=feature.scale,
                    scalar_regression=feature.scalar_regression,
                    basis=feature.basis,
                    feature_singular_values=feature.singular_values,
                    drift=model.drift,
                    diffusion=model.diffusion,
                )
                for phase, (sv, unidentified) in enumerate(
                    zip(model.design_singular_values, model.unidentified_directions, strict=True)
                ):
                    arrays[f"design_singular_{phase}"] = sv
                    arrays[f"unidentified_{phase}"] = unidentified
                for name, array in arrays.items():
                    child.create_dataset(name, data=array, track_times=False)
    payload = output.getvalue()
    if len(payload) > DEVELOPMENT_FIT_MAXIMUM_BYTES:
        raise ValueError("development fitted model artifact exceeds its 64-MiB bound")
    result = ResponseGeometryDevelopmentFitResult(
        context,
        ObjectIdentity.from_record(config.config_id, config),
        report_identities(views),
        tuple(g.representation for g in groups if g.models),
        sha256(payload).hexdigest(),
    )
    return result, payload


def _group(parent: Any, name: str) -> Any:
    if not isinstance(parent.get(name, getlink=True), h5py.HardLink) or not isinstance(
        parent[name], h5py.Group
    ):
        raise ValueError("development fitted model groups must be local hard links")
    return parent[name]


def _array(group: Any, name: str, shape: tuple[int | None, ...]) -> FloatArray:
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise ValueError("development fitted arrays cannot link to another object")
    value = group[name]
    if (
        not isinstance(value, h5py.Dataset)
        or value.is_virtual
        or value.external
        or value.id.get_create_plist().get_nfilters()
        or value.dtype != np.dtype("float64")
    ):
        raise ValueError("development fitted arrays require unfiltered local float64 storage")
    if (
        len(value.shape) != len(shape)
        or any(
            (actual != expected if expected is not None else not 0 <= actual <= 16384)
            for actual, expected in zip(value.shape, shape, strict=True)
        )
        or value.size > 262144
    ):
        raise ValueError("development fitted array exceeds its declared geometry")
    return np.asarray(value[...], dtype=np.float64)


def read_response_geometry_development_fit(result: ResponseGeometryDevelopmentFitResult, payload: bytes) -> tuple[ResponseGeometryDevelopmentModelGroup, ...]:
    if len(payload) > DEVELOPMENT_FIT_MAXIMUM_BYTES or sha256(payload).hexdigest() != result.data_sha256:
        raise ValueError("development fitted model artifact fails its size/digest binding")
    groups = []
    with h5py.File(io.BytesIO(payload), "r") as artifact:
        if (
            response_hdf5_read_text(artifact, "schema") != DEVELOPMENT_FIT_SCHEMA
            or response_hdf5_read_text(artifact, "context") != result.context
            or response_hdf5_read_text(artifact, "method_config_sha256")
            != result.config.object_fingerprint
            or set(artifact) != set(REPRESENTATIONS)
        ):
            raise ValueError("development fitted model artifact changes its identity/roster")
        for representation in REPRESENTATIONS:
            group = _group(artifact, representation)
            raw_metadata = response_hdf5_read_text(group, "metadata")
            if len(raw_metadata) > 16384:
                raise ValueError("development fitted model metadata exceeds its bound")
            metadata = json.loads(raw_metadata)
            if (
                not isinstance(metadata, dict)
                or set(metadata) != {"roots", "selected_index", "reason", "candidates", "parents"}
                or len(metadata["candidates"]) not in (0, 6, 12)
                or metadata["parents"] not in ([], list(PARENTS))
                or set(group) != {"cv_errors", *metadata["parents"]}
            ):
                raise ValueError("development fitted model metadata changes its closed fields/roster")
            roots = tuple(metadata["roots"])
            if any(type(i) is not int for i in roots) or len(roots) > 16:
                raise ValueError("development fit metadata changes its independent-root role")
            errors = _array(group, "cv_errors", (len(metadata["candidates"]), len(roots), 5, 2, 3))
            scores = []
            for index, row in enumerate(metadata["candidates"]):
                if (
                    set(row) != {"dimension", "ridge", "split", "reason"}
                    or row["dimension"] not in (4, 8)
                    or type(row["split"]) is not bool
                ):
                    raise ValueError("development candidate score changes its closed family")
                if (row["reason"] is None and not np.isfinite(errors[index]).all()) or (
                    row["reason"] is not None and not np.isnan(errors[index]).all()
                ):
                    raise ValueError("development score failure and measured errors differ")
                scores.append(
                    ResponseGeometryDevelopmentCandidateScore(
                        row["dimension"],
                        row["ridge"],
                        row["split"],
                        roots,
                        None if row["reason"] is not None else errors[index],
                        row["reason"],
                    )
                )
            selected = metadata["selected_index"]
            models = []
            if metadata["parents"]:
                if (
                    type(selected) is not int
                    or not 0 <= selected < len(scores)
                    or scores[selected].errors is None
                ):
                    raise ValueError("development fitted models lack their selected measured score")
                chosen = scores[selected]
                phases = 2 if chosen.split_at_force_off else 1
                for parent in PARENTS:
                    child = _group(group, parent)
                    if set(child) != {
                        "center",
                        "scale",
                        "scalar_regression",
                        "basis",
                        "feature_singular_values",
                        "drift",
                        "diffusion",
                        *(f"design_singular_{i}" for i in range(phases)),
                        *(f"unidentified_{i}" for i in range(phases)),
                    }:
                        raise ValueError("development fitted model loses a required numerical operand")
                    center = _array(child, "center", (None,))
                    count = len(center)
                    feature = ResponseGeometryDevelopmentFeatureMap(
                        center,
                        _array(child, "scale", (count,)),
                        _array(child, "scalar_regression", (3, count)),
                        _array(child, "basis", (count, None)),
                        _array(child, "feature_singular_values", (None,)),
                        roots,
                    )
                    dimension = chosen.dimension
                    models.append(
                        ResponseGeometryDevelopmentAffineModel(
                            result.context,
                            parent,
                            representation,
                            dimension,
                            chosen.ridge,
                            chosen.split_at_force_off,
                            feature,
                            _array(child, "drift", (phases, dimension + 2, dimension)),
                            _array(child, "diffusion", (phases, dimension, dimension)),
                            tuple(
                                _array(child, f"design_singular_{i}", (dimension + 2,))
                                for i in range(phases)
                            ),
                            tuple(
                                _array(child, f"unidentified_{i}", (None, dimension + 2))
                                for i in range(phases)
                            ),
                        )
                    )
            groups.append(
                ResponseGeometryDevelopmentModelGroup(
                    result.context,
                    representation,
                    roots,
                    tuple(scores),
                    selected,
                    tuple(models),
                    metadata["reason"],
                )
            )
    if tuple(g.representation for g in groups if g.models) != result.available_representations:
        raise ValueError("development fitted model availability differs from its frozen result")
    return tuple(groups)
