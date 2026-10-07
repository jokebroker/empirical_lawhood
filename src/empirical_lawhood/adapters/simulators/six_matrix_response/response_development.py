'The fresh 128-root development panel shares qualified native mechanics.\n\nDevelopment roots have their own identity and explicit numerical inputs.\nConstruction does not grant authority or transfer qualification.\n'

from dataclasses import dataclass
from typing import Callable, ClassVar, cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_sha256, validate_stable_id

from .response_qualification import ResponseGeometryAssayNativeConfig, ResponseGeometryAssayNativeRoot, ResponseGeometryAssayNativeSegmentResult, ResponseGeometryAssayNativeSegment, INNER_SIGNS, PARENTS
from .response_scientific_inputs import require_response_scientific_inputs


DEVELOPMENT_PANEL_ID = "response-geometry-development"
DEVELOPMENT_SOURCE_CONFIG_ID = f"{DEVELOPMENT_PANEL_ID}.source-config"
DEVELOPMENT_SEED_SHA256 = "1b9b0d43a86395d1998c4ae47a136780d33170d7cbc34955e38a3cd2cba68c45"
DEVELOPMENT_HDF5_SCHEMA = 'empirical-lawhood/simulators/six-matrix-response/development-native-observations-hdf5'


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentNativeRoot(ResponseGeometryAssayNativeRoot):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-development-native-root'

    def __post_init__(self) -> None:
        landmarks = {"assembling": 1024, "prepared": 4096}
        if (
            self.context not in landmarks
            or self.landmark_tick != landmarks[self.context]
        ):
            raise ValueError("development requires assay's selected context-indexed landmark")
        if type(self.landmark_tick) is not int or type(self.index) is not int:
            raise ValueError("development root clocks and indices must be integers")
        if not 0 <= self.index < 64:
            raise ValueError("development retains exactly 64 root slots in each context")

    @property
    def root_id(self) -> str:
        return f"{DEVELOPMENT_PANEL_ID}.{self.context}.r{self.index:02d}"

    @property
    def assay(self) -> str:
        return "short-pulse-response"

    @property
    def refinements(self) -> tuple[int, ...]:
        return (1, 2)

    @property
    def restart(self) -> bool:
        return False

    @property
    def covariance(self) -> bool:
        return False

    @property
    def invocation_offset(self) -> int:
        # The same balanced schedule is used in each half of the root split.
        return 384 + 16 * (5 * (self.index % 32) % 9)


def development_roots() -> tuple[ResponseGeometryDevelopmentNativeRoot, ...]:
    return tuple(
        ResponseGeometryDevelopmentNativeRoot(context, landmark, index)
        for context, landmark in (("assembling", 1024), ("prepared", 4096))
        for index in range(64)
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentNativeConfig(ResponseGeometryAssayNativeConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-development-native-config'

    roots: tuple[ResponseGeometryDevelopmentNativeRoot, ...]
    assay_evaluation: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.design_packet_sha256, field_name="design_packet_sha256")
        validate_sha256(self.seed_root_sha256, field_name="seed_root_sha256")
        require_response_scientific_inputs(
            self.scientific_inputs, config_id=self.config_id,
            design_packet_sha256=self.design_packet_sha256,
            seed_root_sha256=self.seed_root_sha256, roots=self.roots,
        )
        if (
            self.roots != development_roots()
            or self.assay_evaluation.object_schema
            != 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-assay-evaluation'
        ):
            raise ValueError(
                "development requires all 128 roots and an explicitly bound assay evaluation; custody is required separately"
            )

    def physical_unit_id(self, root: ResponseGeometryAssayNativeRoot) -> str:
        if root not in self.roots or type(root) is not ResponseGeometryDevelopmentNativeRoot:
            raise ValueError("development physical identity is outside its exact root roster")
        return f"{DEVELOPMENT_PANEL_ID}.{self.seed_root_sha256}.{root.context}.r{root.index:02d}"

    @property
    def observation_schema(self) -> str:
        return DEVELOPMENT_HDF5_SCHEMA

    @property
    def result_type(self) -> type['ResponseGeometryDevelopmentNativeSegmentResult']:
        return ResponseGeometryDevelopmentNativeSegmentResult


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentNativeSegment(ResponseGeometryAssayNativeSegment):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-development-native-segment'
    root: ResponseGeometryDevelopmentNativeRoot


def development_segments() -> tuple[ResponseGeometryDevelopmentNativeSegment, ...]:
    records = []
    for root in development_roots():
        records.append(ResponseGeometryDevelopmentNativeSegment(root, "prefix", None, None))
        for parent in PARENTS:
            records.append(ResponseGeometryDevelopmentNativeSegment(root, "parent", parent, None))
            records.extend(
                ResponseGeometryDevelopmentNativeSegment(root, "inner", parent, sign) for sign in INNER_SIGNS
            )
    return tuple(sorted(records, key=lambda value: value.task_id))


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentNativeSegmentResult(ResponseGeometryAssayNativeSegmentResult):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-development-native-segment-result'
    SOURCE_CONFIG_SCHEMA: ClassVar[str] = ResponseGeometryDevelopmentNativeConfig.SCHEMA
    OBSERVATION_SCHEMA: ClassVar[str] = DEVELOPMENT_HDF5_SCHEMA
    segment: ResponseGeometryDevelopmentNativeSegment


def execute_development_segment(
    config: ResponseGeometryDevelopmentNativeConfig,
    segment: ResponseGeometryDevelopmentNativeSegment,
    predecessor: ResponseGeometryDevelopmentNativeSegmentResult | None,
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[ResponseGeometryDevelopmentNativeSegmentResult, bytes]:
    from .response_source import _execute_response_segment

    if type(config) is not ResponseGeometryDevelopmentNativeConfig or type(segment) is not ResponseGeometryDevelopmentNativeSegment:
        raise ValueError("development execution requires its exact development config and segment types")
    result, payload = _execute_response_segment(
        config, segment, predecessor, progress=progress
    )
    return cast(ResponseGeometryDevelopmentNativeSegmentResult, result), payload
