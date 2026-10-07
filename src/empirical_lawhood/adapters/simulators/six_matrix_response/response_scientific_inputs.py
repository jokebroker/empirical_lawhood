"""Complete numerical response inputs with original and target custody.

The operator supplies a separately verified external export. These records bind
that export to the current declaration; they do not authenticate historical
results, transfer qualification, or derive entropy from public names.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_sha256,
)


def response_scientific_input_declaration_sha256(
    *, config_id: str, design_packet_sha256: str, seed_root_sha256: str,
    roots: tuple[CanonicalRecord, ...],
) -> str:
    """Bind current source coordinates without a circular input fingerprint."""
    return sha256(canonical_json_bytes({
        "config_id": config_id,
        "design_packet_sha256": design_packet_sha256,
        "seed_root_sha256": seed_root_sha256,
        "roots": tuple(root.to_document() for root in roots),
    })).hexdigest()


@dataclass(frozen=True, slots=True)
class ResponseGeometryRootScientificInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/response-geometry-root"

    context: str
    landmark_tick: int
    index: int
    preparation_seed_sha256: str
    continuation_seed_sha256: str
    bridge_seed_sha256: str
    probe_stream_seed_sha256: str
    probe_roster_seed_sha256: str
    covariance_seed_sha256: str | None
    quarter_bridge_seed_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.context not in ("assembling", "prepared")
            or type(self.landmark_tick) is not int
            or self.landmark_tick not in (1024, 4096)
            or type(self.index) is not int
            or not 0 <= self.index < 64
            or not isinstance(self.quarter_bridge_seed_sha256, tuple)
        ):
            raise ValueError("response scientific inputs have another root coordinate")
        for name in (
            "preparation_seed_sha256", "continuation_seed_sha256",
            "bridge_seed_sha256", "probe_stream_seed_sha256",
            "probe_roster_seed_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.covariance_seed_sha256 is not None:
            validate_sha256(self.covariance_seed_sha256, field_name="covariance_seed_sha256")
        for seed in self.quarter_bridge_seed_sha256:
            validate_sha256(seed, field_name="quarter_bridge_seed_sha256")

    def require_domain(self, *, covariance: bool, quarter_tick_count: int) -> None:
        if (
            (self.covariance_seed_sha256 is not None) != covariance
            or len(self.quarter_bridge_seed_sha256) != quarter_tick_count
        ):
            raise ValueError("response scientific input census changes its covariance or quarter-bridge domain")

    def native_seed(self, *, purpose_ordinal: int, counter: int) -> str:
        if type(purpose_ordinal) is not int or type(counter) is not int or counter < 0:
            raise ValueError("response scientific input purpose/counter must be integers")
        if purpose_ordinal == 5:
            if counter >= len(self.quarter_bridge_seed_sha256):
                raise ValueError("missing response quarter-bridge scientific seed")
            return self.quarter_bridge_seed_sha256[counter]
        if not 0 <= purpose_ordinal < 5 or counter != 0:
            raise ValueError("response scientific input is outside the complete native seed census")
        seed = (
            self.preparation_seed_sha256, self.continuation_seed_sha256,
            self.bridge_seed_sha256, self.probe_stream_seed_sha256,
            self.covariance_seed_sha256,
        )[purpose_ordinal]
        if seed is None:
            raise ValueError("response root has no covariance scientific seed")
        return seed


@dataclass(frozen=True, slots=True)
class ResponseGeometryScientificInputs(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/response-geometry"

    original_source_config: ObjectIdentity
    external_export_receipt: ObjectIdentity
    target_declaration_sha256: str
    seed_root_sha256: str
    roots: tuple[ResponseGeometryRootScientificInput, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.target_declaration_sha256, field_name="target_declaration_sha256")
        validate_sha256(self.seed_root_sha256, field_name="seed_root_sha256")
        if (
            not isinstance(self.original_source_config, ObjectIdentity)
            or not isinstance(self.external_export_receipt, ObjectIdentity)
            or self.external_export_receipt.object_schema
            != "empirical-lawhood/runtime/canonical-task-receipt"
            or not isinstance(self.roots, tuple)
            or not self.roots
            or any(not isinstance(root, ResponseGeometryRootScientificInput) for root in self.roots)
        ):
            raise ValueError("response scientific inputs require explicit original custody and an external export receipt")
        coordinates = tuple((root.context, root.landmark_tick, root.index) for root in self.roots)
        if coordinates != tuple(sorted(set(coordinates))):
            raise ValueError("response scientific input coordinates must be complete, ordered and unique")

    def require_source(
        self, *, config_id: str, design_packet_sha256: str,
        seed_root_sha256: str, roots: tuple[CanonicalRecord, ...],
    ) -> None:
        if (
            self.seed_root_sha256 != seed_root_sha256
            or self.target_declaration_sha256 != response_scientific_input_declaration_sha256(
                config_id=config_id, design_packet_sha256=design_packet_sha256,
                seed_root_sha256=seed_root_sha256, roots=roots,
            )
            or tuple((row.context, row.landmark_tick, row.index) for row in self.roots)
            != tuple((root.context, root.landmark_tick, root.index) for root in roots)
        ):
            raise ValueError("response scientific input original-to-current declaration custody differs")
        for row, root in zip(self.roots, roots, strict=True):
            row.require_domain(
                covariance=root.covariance,
                quarter_tick_count=(root.landmark_tick + root.invocation_offset + root.horizon_ticks)
                if 4 in root.refinements else 0,
            )


def require_response_scientific_inputs(
    inputs: ResponseGeometryScientificInputs, *, config_id: str,
    design_packet_sha256: str, seed_root_sha256: str,
    roots: tuple[CanonicalRecord, ...],
) -> None:
    if not isinstance(inputs, ResponseGeometryScientificInputs):
        raise ValueError("response source requires a complete externally verified numerical input export")
    inputs.require_source(
        config_id=config_id, design_packet_sha256=design_packet_sha256,
        seed_root_sha256=seed_root_sha256, roots=roots,
    )
