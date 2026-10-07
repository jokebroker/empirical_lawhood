"""Full numerical bootstrap inputs with separate original and current custody.

The caller must authenticate each original export before issue.  These typed
inputs bind the current scientific operands; they grant no source qualification
or authority and never reconstruct a seed from a public name.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.independent_source_exports import OriginalObjectReference
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes, validate_sha256, validate_stable_id, validate_nonempty


@dataclass(frozen=True, slots=True)
class StructuralBootstrapSeedInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/structural-bootstrap-stream"

    scientific_role: str
    current_target_id: str
    action_id: str
    stage: str
    sample_count: int
    bootstrap_replications: int
    current_context_sha256: str
    full_seed_sha256: str
    source_original_context_sha256: str
    original_source: OriginalObjectReference
    export_receipt: ObjectIdentity

    def __post_init__(self) -> None:
        if self.scientific_role not in ("margin-panel-admission", "grid2op-paired-metric-topology"):
            raise ValueError("structural bootstrap input has another scientific role")
        for name in ("current_target_id", "action_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.stage, field_name="stage")
        if any(type(value) is not int or value < 1 for value in (self.sample_count, self.bootstrap_replications)):
            raise ValueError("structural bootstrap requires positive integer sample and replication counts")
        for name in ("current_context_sha256", "full_seed_sha256", "source_original_context_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if type(self.original_source) is not OriginalObjectReference or type(self.export_receipt) is not ObjectIdentity:
            raise ValueError("structural bootstrap requires explicit original source and export custody")

    def require_current_binding(
        self, *, scientific_role: str, current_target_id: str, action_id: str,
        stage: str, sample_count: int, bootstrap_replications: int,
        current_context_sha256: str,
    ) -> None:
        if (
            self.scientific_role, self.current_target_id, self.action_id, self.stage,
            self.sample_count, self.bootstrap_replications, self.current_context_sha256,
        ) != (
            scientific_role, current_target_id, action_id, stage,
            sample_count, bootstrap_replications, current_context_sha256,
        ):
            raise ValueError("structural bootstrap input binds another current target, action, stage or numeric operand")


@dataclass(frozen=True, slots=True)
class StructuralBootstrapInputCensus(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/scientific-input/structural-bootstrap-census"

    census_id: str
    inputs: tuple[StructuralBootstrapSeedInput, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.census_id, field_name="census_id")
        if type(self.inputs) is not tuple or any(not isinstance(value, StructuralBootstrapSeedInput) for value in self.inputs):
            raise ValueError("structural bootstrap census requires a complete typed tuple")
        keys = tuple((value.current_target_id, value.scientific_role, value.action_id, value.stage) for value in self.inputs)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("structural bootstrap census must be uniquely ordered by target, role, action and stage")


def structural_bootstrap_context_sha256(*operands: object) -> str:
    """Bind current operands without allocating numerical entropy."""
    return sha256(canonical_json_bytes(operands)).hexdigest()


def require_structural_bootstrap_seed_input(
    value: StructuralBootstrapSeedInput | None, **binding: object,
) -> StructuralBootstrapSeedInput:
    if not isinstance(value, StructuralBootstrapSeedInput):
        raise ValueError("structural bootstrap requires its full numerical seed input and authenticated external export custody")
    value.require_current_binding(**binding)
    return value


def require_structural_bootstrap_census(
    value: StructuralBootstrapInputCensus | None,
) -> StructuralBootstrapInputCensus:
    if not isinstance(value, StructuralBootstrapInputCensus):
        raise ValueError("structural prediction requires the complete typed bootstrap census from authenticated original exports")
    return value
