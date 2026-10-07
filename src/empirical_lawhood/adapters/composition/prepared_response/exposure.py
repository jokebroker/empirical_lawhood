"""Bounded exposure metadata for the new prepared-response root namespace.

The operator must replay the listed external metadata before issue. Construction
checks commitments and collisions; it neither inspects storage nor grants access.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.adapters.composition.response_geometry_prospective.exposure import ResponseGeometryAssayExposureSource, exposure_identities
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN, PreparedNativeSpec
from empirical_lawhood.adapters.simulators.prepared_response.source import PURPOSES, prepared_rng


def prepared_prior_exposure_identities(
    document: object,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    "Retain both explicit censuses and nested identities from the bound source qualification anchors."
    if not isinstance(document, dict) or document.get("schema") not in (
        'empirical-lawhood/response-preparation/supervised-response-panel-exposure-inspection',
        PreparedExposureInspection.SCHEMA,
    ):
        raise ValueError("prepared prior exposure requires a recognized source qualification inspection")
    value = document.get("value")
    if not isinstance(value, dict):
        raise ValueError("prepared prior exposure requires its canonical inspection value")
    units, seeds = (set(values) for values in exposure_identities(document))
    for field, target in (
        ("excluded_unit_ids", units),
        ("proposed_unit_ids", units),
        ("excluded_seed_ids", seeds),
        ("proposed_seed_ids", seeds),
    ):
        declared = value.get(field)
        if not isinstance(declared, list):
            raise ValueError(f"prepared prior exposure lacks its declared {field} census")
        require_sorted_unique_strings(tuple(declared), field_name=field)
        for identity in declared:
            validate_stable_id(identity, field_name=field)
        target.update(declared)
    return tuple(sorted(units)), tuple(sorted(seeds))


def prepared_seed_ids(spec: PreparedNativeSpec) -> tuple[str, ...]:
    """Reserve every declared stream, including unused purposes, without draws."""
    values = {f"{CAMPAIGN}.seed.{spec.seed_sha256}"}
    for root in spec.roots:
        for purpose in PURPOSES:
            for bridge in (False,) if purpose == "passive-probes" else (False, True):
                _, receipt = prepared_rng(root, purpose, bridge=bridge)
                values.add(f"seed-purpose.{receipt.purpose_id}.0")
                values.add(f"seed.pcg64dxsm.{receipt.derived_seed_sha256[:32]}")
                if purpose == "passive-probes":
                    probe = root.randomness.passive_probe_seed_sha256
                    values.add(f"seed.pcg64dxsm.{probe[:32]}")
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class PreparedExposureInspection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/prepared-response/prepared-exposure-inspection'
    inspection_id: str
    inspected_at_utc: str
    inspection_implementation_sha256: str
    source_config: PreparedNativeSpec
    scope_roots: tuple[str, ...]
    sources: tuple[ResponseGeometryAssayExposureSource, ...]
    source_inventory_sha256: str
    excluded_unit_ids: tuple[str, ...]
    excluded_seed_ids: tuple[str, ...]
    proposed_unit_ids: tuple[str, ...]
    proposed_seed_ids: tuple[str, ...]
    unit_collisions: tuple[str, ...]
    seed_collisions: tuple[str, ...]
    prior_native_effect_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.inspection_id, field_name="inspection_id")
        parse_utc_timestamp(self.inspected_at_utc, field_name="inspected_at_utc")
        validate_sha256(
            self.inspection_implementation_sha256, field_name="inspection_implementation_sha256"
        )
        require_sorted_unique_strings(self.scope_roots, field_name="scope_roots", allow_empty=False)
        if not self.sources or len(self.sources) > 100_000:
            raise ValueError("prepared exposure inspection requires bounded source evidence")
        require_sorted_unique_strings(
            tuple(s.relative_locator for s in self.sources), field_name="source locators"
        )
        if self.source_inventory_sha256 != sha256(canonical_json_bytes(self.sources)).hexdigest():
            raise ValueError("prepared exposure inventory commitment differs")
        for name in (
            "excluded_unit_ids",
            "excluded_seed_ids",
            "proposed_unit_ids",
            "proposed_seed_ids",
            "unit_collisions",
            "seed_collisions",
            "prior_native_effect_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if not self.excluded_unit_ids or not self.excluded_seed_ids:
            raise ValueError("prepared exposure cannot substitute empty historical exclusions")
        if self.proposed_unit_ids != tuple(
            sorted(root.physical_unit_id for root in self.source_config.roots)
        ) or self.proposed_seed_ids != prepared_seed_ids(self.source_config):
            raise ValueError("prepared exposure changes its root or native stream roster")
        if self.unit_collisions != tuple(
            sorted(set(self.proposed_unit_ids) & set(self.excluded_unit_ids))
        ) or self.seed_collisions != tuple(
            sorted(set(self.proposed_seed_ids) & set(self.excluded_seed_ids))
        ):
            raise ValueError("prepared exposure conceals an identity collision")

    @property
    def units_and_seeds_unexposed(self) -> bool:
        return not (self.unit_collisions or self.seed_collisions or self.prior_native_effect_ids)
