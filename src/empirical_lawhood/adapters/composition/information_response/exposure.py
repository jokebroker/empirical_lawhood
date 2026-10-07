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
from empirical_lawhood.adapters.composition.response_geometry_prospective.exposure import ResponseGeometryAssayExposureSource
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN
from empirical_lawhood.adapters.simulators.information_response.contracts import InformationResponseNativeConfig
from empirical_lawhood.adapters.simulators.prepared_response.source import PURPOSES, prepared_rng
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import probe_roster_seed_sha256


def information_response_seed_ids(spec: InformationResponseNativeConfig) -> tuple[str, ...]:
    """Reserve every declared stream, including unused purposes, without draws."""
    values = {f"{CAMPAIGN}.seed.{spec.recipe.seed_sha256}"}
    for root in spec.roots:
        for purpose in PURPOSES:
            for bridge in (False,) if purpose == "passive-probes" else (False, True):
                _, receipt = prepared_rng(root, purpose, bridge=bridge)
                values.add(f"seed-purpose.{receipt.purpose_id}.0")
                values.add(f"seed.pcg64dxsm.{receipt.derived_seed_sha256[:32]}")
                if purpose == "passive-probes":
                    probe = probe_roster_seed_sha256(
                        config_fingerprint=receipt.derived_seed_sha256,
                        rule_id=f"{CAMPAIGN}.passive-probes",
                        scientific_seed=int(root.randomness.passive_probe_seed_sha256, 16),
                    )
                    values.add(f"seed.pcg64dxsm.{probe[:32]}")
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class InformationResponseExposureInspection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/information-response/information-response-exposure-inspection'
    inspection_id: str
    inspected_at_utc: str
    inspection_implementation_sha256: str
    source_config: InformationResponseNativeConfig
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
        ) or self.proposed_seed_ids != information_response_seed_ids(self.source_config):
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
