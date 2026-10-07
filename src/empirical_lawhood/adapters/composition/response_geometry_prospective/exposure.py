"""assay seed-purpose allocation and inspected historical exclusion identities.

These records carry metadata, not a scientific result or an authority grant.
The operator must replay their bounded source files before attesting exclusion.
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
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import _rng
from empirical_lawhood.adapters.simulators.six_matrix_response.response_panel import IntegrityResponsePanelBinding, SupervisedResponsePanelBinding
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import probe_roster_seed_sha256


def response_geometry_assay_seed_ids(config: ResponseGeometryAssayNativeConfig) -> tuple[str, ...]:
    """Commit every native RNG purpose/counter without sampling or native updates."""
    values = {f"response-geometry.seed.{config.seed_root_sha256}"}
    for root in config.roots:
        purposes = [
            (purpose, 0)
            for purpose in (
                "assay.preparation",
                "assay.continuation",
                "assay.bridge",
                "assay.probes",
            )
        ]
        if root.covariance:
            purposes.append(("assay.covariance", 0))
        if 4 in root.refinements:
            purposes.extend(
                ("assay.quarter-bridge", tick)
                for tick in range(root.landmark_tick + 384 + root.horizon_ticks)
            )
        for purpose, counter in purposes:
            _, receipt = _rng(config, root, purpose, counter)
            values.add(f"seed-purpose.{receipt.purpose_id}.{counter}")
            # PCG64DXSM is initialized from the first 128 bits of this digest.
            values.add(f"seed.pcg64dxsm.{receipt.derived_seed_sha256[:32]}")
            if purpose == "assay.probes":
                probe_seed = probe_roster_seed_sha256(
                    config_fingerprint=receipt.derived_seed_sha256,
                    rule_id="response-geometry-assay.probes",
                    scientific_seed=int(config.scientific_inputs.roots[config.roots.index(root)].probe_roster_seed_sha256, 16),
                )
                values.add(f"seed.pcg64dxsm.{probe_seed[:32]}")
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayExposureSource(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/response-geometry-prospective/response-geometry-assay-exposure-source'
    relative_locator: str
    content_sha256: str
    size_bytes: int
    custody_relative_locator: str | None
    custody_sha256: str | None

    def __post_init__(self) -> None:
        for locator in (self.relative_locator, self.custody_relative_locator):
            if locator is not None and (
                not locator
                or locator.startswith("/")
                or any(part in {"", ".", ".."} for part in locator.split("/"))
            ):
                raise ValueError("exposure source must have a contained relative locator")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        if type(self.size_bytes) is not int or not 0 < self.size_bytes <= 64 * 1024**2:
            raise ValueError("exposure source exceeds its bounded record size")
        if (self.custody_relative_locator is None) != (self.custody_sha256 is None):
            raise ValueError("exposure custody locator and hash must be paired")
        if self.custody_sha256 is not None:
            validate_sha256(self.custody_sha256, field_name="custody_sha256")


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayNoncontactReservationTransfer(CanonicalRecord):
    """One unchanged assay roster transferred after a verified pre-worker refusal.

    This records an application of owner direction, not independent authority.
    The operator must replay the referenced audit and its complete artifact census.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/response-geometry-prospective/response-geometry-assay-noncontact-reservation-transfer'
    transfer_id: str
    recorded_at_utc: str
    owner_grant_id: str
    owner_instruction_sha256: str
    original_issue: ObjectIdentity
    original_run_id: str
    original_run_relative_root: str
    original_implementation_commit: str
    source_config: ResponseGeometryAssayNativeConfig
    replacement_run_id: str
    replacement_implementation_commit: str
    noncontact_audit: ResponseGeometryAssayExposureSource
    run_artifact_inventory_sha256: str
    reserved_unit_ids: tuple[str, ...]
    reserved_seed_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("transfer_id", "owner_grant_id", "original_run_id", "replacement_run_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        parse_utc_timestamp(self.recorded_at_utc, field_name="recorded_at_utc")
        for name in ("owner_instruction_sha256", "run_artifact_inventory_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        for name in ("original_implementation_commit", "replacement_implementation_commit"):
            value = getattr(self, name)
            if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("assay reservation transfer requires exact source commits")
        if self.original_run_id == self.replacement_run_id:
            raise ValueError("assay replacement must have a distinct run identity")
        locator = self.original_run_relative_root
        if (
            locator.startswith("/")
            or any(part in {"", ".", ".."} for part in locator.split("/"))
            or not locator.endswith("/runs/" + self.original_run_id)
        ):
            raise ValueError("assay reservation transfer changes its original run locator")
        if self.reserved_unit_ids != tuple(
            sorted(root.root_id for root in self.source_config.roots)
        ):
            raise ValueError("assay reservation transfer changes the exact root roster")
        if self.reserved_seed_ids != response_geometry_assay_seed_ids(self.source_config):
            raise ValueError("assay reservation transfer changes the exact seed roster")

    def residual_possible_effect_ids(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        """Remove only this audited noncontact reservation, preserving all others."""
        require_sorted_unique_strings(observed, field_name="observed assay possible effects")
        original = "prior-assay-effect." + sha256(self.original_run_relative_root.encode()).hexdigest()
        if original not in observed:
            raise ValueError("assay reservation transfer lacks its observed original reservation")
        return tuple(value for value in observed if value != original)


@dataclass(frozen=True, slots=True)
class ResponsePanelExposureInspection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/response-preparation/response-panel-exposure-inspection'
    inspection_id: str
    inspected_at_utc: str
    inspection_implementation_sha256: str
    source_config: ResponseGeometryAssayNativeConfig
    scope_roots: tuple[str, ...]
    sources: tuple[ResponseGeometryAssayExposureSource, ...]
    source_inventory_sha256: str
    excluded_unit_ids: tuple[str, ...]
    excluded_seed_ids: tuple[str, ...]
    proposed_unit_ids: tuple[str, ...]
    proposed_seed_ids: tuple[str, ...]
    unit_collisions: tuple[str, ...]
    seed_collisions: tuple[str, ...]
    prior_assay_native_effect_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.inspection_id, field_name="inspection_id")
        parse_utc_timestamp(self.inspected_at_utc, field_name="inspected_at_utc")
        validate_sha256(
            self.inspection_implementation_sha256, field_name="inspection_implementation_sha256"
        )
        require_sorted_unique_strings(self.scope_roots, field_name="scope_roots", allow_empty=False)
        if not self.sources or len(self.sources) > 100_000:
            raise ValueError("assay exposure inspection requires bounded source evidence")
        locators = tuple(source.relative_locator for source in self.sources)
        require_sorted_unique_strings(locators, field_name="exposure source locators")
        inventory = sha256(canonical_json_bytes(self.sources)).hexdigest()
        if self.source_inventory_sha256 != inventory:
            raise ValueError("assay exposure inventory commitment differs")
        for name in (
            "excluded_unit_ids",
            "excluded_seed_ids",
            "proposed_unit_ids",
            "proposed_seed_ids",
            "unit_collisions",
            "seed_collisions",
            "prior_assay_native_effect_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if not self.excluded_unit_ids or not self.excluded_seed_ids:
            raise ValueError("assay exposure inspection cannot substitute empty exclusions")
        if self.proposed_unit_ids != self.expected_proposed_unit_ids:
            raise ValueError("assay exposure inspection changes the physical root roster")
        if self.proposed_seed_ids != response_geometry_assay_seed_ids(self.source_config):
            raise ValueError("assay exposure inspection drops a native seed purpose or counter")
        if self.unit_collisions != tuple(
            sorted(set(self.proposed_unit_ids) & set(self.excluded_unit_ids))
        ):
            raise ValueError("assay exposure inspection conceals a unit collision")
        if self.seed_collisions != tuple(
            sorted(set(self.proposed_seed_ids) & set(self.excluded_seed_ids))
        ):
            raise ValueError("assay exposure inspection conceals a seed collision")

    @property
    def expected_proposed_unit_ids(self) -> tuple[str, ...]:
        return tuple(sorted(root.root_id for root in self.source_config.roots))

    @property
    def proposed_panel_has_prior_effects(self) -> bool:
        return bool(self.prior_assay_native_effect_ids)

    @property
    def units_and_seeds_unexposed(self) -> bool:
        return (
            not self.unit_collisions
            and not self.seed_collisions
            and not self.prior_assay_native_effect_ids
        )


@dataclass(frozen=True, slots=True)
class PriorResponsePanelExposure(CanonicalRecord):
    "An audited prior reservation/execution retained in the prospective assay's exclusions."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/response-preparation/prior-response-panel-exposure'

    run_id: str
    run_relative_root: str
    issue: ObjectIdentity
    implementation_commit: str
    source_config: ResponseGeometryAssayNativeConfig
    audit: ResponseGeometryAssayExposureSource
    physical_unit_ids: tuple[str, ...]
    seed_ids: tuple[str, ...]
    launched_physical_unit_ids: tuple[str, ...]
    completed_native_updates_lower_bound: int

    def __post_init__(self) -> None:
        validate_stable_id(self.run_id, field_name="run_id")
        if (
            self.run_relative_root.startswith("/")
            or any(part in {"", ".", ".."} for part in self.run_relative_root.split("/"))
            or not self.run_relative_root.endswith("/runs/" + self.run_id)
        ):
            raise ValueError("prior assay exposure changes its contained run locator")
        if len(self.implementation_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.implementation_commit
        ):
            raise ValueError("prior assay exposure requires its exact implementation commit")
        if self.physical_unit_ids != self.expected_physical_unit_ids or (
            self.seed_ids != response_geometry_assay_seed_ids(self.source_config)
        ):
            raise ValueError("prior assay exposure changes the original unit/seed roster")
        require_sorted_unique_strings(
            self.launched_physical_unit_ids, field_name="launched_physical_unit_ids"
        )
        if not set(self.launched_physical_unit_ids) <= set(self.physical_unit_ids):
            raise ValueError("prior assay exposure launches an undeclared physical unit")
        count = self.completed_native_updates_lower_bound
        if type(count) is not int or count < 0 or count and not self.launched_physical_unit_ids:
            raise ValueError("prior assay native progress differs from its launch roster")

    @property
    def possible_effect_id(self) -> str:
        return "prior-assay-effect." + sha256(self.run_relative_root.encode()).hexdigest()

    @property
    def expected_physical_unit_ids(self) -> tuple[str, ...]:
        return tuple(sorted(root.root_id for root in self.source_config.roots))


@dataclass(frozen=True, slots=True)
class MappedPriorResponsePanelExposure(PriorResponsePanelExposure):
    """Retained integrity-r2 exposure, including its explicit slot/instance map."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/response-preparation/mapped-prior-response-panel-exposure'
    panel_binding: IntegrityResponsePanelBinding

    def __post_init__(self) -> None:
        PriorResponsePanelExposure.__post_init__(self)
        if self.panel_binding.source_config != self.source_config:
            raise ValueError("prior mapped assay exposure binds another source configuration")

    @property
    def expected_physical_unit_ids(self) -> tuple[str, ...]:
        return self.panel_binding.physical_unit_ids


@dataclass(frozen=True, slots=True)
class MappedResponsePanelExposureInspection(ResponsePanelExposureInspection):
    """Fresh assay units with an explicit map and complete, unmodified prior exclusions.

    Prior effects remain present. Only an exactly audited, disjoint earlier panel
    is distinguishable from contact with this new panel. Unknown effects fail shut.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/response-preparation/mapped-response-panel-exposure-inspection'

    panel_binding: IntegrityResponsePanelBinding
    prior_panels: tuple[PriorResponsePanelExposure, ...]
    scope_amendment: ResponseGeometryAssayExposureSource
    owner_direction: ResponseGeometryAssayExposureSource

    def __post_init__(self) -> None:
        ResponsePanelExposureInspection.__post_init__(self)
        if self.panel_binding.source_config != self.source_config:
            raise ValueError("prospective assay exposure binds another native configuration")
        if not self.prior_panels or tuple(
            sorted(set(prior.run_id for prior in self.prior_panels))
        ) != tuple(prior.run_id for prior in self.prior_panels):
            raise ValueError("prospective assay requires a sorted, unique prior-panel audit roster")
        required = (
            self.scope_amendment,
            self.owner_direction,
            *(prior.audit for prior in self.prior_panels),
        )
        if any(value not in self.sources for value in required):
            raise ValueError("prospective assay drops its amendment, direction or prior audit source")
        if any(
            not set(prior.physical_unit_ids) <= set(self.excluded_unit_ids)
            or not set(prior.seed_ids) <= set(self.excluded_seed_ids)
            for prior in self.prior_panels
        ):
            raise ValueError("prospective assay conceals a prior unit or native seed exclusion")

    @property
    def expected_proposed_unit_ids(self) -> tuple[str, ...]:
        return self.panel_binding.physical_unit_ids

    @property
    def proposed_panel_has_prior_effects(self) -> bool:
        return bool(
            self.unit_collisions or self.seed_collisions
        ) or self.prior_assay_native_effect_ids != tuple(
            sorted(prior.possible_effect_id for prior in self.prior_panels)
        )

    @property
    def units_and_seeds_unexposed(self) -> bool:
        return not self.proposed_panel_has_prior_effects


@dataclass(frozen=True, slots=True)
class SupervisedResponsePanelExposureInspection(MappedResponsePanelExposureInspection):
    """Supervised-r3 metadata retains original and mapped predecessor exposures."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/response-preparation/supervised-response-panel-exposure-inspection'
    panel_binding: SupervisedResponsePanelBinding
    prior_panels: tuple[PriorResponsePanelExposure | MappedPriorResponsePanelExposure, ...]


def exposure_identities(document: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read only unit/seed metadata, preserving native purpose and digest identities."""
    units: set[str] = set()
    seeds: set[str] = set()

    def seed_value(value: object) -> None:
        if isinstance(value, str):
            try:
                validate_stable_id(value, field_name="inspected seed identity")
            except ValueError:
                return
            seeds.add(value)
            if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
                seeds.add(f"seed.pcg64dxsm.{value[:32]}")
        elif isinstance(value, (list, tuple)):
            for item in value:
                seed_value(item)
        elif type(value) is int and 0 <= value < 2**128:
            seeds.add(f"seed.pcg64dxsm.{value:032x}")

    pending = [document]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if {"purpose_id", "stream_index", "derived_seed_sha256"} <= value.keys():
                seeds.add(f"seed-purpose.{value['purpose_id']}.{value['stream_index']}")
            for key, operand in value.items():
                if not isinstance(key, str):
                    raise ValueError("exposure source has a non-string key")
                if key in {
                    "physical_independent_unit_id",
                    "physical_independent_unit_ids",
                    "physical_unit_id",
                    "physical_unit_ids",
                    "evaluation_unit_ids",
                    "development_unit_ids",
                    "root_id",
                    "root_ids",
                    # Conservative historical exclusion labels; they do not
                    # establish an independent-unit count or merge denominators.
                    "root_block_id",
                    "history_id",
                    "history_ids",
                }:
                    items = operand if isinstance(operand, list) else (operand,)
                    for item in items:
                        if isinstance(item, str):
                            validate_stable_id(item, field_name="inspected unit identity")
                            units.add(item)
                if "seed" in key or "purpose" in key:
                    seed_value(operand)
                if isinstance(operand, (dict, list)):
                    pending.append(operand)
        elif isinstance(value, list):
            pending.extend(value)
    return tuple(sorted(units)), tuple(sorted(seeds))
