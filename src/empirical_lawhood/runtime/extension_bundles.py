"""Closed static composition records for reusable backbone extensions.

Build-time tooling may discover allowlisted ``extension_bundle.py`` modules.
Runtime consumes only the generated module and these canonical records: there
is no filesystem scan, dynamic import, executable config or authority grant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from typing import ClassVar, cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_profiles import EvidenceWorldProfileRegistry

from .candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from .capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from .datasets import DatasetCapabilityRegistry


MAX_EXTENSION_BUNDLE_MEMBERS = 1024


class ExtensionBundleAggregation(StrEnum):
    MANUAL_PILOT = "MANUAL_PILOT"
    GENERATED_RELEASE = "GENERATED_RELEASE"


class ExtensionProfileKind(StrEnum):
    SOURCE_PIPELINE = "SOURCE_PIPELINE"
    LINKED_CAMPAIGN = "LINKED_CAMPAIGN"
    OBJECTIVE_TERMINAL = "OBJECTIVE_TERMINAL"


class ExtensionComponentKind(StrEnum):
    METHOD = "METHOD"
    CONFIG_DECODER = "CONFIG_DECODER"
    ARTIFACT_VALIDATOR = "ARTIFACT_VALIDATOR"
    RUNTIME_PROVIDER = "RUNTIME_PROVIDER"
    PROGRAMME_AUTHOR = "PROGRAMME_AUTHOR"


class ExtensionContributionKind(StrEnum):
    SOURCE = "SOURCE"
    METHOD = "METHOD"
    CAMPAIGN = "CAMPAIGN"
    STRUCTURAL = "STRUCTURAL"


@dataclass(frozen=True, slots=True)
class ExtensionProfileRegistration(CanonicalRecord):
    """A reusable profile type, not one experiment-specific profile instance."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/extension-profile-registration'

    registration_id: str
    profile_key: str
    profile_version: str
    kind: ExtensionProfileKind
    profile_schema: str
    profile_schema_sha256: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.registration_id, field_name="registration_id")
        validate_stable_id(self.profile_key, field_name="profile_key")
        validate_semantic_version(self.profile_version)
        validate_schema(self.profile_schema)
        validate_sha256(self.profile_schema_sha256, field_name="profile_schema_sha256")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        expected = hashlib.sha256(self.profile_schema.encode("utf-8")).hexdigest()
        if self.profile_schema_sha256 != expected:
            raise ValueError("extension profile schema digest differs")


@dataclass(frozen=True, slots=True)
class ExtensionComponentRegistration(CanonicalRecord):
    """One non-discovering implementation role in a generated bundle."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/extension-component-registration'

    registration_id: str
    component_key: str
    component_version: str
    kind: ExtensionComponentKind
    input_schema_ids: tuple[str, ...]
    output_schema_ids: tuple[str, ...]
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.registration_id, field_name="registration_id")
        validate_stable_id(self.component_key, field_name="component_key")
        validate_semantic_version(self.component_version)
        for name, values in (
            ("input_schema_ids", self.input_schema_ids),
            ("output_schema_ids", self.output_schema_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
            for value in values:
                validate_schema(value)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")


def _identity_group(
    values: tuple[ObjectIdentity, ...],
    *,
    field_name: str,
    allow_empty: bool = True,
) -> None:
    if not values and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    require_sorted_unique_ids(values, attribute="object_id", field_name=field_name)


@dataclass(frozen=True, slots=True)
class ExtensionBundleContribution(CanonicalRecord):
    """One code-owned, nonexecuting descriptor consumed by the generator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/extension-bundle-contribution'

    contribution_id: str
    contribution_version: str
    kind: ExtensionContributionKind
    evidence_profile_registries: tuple[EvidenceWorldProfileRegistry, ...]
    capability_manifests: tuple[CapabilityManifest, ...]
    candidate_capability_registrations: tuple[CandidateCapabilityRegistration, ...]
    dataset_capability_registries: tuple[DatasetCapabilityRegistry, ...]
    profile_registrations: tuple[ExtensionProfileRegistration, ...]
    method_registrations: tuple[ObjectIdentity, ...]
    config_decoders: tuple[ObjectIdentity, ...]
    artifact_validators: tuple[ObjectIdentity, ...]
    runtime_providers: tuple[ObjectIdentity, ...]
    study_authors: tuple[ObjectIdentity, ...]
    grants_authority: bool
    embeds_scientific_payload: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.contribution_id, field_name="contribution_id")
        validate_semantic_version(self.contribution_version)
        require_sorted_unique_ids(
            self.evidence_profile_registries,
            attribute="registry_id",
            field_name="evidence_profile_registries",
        )
        require_sorted_unique_ids(
            self.capability_manifests,
            attribute="registry_id",
            field_name="capability_manifests",
        )
        require_sorted_unique_ids(
            self.candidate_capability_registrations,
            attribute="registration_id",
            field_name="candidate_capability_registrations",
        )
        require_sorted_unique_ids(
            self.dataset_capability_registries,
            attribute="registry_id",
            field_name="dataset_capability_registries",
        )
        require_sorted_unique_ids(
            self.profile_registrations,
            attribute="registration_id",
            field_name="profile_registrations",
        )
        for name in (
            "method_registrations",
            "config_decoders",
            "artifact_validators",
            "runtime_providers",
            'study_authors',
        ):
            _identity_group(getattr(self, name), field_name=name)
        manifests = {value.registry_id: value for value in self.capability_manifests}
        if any(
            manifests.get(value.registration_id) != value.manifest
            for value in self.candidate_capability_registrations
        ):
            raise ValueError("candidate registration lacks its exact capability manifest")
        forbidden_kinds = {CapabilityKind.ACTUATOR, CapabilityKind.APPROVAL_GATE}
        forbidden_permissions = {
            CapabilityPermission.APPROVE_NONACTUATING,
            CapabilityPermission.COMMAND_ACTUATOR,
        }
        # A sealed-read requirement is needed for a continuation to restore
        # its own sealed predecessor. Declaring that capability is not an
        # authority act; issued execution and the worker input gate still
        # enforce the actual grant. Likewise, a declared evaluator reveal
        # requirement grants no reveal authority. The runtime replays that
        # separate authority before its protected input read.
        if any(
            value.kind in forbidden_kinds or bool(set(value.permissions) & forbidden_permissions)
            for value in self.capability_manifests
        ):
            raise ValueError("extension bundle contribution attempts to grant authority")
        if self.grants_authority or self.embeds_scientific_payload:
            raise ValueError("extension descriptor cannot grant authority or embed payload")


@dataclass(frozen=True, slots=True)
class ExtensionBundle(CanonicalRecord):
    """Frozen generated inventory for the proven F2/F3 extension roles."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/extension-bundle'

    bundle_id: str
    bundle_version: str
    aggregation: ExtensionBundleAggregation
    evidence_profile_registry: ObjectIdentity
    capability_registry: ObjectIdentity
    dataset_capability_registries: tuple[ObjectIdentity, ...]
    profile_registrations: tuple[ObjectIdentity, ...]
    method_registrations: tuple[ObjectIdentity, ...]
    config_decoders: tuple[ObjectIdentity, ...]
    artifact_validators: tuple[ObjectIdentity, ...]
    runtime_providers: tuple[ObjectIdentity, ...]
    study_authors: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        validate_semantic_version(self.bundle_version)
        if self.evidence_profile_registry.object_schema != EvidenceWorldProfileRegistry.SCHEMA:
            raise ValueError("extension bundle binds another evidence registry schema")
        if self.capability_registry.object_schema != CapabilityRegistry.SCHEMA:
            raise ValueError("extension bundle binds another capability registry schema")
        for name in (
            "dataset_capability_registries",
            "profile_registrations",
            "method_registrations",
            "config_decoders",
            "artifact_validators",
            "runtime_providers",
            'study_authors',
        ):
            _identity_group(getattr(self, name), field_name=name, allow_empty=False)
        if any(
            value.object_schema != DatasetCapabilityRegistry.SCHEMA
            for value in self.dataset_capability_registries
        ):
            raise ValueError("extension bundle contains another dataset registry schema")
        if any(
            value.object_schema != ExtensionProfileRegistration.SCHEMA
            for value in self.profile_registrations
        ):
            raise ValueError("extension bundle contains another profile registration schema")
        all_identities = (
            self.evidence_profile_registry,
            self.capability_registry,
            *self.dataset_capability_registries,
            *self.profile_registrations,
            *self.method_registrations,
            *self.config_decoders,
            *self.artifact_validators,
            *self.runtime_providers,
            *self.study_authors,
        )
        if len(all_identities) > MAX_EXTENSION_BUNDLE_MEMBERS:
            raise ValueError("extension bundle exceeds its member-count bound")
        exact = {
            (
                value.object_id,
                value.object_schema,
                value.object_version,
                value.object_fingerprint,
            )
            for value in all_identities
        }
        if len(exact) != len(all_identities):
            raise ValueError("extension bundle repeats an exact member identity")


@dataclass(frozen=True, slots=True)
class CapabilityBundleBinding(CanonicalRecord):
    """Public discovery provenance for one candidate capability."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/capability-bundle-binding'

    binding_id: str
    capability_key: str
    capability_version: str
    bundle_id: str
    bundle_version: str
    bundle_fingerprint: str
    profile_registrations: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        validate_semantic_version(self.bundle_version)
        validate_sha256(self.bundle_fingerprint, field_name="bundle_fingerprint")
        _identity_group(
            self.profile_registrations,
            field_name="profile_registrations",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class GeneratedExtensionBundleAggregate(CanonicalRecord):
    """Generated static roots and discovery view; never an executable provider."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/generated-extension-bundle-aggregate'

    aggregate_id: str
    aggregate_version: str
    contributions: tuple[ObjectIdentity, ...]
    bundle: ExtensionBundle
    evidence_profile_registry: EvidenceWorldProfileRegistry
    capability_registry: CapabilityRegistry
    dataset_capability_registries: tuple[DatasetCapabilityRegistry, ...]
    profile_registrations: tuple[ExtensionProfileRegistration, ...]
    candidate_capability_catalog: CandidateCapabilityCatalog
    capability_bindings: tuple[CapabilityBundleBinding, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.aggregate_id, field_name="aggregate_id")
        validate_semantic_version(self.aggregate_version)
        _identity_group(self.contributions, field_name="contributions", allow_empty=False)
        require_sorted_unique_ids(
            self.dataset_capability_registries,
            attribute="registry_id",
            field_name="dataset_capability_registries",
        )
        require_sorted_unique_ids(
            self.profile_registrations,
            attribute="registration_id",
            field_name="profile_registrations",
        )
        require_sorted_unique_ids(
            self.capability_bindings,
            attribute="binding_id",
            field_name="capability_bindings",
        )
        if self.bundle.aggregation is not ExtensionBundleAggregation.GENERATED_RELEASE:
            raise ValueError("generated aggregate contains a manual bundle")
        expected = (
            ObjectIdentity.from_record(
                self.evidence_profile_registry.registry_id,
                self.evidence_profile_registry,
            ),
            ObjectIdentity.from_record(
                self.capability_registry.registry_id,
                self.capability_registry,
            ),
            tuple(
                ObjectIdentity.from_record(value.registry_id, value)
                for value in self.dataset_capability_registries
            ),
            tuple(
                ObjectIdentity.from_record(value.registration_id, value)
                for value in self.profile_registrations
            ),
        )
        observed = (
            self.bundle.evidence_profile_registry,
            self.bundle.capability_registry,
            self.bundle.dataset_capability_registries,
            self.bundle.profile_registrations,
        )
        if observed != expected:
            raise ValueError("generated aggregate roots differ from its bundle identities")

    def binding(
        self, capability_key: str, capability_version: str
    ) -> CapabilityBundleBinding | None:
        return next(
            (
                value
                for value in self.capability_bindings
                if value.capability_key == capability_key
                and value.capability_version == capability_version
            ),
            None,
        )


def _merge_exact(values: tuple[object, ...], attribute: str, label: str) -> tuple[object, ...]:
    merged: dict[str, object] = {}
    for value in values:
        key = getattr(value, attribute)
        previous = merged.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"extension contributions conflict for {label} {key}")
        if previous is not None:
            raise ValueError(f"extension contributions duplicate {label} {key}")
        merged[key] = value
    return tuple(merged[key] for key in sorted(merged))


def _merge_identity_groups(
    contributions: tuple[ExtensionBundleContribution, ...],
    field_name: str,
) -> tuple[ObjectIdentity, ...]:
    values = tuple(
        value for contribution in contributions for value in getattr(contribution, field_name)
    )
    return cast(tuple[ObjectIdentity, ...], _merge_exact(values, "object_id", field_name))


def compose_extension_bundle_aggregate(
    contributions: tuple[ExtensionBundleContribution, ...],
) -> GeneratedExtensionBundleAggregate:
    """Build the sole deterministic aggregate from explicitly imported descriptors."""

    ordered = tuple(sorted(contributions, key=lambda value: value.contribution_id))
    require_sorted_unique_ids(ordered, attribute="contribution_id", field_name="contributions")
    if not ordered:
        raise ValueError("generated extension aggregate requires contributions")
    evidence = tuple(
        value for contribution in ordered for value in contribution.evidence_profile_registries
    )
    if len(evidence) != 1:
        raise ValueError("generated extension aggregate requires one evidence registry")
    manifests = cast(
        tuple[CapabilityManifest, ...],
        _merge_exact(
            tuple(value for contribution in ordered for value in contribution.capability_manifests),
            "registry_id",
            "capability",
        ),
    )
    datasets = cast(
        tuple[DatasetCapabilityRegistry, ...],
        _merge_exact(
            tuple(
                value
                for contribution in ordered
                for value in contribution.dataset_capability_registries
            ),
            "registry_id",
            "dataset registry",
        ),
    )
    profiles = cast(
        tuple[ExtensionProfileRegistration, ...],
        _merge_exact(
            tuple(
                value for contribution in ordered for value in contribution.profile_registrations
            ),
            "registration_id",
            "profile registration",
        ),
    )
    candidates = cast(
        tuple[CandidateCapabilityRegistration, ...],
        _merge_exact(
            tuple(
                value
                for contribution in ordered
                for value in contribution.candidate_capability_registrations
            ),
            "registration_id",
            "candidate registration",
        ),
    )
    if not manifests or not datasets or not profiles or not candidates:
        raise ValueError("generated extension aggregate omits a required static role")
    capability_registry = CapabilityRegistry(
        registry_id="registry.backbone-extensions.generated",
        capabilities=manifests,
    )
    candidate_catalog = CandidateCapabilityCatalog(
        catalog_id="catalog.backbone-extensions.generated",
        registrations=candidates,
        templates=(),
    )
    method_registrations = _merge_identity_groups(ordered, "method_registrations")
    config_decoders = _merge_identity_groups(ordered, "config_decoders")
    artifact_validators = _merge_identity_groups(ordered, "artifact_validators")
    runtime_providers = _merge_identity_groups(ordered, "runtime_providers")
    programme_authors = _merge_identity_groups(ordered, 'study_authors')
    profile_identities = tuple(
        ObjectIdentity.from_record(value.registration_id, value) for value in profiles
    )
    bundle = ExtensionBundle(
        bundle_id="extension-bundle.consolidated-backbone",
        bundle_version="1.0.0",
        aggregation=ExtensionBundleAggregation.GENERATED_RELEASE,
        evidence_profile_registry=ObjectIdentity.from_record(evidence[0].registry_id, evidence[0]),
        capability_registry=ObjectIdentity.from_record(
            capability_registry.registry_id,
            capability_registry,
        ),
        dataset_capability_registries=tuple(
            ObjectIdentity.from_record(value.registry_id, value) for value in datasets
        ),
        profile_registrations=profile_identities,
        method_registrations=method_registrations,
        config_decoders=config_decoders,
        artifact_validators=artifact_validators,
        runtime_providers=runtime_providers,
        study_authors=programme_authors,
    )
    bundle_fingerprint = bundle.fingerprint()
    bindings = []
    for contribution in ordered:
        contribution_profiles = tuple(
            ObjectIdentity.from_record(value.registration_id, value)
            for value in contribution.profile_registrations
        )
        for registration in contribution.candidate_capability_registrations:
            bindings.append(
                CapabilityBundleBinding(
                    binding_id=(
                        f"bundle-binding.{registration.manifest.capability_key}."
                        f"{registration.manifest.capability_version.replace('.', '-')}"
                    ),
                    capability_key=registration.manifest.capability_key,
                    capability_version=registration.manifest.capability_version,
                    bundle_id=bundle.bundle_id,
                    bundle_version=bundle.bundle_version,
                    bundle_fingerprint=bundle_fingerprint,
                    profile_registrations=contribution_profiles,
                )
            )
    return GeneratedExtensionBundleAggregate(
        aggregate_id="extension-aggregate.consolidated-backbone",
        aggregate_version="1.0.0",
        contributions=tuple(
            ObjectIdentity.from_record(value.contribution_id, value) for value in ordered
        ),
        bundle=bundle,
        evidence_profile_registry=evidence[0],
        capability_registry=capability_registry,
        dataset_capability_registries=datasets,
        profile_registrations=profiles,
        candidate_capability_catalog=candidate_catalog,
        capability_bindings=tuple(sorted(bindings, key=lambda value: value.binding_id)),
    )


__all__ = [
    'CapabilityBundleBinding',
    'ExtensionBundleAggregation',
    'ExtensionBundleContribution',
    'ExtensionBundle',
    'ExtensionComponentKind',
    'ExtensionComponentRegistration',
    'ExtensionContributionKind',
    'ExtensionProfileKind',
    'ExtensionProfileRegistration',
    'GeneratedExtensionBundleAggregate',
    'compose_extension_bundle_aggregate',
]
