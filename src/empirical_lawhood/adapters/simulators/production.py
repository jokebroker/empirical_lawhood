"""Authenticated static composition for one held simulator registration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import ClassVar, Final

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.dataset_authority import DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA
from empirical_lawhood.planning.dataset_manifests import DatasetRegistrationManifest
from empirical_lawhood.runtime.capabilities import CapabilityConfigRef, ImplementationSourceClosure

from .manifests import (
    SimulatorHeldRegistrationDefinition,
    SimulatorRegistrationManifestDependencies,
    build_simulator_held_registration_manifest,
)
from .registry import (
    SIMULATOR_DATASET_CONFIG_KEY,
    SIMULATOR_DATASET_ENVIRONMENT_KEY,
    SIMULATOR_DATASET_RUNTIME_KEY,
    SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID,
    SimulatorDatasetRegistryProfile,
    build_simulator_dataset_registry_profile,
    simulator_dataset_capability_bindings,
)


SIMULATOR_DATASET_PRODUCTION_PROFILE_ID: Final = "simulator-directory-registration"
SIMULATOR_DATASET_CONFIG_ARTIFACT_ID: Final = "artifact.simulator-dataset-config"
SIMULATOR_DATASET_INSPECTOR_CLOSURE_ID: Final = "implementation.simulator-dataset-inspector"
SIMULATOR_DATASET_EVIDENCE_CLOSURE_ID: Final = (
    "implementation.simulator-dataset-evidence-verifier"
)


@dataclass(frozen=True, slots=True)
class SimulatorDatasetCapabilityConfig(CanonicalRecord):
    """Closed zero-network configuration for held simulator registration."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/simulator-dataset-capability-config'

    profile_id: str
    format_profile_id: str
    literal_allowlist_required: bool
    source_mutation_allowed: bool
    network_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_stable_id(self.format_profile_id, field_name="format_profile_id")
        if self.profile_id != SIMULATOR_DATASET_PRODUCTION_PROFILE_ID:
            raise ValueError("simulator dataset config has another production profile")
        if self.format_profile_id != SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID:
            raise ValueError("simulator dataset config has another directory profile")
        if self.literal_allowlist_required is not True:
            raise ValueError("simulator dataset config requires a literal allowlist")
        if self.source_mutation_allowed is not False:
            raise ValueError("held simulator source mutation must remain disabled")
        if self.network_required is not False:
            raise ValueError("held simulator registration must use zero network")


def simulator_dataset_capability_config() -> SimulatorDatasetCapabilityConfig:
    return SimulatorDatasetCapabilityConfig(
        profile_id=SIMULATOR_DATASET_PRODUCTION_PROFILE_ID,
        format_profile_id=SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID,
        literal_allowlist_required=True,
        source_mutation_allowed=False,
        network_required=False,
    )


def _config_ref(config: SimulatorDatasetCapabilityConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=SIMULATOR_DATASET_CONFIG_KEY,
        config_schema=config.SCHEMA,
        config_schema_sha256=hashlib.sha256(config.SCHEMA.encode("utf-8")).hexdigest(),
        content_sha256=config.fingerprint(),
        artifact_id=SIMULATOR_DATASET_CONFIG_ARTIFACT_ID,
    )


def _require_artifact(
    artifact: ArtifactIdentity,
    *,
    expected_id: str,
    field_name: str,
) -> None:
    if not isinstance(artifact, ArtifactIdentity) or artifact.artifact_id != expected_id:
        raise ValueError(f"{field_name} has another exact artifact identity")


def _require_closure(
    closure: ImplementationSourceClosure,
    *,
    expected_id: str,
    expected_commit: str,
    field_name: str,
) -> None:
    if (
        not isinstance(closure, ImplementationSourceClosure)
        or closure.closure_id != expected_id
        or closure.implementation_commit != expected_commit
    ):
        raise ValueError(f"{field_name} has another exact source closure")


@dataclass(frozen=True, slots=True)
class SimulatorRegistrationStaticComposition(CanonicalRecord):
    """Exact non-writing closure for one directory-backed registration."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/simulator-registration-static-composition'

    implementation_commit: str
    storage_root: ObjectIdentity
    definition: SimulatorHeldRegistrationDefinition
    config: SimulatorDatasetCapabilityConfig
    runtime_artifact: ArtifactIdentity
    environment_artifact: ArtifactIdentity
    inspector_closure: ImplementationSourceClosure
    evidence_verifier_closure: ImplementationSourceClosure
    registry_profile: SimulatorDatasetRegistryProfile
    registration_manifest: DatasetRegistrationManifest

    def __post_init__(self) -> None:
        if (
            not isinstance(self.storage_root, ObjectIdentity)
            or self.storage_root.object_schema != DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA
        ):
            raise ValueError("storage_root must identify a current external root")
        if not isinstance(self.definition, SimulatorHeldRegistrationDefinition):
            raise ValueError("definition must be a held simulator registration")
        if self.config != simulator_dataset_capability_config():
            raise ValueError("config differs from the simulator production config")
        _require_artifact(
            self.runtime_artifact,
            expected_id=SIMULATOR_DATASET_RUNTIME_KEY,
            field_name="runtime_artifact",
        )
        _require_artifact(
            self.environment_artifact,
            expected_id=SIMULATOR_DATASET_ENVIRONMENT_KEY,
            field_name="environment_artifact",
        )
        _require_closure(
            self.inspector_closure,
            expected_id=SIMULATOR_DATASET_INSPECTOR_CLOSURE_ID,
            expected_commit=self.implementation_commit,
            field_name="inspector_closure",
        )
        _require_closure(
            self.evidence_verifier_closure,
            expected_id=SIMULATOR_DATASET_EVIDENCE_CLOSURE_ID,
            expected_commit=self.implementation_commit,
            field_name="evidence_verifier_closure",
        )
        expected_profile = build_simulator_dataset_registry_profile(
            config=_config_ref(self.config),
            runtime=ObjectIdentity.from_record(
                self.runtime_artifact.artifact_id,
                self.runtime_artifact,
            ),
            environment=ObjectIdentity.from_record(
                self.environment_artifact.artifact_id,
                self.environment_artifact,
            ),
            inspector_implementation_sha256=self.inspector_closure.fingerprint(),
            evidence_verifier_implementation_sha256=(self.evidence_verifier_closure.fingerprint()),
        )
        if self.registry_profile != expected_profile:
            raise ValueError("registry_profile differs from the exact production closure")
        expected_manifest = build_simulator_held_registration_manifest(
            self.definition,
            SimulatorRegistrationManifestDependencies(
                storage_root=self.storage_root,
                capabilities=simulator_dataset_capability_bindings(expected_profile),
            ),
        )
        if self.registration_manifest != expected_manifest:
            raise ValueError("registration_manifest differs from the exact composition")


def build_simulator_registration_static_composition(
    *,
    implementation_commit: str,
    storage_root: ObjectIdentity,
    definition: SimulatorHeldRegistrationDefinition,
    runtime_artifact: ArtifactIdentity,
    environment_artifact: ArtifactIdentity,
    inspector_closure: ImplementationSourceClosure,
    evidence_verifier_closure: ImplementationSourceClosure,
) -> SimulatorRegistrationStaticComposition:
    config = simulator_dataset_capability_config()
    profile = build_simulator_dataset_registry_profile(
        config=_config_ref(config),
        runtime=ObjectIdentity.from_record(runtime_artifact.artifact_id, runtime_artifact),
        environment=ObjectIdentity.from_record(
            environment_artifact.artifact_id,
            environment_artifact,
        ),
        inspector_implementation_sha256=inspector_closure.fingerprint(),
        evidence_verifier_implementation_sha256=evidence_verifier_closure.fingerprint(),
    )
    manifest = build_simulator_held_registration_manifest(
        definition,
        SimulatorRegistrationManifestDependencies(
            storage_root=storage_root,
            capabilities=simulator_dataset_capability_bindings(profile),
        ),
    )
    return SimulatorRegistrationStaticComposition(
        implementation_commit=implementation_commit,
        storage_root=storage_root,
        definition=definition,
        config=config,
        runtime_artifact=runtime_artifact,
        environment_artifact=environment_artifact,
        inspector_closure=inspector_closure,
        evidence_verifier_closure=evidence_verifier_closure,
        registry_profile=profile,
        registration_manifest=manifest,
    )


__all__ = [
    "SIMULATOR_DATASET_CONFIG_ARTIFACT_ID",
    "SIMULATOR_DATASET_EVIDENCE_CLOSURE_ID",
    "SIMULATOR_DATASET_INSPECTOR_CLOSURE_ID",
    "SIMULATOR_DATASET_PRODUCTION_PROFILE_ID",
    "SimulatorDatasetCapabilityConfig",
    "SimulatorRegistrationStaticComposition",
    "build_simulator_registration_static_composition",
    "simulator_dataset_capability_config",
]
