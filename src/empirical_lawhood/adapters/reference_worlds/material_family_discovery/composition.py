"""Write-free composition for one separately issued SDCB-SC phase."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    canonical_json_bytes,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodCatalog,
)
from empirical_lawhood.planning.formal_gaps import FormalGapRegister
from empirical_lawhood.planning.study_authoring import DesignInputRecord
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind
from empirical_lawhood.runtime.artifacts import TabularPayloadContract
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.study_issue import StudySourceClosureInspector

from .authoring import MaterialFamilyDiscoveryAuthoringBundle, build_material_family_authoring_bundle
from .codecs import table_payload_contracts
from .contracts import MaterialFamilyConfig, MaterialSourceManifest, MaterialFamilyDiscoveryAdjudicationConfig, MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION, SourceQualification
from .protocol import world_configs
from .runtime_provider import MaterialFamilyDiscoveryCampaignRuntimeProvider, MaterialFamilyDiscoveryConfigDecoder, material_family_config_decoders
from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import (
    DiscoveryPolicyConfig,
)


MATERIAL_FAMILY_IMPLEMENTATION_SOURCE_CLOSURE_PATHS = tuple(
    sorted(
        (
            "pyproject.toml",
            "src/empirical_lawhood/api/composition.py",
            "src/empirical_lawhood/api/execution.py",
            "src/empirical_lawhood/adapters/methods/budgeted_first_discovery/contracts.py",
            "src/empirical_lawhood/adapters/methods/budgeted_first_discovery/metrics.py",
            "src/empirical_lawhood/adapters/methods/budgeted_first_discovery/selectors.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/authoring.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/codecs.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/composition.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/contracts.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/descriptors.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/protocol.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/records.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/registration.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/runtime_provider.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/source.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/system.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/workflow.py",
            "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery/worlds.py",
            "src/empirical_lawhood/infrastructure/artifacts.py",
            "src/empirical_lawhood/infrastructure/execution.py",
            "src/empirical_lawhood/infrastructure/task_receipts.py",
            "src/empirical_lawhood/kernel/evidence.py",
            "src/empirical_lawhood/kernel/serialization.py",
            "src/empirical_lawhood/runtime/artifacts.py",
            "src/empirical_lawhood/runtime/capabilities.py",
            "src/empirical_lawhood/runtime/compiler.py",
            "src/empirical_lawhood/runtime/execution.py",
            "src/empirical_lawhood/runtime/plans.py",
            "src/empirical_lawhood/runtime/providers.py",
            "uv.lock",
        )
    )
)
MATERIAL_FAMILY_IMPLEMENTATION_MANIFEST_ID = f"manifest.material-family-discovery-implementation-source-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"
MATERIAL_FAMILY_IMPLEMENTATION_MANIFEST_SCHEMA = 'empirical-lawhood/reference-worlds/material-family-discovery/implementation-source-manifest'


def material_family_implementation_sha256(source_files: Mapping[str, bytes]) -> str:
    """Fingerprint the exact SC worker sources and pinned Python environment."""

    if tuple(sorted(source_files)) != MATERIAL_FAMILY_IMPLEMENTATION_SOURCE_CLOSURE_PATHS:
        raise ValueError("SC implementation source roster differs")
    return sha256(
        canonical_json_bytes(
            tuple(
                (path, sha256(source_files[path]).hexdigest())
                for path in MATERIAL_FAMILY_IMPLEMENTATION_SOURCE_CLOSURE_PATHS
            )
        )
    ).hexdigest()


def material_family_implementation_manifest_identity(implementation_sha256: str) -> ObjectIdentity:
    """Name the exact source roster whose member bytes produce the implementation ID."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    return ObjectIdentity(
        object_id=MATERIAL_FAMILY_IMPLEMENTATION_MANIFEST_ID,
        object_schema=MATERIAL_FAMILY_IMPLEMENTATION_MANIFEST_SCHEMA,
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )


class MaterialFamilyDiscoveryImplementationSourceClosureInspector:
    """Re-read the exact bounded SC roster without requiring an unrelated clean tree."""

    _MAXIMUM_SOURCE_BYTES = 32 * 1024**2

    def __init__(self, repo_root: Path, implementation_sha256: str) -> None:
        self._repo_root = repo_root.resolve(strict=True)
        self._implementation_sha256 = implementation_sha256
        validate_sha256(implementation_sha256, field_name="implementation_sha256")

    def observe(
        self,
        expected: ImplementationSourceClosure,
    ) -> ImplementationSourceClosure:
        if expected.implementation_sha256 != self._implementation_sha256:
            raise PermissionError("SC source-closure record differs")
        if expected.kind is SourceClosureKind.EXACT_SOURCE_CLOSURE and (
            expected.source_tree_sha256 != self._implementation_sha256
            or expected.clean_worktree
            or expected.exact_source_manifest
            != material_family_implementation_manifest_identity(self._implementation_sha256)
        ):
            raise PermissionError("SC exact-source closure record differs")
        source_files: dict[str, bytes] = {}
        for relative_path in MATERIAL_FAMILY_IMPLEMENTATION_SOURCE_CLOSURE_PATHS:
            path = self._repo_root / relative_path
            observed = path.lstat()
            if path.is_symlink() or not path.is_file():
                raise PermissionError("SC source closure contains a non-regular member")
            if observed.st_size <= 0 or observed.st_size > self._MAXIMUM_SOURCE_BYTES:
                raise PermissionError("SC source closure member violates its byte bound")
            payload = path.read_bytes()
            if len(payload) != observed.st_size:
                raise PermissionError("SC source closure member changed while reading")
            source_files[relative_path] = payload
        if material_family_implementation_sha256(source_files) != self._implementation_sha256:
            raise PermissionError("SC implementation changed before issue")
        return expected


class MaterialFamilyStudySourceClosureInspector:
    """Compose exact SC-byte replay with clean-Git replay where declared."""

    def __init__(
        self,
        repo_root: Path,
        implementation_sha256: str,
        clean_git_inspector: StudySourceClosureInspector,
    ) -> None:
        self._implementation_inspector = MaterialFamilyDiscoveryImplementationSourceClosureInspector(
            repo_root,
            implementation_sha256,
        )
        self._clean_git_inspector = clean_git_inspector

    def observe(
        self,
        expected: ImplementationSourceClosure,
    ) -> ImplementationSourceClosure:
        observed = self._implementation_inspector.observe(expected)
        if expected.kind is SourceClosureKind.CLEAN_GIT_COMMIT:
            return self._clean_git_inspector.observe(observed)
        return observed


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryPhaseComposition:
    """Exact authoring, candidate, runtime and wire contracts for one SC phase."""

    phase_id: str
    implementation_sha256: str
    bundle: MaterialFamilyDiscoveryAuthoringBundle
    config_decoders: tuple[MaterialFamilyDiscoveryConfigDecoder, ...]
    runtime_provider: MaterialFamilyDiscoveryCampaignRuntimeProvider
    candidate_runtime_provider: MaterialFamilyDiscoveryCampaignRuntimeProvider

    def __post_init__(self) -> None:
        validate_stable_id(self.phase_id, field_name="phase_id")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.bundle.context.implementation_sha256 != self.implementation_sha256:
            raise ValueError("SC composition implementation differs from authoring")
        if self.runtime_provider.registry != self.bundle.registry:
            raise ValueError("SC composition authored registry differs")
        candidate_registry = self.bundle.catalog.compose_registry(
            self.bundle.draft.capability_selections
        )
        if self.candidate_runtime_provider.registry != candidate_registry:
            raise ValueError("SC composition candidate registry differs")

    @property
    def runtime_providers(self) -> tuple[MaterialFamilyDiscoveryCampaignRuntimeProvider, ...]:
        by_hash = {
            value.registry_sha256: value
            for value in (self.runtime_provider, self.candidate_runtime_provider)
        }
        return tuple(by_hash[key] for key in sorted(by_hash))

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.catalog

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        return self.bundle.candidate_input_payloads

    @property
    def known_design_inputs(self) -> tuple[DesignInputRecord, ...]:
        return self.bundle.design_inputs

    @property
    def formal_methods(self) -> FormalMethodCatalog:
        return self.bundle.formal_methods

    @property
    def formal_source_inventories(
        self,
    ) -> tuple[FormalGapSourceCapabilityInventory, ...]:
        return (self.bundle.inventory,)

    @property
    def numpy_payload_contracts(self) -> tuple[tuple[str, str], ...]:
        return ()

    @property
    def tabular_payload_contracts(self) -> tuple[TabularPayloadContract, ...]:
        return table_payload_contracts()


def compose_material_family_phase(
    *,
    family_config: MaterialFamilyConfig,
    adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig,
    policies: tuple[DiscoveryPolicyConfig, ...],
    source_manifest: MaterialSourceManifest,
    source_qualification: SourceQualification,
    corpus_payload: bytes,
    source_files: Mapping[str, bytes],
    register: FormalGapRegister,
) -> MaterialFamilyDiscoveryPhaseComposition:
    """Bind one phase without source acquisition, writes, issue, execution or reveal."""

    implementation = material_family_implementation_sha256(source_files)
    bundle = build_material_family_authoring_bundle(
        family_config=family_config,
        adjudication_config=adjudication_config,
        policies=policies,
        source_manifest=source_manifest,
        source_qualification=source_qualification,
        corpus_payload=corpus_payload,
        register=register,
        implementation_sha256=implementation,
    )
    candidate_registry = bundle.catalog.compose_registry(bundle.draft.capability_selections)

    def provider(registry: CapabilityRegistry) -> MaterialFamilyDiscoveryCampaignRuntimeProvider:
        return MaterialFamilyDiscoveryCampaignRuntimeProvider(
            registry=registry,
            family_config=family_config,
            adjudication_config=adjudication_config,
            source_manifest=source_manifest,
            source_qualification=source_qualification,
            corpus_payload=corpus_payload,
            corpus_sha256=bundle.corpus_sha256,
            corpus_size_bytes=len(corpus_payload),
            policies=policies,
            world_configs=world_configs(family_config),
        )

    return MaterialFamilyDiscoveryPhaseComposition(
        phase_id=(f"material-family-discovery.{family_config.phase.value.lower()}.{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"),
        implementation_sha256=implementation,
        bundle=bundle,
        config_decoders=material_family_config_decoders(
            registry=bundle.registry,
            family_config=family_config,
            adjudication_config=adjudication_config,
            policies=policies,
            world_configs=bundle.world_configs,
        ),
        runtime_provider=provider(bundle.registry),
        candidate_runtime_provider=provider(candidate_registry),
    )


__all__ = [
    'MATERIAL_FAMILY_IMPLEMENTATION_MANIFEST_ID',
    'MATERIAL_FAMILY_IMPLEMENTATION_MANIFEST_SCHEMA',
    'MATERIAL_FAMILY_IMPLEMENTATION_SOURCE_CLOSURE_PATHS',
    'MaterialFamilyDiscoveryImplementationSourceClosureInspector',
    'MaterialFamilyStudySourceClosureInspector',
    'MaterialFamilyDiscoveryPhaseComposition',
    'compose_material_family_phase',
    'material_family_implementation_sha256',
    'material_family_implementation_manifest_identity',
]
