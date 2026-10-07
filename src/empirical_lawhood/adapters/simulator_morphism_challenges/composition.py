"""Write-free composition for one separately issued simulator morphism challenges phase.

The four phase DAGs have different registries and external inputs.  Composition
therefore binds exactly one phase at a time; it never searches for an upstream
"latest" record or invents a commitment, qualification, freeze, or outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Mapping

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodCatalog,
)
from empirical_lawhood.planning.formal_gaps import FormalGapRegister
from empirical_lawhood.planning.study_authoring import DesignInputRecord
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.candidate_composition import CandidateContextResolution, StandardCandidateContextResolution
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution
from empirical_lawhood.runtime.artifacts import ArtifactWriter
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .authoring import SimulatorMorphismChallengeExternalRecord, config_decoders
from .contracts import SimulatorMorphismChallengeConfig
from .contracts import SimulatorMorphismChallengePhase
from .numeric_inputs import RCChallengeNumericInput
from .retained_results import RCChallengeRetainedResult
from .issued_inputs import CAPABILITY_INPUT_TYPES, RCChallengeIssuedExternalInput, RCChallengeIssuedInputs, RCChallengeRuntimeInputPort, capability_port_key
from .extension_bundle import INSTALLED_IMPLEMENTATION_SHA256
from .capability_configs import CAPABILITY_CONFIG_TYPES
from .study_authoring import SimulatorMorphismChallengeStandardAuthoringBundle, build_simulator_morphism_challenges_standard_authoring_bundle
from .runtime_provider import SimulatorMorphismChallengeCampaignRuntimeProvider, implementation_closures


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengePhaseComposition:
    """Exact candidate/runtime bindings and control bytes for one simulator morphism challenge phase."""

    phase_id: str
    implementation_sha256: str
    bundle: SimulatorMorphismChallengeStandardAuthoringBundle
    config_decoders: tuple[CandidateCapabilityConfigDecoder, ...]
    runtime_provider: SimulatorMorphismChallengeCampaignRuntimeProvider
    issued_inputs: RCChallengeIssuedInputs

    def __post_init__(self) -> None:
        validate_stable_id(self.phase_id, field_name="phase_id")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.bundle.base.context.implementation_sha256 != self.implementation_sha256:
            raise ValueError("simulator morphism challenges phase composition implementation differs")
        if self.runtime_provider.registry != self.bundle.base.registry:
            raise ValueError("simulator morphism challenges phase composition registry differs")

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return self.bundle.base.catalog

    @property
    def known_design_inputs(self) -> tuple[DesignInputRecord, ...]:
        return self.bundle.base.draft.design_inputs

    @property
    def formal_methods(self) -> FormalMethodCatalog:
        return self.bundle.formal_methods

    @property
    def formal_source_inventories(
        self,
    ) -> tuple[FormalGapSourceCapabilityInventory, ...]:
        return (self.bundle.formal_source_inventory,)

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        """Return every compact byte string required by public resolution."""

        base = self.bundle.base
        records: tuple[CanonicalRecord, ...] = (
            base.config,
            *(value.record for value in base.external_records),
            *base.source_configs,
            *base.qualifications,
            *self.payloads,
        )
        by_digest = {
            sha256(value.canonical_bytes()).hexdigest(): value.canonical_bytes()
            for value in records
        }
        return tuple(by_digest[key] for key in sorted(by_digest))

    @property
    def payloads(self) -> tuple[CanonicalRecord, ...]:
        return tuple(record for manifest in self.bundle.base.registry.capabilities
                     for record in (CAPABILITY_CONFIG_TYPES[manifest.capability_key](self.issued_inputs.config),
                                    CAPABILITY_INPUT_TYPES[manifest.capability_key](self.issued_inputs)))

    @property
    def decoder_registrations(self):
        from .executable_binding import BINDINGS

        selected = {value.capability_key for value in self.bundle.base.registry.capabilities}
        return tuple(decoder for binding in BINDINGS if binding.capability_key in selected
                     for decoder in binding.issued_decoder_registrations)

    @property
    def authoring(self) -> ExecutableStudyDefinition:
        package = self.bundle.authoring_package
        by_schema = {value.payload_schema: value for value in self.decoder_registrations}
        extensions = tuple(ProposedStudyExtension(
            f"{self.phase_id}.extension.{index:02d}", f"{self.phase_id}.namespace.{index:02d}",
            ObjectIdentity.from_record(f"{self.phase_id}.payload.{index:02d}", payload),
            len(payload.canonical_bytes()), by_schema[payload.SCHEMA].decoder_key,
            by_schema[payload.SCHEMA].decoder_version, by_schema[payload.SCHEMA].config_sha256,
            True, OutcomeAccess.OUTCOME_BLIND, VisibilityCeiling.PROSPECTIVE,
        ) for index, payload in enumerate(self.payloads))
        return ExecutableStudyDefinition(f"{self.phase_id}.executable-study-definition", package,
            ProposedStudyExtensionSet(f"{self.phase_id}.extensions",
                ObjectIdentity.from_record(package.package_id, package),
                tuple(value.namespace_id for value in extensions), extensions))


@dataclass(frozen=True, slots=True)
class RCChallengeCandidateContextProvider:
    composition: SimulatorMorphismChallengePhaseComposition

    @property
    def catalog(self):
        return self.composition.catalog

    def resolve(self, draft):
        if draft != self.composition.bundle.base.draft:
            raise ValueError("RC challenge draft differs from its frozen phase context")
        context = self.composition.bundle.base.context
        return CandidateContextResolution(context, (), CandidateSourceResolution(context.qualifications, (), ()))

    def resolve_standard(self, package):
        if package != self.composition.bundle.authoring_package:
            raise ValueError("RC challenge package differs from its frozen phase context")
        context = self.composition.bundle.context
        return StandardCandidateContextResolution(context, (), CandidateSourceResolution(context.base.qualifications, (), ()))


def runtime_platform_ports(*, config: SimulatorMorphismChallengeConfig,
                           source_files: Mapping[str, bytes], artifact_writer: ArtifactWriter,
                           prerequisite_access_verifier: Callable[[RCChallengeRetainedResult], None] | None = None):
    from .authoring import simulator_morphism_challenges_phase_registry

    registry = simulator_morphism_challenges_phase_registry(implementation_sha256=INSTALLED_IMPLEMENTATION_SHA256, config=config)
    port = RCChallengeRuntimeInputPort(source_files, artifact_writer, prerequisite_access_verifier)
    return tuple(ExecutablePlatformPort(capability_port_key(manifest.capability_key), port) for manifest in registry.capabilities)


def compose_simulator_morphism_challenges_phase(
    *,
    config: SimulatorMorphismChallengeConfig,
    external_records: tuple[SimulatorMorphismChallengeExternalRecord, ...],
    source_files: Mapping[str, bytes],
    register: FormalGapRegister,
    implementation_sha256: str = INSTALLED_IMPLEMENTATION_SHA256,
    numeric_input: RCChallengeNumericInput | None = None,
    artifact_writer: ArtifactWriter | None = None,
    prerequisite_custody: tuple[RCChallengeRetainedResult, ...] = (),
    prerequisite_access_verifier: Callable[[RCChallengeRetainedResult], None] | None = None,
) -> SimulatorMorphismChallengePhaseComposition:
    """Bind one phase without writes, source acquisition, reveal, or authority."""

    closures = implementation_closures(source_files)
    issued = RCChallengeIssuedInputs(
        f"simulator-morphism-challenges.{config.phase.value.lower()}.issued-inputs", config,
        tuple(RCChallengeIssuedExternalInput.from_external(value) for value in sorted(external_records, key=lambda value: value.input_id)),
        numeric_input, closures.complete_sha256, prerequisite_custody,
    )
    if numeric_input is not None:
        if artifact_writer is None:
            raise ValueError("RC phase composition requires the actual external custody verifier")
        numeric_input.authenticate(artifact_writer)
    for retained in prerequisite_custody:
        if artifact_writer is None:
            raise ValueError("RC prerequisite custody requires the selected external verifier")
        if retained.requires_outcome_authority:
            if prerequisite_access_verifier is None:
                raise PermissionError("RC protected prerequisite requires actual current outcome authority")
            prerequisite_access_verifier(retained)
        retained.authenticate(artifact_writer)
    scientific_inputs = () if config.phase in (SimulatorMorphismChallengePhase.CANARY, SimulatorMorphismChallengePhase.NOMINATION) else numeric_input.scientific_inputs(config) if numeric_input is not None else None
    bundle = build_simulator_morphism_challenges_standard_authoring_bundle(
        config=config,
        implementation_sha256=implementation_sha256,
        manifest_implementation_sha256=INSTALLED_IMPLEMENTATION_SHA256,
        register=register,
        external_records=external_records,
    )
    return SimulatorMorphismChallengePhaseComposition(
        phase_id=f"simulator-morphism-challenges.{config.phase.value.lower()}",
        implementation_sha256=implementation_sha256,
        bundle=bundle,
        config_decoders=config_decoders(bundle.base.catalog, config),
        runtime_provider=SimulatorMorphismChallengeCampaignRuntimeProvider(
            registry=bundle.base.registry,
            config=config,
            external_records=external_records,
            source_files=source_files,
            injected_nomination_seeds=numeric_input.source.seeds if numeric_input is not None and config.phase is SimulatorMorphismChallengePhase.NOMINATION else None,
            scientific_inputs=scientific_inputs,
            manifest_implementation_sha256=INSTALLED_IMPLEMENTATION_SHA256,
        ),
        issued_inputs=issued,
    )


__all__ = ['SimulatorMorphismChallengePhaseComposition', 'RCChallengeCandidateContextProvider', 'compose_simulator_morphism_challenges_phase', 'runtime_platform_ports']
