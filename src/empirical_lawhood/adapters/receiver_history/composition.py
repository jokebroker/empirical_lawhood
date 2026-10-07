"""Write-free composition for one separately issued receiver-history phase.

The five phase DAGs have different registries and external inputs.  Composition
therefore binds exactly one phase at a time; it never searches for an upstream
"latest" record or invents a commitment, qualification, freeze, or outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodCatalog,
)
from empirical_lawhood.planning.formal_gaps import FormalGapRegister
from empirical_lawhood.planning.study_authoring import DesignInputRecord
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .authoring import ReceiverHistoryExternalRecord, config_decoders
from .contracts import ReceiverHistoryConfig
from .study_authoring import ReceiverHistoryStandardAuthoringBundle, build_receiver_history_standard_authoring_bundle
from .runtime_provider import (
    ReceiverHistoryCampaignRuntimeProvider,
    implementation_closures,
)
from .runtime_contracts import FLOAT64_ARRAY_PAYLOAD_SCHEMA, INT64_ARRAY_PAYLOAD_SCHEMA


@dataclass(frozen=True, slots=True)
class ReceiverHistoryPhaseComposition:
    """Exact candidate/runtime bindings and control bytes for one receiver-history phase."""

    phase_id: str
    implementation_sha256: str
    bundle: ReceiverHistoryStandardAuthoringBundle
    config_decoders: tuple[CandidateCapabilityConfigDecoder, ...]
    runtime_provider: ReceiverHistoryCampaignRuntimeProvider
    candidate_runtime_provider: ReceiverHistoryCampaignRuntimeProvider

    def __post_init__(self) -> None:
        validate_stable_id(self.phase_id, field_name="phase_id")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.bundle.base.context.implementation_sha256 != self.implementation_sha256:
            raise ValueError("receiver-history phase composition implementation differs")
        if self.runtime_provider.registry != self.bundle.base.registry:
            raise ValueError("receiver-history phase composition registry differs")
        candidate_registry = self.bundle.base.catalog.compose_registry(
            self.bundle.base.draft.capability_selections
        )
        if self.candidate_runtime_provider.registry != candidate_registry:
            raise ValueError("receiver-history candidate runtime registry differs")

    @property
    def runtime_providers(self) -> tuple[CampaignRuntimeProvider, ...]:
        """Bind both authored and compiler-derived issued registry identities."""

        return tuple(
            sorted(
                (self.runtime_provider, self.candidate_runtime_provider),
                key=lambda value: value.registry_sha256,
            )
        )

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
        )
        by_digest = {
            sha256(value.canonical_bytes()).hexdigest(): value.canonical_bytes()
            for value in records
        }
        return tuple(by_digest[key] for key in sorted(by_digest))

    @property
    def numpy_payload_contracts(self) -> tuple[tuple[str, str], ...]:
        """Declare the closed binary payload schemas used by this phase family."""

        return (
            (FLOAT64_ARRAY_PAYLOAD_SCHEMA, "float64"),
            (INT64_ARRAY_PAYLOAD_SCHEMA, "int64"),
        )


def compose_receiver_history_phase(
    *,
    config: ReceiverHistoryConfig,
    external_records: tuple[ReceiverHistoryExternalRecord, ...],
    source_files: Mapping[str, bytes],
    register: FormalGapRegister,
    injected_nomination_seeds: tuple[bytes, ...] | None = None,
) -> ReceiverHistoryPhaseComposition:
    """Bind one phase without writes, source acquisition, reveal, or authority."""

    closures = implementation_closures(source_files)
    bundle = build_receiver_history_standard_authoring_bundle(
        config=config,
        implementation_sha256=closures.complete_sha256,
        register=register,
        external_records=external_records,
    )
    candidate_registry = bundle.base.catalog.compose_registry(
        bundle.base.draft.capability_selections
    )
    return ReceiverHistoryPhaseComposition(
        phase_id=f"receiver-history.{config.phase.value.lower()}",
        implementation_sha256=closures.complete_sha256,
        bundle=bundle,
        config_decoders=config_decoders(bundle.base.catalog, config),
        runtime_provider=ReceiverHistoryCampaignRuntimeProvider(
            registry=bundle.base.registry,
            config=config,
            external_records=external_records,
            source_files=source_files,
            injected_nomination_seeds=injected_nomination_seeds,
        ),
        candidate_runtime_provider=ReceiverHistoryCampaignRuntimeProvider(
            registry=candidate_registry,
            config=config,
            external_records=external_records,
            source_files=source_files,
            injected_nomination_seeds=injected_nomination_seeds,
        ),
    )


__all__ = ["ReceiverHistoryPhaseComposition", "compose_receiver_history_phase"]
