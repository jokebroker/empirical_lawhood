"""Fingerprint-bound Prospective nomination, design and adjudication capabilities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.planning.design import ExperimentProposal
from empirical_lawhood.planning.exploration import HypothesisSet, ProspectiveNomination
from empirical_lawhood.planning.prospective import (
    NominationObligations,
    ProspectiveAdjudication,
    ProspectiveDesignContext,
    ProspectiveEvidenceResult,
    ProspectiveNominationDecision,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .adjudication import ProspectiveAdjudicator
from .designers import default_designer_registry
from .nominator import ProspectiveNominator


def prospective_capability_keys() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                ProspectiveNominator.capability_key,
                ProspectiveAdjudicator.capability_key,
                *(designer.capability_key for designer in default_designer_registry().designers),
            }
        )
    )


def _schema_sha256(schema: str) -> str:
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def _resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=2,
        memory_bytes=2_000_000_000,
        gpu_devices=0,
        wall_time_seconds=600,
        source_scan_bytes=1_000_000_000,
        output_bytes=100_000_000,
    )


def _manifest(
    *,
    key: str,
    kind: CapabilityKind,
    config_schema: str,
    input_schemas: tuple[str, ...],
    output_schemas: tuple[str, ...],
    checks: tuple[str, ...],
    implementation_sha256: str,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version="1.0.0",
        kind=kind,
        config_schema=config_schema,
        config_schema_sha256=_schema_sha256(config_schema),
        input_schema_ids=tuple(sorted(input_schemas)),
        output_schema_ids=tuple(sorted(output_schemas)),
        permissions=(CapabilityPermission.READ_OUTCOME_VISIBLE,),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        resource_ceiling=_resources(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11",
        requires_clean_commit=True,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=tuple(sorted(checks)),
        implementation_sha256=implementation_sha256,
    )


def prospective_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    expected = set(prospective_capability_keys())
    if set(implementation_sha256_by_key) != expected:
        raise ValueError("Prospective implementation identities must cover the exact capability family")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    manifests = [
        _manifest(
            key=ProspectiveNominator.capability_key,
            kind=CapabilityKind.PROSPECTIVE_NOMINATOR,
            config_schema=ProspectiveDesignContext.SCHEMA,
            input_schemas=(HypothesisSet.SCHEMA,),
            output_schemas=(
                NominationObligations.SCHEMA,
                ProspectiveNomination.SCHEMA,
                ProspectiveNominationDecision.SCHEMA,
            ),
            checks=(
                "prospective-complete-fresh-evidence-contract",
                "prospective-no-acquisition-stop",
                "prospective-outcome-visible-nonpromotion",
            ),
            implementation_sha256=implementation_sha256_by_key[ProspectiveNominator.capability_key],
        ),
        _manifest(
            key=ProspectiveAdjudicator.capability_key,
            kind=CapabilityKind.HYPOTHESIS_ADJUDICATOR,
            config_schema=ProspectiveEvidenceResult.SCHEMA,
            input_schemas=(
                ExperimentProposal.SCHEMA,
                HypothesisSet.SCHEMA,
                NominationObligations.SCHEMA,
                ProspectiveEvidenceResult.SCHEMA,
                ProspectiveNomination.SCHEMA,
            ),
            output_schemas=(ProspectiveAdjudication.SCHEMA,),
            checks=(
                "prospective-predeclared-terminal-interpretation",
                "prospective-source-fingerprint-preservation",
                "prospective-three-way-adjudication",
            ),
            implementation_sha256=implementation_sha256_by_key[
                ProspectiveAdjudicator.capability_key
            ],
        ),
    ]
    for designer in default_designer_registry().designers:
        manifests.append(
            _manifest(
                key=designer.capability_key,
                kind=CapabilityKind.EXPERIMENT_DESIGNER,
                config_schema=NominationObligations.SCHEMA,
                input_schemas=(NominationObligations.SCHEMA, ProspectiveNomination.SCHEMA),
                output_schemas=(ExperimentProposal.SCHEMA, ExperimentSpec.SCHEMA),
                checks=(
                    "prospective-advisory-only",
                    "prospective-native-measurement-and-controls",
                    "prospective-sealed-fresh-claim",
                ),
                implementation_sha256=implementation_sha256_by_key[designer.capability_key],
            )
        )
    return tuple(sorted(manifests, key=lambda item: item.registry_id))


def prospective_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="prospective-prospective-confirmation-capabilities",
        capabilities=prospective_capability_manifests(implementation_sha256_by_key),
    )
