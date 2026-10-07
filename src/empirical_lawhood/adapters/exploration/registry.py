"""Fingerprint-bound Exploration detector, template, planner and synthesis capabilities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import EvidenceSnapshot
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.planning.discovery import (
    AnalysisObligations,
    ExplorationDiagnosticProjection,
    HypothesisSynthesis,
    PortfolioCandidate,
    PortfolioDecision,
    PortfolioPolicy,
    SkepticReport,
)
from empirical_lawhood.planning.exploration import (
    AnalysisSpec,
    AnomalySignal,
    ExplorationPlan,
    ExploratoryFinding,
    HypothesisSet,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .detectors import DetectorDefinition, default_detector_registry

PIPELINE_KEYS = (
    "exploration.admission-structure",
    "exploration.information-limit",
    "exploration.model-set-structure",
    "exploration.reachability-structure",
    "exploration.response-dynamics",
    "exploration.response-geometry",
    "exploration.support-structure",
    "exploration.transport-structure",
)
PLANNER_KEY = "exploration.pareto-portfolio-planner"
SKEPTIC_KEY = "exploration.automatic-skeptic"
SYNTHESIS_KEY = "exploration.hypothesis-synthesis"


def exploration_capability_keys() -> tuple[str, ...]:
    detector_keys = tuple(
        definition.detector_key for definition in default_detector_registry().detectors
    )
    return tuple(sorted((*detector_keys, *PIPELINE_KEYS, PLANNER_KEY, SKEPTIC_KEY, SYNTHESIS_KEY)))


def _schema_sha256(schema: str) -> str:
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def _resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=4,
        memory_bytes=8_000_000_000,
        gpu_devices=0,
        wall_time_seconds=3_600,
        source_scan_bytes=100_000_000_000,
        output_bytes=1_000_000_000,
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
        permissions=(
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.READ_OUTCOME_VISIBLE,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        resource_ceiling=_resources(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=tuple(sorted(checks)),
        implementation_sha256=implementation_sha256,
    )


def exploration_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    expected = set(exploration_capability_keys())
    if set(implementation_sha256_by_key) != expected:
        raise ValueError("Exploration implementation identities must cover the exact capability family")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    manifests: list[CapabilityManifest] = []
    for definition in default_detector_registry().detectors:
        manifests.append(
            _manifest(
                key=definition.detector_key,
                kind=CapabilityKind.ANOMALY_DETECTOR,
                config_schema=DetectorDefinition.SCHEMA,
                input_schemas=(ExplorationDiagnosticProjection.SCHEMA,),
                output_schemas=(AnomalySignal.SCHEMA,),
                checks=(
                    "exploration-detector-measure-semantics",
                    "exploration-no-positivity-objective",
                    "exploration-outcome-visible-nonpromotion",
                ),
                implementation_sha256=implementation_sha256_by_key[definition.detector_key],
            )
        )
    for key in PIPELINE_KEYS:
        manifests.append(
            _manifest(
                key=key,
                kind=CapabilityKind.ANALYSIS,
                config_schema=AnalysisSpec.SCHEMA,
                input_schemas=(ExplorationDiagnosticProjection.SCHEMA,),
                output_schemas=(ExploratoryFinding.SCHEMA,),
                checks=(
                    "exploration-complete-search-family",
                    "exploration-no-arbitrary-code",
                    "exploration-physical-unit-scope",
                ),
                implementation_sha256=implementation_sha256_by_key[key],
            )
        )
    manifests.extend(
        (
            _manifest(
                key=PLANNER_KEY,
                kind=CapabilityKind.PORTFOLIO_PLANNER,
                config_schema=PortfolioPolicy.SCHEMA,
                input_schemas=(
                    EvidenceSnapshot.SCHEMA,
                    PortfolioCandidate.SCHEMA,
                ),
                output_schemas=(ExplorationPlan.SCHEMA, PortfolioDecision.SCHEMA),
                checks=(
                    "exploration-complete-dispositions",
                    "exploration-pareto-not-scalar-reward",
                    "exploration-stop-no-analysis",
                ),
                implementation_sha256=implementation_sha256_by_key[PLANNER_KEY],
            ),
            _manifest(
                key=SKEPTIC_KEY,
                kind=CapabilityKind.FALSIFIER,
                config_schema=AnalysisObligations.SCHEMA,
                input_schemas=(ExploratoryFinding.SCHEMA,),
                output_schemas=(SkepticReport.SCHEMA,),
                checks=(
                    "exploration-family-completeness",
                    "exploration-influence-and-artifact-controls",
                    "exploration-matched-nulls",
                ),
                implementation_sha256=implementation_sha256_by_key[SKEPTIC_KEY],
            ),
            _manifest(
                key=SYNTHESIS_KEY,
                kind=CapabilityKind.HYPOTHESIS_SYNTHESIZER,
                config_schema=ExplorationPlan.SCHEMA,
                input_schemas=(ExploratoryFinding.SCHEMA,),
                output_schemas=(HypothesisSet.SCHEMA, HypothesisSynthesis.SCHEMA),
                checks=(
                    "exploration-competing-hypotheses",
                    "exploration-disconfirming-observations",
                    "exploration-outcome-visible-nonpromotion",
                ),
                implementation_sha256=implementation_sha256_by_key[SYNTHESIS_KEY],
            ),
        )
    )
    return tuple(sorted(manifests, key=lambda manifest: manifest.registry_id))


def exploration_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="exploration-automated-exploration-capabilities",
        capabilities=exploration_capability_manifests(implementation_sha256_by_key),
    )
