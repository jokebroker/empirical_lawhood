"""Closed capability manifests for the reusable receiver-conditioned methods."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.planning.identification_evidence import IdentificationEvidenceProjection
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .acquisition_scores import (
    DOptimalCandidate,
    DOptimalConfig,
    DOptimalScoreSet,
)
from .contracts import (
    CanonicalMatrix,
    ControlledIOMember,
    ControlledIOQualificationConfig,
    ControlledIOVersionSet,
    CoordinateBasis,
    MarkovKernelFamily,
)
from .farthest_point_maximin import (
    FarthestPointConfig,
    FarthestPointOrder,
    MaximinCandidateSet,
    MaximinConfig,
    NativeActionPoint,
)
from .finite_action_mpc import FiniteMPCConfig, FiniteMPCMemberPrediction, FiniteMPCProposal
from .jacobi import JacobiConfig, JacobiWitness
from .mechanistic_evidence import (
    MechanisticEvidencePlan,
    MechanisticEvidenceResult,
    MechanisticProductResult,
)
from .nonnormal import (
    NonnormalAudit,
    NonnormalAuditConfig,
    PrefixPathwiseMargin,
)
from .receiver_metric import ReceiverRieszFamily, StateMetric, StateMetricConfig
from .version_set import ControlledIOVersionSetConfig


CONTROLLED_IO_IDENTIFIER_KEY = "response-law.controlled-io-finite"
CONTROLLED_IO_QUALIFIER_KEY = "response-law.controlled-io-version-qualifier"
CONTROLLED_IO_EVALUATOR_KEY = "response-law.controlled-io-evaluator"
RECEIVER_METRIC_ANALYSIS_KEY = "response-law.receiver-metric-analysis"
RECEIVER_JACOBI_ANALYSIS_KEY = "response-law.receiver-jacobi-analysis"
NONNORMAL_AUDIT_KEY = "response-law.nonnormal-pathwise-audit"
MECHANISTIC_EVIDENCE_KEY = "response-law.mechanistic-evidence"
D_OPTIMAL_DESIGN_KEY = "response-law.d-optimal-design"
FARTHEST_POINT_DESIGN_KEY = "response-law.farthest-point-design"
FINITE_MPC_SYNTHESIS_KEY = "response-law.finite-chart-mpc"
MAXIMIN_SYNTHESIS_KEY = "response-law.farthest-maximin"

_KEYS = (
    CONTROLLED_IO_IDENTIFIER_KEY,
    CONTROLLED_IO_QUALIFIER_KEY,
    CONTROLLED_IO_EVALUATOR_KEY,
    D_OPTIMAL_DESIGN_KEY,
    FARTHEST_POINT_DESIGN_KEY,
    FINITE_MPC_SYNTHESIS_KEY,
    MAXIMIN_SYNTHESIS_KEY,
    MECHANISTIC_EVIDENCE_KEY,
    NONNORMAL_AUDIT_KEY,
    RECEIVER_JACOBI_ANALYSIS_KEY,
    RECEIVER_METRIC_ANALYSIS_KEY,
)


def _schema_digest(schema: str) -> str:
    return hashlib.sha256(schema.encode()).hexdigest()


def receiver_conditioned_method_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    if set(implementation_sha256_by_key) != set(_KEYS):
        raise ValueError("receiver-conditioned implementation roster is incomplete")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    specifications = {
        CONTROLLED_IO_IDENTIFIER_KEY: (
            CapabilityKind.LAW_IDENTIFIER,
            ControlledIOQualificationConfig.SCHEMA,
            (IdentificationEvidenceProjection.SCHEMA,),
            (ControlledIOMember.SCHEMA,),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        CONTROLLED_IO_QUALIFIER_KEY: (
            CapabilityKind.NUMERICAL_QUALIFIER,
            ControlledIOVersionSetConfig.SCHEMA,
            (ControlledIOMember.SCHEMA,),
            (ControlledIOVersionSet.SCHEMA,),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        CONTROLLED_IO_EVALUATOR_KEY: (
            CapabilityKind.ANALYSIS,
            ControlledIOQualificationConfig.SCHEMA,
            (ControlledIOMember.SCHEMA,),
            (MarkovKernelFamily.SCHEMA,),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        RECEIVER_METRIC_ANALYSIS_KEY: (
            CapabilityKind.ANALYSIS,
            StateMetricConfig.SCHEMA,
            tuple(sorted((CanonicalMatrix.SCHEMA, CoordinateBasis.SCHEMA))),
            tuple(sorted((ReceiverRieszFamily.SCHEMA, StateMetric.SCHEMA))),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        RECEIVER_JACOBI_ANALYSIS_KEY: (
            CapabilityKind.ANALYSIS,
            JacobiConfig.SCHEMA,
            tuple(
                sorted(
                    (
                        ControlledIOMember.SCHEMA,
                        StateMetric.SCHEMA,
                        ReceiverRieszFamily.SCHEMA,
                    )
                )
            ),
            (JacobiWitness.SCHEMA,),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        NONNORMAL_AUDIT_KEY: (
            CapabilityKind.ANALYSIS,
            NonnormalAuditConfig.SCHEMA,
            tuple(
                sorted(
                    (
                        CanonicalMatrix.SCHEMA,
                        ControlledIOMember.SCHEMA,
                        JacobiWitness.SCHEMA,
                        MarkovKernelFamily.SCHEMA,
                        PrefixPathwiseMargin.SCHEMA,
                        StateMetric.SCHEMA,
                    )
                )
            ),
            (NonnormalAudit.SCHEMA,),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        MECHANISTIC_EVIDENCE_KEY: (
            CapabilityKind.ANALYSIS,
            MechanisticEvidencePlan.SCHEMA,
            tuple(sorted((MechanisticEvidencePlan.SCHEMA, MechanisticProductResult.SCHEMA))),
            (MechanisticEvidenceResult.SCHEMA,),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        D_OPTIMAL_DESIGN_KEY: (
            CapabilityKind.EXPERIMENT_DESIGNER,
            DOptimalConfig.SCHEMA,
            tuple(sorted((CanonicalMatrix.SCHEMA, DOptimalCandidate.SCHEMA))),
            (DOptimalScoreSet.SCHEMA,),
            EvidenceCeiling.RESPONSE,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        FARTHEST_POINT_DESIGN_KEY: (
            CapabilityKind.EXPERIMENT_DESIGNER,
            FarthestPointConfig.SCHEMA,
            (NativeActionPoint.SCHEMA,),
            (FarthestPointOrder.SCHEMA,),
            EvidenceCeiling.RESPONSE,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        FINITE_MPC_SYNTHESIS_KEY: (
            CapabilityKind.CONTROLLER_SYNTHESIZER,
            FiniteMPCConfig.SCHEMA,
            tuple(sorted((OccurrenceActionWord.SCHEMA, FiniteMPCMemberPrediction.SCHEMA))),
            (FiniteMPCProposal.SCHEMA,),
            EvidenceCeiling.ADMISSION,
            OutcomeAccess.OUTCOME_BLIND,
        ),
        MAXIMIN_SYNTHESIS_KEY: (
            CapabilityKind.CONTROLLER_SYNTHESIZER,
            MaximinConfig.SCHEMA,
            (NamedDecimal.SCHEMA,),
            (MaximinCandidateSet.SCHEMA,),
            EvidenceCeiling.ADMISSION,
            OutcomeAccess.OUTCOME_BLIND,
        ),
    }
    manifests = []
    for key in _KEYS:
        kind, config_schema, inputs, outputs, ceiling, access = specifications[key]
        manifests.append(
            CapabilityManifest(
                capability_key=key,
                capability_version="1.0.0",
                kind=kind,
                config_schema=config_schema,
                config_schema_sha256=_schema_digest(config_schema),
                input_schema_ids=inputs,
                output_schema_ids=outputs,
                permissions=(CapabilityPermission.READ_DEVELOPMENT,),
                maximum_evidence_ceiling=ceiling,
                maximum_outcome_access=access,
                resource_ceiling=ResourceBudget(
                    cpu_cores=4,
                    memory_bytes=4 * 1024**3,
                    gpu_devices=0,
                    wall_time_seconds=3600,
                    source_scan_bytes=1024**3,
                    output_bytes=256 * 1024**2,
                ),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id="cpython-3.11-receiver-conditioned-io",
                requires_clean_commit=True,
                requires_active_mount=False,
                requires_network=False,
                conformance_check_ids=(
                    "exact-member-version-refinement-axes",
                    "full-map-adverse-veto",
                    "no-final-law-gate-or-commitment-authority",
                    "receiver-geometry-not-controlled-admittance",
                    "truth-known-refusal-semantics",
                ),
                implementation_sha256=implementation_sha256_by_key[key],
            )
        )
    return tuple(sorted(manifests, key=lambda value: value.registry_id))


def receiver_conditioned_method_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="response-law-receiver-conditioned-methods",
        capabilities=receiver_conditioned_method_manifests(implementation_sha256_by_key),
    )
