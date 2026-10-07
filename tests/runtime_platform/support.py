# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import EvidenceSnapshot
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.approval import (
    CompleteApprovalService,
    DurableAuthorizationRecord,
    FrozenApprovalProposal,
)
from empirical_lawhood.planning.authority import ApprovalRequest, AuthorizationRecord
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.planning.design import ExperimentProposal
from empirical_lawhood.planning.exploration import ExplorationPlan
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan, ProtocolTemplate, ProtocolRunPlan, SnapshotVerification

IMPLEMENTATION_COMMIT = "a" * 40


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def budget(
    *,
    wall_time_seconds: int = 1,
    output_bytes: int = 1_000,
    source_scan_bytes: int = 0,
) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=1_000_000,
        gpu_devices=0,
        wall_time_seconds=wall_time_seconds,
        source_scan_bytes=source_scan_bytes,
        output_bytes=output_bytes,
    )


def permissions(*values: CapabilityPermission) -> tuple[CapabilityPermission, ...]:
    return tuple(sorted(values))


def capability_config(key: str, schema: str) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=f"config.{key}",
        config_schema=schema,
        config_schema_sha256=digest(schema),
        content_sha256=digest(f"config payload for {key}"),
        artifact_id=f"config-artifact.{key}",
    )


@dataclass(frozen=True, slots=True)
class ProtocolFixture:
    system: SystemSpec
    proposal: ExperimentProposal
    approval_request: ApprovalRequest
    comparison_authorization: AuthorizationRecord
    frozen_proposal: FrozenApprovalProposal
    authorization: DurableAuthorizationRecord
    approval_service: CompleteApprovalService
    experiment: ExperimentSpec
    campaign: CampaignSpec
    template: ProtocolTemplate
    registry: CapabilityRegistry
    run_plan: ProtocolRunPlan
    execution_plan: ProtocolExecutionPlan


@dataclass(frozen=True, slots=True)
class ExplorationFixture:
    snapshot: EvidenceSnapshot
    verification: SnapshotVerification
    plan: ExplorationPlan
    registry: CapabilityRegistry
