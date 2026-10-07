"""Strict config codecs and frozen default policy semantics for SDCB-SC."""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import (
    DiscoveryPolicyConfig,
    PolicyKind,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

from .contracts import MaterialFamilyConfig, MaterialSourceManifest, MaterialFamilyDiscoveryAdjudicationConfig, SourceQualification
from .records import MaterialFamilyDiscoveryWorldBuildConfig


_MAXIMUM_CONFIG_BYTES = 256 * 1024


def default_policy_config(kind: PolicyKind) -> DiscoveryPolicyConfig:
    """Return the pre-development CPU-only composition-view policy contract."""

    return DiscoveryPolicyConfig(
        config_id=f"policy.material-family-discovery.{kind.value.lower().replace('_', '-')}",
        policy_kind=kind,
        batch_size=10,
        total_query_budget=100,
        seed=20_260_814,
        query_cost=Decimal(1),
        ensemble_members=64,
        ensemble_min_leaf=2,
        ucb_beta=Decimal("1.5"),
        local_neighbor_count=32,
        local_min_effective_rank=4,
        local_anchor_count=32,
        local_anchor_quantile=Decimal("0.8"),
        local_ridge=Decimal("0.01"),
        local_support_multiplier=Decimal("1.25"),
        local_boundary_weight=Decimal(5),
        information_queries_per_batch=2,
        bo_training_limit=512,
        bo_projection_dimensions=16,
        bo_fit_max_iterations=50,
        feature_schema_id="features.nims-stoichiometry-presence-f32-236",
        implementation_id=(
            'implementation.material-family-local-law'
            if kind is PolicyKind.LOCAL_LAW_BOUNDARY
            else "implementation.botorch-0.18.1-discrete-ucb"
            if kind is PolicyKind.BOTORCH_DISCRETE_UCB
            else "implementation.scikit-learn-extra-trees"
            if kind is PolicyKind.GREEDY_SCALAR_ENSEMBLE
            else 'implementation.material-family-deterministic-selector'
        ),
    )


def decode_policy_config(payload: bytes) -> DiscoveryPolicyConfig:
    return decode_canonical_bytes(
        payload,
        DiscoveryPolicyConfig,
        maximum_bytes=_MAXIMUM_CONFIG_BYTES,
    )


def decode_family_config(payload: bytes) -> MaterialFamilyConfig:
    return decode_canonical_bytes(
        payload,
        MaterialFamilyConfig,
        maximum_bytes=_MAXIMUM_CONFIG_BYTES,
    )


def decode_source_manifest(payload: bytes) -> MaterialSourceManifest:
    return decode_canonical_bytes(
        payload,
        MaterialSourceManifest,
        maximum_bytes=_MAXIMUM_CONFIG_BYTES,
    )


def decode_source_qualification(payload: bytes) -> SourceQualification:
    return decode_canonical_bytes(
        payload,
        SourceQualification,
        maximum_bytes=_MAXIMUM_CONFIG_BYTES,
    )


def decode_adjudication_config(payload: bytes) -> MaterialFamilyDiscoveryAdjudicationConfig:
    return decode_canonical_bytes(
        payload,
        MaterialFamilyDiscoveryAdjudicationConfig,
        maximum_bytes=_MAXIMUM_CONFIG_BYTES,
    )


def decode_world_config(payload: bytes) -> MaterialFamilyDiscoveryWorldBuildConfig:
    return decode_canonical_bytes(
        payload,
        MaterialFamilyDiscoveryWorldBuildConfig,
        maximum_bytes=_MAXIMUM_CONFIG_BYTES,
    )


__all__ = [
    "decode_family_config",
    "decode_adjudication_config",
    "decode_policy_config",
    "decode_source_manifest",
    "decode_source_qualification",
    "decode_world_config",
    "default_policy_config",
]
