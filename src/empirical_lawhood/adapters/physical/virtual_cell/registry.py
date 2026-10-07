"""Closed capability registry for the 2025 Virtual Cell benchmark route."""

from __future__ import annotations

from collections.abc import Mapping

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .config import (
    CAPABILITY_VERSION,
    CONFIG_SCHEMA_SHA256,
    EVALUATOR_CAPABILITY_KEY,
    FALSIFIER_CAPABILITY_KEY,
    MODEL_CAPABILITY_KEY,
    PLACEMENT_CAPABILITY_KEY,
    PREDICTION_CAPABILITY_KEY,
    REPORTER_CAPABILITY_KEY,
    SELECTION_CAPABILITY_KEY,
    SOURCE_CAPABILITY_KEY,
    SUMMARY_CAPABILITY_KEY,
)
from .contracts import LeaderboardSnapshot, PlacementAdjudication, ProvenanceBoundVirtualCellPipelineConfig, VirtualCellSourceManifest, VirtualCellSourceObject
from .records import (
    CONTROL_RESERVOIR_TABLE_SCHEMA,
    FalsifierPanel,
    MODEL_SAFETENSORS_SCHEMA,
    ModelDevelopmentRecord,
    ModelSelectionRecord,
    OFFICIAL_METRICS_TABLE_SCHEMA,
    OfficialEvaluationRecord,
    PREDICTED_MEAN_TABLE_SCHEMA,
    PREDICTION_H5AD_ENVELOPE_SCHEMA,
    PreparedSourceRecord,
    PredictionFreezeRecord,
    RESPONSE_SUMMARY_TABLE_SCHEMA,
    ResponseSummaryRecord,
    TARGET_FEATURE_TABLE_SCHEMA,
    TEST_ROSTER_TEXT_SCHEMA,
    VirtualCellDevelopmentCloseout,
    VirtualCellRunCloseout,
)


def virtual_cell_capability_keys() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                SOURCE_CAPABILITY_KEY,
                SUMMARY_CAPABILITY_KEY,
                MODEL_CAPABILITY_KEY,
                PLACEMENT_CAPABILITY_KEY,
                FALSIFIER_CAPABILITY_KEY,
                SELECTION_CAPABILITY_KEY,
                PREDICTION_CAPABILITY_KEY,
                EVALUATOR_CAPABILITY_KEY,
                REPORTER_CAPABILITY_KEY,
            )
        )
    )


def _budget(
    *,
    cpu: int,
    memory: int,
    wall: int,
    scan: int,
    output: int,
) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=cpu,
        memory_bytes=memory,
        gpu_devices=0,
        wall_time_seconds=wall,
        source_scan_bytes=scan,
        output_bytes=output,
    )


def _manifest(
    *,
    key: str,
    kind: CapabilityKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    permissions: tuple[CapabilityPermission, ...],
    access: OutcomeAccess,
    budget: ResourceBudget,
    checks: tuple[str, ...],
    implementation_sha256: str,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version=CAPABILITY_VERSION,
        kind=kind,
        config_schema=ProvenanceBoundVirtualCellPipelineConfig.SCHEMA,
        config_schema_sha256=CONFIG_SCHEMA_SHA256,
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_ids=tuple(sorted(outputs)),
        permissions=tuple(sorted(permissions)),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=access,
        resource_ceiling=budget,
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


def virtual_cell_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    if set(implementation_sha256_by_key) != set(virtual_cell_capability_keys()):
        raise ValueError(
            "Virtual Cell implementation identities must cover the exact registry"
        )
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    read_write = (
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    development = (*read_write, CapabilityPermission.READ_DEVELOPMENT)
    manifests = (
        _manifest(
            key=SOURCE_CAPABILITY_KEY,
            kind=CapabilityKind.SOURCE,
            inputs=(VirtualCellSourceManifest.SCHEMA,),
            outputs=(PreparedSourceRecord.SCHEMA,),
            permissions=read_write,
            access=OutcomeAccess.OUTCOME_BLIND,
            budget=_budget(
                cpu=2,
                memory=2 * 1024**3,
                wall=3_600,
                scan=36 * 1024**3,
                output=16 * 1024**2,
            ),
            checks=(
                "vcc-exact-seven-object-manifest",
                "vcc-segmented-source-byte-closure",
                "vcc-test-content-not-opened",
            ),
            implementation_sha256=implementation_sha256_by_key[SOURCE_CAPABILITY_KEY],
        ),
        _manifest(
            key=SUMMARY_CAPABILITY_KEY,
            kind=CapabilityKind.TRANSFORM,
            inputs=(PreparedSourceRecord.SCHEMA,),
            outputs=(
                CONTROL_RESERVOIR_TABLE_SCHEMA,
                RESPONSE_SUMMARY_TABLE_SCHEMA,
                ResponseSummaryRecord.SCHEMA,
            ),
            permissions=development,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            budget=_budget(
                cpu=8,
                memory=12 * 1024**3,
                wall=8 * 3_600,
                # Train is scanned once for summaries and once for the frozen
                # control reservoir; validation is scanned once.  The exact
                # held byte sum is about 37.9 GB, so 40 GiB closes the real I/O.
                scan=40 * 1024**3,
                output=2 * 1024**3,
            ),
            checks=(
                "vcc-batch-ceiling-not-cell-replication",
                "vcc-fixed-total-normalization",
                "vcc-sparse-dense-equivalence",
            ),
            implementation_sha256=implementation_sha256_by_key[SUMMARY_CAPABILITY_KEY],
        ),
        _manifest(
            key=MODEL_CAPABILITY_KEY,
            kind=CapabilityKind.ANALYSIS,
            inputs=(
                CONTROL_RESERVOIR_TABLE_SCHEMA,
                PreparedSourceRecord.SCHEMA,
                RESPONSE_SUMMARY_TABLE_SCHEMA,
                ResponseSummaryRecord.SCHEMA,
                TARGET_FEATURE_TABLE_SCHEMA,
            ),
            outputs=(
                MODEL_SAFETENSORS_SCHEMA,
                ModelDevelopmentRecord.SCHEMA,
                OFFICIAL_METRICS_TABLE_SCHEMA,
            ),
            permissions=development,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            budget=_budget(
                cpu=8,
                # Exact family selection opens validation once and runs the
                # pinned broad scorer for all five candidates.  cell-eval's
                # pseudobulk path transiently densifies the 98,927 x 18,080
                # validation matrix, so this uses the campaign's measured
                # 30 GiB local ceiling rather than the summary-only estimate.
                memory=30 * 1024**3,
                wall=8 * 3_600,
                # 6.93 GB raw validation plus bounded summaries, features and
                # the control reservoir; the source is opened exactly once.
                scan=10 * 1024**3,
                output=2 * 1024**3,
            ),
            checks=(
                "vcc-development-target-grouping",
                "vcc-exact-official-validation-score",
                "vcc-expected-admission-cross-fit",
                "vcc-final-outcome-lineage-empty",
                "vcc-validation-not-training",
            ),
            implementation_sha256=implementation_sha256_by_key[MODEL_CAPABILITY_KEY],
        ),
        _manifest(
            key=FALSIFIER_CAPABILITY_KEY,
            kind=CapabilityKind.FALSIFIER,
            inputs=(
                CONTROL_RESERVOIR_TABLE_SCHEMA,
                MODEL_SAFETENSORS_SCHEMA,
                ModelDevelopmentRecord.SCHEMA,
                RESPONSE_SUMMARY_TABLE_SCHEMA,
                ResponseSummaryRecord.SCHEMA,
                TARGET_FEATURE_TABLE_SCHEMA,
            ),
            outputs=(FalsifierPanel.SCHEMA,),
            permissions=development,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            budget=_budget(
                cpu=4,
                memory=8 * 1024**3,
                wall=2 * 3_600,
                scan=2 * 1024**3,
                output=64 * 1024**2,
            ),
            checks=(
                "vcc-gene-and-target-permutation",
                "vcc-leakage-and-cutoff-sentinels",
                "vcc-wrong-control-and-label-shuffle",
            ),
            implementation_sha256=implementation_sha256_by_key[
                FALSIFIER_CAPABILITY_KEY
            ],
        ),
        _manifest(
            key=SELECTION_CAPABILITY_KEY,
            kind=CapabilityKind.NUMERICAL_QUALIFIER,
            inputs=(FalsifierPanel.SCHEMA, ModelDevelopmentRecord.SCHEMA),
            outputs=(ModelSelectionRecord.SCHEMA,),
            permissions=development,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            budget=_budget(
                cpu=1,
                memory=1024**3,
                wall=600,
                scan=128 * 1024**2,
                output=16 * 1024**2,
            ),
            checks=(
                "vcc-frozen-validation-only-selection",
                "vcc-one-standard-error-tie-break",
            ),
            implementation_sha256=implementation_sha256_by_key[
                SELECTION_CAPABILITY_KEY
            ],
        ),
        _manifest(
            key=PREDICTION_CAPABILITY_KEY,
            # This step serializes already predicted means into the exact
            # submission envelope at the FREEZE barrier.  It does not observe
            # the source system, so the shared stage ontology classifies it as
            # a deterministic transform rather than an observation operator.
            kind=CapabilityKind.TRANSFORM,
            inputs=(
                CONTROL_RESERVOIR_TABLE_SCHEMA,
                MODEL_SAFETENSORS_SCHEMA,
                ModelSelectionRecord.SCHEMA,
                TARGET_FEATURE_TABLE_SCHEMA,
                TEST_ROSTER_TEXT_SCHEMA,
            ),
            outputs=(
                PREDICTED_MEAN_TABLE_SCHEMA,
                PREDICTION_H5AD_ENVELOPE_SCHEMA,
                PredictionFreezeRecord.SCHEMA,
            ),
            permissions=development,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            budget=_budget(
                cpu=8,
                memory=12 * 1024**3,
                wall=4 * 3_600,
                scan=3 * 1024**3,
                output=2_000_000_000,
            ),
            checks=(
                "vcc-compiler-mean-preservation",
                "vcc-prediction-schema-and-range",
                "vcc-test-outcome-read-false",
                "vcc-vfat-safe-output-bound",
            ),
            implementation_sha256=implementation_sha256_by_key[
                PREDICTION_CAPABILITY_KEY
            ],
        ),
        _manifest(
            key=EVALUATOR_CAPABILITY_KEY,
            kind=CapabilityKind.EVALUATOR,
            inputs=(
                PREDICTION_H5AD_ENVELOPE_SCHEMA,
                PredictionFreezeRecord.SCHEMA,
                VirtualCellSourceObject.SCHEMA,
            ),
            outputs=(
                OFFICIAL_METRICS_TABLE_SCHEMA,
                OfficialEvaluationRecord.SCHEMA,
                ScientificAdjudicationRecord.SCHEMA,
            ),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            access=OutcomeAccess.EVALUATOR_REVEAL,
            budget=_budget(
                cpu=8,
                memory=30 * 1024**3,
                wall=8 * 3_600,
                scan=20 * 1024**3,
                output=2 * 1024**3,
            ),
            checks=(
                "vcc-cell-eval-0.6.6-pdex-0.1.26",
                "vcc-evaluator-only-test-open",
                "vcc-exact-final-normalization",
            ),
            implementation_sha256=implementation_sha256_by_key[
                EVALUATOR_CAPABILITY_KEY
            ],
        ),
        _manifest(
            key=PLACEMENT_CAPABILITY_KEY,
            kind=CapabilityKind.REPORTER,
            inputs=(
                LeaderboardSnapshot.SCHEMA,
                OfficialEvaluationRecord.SCHEMA,
                PredictionFreezeRecord.SCHEMA,
            ),
            outputs=(PlacementAdjudication.SCHEMA,),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            access=OutcomeAccess.EVALUATION_REVEALED,
            budget=_budget(
                cpu=1,
                memory=1024**3,
                wall=600,
                scan=64 * 1024**2,
                output=16 * 1024**2,
            ),
            checks=(
                "vcc-leaderboard-lineage-post-score-only",
                "vcc-partial-field-placement-bounded",
            ),
            implementation_sha256=implementation_sha256_by_key[
                PLACEMENT_CAPABILITY_KEY
            ],
        ),
        _manifest(
            key=REPORTER_CAPABILITY_KEY,
            kind=CapabilityKind.REPORTER,
            inputs=(
                FalsifierPanel.SCHEMA,
                ModelDevelopmentRecord.SCHEMA,
                ModelSelectionRecord.SCHEMA,
                OfficialEvaluationRecord.SCHEMA,
                PlacementAdjudication.SCHEMA,
                PredictionFreezeRecord.SCHEMA,
            ),
            outputs=(
                ScientificAdjudicationRecord.SCHEMA,
                VirtualCellDevelopmentCloseout.SCHEMA,
                VirtualCellRunCloseout.SCHEMA,
            ),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_DEVELOPMENT,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            access=OutcomeAccess.EVALUATION_REVEALED,
            budget=_budget(
                cpu=1,
                memory=1024**3,
                wall=600,
                scan=64 * 1024**2,
                output=16 * 1024**2,
            ),
            checks=(
                "vcc-development-to-reveal-handoff",
                "vcc-placement-lineage-bound",
                'virtual-cell-physical-validation-not-promoted',
            ),
            implementation_sha256=implementation_sha256_by_key[REPORTER_CAPABILITY_KEY],
        ),
    )
    return tuple(sorted(manifests, key=lambda value: value.registry_id))


def virtual_cell_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="virtual-cell-2025-tier-l0-capabilities",
        capabilities=virtual_cell_capability_manifests(implementation_sha256_by_key),
    )


__all__ = [
    "virtual_cell_capability_keys",
    "virtual_cell_capability_manifests",
    "virtual_cell_capability_registry",
]
