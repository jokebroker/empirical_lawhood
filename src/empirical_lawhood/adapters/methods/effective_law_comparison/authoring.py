# SPDX-License-Identifier: MPL-2.0
"""Deterministic authoring of the qualified portfolio freeze and execution DAG."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    canonical_json_bytes,
    validate_relative_locator,
    validate_sha256,
)
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.posthoc import (
    POSTHOC_ANALYSIS_IDS,
    ParentEligibilityRecord,
    ParentEligibilityStatus,
    PosthocAnalysisSpec,
    PosthocEvidencePartition,
    PosthocExecutionAuthorization,
    PosthocInferenceMode,
    PosthocPortfolioPackage,
    PosthocSourceArtifact,
    PosthocTrancheFreeze,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import CandidateExecutionPlan
from empirical_lawhood.runtime.posthoc import (
    ANALYSIS_RESULT_SCHEMA,
    compile_posthoc_portfolio_execution_plan,
)

from .execution import _qualified_source_export
from .analysis_questions import ANALYSIS_CONTRACTS
from .provider import posthoc_capability_registry
from .sources import ParentSourceDefinition


def _canonical_container(value: object) -> object:
    if isinstance(value, list):
        return tuple(_canonical_container(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_container(item) for key, item in value.items()}
    return value


def repository_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        text=True,
    ).strip()


def implementation_source_closure(
    repository_root: Path, implementation_files: tuple[str, ...]
) -> dict[str, object]:
    rows = []
    accumulator = sha256()
    if not implementation_files or len(set(implementation_files)) != len(implementation_files):
        raise ValueError("post-hoc implementation source roster is empty or duplicated")
    repository_root = repository_root.resolve(strict=True)
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    if status:
        raise PermissionError("post-hoc freeze requires a clean source commit")
    for relative in implementation_files:
        validate_relative_locator(relative)
        path = repository_root / relative
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(repository_root):
            raise PermissionError("post-hoc implementation source escapes target")
        if not path.is_file():
            raise ValueError(f"implementation source is absent: {relative}")
        payload = path.read_bytes()
        digest = sha256(payload).hexdigest()
        accumulator.update(relative.encode("utf-8"))
        accumulator.update(b"\0")
        accumulator.update(bytes.fromhex(digest))
        rows.append({"relative_path": relative, "sha256": digest, "size_bytes": len(payload)})
    return {
        "schema": 'empirical-lawhood/methods/posthoc-response-composition/implementation-source-closure',
        "version": "1.0.0",
        "value": {
            "base_commit": repository_head(repository_root),
            "implementation_sha256": accumulator.hexdigest(),
            "source_files": rows,
        },
    }


def _identity(object_id: str, document: Mapping[str, object]) -> ObjectIdentity:
    schema = document.get("schema")
    version = document.get("version")
    if not isinstance(schema, str) or not isinstance(version, str):
        raise ValueError("identity document is not canonical")
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version=version,
        object_fingerprint=sha256(canonical_json_bytes(_canonical_container(document))).hexdigest(),
    )


def build_freeze(
    *,
    repository_root: Path,
    qualification: Mapping[str, object],
    frozen_at_utc: str,
    parent_sources: tuple[ParentSourceDefinition, ...],
    implementation_files: tuple[str, ...],
    plan_sha256: str,
    freeze_id: str,
) -> tuple[PosthocTrancheFreeze, dict[str, object]]:
    validate_sha256(plan_sha256, field_name="plan_sha256")
    closure = implementation_source_closure(repository_root, implementation_files)
    closure_value = closure["value"]
    assert isinstance(closure_value, Mapping)
    qualification_value = qualification.get("value")
    if not isinstance(qualification_value, Mapping):
        raise ValueError("source qualification value is absent")
    rows = qualification_value.get("parents")
    if not isinstance(rows, list):
        raise ValueError("source qualification parent roster is absent")
    definitions = {value.parent_id: value for value in parent_sources}
    if set(definitions) != {str(row["parent_id"]) for row in rows if isinstance(row, Mapping)}:
        raise ValueError("post-hoc qualification differs from its explicit source roster")
    parents: list[ParentEligibilityRecord] = []
    partitions: list[PosthocEvidencePartition] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("source qualification parent row changed shape")
        parent_id = str(row["parent_id"])
        definition = definitions[parent_id]
        status = ParentEligibilityStatus(str(row["status"]))
        members = row.get("members")
        if not isinstance(members, list):
            raise ValueError("source qualification member roster is absent")
        source_artifacts = []
        for member in members:
            if not isinstance(member, Mapping) or member.get("status") == "ABSENT":
                continue
            export = _qualified_source_export(member)
            supplied_member = next((value for value in definition.members if value.artifact_id == member.get("artifact_id")), None)
            if supplied_member is None or supplied_member.source_export != export:
                raise ValueError("post-hoc freeze source/export differs from its explicit current input roster")
            source_artifacts.append(
                PosthocSourceArtifact(
                    artifact_id=str(member["artifact_id"]),
                    role_id=str(member["role_id"]),
                    relative_locator=str(member["relative_locator"]),
                    payload_schema=(
                        str(member["payload_schema"])
                        if member.get("payload_schema") is not None
                        else None
                    ),
                    payload_version=(
                        str(member["payload_version"])
                        if member.get("payload_version") is not None
                        else None
                    ),
                    identity_version=(
                        str(member["identity_version"])
                        if member.get("identity_version") is not None
                        else None
                    ),
                    version_normalization=(
                        str(member["version_normalization"])
                        if member.get("version_normalization") is not None
                        else None
                    ),
                    qualification_status=str(member["status"]),
                    content_sha256=str(member["content_sha256"]),
                    size_bytes=int(member["size_bytes"]),
                    custody_sha256=(
                        str(member["custody_sha256"])
                        if member.get("custody_sha256") is not None
                        else None
                    ),
                )
            )
        if not source_artifacts:
            raise ValueError(f"parent lacks even a controlling source record: {parent_id}")
        source_artifacts.sort(key=lambda value: value.artifact_id)
        partition_id = (
            None if status is ParentEligibilityStatus.INELIGIBLE else f"partition.{parent_id}"
        )
        analysis_ids = (
            () if status is ParentEligibilityStatus.INELIGIBLE else definition.analysis_ids
        )
        parents.append(
            ParentEligibilityRecord(
                parent_id=parent_id,
                lane_id=definition.lane_id,
                partition_id=partition_id,
                controlling_artifact_id=source_artifacts[0].artifact_id,
                source_artifacts=tuple(source_artifacts),
                status=status,
                available_role_ids=definition.role_ids,
                analysis_ids=analysis_ids,
                independent_unit_count=definition.expected_unit_count,
                nested_observations_count_as_units=False,
                reason_codes=tuple(sorted(str(value) for value in row.get("reason_codes", []))),
                claim_promotion_allowed=False,
            )
        )
        if partition_id is not None:
            partitions.append(
                PosthocEvidencePartition(
                    partition_id=partition_id,
                    study_id=definition.study_id,
                    target_id=definition.target_id,
                    world_id=definition.world_id,
                    world_kind=WorldKind(str(row["world_kind"])),
                    independent_unit_id=definition.independent_unit_id,
                    source_artifact_ids=tuple(value.artifact_id for value in source_artifacts),
                    outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                    visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                    numerical_pooling_allowed=False,
                )
            )
    parents.sort(key=lambda value: value.parent_id)
    partitions.sort(key=lambda value: value.partition_id)
    analyses = []
    for analysis_id in POSTHOC_ANALYSIS_IDS:
        contract = ANALYSIS_CONTRACTS[analysis_id]
        question, modes, roles, stop_codes, counterexample_rule = contract
        eligible_parent_ids = tuple(
            parent.parent_id for parent in parents if analysis_id in parent.analysis_ids
        )
        analyses.append(
            PosthocAnalysisSpec(
                analysis_id=analysis_id,
                question=question,
                inference_modes=tuple(sorted(modes)),
                required_role_ids=tuple(sorted(roles)),
                eligible_parent_ids=eligible_parent_ids,
                family_coordinate_ids=(f"family.{analysis_id}.frozen",),
                typed_stop_codes=tuple(sorted(stop_codes)),
                decisive_counterexample_rule=counterexample_rule,
                output_schema=ANALYSIS_RESULT_SCHEMA,
                seed=2_026_081_101 if PosthocInferenceMode.WITHIN_PARTITION_UNIT in modes else None,
            )
        )
    freeze = PosthocTrancheFreeze(
        freeze_id=freeze_id,
        plan_sha256=plan_sha256,
        implementation_commit=str(closure_value["base_commit"]),
        implementation_sha256=str(closure_value["implementation_sha256"]),
        source_qualification=_identity(
            f"{freeze_id}.source-qualification", qualification
        ),
        partitions=tuple(partitions),
        parents=tuple(parents),
        analyses=tuple(sorted(analyses, key=lambda value: value.analysis_id)),
        synthesis_codebook_sha256=sha256((Path(__file__).parent / "analysis.py").read_bytes()).hexdigest(),
        frozen_at_utc=frozen_at_utc,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        frozen=True,
    )
    return freeze, closure


def build_authorized_package(
    freeze: PosthocTrancheFreeze,
    *,
    authorization: PosthocExecutionAuthorization,
    run_root: str,
    package_id: str,
) -> tuple[CandidateExecutionPlan, PosthocPortfolioPackage, CapabilityRegistry]:
    """Compile the exact post-hoc DAG under a separately supplied typed authority."""

    if authorization.freeze != ObjectIdentity.from_record(freeze.freeze_id, freeze):
        raise PermissionError("post-hoc authorization binds another freeze")
    if authorization.external_run_root != run_root:
        raise PermissionError("post-hoc authorization binds another run root")
    registry = posthoc_capability_registry(freeze.implementation_sha256)
    plan = compile_posthoc_portfolio_execution_plan(
        freeze,
        run_root=run_root,
        registry_sha256=registry.fingerprint(),
    )
    for task in plan.tasks:
        registry.require(task.capability)
    package = PosthocPortfolioPackage(
        package_id=package_id,
        freeze=freeze,
        authorization=authorization,
        execution_plan=ObjectIdentity.from_record(plan.execution_plan_id, plan),
    )
    return plan, package, registry


__all__ = [
    "build_authorized_package",
    "build_freeze",
    "implementation_source_closure",
    "repository_head",
]
