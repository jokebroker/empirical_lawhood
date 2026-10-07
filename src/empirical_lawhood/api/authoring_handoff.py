"""Shared application mechanics for typed native authoring handoffs.

SPDX-License-Identifier: MPL-2.0
"""

import os
from pathlib import Path
import pwd

from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import GENERATED_EXTENSION_BUNDLE_AGGREGATE
from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.infrastructure.bounded_io import MAX_RUNTIME_PLAN_JSON_BYTES, read_bounded_bytes
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ExternalRootContract
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog, CandidateContextProvider
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile, resolve_external_root_contract
from empirical_lawhood.runtime.plans import CandidateExecutionPlan


def write_exclusive_record(directory: Path, name: str, record: CanonicalRecord) -> Path:
    path = directory / name
    with path.open("xb") as handle:
        handle.write(record.canonical_bytes())
        handle.flush()
    return path


def create_authoring_api(
    *, repo_root: Path, candidate_context_provider: CandidateContextProvider,
    candidate_capability_catalog: CandidateCapabilityCatalog,
) -> EmpiricalLawhoodApi:
    return EmpiricalLawhoodApi(
        repo_root=repo_root,
        external_root=None,
        candidate_context_provider=candidate_context_provider,
        candidate_capability_catalog=candidate_capability_catalog,
        extension_bundle_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,
        executable_binding_aggregate=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.aggregate,
        executable_factory_registry=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
        study_extension_codec_registry=EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    )


def preflight_output_directory(
    *, directory: Path,
    allow_existing_empty: bool = False,
) -> Path:
    """Check a fresh real-directory destination without creating it or reading inputs."""
    if ".." in directory.parts or any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError("authoring output refuses parent traversal and symbolic links")
    selected = directory.absolute()
    if selected.exists() and (
        not allow_existing_empty or not selected.is_dir() or any(selected.iterdir())
    ):
        raise FileExistsError("authoring output must be a fresh directory")
    if any(parent.exists() and not parent.is_dir() for parent in selected.parents):
        raise NotADirectoryError("authoring output has a non-directory parent")
    return selected


def preflight_authoring_output_directory(
    *, directory: Path, repo_root: Path, artifact_writer,
    allow_existing_empty: bool = False,
) -> Path:
    """Validate a guarded output before upstream reads; create no directory."""
    from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane

    if not isinstance(artifact_writer, ExternalArtifactPlane):
        raise TypeError("current authoring requires its actual guarded external artifact plane")
    selected = preflight_output_directory(directory=directory,
        allow_existing_empty=allow_existing_empty)
    guarded = Path(artifact_writer.root.contract.canonical_path)
    resolved = selected.resolve(strict=False)
    if not selected.is_relative_to(guarded) or not resolved.is_relative_to(guarded.resolve(strict=True)):
        raise ValueError("authoring output must be under the selected guarded external root")
    if resolved.is_relative_to(repo_root.resolve(strict=True)):
        raise ValueError("authoring output must be outside the selected checkout")
    artifact_writer.root.verify(for_write=True)
    artifact_writer.root.resolve(selected.relative_to(guarded).as_posix(), for_write=True)
    return selected


def resolve_authoring_directory(
    directory: Path, *, repo_root: Path, storage_profile: OperatorStorageProfile,
    escape_message: str,
) -> tuple[Path, ExternalRootContract]:
    directory = directory.resolve(strict=True)
    contract = resolve_external_root_contract(
        storage_profile, repo_root=repo_root,
        home_root=Path(pwd.getpwuid(os.geteuid()).pw_dir),
    )
    if not directory.is_relative_to(Path(contract.canonical_path)):
        raise ValueError(escape_message)
    return directory, contract


def load_authoring_execution_projection(directory: Path) -> CandidateExecutionPlan:
    return decode_canonical_bytes(
        read_bounded_bytes(directory / "preissue-execution-plan.json", maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES),
        CandidateExecutionPlan,
        maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES,
    )
