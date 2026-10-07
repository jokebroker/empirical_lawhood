"""Target-owned Gym--TORAX source closure for the installed adapter.

The manifest authenticates the listed target files. It makes no donor-commit,
archive or historical-result claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters._bounded_files import read_bounded_contained

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)


GYM_TORAX_SURGICAL_EXTRACTION_MANIFEST_ID = 'source-manifest.tokamak-control.current-native-source-closure'
_MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
_ENTRY_PATHS = (
    'src/empirical_lawhood/adapters/composition/tokamak_control_replication/__init__.py',
    'src/empirical_lawhood/adapters/composition/tokamak_control_replication/action_words.py',
    'src/empirical_lawhood/adapters/composition/tokamak_control_replication/campaign_design.py',
    'src/empirical_lawhood/adapters/composition/tokamak_control_replication/rosters.py',
    'src/empirical_lawhood/adapters/composition/tokamak_control_replication/scientific_inputs.py',
    'src/empirical_lawhood/adapters/methods/confirmatory_finite_action.py',
    'src/empirical_lawhood/adapters/methods/confirmatory_finite_action_region_support.py',
    'src/empirical_lawhood/adapters/methods/confirmatory_finite_action_registration.py',
    'src/empirical_lawhood/adapters/methods/evidence_projection_templates.py',
    'src/empirical_lawhood/adapters/methods/qualification_profiles.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/__init__.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/action_word.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/diagnostic_contracts.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/executable_binding.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/extension_bundle.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/extraction_manifest.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/field_metadata.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/field_metadata_contracts.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/finite_action_recurrence_evaluation_recovery.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/finite_action_recurrence_execution.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/finite_action_recurrence_protocol.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/finite_action_recurrence_recovery.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/finite_action_recurrence_science.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/metadata_barrier.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/native_quickstart.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/observation.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/parameterised_binding.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/projection.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/protocol_scientific_inputs.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/provider.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/retained_inputs.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/runtime.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/selected_parent_execution.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/selected_parent_protocol.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/selected_parent_science.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/source.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/source_assessment_adjudication.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/source_assessment_execution.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/source_assessment_protocol.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/source_assessment_seeds.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/source_qualification.py',
    'src/empirical_lawhood/adapters/simulators/gym_torax_native/system.py',
    'src/empirical_lawhood/planning/controller_study.py',
    'src/empirical_lawhood/planning/finite_action_alternate_branch_selection.py',
    'src/empirical_lawhood/planning/finite_action_occurrence.py',
    'src/empirical_lawhood/planning/finite_action_recurrence.py',
    'src/empirical_lawhood/planning/native_hold_decision_cell_calibration.py',
    'src/empirical_lawhood/planning/preissue_branch_selection.py',
    'src/empirical_lawhood/planning/source_pipelines.py',
    'src/empirical_lawhood/runtime/finite_action_recurrence.py',
    'src/empirical_lawhood/runtime/native_hold_decision_cell_calibration.py',
)


@dataclass(frozen=True, slots=True)
class GymToraxBoundedExtractionEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-bounded-extraction-entry'

    entry_id: str
    target_relative_path: str
    target_file_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.entry_id, field_name="entry_id")
        validate_relative_locator(self.target_relative_path)
        validate_sha256(self.target_file_sha256, field_name="target_file_sha256")
        if self.target_relative_path not in _ENTRY_PATHS:
            raise ValueError("source entry is outside the target package roster")


@dataclass(frozen=True, slots=True)
class GymToraxBoundedExtractionManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-bounded-extraction-manifest'

    manifest_id: str
    entries: tuple[GymToraxBoundedExtractionEntry, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling
    grants_source_access: bool
    grants_execution_authority: bool
    grants_scientific_promotion: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        if self.manifest_id != GYM_TORAX_SURGICAL_EXTRACTION_MANIFEST_ID:
            raise ValueError("unexpected target source manifest identity")
        require_sorted_unique_ids(self.entries, attribute="entry_id", field_name="entries")
        if tuple(sorted(row.target_relative_path for row in self.entries)) != tuple(sorted(_ENTRY_PATHS)):
            raise ValueError("target source file roster is not closed")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("target source manifest must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("target source manifest must remain prospective")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("source bytes are not scientific evidence")
        if self.grants_source_access or self.grants_execution_authority or self.grants_scientific_promotion:
            raise ValueError("target source manifest cannot grant authority")


def _read_exact_regular_file(repository_root: Path, relative_path: str) -> bytes:
    root = repository_root.resolve(strict=True)
    validate_relative_locator(relative_path)
    payload = read_bounded_contained(
        root, root / relative_path, maximum_bytes=_MAX_SOURCE_FILE_BYTES
    )
    if not payload:
        raise ValueError("target source entry exceeds its exact byte bound")
    return payload


def gym_torax_current_implementation_sha256() -> str:
    """Bind listed package bytes in both source and installed distributions."""
    package_root = Path(__file__).resolve(strict=True).parents[3]
    digest = sha256()
    for source_path in sorted(_ENTRY_PATHS):
        relative_path = source_path.removeprefix("src/empirical_lawhood/")
        if relative_path == source_path:
            raise ValueError("implementation source path is outside the installed package")
        name = relative_path.encode("utf-8")
        payload = _read_exact_regular_file(package_root, relative_path)
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_gym_torax_extraction_manifest(
    repository_root: Path,
) -> GymToraxBoundedExtractionManifest:
    """Fingerprint the explicit source roster in the inspected target checkout."""
    entries = tuple(
        sorted(
            (
                GymToraxBoundedExtractionEntry(
                    entry_id='source.tokamak-control.target.' + path.replace("/", "-").replace("_", "-").replace(".", "-"),
                    target_relative_path=path,
                    target_file_sha256=sha256(_read_exact_regular_file(repository_root, path)).hexdigest(),
                )
                for path in _ENTRY_PATHS
            ),
            key=lambda entry: entry.entry_id,
        )
    )
    return GymToraxBoundedExtractionManifest(
        manifest_id=GYM_TORAX_SURGICAL_EXTRACTION_MANIFEST_ID,
        entries=entries,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        grants_source_access=False,
        grants_execution_authority=False,
        grants_scientific_promotion=False,
    )


def verify_gym_torax_extraction_manifest(
    manifest: GymToraxBoundedExtractionManifest,
    *,
    expected_identity: ObjectIdentity,
    repository_root: Path,
) -> None:
    if expected_identity != ObjectIdentity.from_record(manifest.manifest_id, manifest):
        raise ValueError("target source manifest identity diverged")
    observed = build_gym_torax_extraction_manifest(repository_root)
    if observed != manifest:
        raise ValueError("target source bytes diverged")


__all__ = [
    'GYM_TORAX_SURGICAL_EXTRACTION_MANIFEST_ID',
    'GymToraxBoundedExtractionEntry',
    'GymToraxBoundedExtractionManifest',
    'build_gym_torax_extraction_manifest',
    'verify_gym_torax_extraction_manifest',
]
