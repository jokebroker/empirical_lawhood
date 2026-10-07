"""Bounded development-native publication through the existing artifact plane.

These operations create actual receipts for completed numerical effects.  They
convey no prospective issue, execution or reveal authority.  An unreceipted
partial publication refuses recovery rather than repeating an uncertain effect.
"""

from hashlib import sha256
from importlib.resources import files
from dataclasses import dataclass
from typing import ClassVar
import json

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import MAX_ARTIFACT_MANIFEST_BYTES, read_bounded_bytes
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore, decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactManifest, ArtifactProfile, ArtifactWriteRequest, CanonicalTaskReceipt, ReceiptCheck


def read_matrix_native_payload(writer: ExternalArtifactPlane, manifest: ArtifactManifest, *, maximum_bytes: int) -> bytes:
    if (manifest.logical.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
        or manifest.logical.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        or manifest.materialization.size_bytes > maximum_bytes):
        raise ValueError("Matrix native input has another access role or exceeds its bound")
    writer.verify_manifest(manifest)
    raw = read_bounded_bytes(writer.root.resolve(manifest.materialization.relative_path, for_write=False), maximum_bytes=maximum_bytes)
    if (len(raw) != manifest.materialization.size_bytes
        or sha256(raw).hexdigest() != manifest.materialization.physical_sha256
        or sha256(raw).hexdigest() != manifest.logical.content_sha256):
        raise ValueError("Matrix native bytes changed after exact publication authentication")
    return raw


def preflight_matrix_native_store(writer: ExternalArtifactPlane, *, minimum_free_bytes: int) -> None:
    if not isinstance(writer, ExternalArtifactPlane):
        raise TypeError("Matrix native operations require an actual guarded ExternalArtifactPlane")
    writer.root.verify(for_write=True, operation_minimum_free_bytes=minimum_free_bytes)


@dataclass(frozen=True, slots=True)
class MatrixNativeEffectInput(CanonicalRecord):
    """Persisted contact boundary, not completion or retry permission."""
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-native/effect-input"
    run_id: str
    task_id: str
    configuration: ObjectIdentity


@dataclass(frozen=True, slots=True)
class MatrixNativeRuntimeObservation(CanonicalRecord):
    """Actual native-entry observation, retained separately from source/lock identities."""
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-native/runtime-observation"
    observation_json: str

    def __post_init__(self):
        value = json.loads(self.observation_json)
        if canonical_json_bytes(_immutable_json(value)).decode() != self.observation_json:
            raise ValueError("Matrix runtime observation requires exact canonical JSON")
        if tuple(value.get(key) for key in ("python", "numpy", "scipy", "system", "machine")) != (
            "3.11.14", "2.4.6", "1.17.1", "linux", "x86_64"):
            raise ValueError("Matrix runtime observation differs from its selected numerical profile")
        if not isinstance(value.get("threadpools"), list) or any(pool.get("num_threads") != 1 for pool in value["threadpools"]):
            raise ValueError("Matrix runtime observation lacks its actual single-thread bootstrap")


def _immutable_json(value):
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_json(item) for item in value)
    if isinstance(value, dict):
        return {key: _immutable_json(item) for key, item in value.items()}
    return value


def retain_matrix_native_runtime(custody, observation):
    record = MatrixNativeRuntimeObservation(canonical_json_bytes(_immutable_json(observation)).decode())
    existing = custody.load("runtime-observation", MatrixNativeRuntimeObservation)
    if existing is None:
        custody.retain("runtime-observation", record)
    elif existing != record:
        raise ValueError("Matrix native recovery runtime differs from its actual retained observation")



def matrix_scientific_code_sha256() -> str:
    """Exact scientific owner subset, distinct from a full checkout closure."""
    owners = (
        "adapters/methods/matrix_response_study/anisotropic_feasibility.py",
        "adapters/methods/matrix_response_study/geometry_inputs.py",
        "adapters/methods/matrix_response_study/numerical_qualification.py",
        "adapters/methods/matrix_response_study/scientific_seed_inputs.py",
        "adapters/methods/matrix_response_study/contracts.py",
        "adapters/methods/matrix_response_study/shooting_committor.py",
        "adapters/composition/matrix_response_study/design.py",
        "adapters/composition/matrix_response_study/selected_event_inputs.py",
        "adapters/simulators/six_matrix_response/contracts.py",
        "adapters/simulators/six_matrix_response/history_preparation.py",
        "adapters/simulators/six_matrix_response/scientific_inputs.py",
        "adapters/simulators/six_matrix_response/provider.py",
        "adapters/simulators/six_matrix_response/executable_binding.py",
        "adapters/simulators/six_matrix_response/extension_bundle.py",
        "adapters/simulators/six_matrix_response/shooting.py",
        "adapters/simulators/six_matrix_response/simulation.py",
        "adapters/simulators/six_matrix_response/gradients.py",
        "adapters/simulators/six_matrix_response/spectral.py",
        "adapters/simulators/six_matrix_response/model.py",
        "api/matrix_native_custody.py", "api/matrix_geometry.py", "api/selected_events.py",
    )
    package = files("empirical_lawhood")
    rows = []
    for owner in owners:
        digest = sha256()
        total_bytes = 0
        with package.joinpath(owner).open("rb") as stream:
            while chunk := stream.read(65536):
                total_bytes += len(chunk)
                if total_bytes > 16 * 1024**2:
                    raise ValueError("Matrix scientific source owner exceeds its exact byte bound")
                digest.update(chunk)
        rows.append((owner, digest.hexdigest()))
    return sha256(canonical_json_bytes(tuple(rows))).hexdigest()


class MatrixNativeCustody:
    """Existing artifact/receipt contracts with exact config-bound locators."""

    def __init__(self, *, writer: ExternalArtifactPlane, config: CanonicalRecord,
                 run_id: str, implementation_commit: str, minimum_free_bytes: int,
                 maximum_output_bytes: int, read_only: bool = False):
        if not isinstance(writer, ExternalArtifactPlane):
            raise TypeError("Matrix native operations require an actual guarded ExternalArtifactPlane")
        if len(implementation_commit) != 40 or any(c not in "0123456789abcdef" for c in implementation_commit):
            raise ValueError("Matrix native operation requires an exact lowercase producing Git commit")
        self.writer, self.config, self.run_id = writer, config, run_id
        self.implementation_commit = implementation_commit
        self.minimum_free_bytes, self.maximum_output_bytes = minimum_free_bytes, maximum_output_bytes
        self.store = ExternalTaskReceiptStore(writer, minimum_free_bytes=minimum_free_bytes)
        self.parent = ArtifactLineageParent(ObjectIdentity.from_record(config.config_id, config),
            VisibilityCeiling.DEVELOPMENT_ONLY, OutcomeAccess.DEVELOPMENT_VISIBLE)
        relative = f"runs/{run_id}/inputs/config.json"
        raw = config.canonical_bytes()
        existing = self._manifest(relative)
        if existing is None:
            if read_only:
                raise ValueError("Matrix native result lacks its retained exact input")
            published = writer.write(ArtifactWriteRequest(
                f"{run_id}.config", relative, config.SCHEMA, ArtifactProfile.CANONICAL_JSON,
                "application/json", f"{run_id}.inputs", f"runs/{run_id}/inputs", raw,
                VisibilityCeiling.DEVELOPMENT_ONLY, (), OutcomeAccess.DEVELOPMENT_VISIBLE,
                minimum_free_bytes=minimum_free_bytes))
            existing = self._manifest(relative)
            if existing is None or existing.materialization != published.materialization:
                raise ValueError("Matrix native input publication is incomplete")
        if (existing.logical.payload_schema != config.SCHEMA or existing.logical.content_sha256 != config.fingerprint()
            or read_matrix_native_payload(writer, existing, maximum_bytes=16 * 1024**2) != raw):
            raise ValueError("Matrix native retained input differs from its selected allocation/config")
        self.input_manifest = existing

    def _manifest(self, relative: str) -> ArtifactManifest | None:
        path = self.writer.root.resolve(relative, for_write=False)
        manifest_path = self.writer.root.resolve(relative + ".manifest.json", for_write=False)
        if not path.exists() and not manifest_path.exists():
            return None
        if not path.is_file() or not manifest_path.is_file():
            raise ValueError("Matrix native artifact publication is partial")
        manifest = decode_artifact_manifest(read_bounded_bytes(manifest_path, maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES))
        if manifest.materialization.relative_path != relative:
            raise ValueError("Matrix native manifest changes its exact selected path")
        return manifest

    def load(self, task_id: str, record_type: type[CanonicalRecord], *, in_progress: bool = False):
        # Native scientific rollout IDs remain unchanged in their records.
        # Compact operational locators keep receipt staging below NAME_MAX.
        task_id = self.task_locator(task_id)
        attempt_id = f"{self.run_id}.{task_id}.attempt-001"
        receipt = self.store.read_by_receipt_id(self.run_id, task_id, f"receipt.{attempt_id}")
        relative = f"runs/{self.run_id}/outputs/{task_id}.json"
        manifest = self._manifest(relative)
        if receipt is None:
            if manifest is not None:
                raise ValueError("Matrix native unreceipted effect requires inspection; it cannot be repeated")
            if not in_progress and self._manifest(f"runs/{self.run_id}/inputs/effects/{task_id}.json") is not None:
                raise ValueError("Matrix native interrupted contact lacks a completion receipt; uncertain effects cannot be repeated")
            return None
        if manifest is None or receipt.implementation_commit != self.implementation_commit:
            raise ValueError("Matrix native recovery changes source or omits its exact output")
        logical = manifest.logical
        if (receipt.operational_status is not OperationalStatus.SUCCEEDED
            or receipt.input_materialization_ids != (self.input_manifest.materialization.materialization_id,)
            or receipt.output_materializations != (manifest.materialization,)
            or receipt.output_logical_artifacts != (logical,)
            or logical.logical_artifact_id != f"{self.run_id}.{task_id}.output"
            or logical.payload_schema != record_type.SCHEMA or logical.profile is not ArtifactProfile.CANONICAL_JSON
            or logical.media_type != "application/json"
            or logical.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
            or logical.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or logical.lineage_parents != (self.parent,)
            or manifest.materialization.size_bytes > self.maximum_output_bytes):
            raise ValueError("Matrix native receipt/output changes its frozen census, role or configuration")
        return decode_canonical_bytes(read_matrix_native_payload(self.writer, manifest, maximum_bytes=self.maximum_output_bytes), record_type,
            maximum_bytes=self.maximum_output_bytes)

    def retain(self, task_id: str, record: CanonicalRecord) -> CanonicalRecord:
        task_id = self.task_locator(task_id)
        if self.load(task_id, type(record), in_progress=True) is not None:
            raise ValueError("Matrix native completed effect is immutable")
        raw = record.canonical_bytes()
        if len(raw) > self.maximum_output_bytes:
            raise ValueError("Matrix native output exceeds its declared work contract")
        written = self.writer.write(ArtifactWriteRequest(
            f"{self.run_id}.{task_id}.output", f"runs/{self.run_id}/outputs/{task_id}.json",
            record.SCHEMA, ArtifactProfile.CANONICAL_JSON, "application/json",
            f"{self.run_id}.outputs", f"runs/{self.run_id}/outputs", raw,
            VisibilityCeiling.DEVELOPMENT_ONLY, (VisibilityCeiling.DEVELOPMENT_ONLY,),
            OutcomeAccess.DEVELOPMENT_VISIBLE, lineage_parents=(self.parent,),
            minimum_free_bytes=self.minimum_free_bytes))
        attempt = f"{self.run_id}.{task_id}.attempt-001"
        receipt = CanonicalTaskReceipt(f"receipt.{attempt}", self.run_id, task_id, attempt,
            self.implementation_commit, (self.input_manifest.materialization.materialization_id,),
            (written.materialization,), (written.logical,),
            (ReceiptCheck("complete-typed-native-product", True, ()),), OperationalStatus.SUCCEEDED, ())
        self.store.commit(receipt, visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE)
        return record

    def begin_native_effect(self, task_id: str) -> None:
        task_id = self.task_locator(task_id)
        marker = MatrixNativeEffectInput(self.run_id, task_id, self.parent.identity)
        relative = f"runs/{self.run_id}/inputs/effects/{task_id}.json"
        if self._manifest(relative) is not None:
            raise ValueError("Matrix native contact is already recorded; absence of a receipt grants no retry")
        self.writer.write(ArtifactWriteRequest(f"{self.run_id}.{task_id}.effect-input", relative,
            marker.SCHEMA, ArtifactProfile.CANONICAL_JSON, "application/json",
            f"{self.run_id}.effect-inputs", f"runs/{self.run_id}/inputs/effects", marker.canonical_bytes(),
            VisibilityCeiling.DEVELOPMENT_ONLY, (VisibilityCeiling.DEVELOPMENT_ONLY,), OutcomeAccess.DEVELOPMENT_VISIBLE,
            lineage_parents=(self.parent,), minimum_free_bytes=self.minimum_free_bytes))

    @staticmethod
    def task_locator(scientific_id: str) -> str:
        return scientific_id if len(scientific_id) < 80 else f"cell-{sha256(scientific_id.encode()).hexdigest()[:32]}"
