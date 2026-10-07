# SPDX-License-Identifier: MPL-2.0

"""Spawn-safe experiment port assembly over guarded production stores.

Only composition and I/O live here. Scientific meanings stay in registered owners.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from empirical_lawhood.adapters.methods.reactor_causal_response.control_services import EmpiricalControllerBindingConfig
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.canonical_record_archive import (
    ExternalCanonicalRecordArchive,
)
from empirical_lawhood.infrastructure.dependency_manifests import (
    DependencyManifestReader,
)
from empirical_lawhood.infrastructure.prepared_execution_events import (
    ExternalPreparedExecutionEventStore,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.runtime.controller_evaluation_trajectory import TrajectoryCallbackLink
from empirical_lawhood.runtime.execution import (
    TaskContext,
    WorkerInputBinding,
    WorkerInputKind,
)

from .custody_ports import ReactorCustodyPorts


@dataclass(frozen=True)
class EmpiricalPayloadCustodyPorts(ReactorCustodyPorts):
    """Resolve only this worker's authorized subset of the 114 output locators."""

    def read_dependency(
        self, context: TaskContext, binding: WorkerInputBinding
    ) -> tuple[ArtifactManifest, CanonicalTaskReceipt]:
        allowed = {
            v.artifact_id
            for v in context.input_bindings
            if v.kind is WorkerInputKind.DEPENDENCY
        }
        outputs = tuple(o for o in self.outputs if o.logical_artifact_id in allowed)
        if len(outputs) != len(allowed):
            raise ValueError("worker dependency census is outside its compiled outputs")
        # Resolve one already authorized binding per call. A reducer may have
        # hundreds of inputs; the bounded manifest reader never scans them.
        selected = tuple(
            o for o in outputs if o.logical_artifact_id == binding.artifact_id
        )
        return DependencyManifestReader(
            ExternalArtifactPlane(
                self.root, semantic_validations=self.semantic_validations
            ),
            selected,
        ).read_dependency(context, binding)


@dataclass(frozen=True)
class EmpiricalControlCustodyPorts:
    root: GuardedExternalRoot
    run_id: str
    authority: ObjectIdentity
    resources: ObjectIdentity
    minimum_free_bytes: int = 100 * 1024**3

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_stable_id

        validate_stable_id(self.run_id)
        # Callers supply replayed operation authority and the issued execution
        # envelope. Merely constructing this bounded port grants neither.
        if (
            self.authority.object_schema
            != 'empirical-lawhood/planning/study-operation-authority'
            or self.resources.object_schema
            not in {
                'empirical-lawhood/runtime/execution-resource-envelope-spec',
                'empirical-lawhood/runtime/run-execution-resource-envelope',
            }
        ):
            raise ValueError(
                "control custody requires typed production authority and resources"
            )

    def open_control_store(self, root: str) -> ExternalCanonicalRecordArchive:
        if root not in {f"reactor-empirical-confirmation-{i:03d}" for i in range(8)}:
            raise ValueError("control custody root outside the frozen allocation")
        namespace = f"runs/{self.run_id}/control/{root}"
        return ExternalCanonicalRecordArchive(
            ExternalPreparedExecutionEventStore(
                ExternalArtifactPlane(self.root),
                state_root_relative_path=namespace,
                record_schemas=tuple(
                    sorted(
                        (
                            CanonicalRecordArchive.SCHEMA,
                            TrajectoryCallbackLink.SCHEMA,
                            EmpiricalControllerBindingConfig.SCHEMA,
                        )
                    )
                ),
                maximum_record_bytes=64 * 1024**2,
                minimum_free_bytes=self.minimum_free_bytes,
            ),
            root,
            EmpiricalArchiveAllocation(self.root, namespace),
        )


@dataclass
class EmpiricalArchiveAllocation:
    """One root has one publication worker; recovery counts retained allocations.

    Reserve VFAT allocation units, including manifests and bounded directories.
    An interrupted write never frees its reservation within a running task.
    This port neither deletes nor overwrites an artifact.
    """

    root: GuardedExternalRoot
    namespace: str
    maximum_bytes: int = 3 * 1024**3
    _allocations: dict[str, int] = field(default_factory=dict, init=False)
    _loaded: bool = field(default=False, init=False)
    _total: int = field(default=0, init=False)

    def reserve(self, relative_path: str, payload_bytes: int) -> None:
        base = self.root.resolve(
            self.namespace, for_write=True, operation_minimum_free_bytes=100 * 1024**3
        )
        path = self.root.resolve(
            relative_path, for_write=True, operation_minimum_free_bytes=100 * 1024**3
        )
        if (
            not path.is_relative_to(base)
            or payload_bytes <= 0
            or payload_bytes > 64 * 1024**2
        ):
            raise ValueError("archive allocation escaped its bounded root/record")
        unit = os.statvfs(
            base if base.exists() else self.root.contract.canonical_path
        ).f_frsize
        if unit != 32768:
            raise ValueError(
                "archive quota proof requires the declared VFAT allocation unit"
            )
        if not self._loaded:
            files = 0
            directories = 0
            if base.exists():
                for directory, dirs, names in os.walk(base):
                    directories += 1
                    if (
                        directories > 1280
                        or Path(directory).is_symlink()
                        or len(dirs) > 256
                        or any((Path(directory) / name).is_symlink() for name in dirs)
                    ):
                        raise ValueError("archive quota traversal changed")
                    for name in names:
                        files += 1
                        if files > 90000:
                            raise ValueError(
                                "archive publication census exceeds full trajectory bound"
                            )
                        member = self.root.resolve(
                            str(
                                (Path(directory) / name).relative_to(
                                    Path(self.root.contract.canonical_path)
                                )
                            ),
                            for_write=False,
                        )
                        allocated = member.stat().st_blocks * 512
                        if allocated:
                            self._allocations[str(member)] = allocated
            self._total = sum(self._allocations.values()) + 40 * 1024**2
            self._loaded = True
        # Production publishes one payload, one manifest and one commit marker.
        # Its two lock files are empty. The full six-record callback census has
        # 86,400 files and 1,025 directories, not merely two files per record.
        # Each bounded metadata record fits one 32 KiB allocation unit.
        size = ((payload_bytes + unit - 1) // unit) * unit
        additions = {str(path): size, str(path) + ".manifest.json": unit}
        manifest_path = Path(str(path) + ".manifest.json")
        commit_key = str(path) + "::publication-commit"
        if manifest_path.exists():
            from empirical_lawhood.infrastructure.task_receipts import (
                decode_artifact_manifest,
            )

            guarded_manifest = self.root.resolve(
                str(manifest_path.relative_to(Path(self.root.contract.canonical_path))),
                for_write=False,
            )
            if guarded_manifest.stat().st_size > unit:
                raise ValueError("prepared manifest exceeds its allocation proof")
            manifest = decode_artifact_manifest(guarded_manifest.read_bytes())
            if manifest.publication is None:
                raise ValueError(
                    "prepared manifest lacks production publication custody"
                )
            commit_key = str(
                self.root.resolve(
                    manifest.publication.commit_marker_relative_path, for_write=False
                )
            )
            pending_key = str(path) + "::publication-commit"
            if pending_key in self._allocations:
                reserved = self._allocations.pop(pending_key)
                existing = self._allocations.get(commit_key, 0)
                self._allocations[commit_key] = max(existing, reserved)
                self._total -= min(existing, reserved)
        additions[commit_key] = unit
        extra = sum(
            max(0, n - self._allocations.get(p, 0)) for p, n in additions.items()
        )
        if len(set(self._allocations) | set(additions)) > 54000:
            raise RuntimeError("Bounded control publication count exhausted")
        # One largest in-flight atomic payload is reserved independently.
        if self._total + extra + 64 * 1024**2 > self.maximum_bytes:
            raise RuntimeError("Frozen per-root control artifact allocation exhausted")
        self._total += extra
        for p, n in additions.items():
            self._allocations[p] = max(n, self._allocations.get(p, 0))
