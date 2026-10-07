"""Bounded publication metadata lookup for already authorized task inputs."""

from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.runtime.execution import WorkerInputBinding, WorkerInputKind, TaskContext
from empirical_lawhood.runtime.plans import ArtifactOutputSpec
from .artifacts import ExternalArtifactPlane, ArtifactIdentityConflict
from .bounded_io import MAX_ARTIFACT_MANIFEST_BYTES, read_bounded_bytes
from .task_receipts import decode_artifact_manifest, ExternalTaskReceiptStore


class DependencyManifestReader:
    """Resolve only compiled output locators and require the task's exact binding.

    Construction performs no I/O. This port supplies custody metadata; payload
    access remains through the scheduler's separately authorized input stream.
    """

    def __init__(
        self, plane: ExternalArtifactPlane, outputs: tuple[ArtifactOutputSpec, ...]
    ) -> None:
        if (
            not outputs
            or len(outputs) > 100
            or len({o.logical_artifact_id for o in outputs}) != len(outputs)
        ):
            raise ValueError("dependency manifest lookup requires a bounded unique output roster")
        self.plane = plane
        self.outputs = outputs

    def read_dependency(
        self, context: TaskContext, binding: WorkerInputBinding
    ) -> tuple[ArtifactManifest, CanonicalTaskReceipt]:
        if binding not in context.input_bindings:
            raise ValueError("dependency binding is absent from the authorized task")
        dependencies = tuple(
            d
            for d in context.dependency_receipts
            if binding.materialization_id in d.output_materialization_ids
        )
        if len(dependencies) != 1:
            raise ValueError("dependency metadata requires one exact task receipt")
        dependency = dependencies[0]
        relative = f"runs/{context.run_id}/receipts/{dependency.task_id}"
        directory = self.plane.root.resolve(relative, for_write=False)
        candidates = []
        for path in directory.iterdir():
            if path.name.endswith(".json") and not path.name.endswith(".manifest.json"):
                candidates.append(path)
                if len(candidates) > 64:
                    raise ArtifactIdentityConflict(
                        "dependency receipt attempt roster exceeds bound"
                    )
        matches = []
        store = ExternalTaskReceiptStore(self.plane)
        for path in sorted(candidates):
            receipt = store.read(
                context.run_id, dependency.task_id, path.name.removesuffix(".json")
            )
            if receipt is not None and receipt.receipt_id == dependency.receipt_id:
                matches.append(receipt)
        if len(matches) != 1:
            raise ArtifactIdentityConflict(
                "authenticated dependency receipt is absent or ambiguous"
            )
        receipt = matches[0]
        manifest = self.read_manifest(binding)
        if (
            manifest.logical not in receipt.output_logical_artifacts
            or manifest.materialization not in receipt.output_materializations
            # Scientific dependency edges deliberately expose only selected outputs of
            # a producer. Authenticate that exact subset against its complete
            # immutable receipt; never grant access to the remaining payloads.
            or not set(dependency.output_materialization_ids).issubset(
                {m.materialization_id for m in receipt.output_materializations}
            )
        ):
            raise ArtifactIdentityConflict("dependency manifest differs from its committed receipt")
        return manifest, receipt

    def read_manifest(self, binding: WorkerInputBinding) -> ArtifactManifest:
        if binding.kind is not WorkerInputKind.DEPENDENCY:
            raise ValueError("dependency metadata requires a scheduler dependency binding")
        output = next(
            (o for o in self.outputs if o.logical_artifact_id == binding.artifact_id), None
        )
        if output is None:
            raise ValueError("dependency metadata is outside the compiled output roster")
        path = self.plane.root.resolve(f"{output.relative_path}.manifest.json", for_write=False)
        manifest = decode_artifact_manifest(
            read_bounded_bytes(path, maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES)
        )
        logical, materialization = manifest.logical, manifest.materialization
        if (
            manifest.publication is None
            or logical.logical_artifact_id != output.logical_artifact_id
            or materialization.relative_path != output.relative_path
            or materialization.materialization_id != binding.materialization_id
            or materialization.size_bytes != binding.size_bytes
            or logical.payload_schema != binding.payload_schema
            or logical.payload_schema != output.payload_schema
            or logical.media_type != binding.media_type
            or logical.media_type != output.media_type
            or logical.profile != output.profile
            or logical.visibility_ceiling != binding.visibility_ceiling
            or logical.visibility_ceiling != output.visibility_ceiling
            or logical.outcome_access != binding.outcome_access
            or logical.outcome_access != output.outcome_access
        ):
            raise ArtifactIdentityConflict(
                "dependency publication differs from its scoped input binding"
            )
        self.plane.verify_manifest(manifest)
        return manifest
