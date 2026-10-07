"""Spawn-safe current custody and a closed task census for the existing scheduler."""

from contextlib import contextmanager
from dataclasses import dataclass

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.candidate_payloads import ExternalCandidatePayloadPlane
from empirical_lawhood.infrastructure.dependency_manifests import DependencyManifestReader
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent
from empirical_lawhood.runtime.plans import ArtifactOutputSpec
from empirical_lawhood.adapters.composition.preparation_applicability.inputs import preparation_input_template
from empirical_lawhood.adapters.methods.preparation_applicability.config import preparation_run_id
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityTaskCensus:
    """Validate entry; the scheduler enforces the declared execution resource cells."""
    task_ids: tuple[str, ...]

    @contextmanager
    def task(self, task_id):
        if task_id not in self.task_ids:
            raise ValueError("applicability task is outside the frozen execution census")
        yield


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityCustodyPorts:
    root: GuardedExternalRoot
    run_id: str
    parent: ArtifactLineageParent
    outputs: tuple[ArtifactOutputSpec, ...]

    def _models(self):
        return ExternalCandidatePayloadPlane(
            ExternalArtifactPlane(self.root), f"runs/{self.run_id}/candidate-models",
            f"{self.run_id}.candidate-models", VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.EVALUATION_SEALED, (self.parent,), 0,
        )

    def publish_candidate_payload(self, **kwargs):
        return self._models().publish_candidate_payload(**kwargs)

    def read_candidate_payload(self, receipt):
        return self._models().read_candidate_payload(receipt)

    def read_dependency(self, context, binding):
        # Each read is scoped to the one authorized input. Full protocols may
        # contain 386 outputs; no global reader bound is relaxed for them.
        outputs = tuple(output for output in self.outputs if output.logical_artifact_id == binding.artifact_id)
        return DependencyManifestReader(ExternalArtifactPlane(self.root), outputs).read_dependency(context, binding)


def runtime_platform_ports(*, selection, bundle, projection, root):
    stage = selection.stage
    if projection.source_plan.object_id!=preparation_run_id(stage):
        raise ValueError("preparation runtime projection changes its exact allocated input custody")
    constructed = stage.SCHEMA.startswith("empirical-lawhood/constructed-")
    prefix = "bp" if constructed else "pa"
    template = bundle.catalog.templates[0]
    externals = {value.logical_artifact_id: value for value in template.graph.external_inputs}

    def payload(record, logical_id, *, lineage=()):
        external = externals[logical_id]
        from empirical_lawhood.runtime.artifacts import ArtifactProfile
        return preparation_input_template(
            logical_artifact_id=logical_id, payload_schema=record.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON, media_type=external.media_type,
            payload=record.canonical_bytes(), visibility_ceiling=external.visibility_ceiling,
            outcome_access=external.outcome_access,
            parent_visibility_ceilings=tuple(value.visibility_ceiling for value in lineage),
            lineage_parents=lineage,
        )

    source_id = next(value.logical_artifact_id for value in template.graph.external_inputs
                     if value.payload_schema == selection.source.SCHEMA)
    method_inputs = []
    native_inputs = [payload(selection.source, source_id)]
    for value in selection.upstream_records:
        if value.key in ("exposure", "conformance"):
            continue
        logical = value.manifest.logical
        source = payload(value.record, logical.logical_artifact_id, lineage=logical.lineage_parents)
        # The exact shared Q input has one publisher: the method binding,
        # which precedes native in the composite provider's closed roster.
        # Native task contexts still receive this same global committed input
        # through their explicit graph edges and check it before prefix contact.
        method_inputs.append(source)
    from empirical_lawhood.kernel.provenance import ObjectIdentity
    parent = ArtifactLineageParent(ObjectIdentity.from_record(stage.config_id, stage),
                                   VisibilityCeiling.PROSPECTIVE, OutcomeAccess.EVALUATION_SEALED)
    custody = PreparationApplicabilityCustodyPorts(root, projection.source_plan.object_id, parent,
        tuple(output for task in projection.tasks for output in task.outputs))
    limits = PreparationApplicabilityTaskCensus(tuple(task.task_id for task in projection.tasks))
    values = {
        f"{prefix}-method-inputs": tuple(method_inputs), f"{prefix}-method-custody": custody,
        f"{prefix}-method-limits": limits, f"{prefix}-native-inputs": tuple(native_inputs),
        f"{prefix}-native-custody": custody, f"{prefix}-native-limits": limits,
        "candidate-payload-publisher": custody, "candidate-payload-reader": custody,
    }
    return tuple(ExecutablePlatformPort(key, value) for key, value in sorted(values.items()))
