"""Candidate authoring for nine unchanged methods over committed development data."""
from empirical_lawhood.kernel.authority import AuthorityAction


from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition, ProposedStudyExtension, ProposedStudyExtensionSet
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import OperationalFailureClass
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, NonTimeResourceBudget
from empirical_lawhood.runtime.plans import ProtocolTemplate, ScientificInputRole
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.matrix_preparation.executable_binding import METHOD_BINDINGS
from empirical_lawhood.adapters.methods.matrix_preparation.extension_bundle import METHOD_CAPABILITIES
from empirical_lawhood.adapters.methods.matrix_preparation.method_continuation import PreparationMethodContinuation, METHOD_CONTINUATION_ROLES, METHOD_IMPORT_TYPES, retained_method_artifact_id, unfinished_method_tasks
from empirical_lawhood.adapters.methods.matrix_preparation.topology import METHOD_CONFIG_TYPES, METHOD_ROLES, expected_method_inputs
from .authoring import PreparationAuthoringBundle
from .entry import preparation_entry
from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign as _campaign
from .science import DEVELOPMENT_BUDGET


def build_preparation_method_authoring(
    continuation: PreparationMethodContinuation,
    original: PreparationAuthoringBundle,
    *,
    implementation_sha256: str,
) -> PreparationAuthoringBundle:
    prefix = continuation.run_id
    if continuation.source_authoring != ObjectIdentity.from_record(original.authoring.package_id, original.authoring):
        raise ValueError("method completion requires its exact current target authoring export before composition")
    original_draft = original.authoring.base.draft
    experiment = original_draft.experiment
    assert experiment is not None and original_draft.system is not None
    configs = {
        role: next(r for r in original.payloads if type(r) is METHOD_CONFIG_TYPES[role])
        for role in METHOD_CONTINUATION_ROLES
    }
    assessment = configs["description"]
    if continuation.assessment != ObjectIdentity.from_record(
        str(getattr(assessment, "config_id")), assessment
    ):
        raise ValueError("method continuation changed the original scientific configuration")
    manifests = tuple(
        m
        for role, m in zip(METHOD_ROLES, METHOD_CAPABILITIES, strict=True)
        if role in METHOD_CONTINUATION_ROLES
    )
    registry = CapabilityRegistry(
        f"{prefix}.registry", tuple(sorted(manifests, key=lambda m: m.registry_id))
    )
    declarations = {t.task_id: t for t in unfinished_method_tasks()}
    old_template = original.standard_context.base.template(original_draft.dag_template_key)
    assert old_template is not None
    terminal = next(t.task_id for t in declarations.values() if t.role == "evaluation")
    steps = {}
    for old in old_template.protocol.steps:
        if old.step_id not in declarations:
            continue
        task = declarations[old.step_id]
        dependencies = tuple(d for d in old.dependency_step_ids if d in declarations)
        config = configs[task.role]
        # Retain the original safe envelope; the completed producers become
        # authenticated external operands and cannot be scheduled again.
        steps[old.step_id] = replace(
            old,
            dependency_step_ids=dependencies,
            obligation_ids=(f"{prefix}.single-terminal",)
            if task.role == "evaluation"
            else old.obligation_ids,
        )
    protocol = ProtocolTemplate(
        f"{prefix}.protocol", "1.0.0", tuple(steps[k] for k in sorted(steps)), False, False, True
    )
    external = {}
    edges = []
    for child in protocol.steps:
        task = declarations[child.step_id]
        config = configs[task.role]
        expected = expected_method_inputs(task, str(getattr(config, "config_id")), config.SCHEMA)
        for row in continuation.inputs_for_task(child.step_id):
            artifact = row.artifact
            imported = retained_method_artifact_id(prefix, task.role, row.slot_id)
            external[imported] = CandidateGraphExternalInput(
                imported,
                ScientificInputRole.OUTCOME,
                imported,
                ContentIdentityPolicy.EXACT_SHA256,
                artifact.sha256,
                artifact.payload_schema,
                artifact.media_type,
                artifact.size_bytes,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.OUTCOME_VISIBLE,
            )
            edges.append(
                CandidateGraphEdge(
                    f"edge.{row.slot_id}.{child.step_id}",
                    None,
                    None,
                    imported,
                    child.step_id,
                    f"input.{row.slot_id}",
                    ScientificInputRole.OUTCOME,
                    imported,
                    artifact.payload_schema,
                    artifact.media_type,
                    artifact.size_bytes,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    child.barrier,
                )
            )
        for parent_id in child.dependency_step_ids:
            parent = steps[parent_id]
            for output in parent.outputs:
                slot = f"{parent_id}.{output.output_id}"
                if slot not in expected:
                    continue
                edges.append(
                    CandidateGraphEdge(
                        f"edge.{slot}.{child.step_id}",
                        parent_id,
                        output.output_id,
                        None,
                        child.step_id,
                        f"input.{slot}",
                        ScientificInputRole.OUTCOME,
                        f"artifact.{slot}",
                        output.payload_schema,
                        output.media_type,
                        parent.resource_budget.output_bytes,
                        child.requested_outcome_access,
                        VisibilityCeiling.OUTCOME_VISIBLE,
                        child.barrier,
                    )
                )
    declaration = ObjectIdentity.from_record(continuation.config_id, continuation)
    declaration_input = CandidateGraphExternalInput(
        continuation.config_id,
        ScientificInputRole.MODEL,
        f"config-artifact.{continuation.config_id}",
        ContentIdentityPolicy.EXACT_SHA256,
        declaration.fingerprint(),
        ObjectIdentity.SCHEMA,
        "application/vnd.empirical-lawhood.canonical+json",
        len(declaration.canonical_bytes()),
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    external[continuation.config_id] = declaration_input
    for child in protocol.steps:
        if declarations[child.step_id].role == "description":
            edges.append(
                CandidateGraphEdge(
                    f"edge.{continuation.config_id}.{child.step_id}",
                    None,
                    None,
                    continuation.config_id,
                    child.step_id,
                    "retained-input-declaration",
                    ScientificInputRole.MODEL,
                    declaration_input.logical_artifact_id,
                    ObjectIdentity.SCHEMA,
                    declaration_input.media_type,
                    declaration_input.maximum_size_bytes,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    child.barrier,
                )
            )
    nodes = tuple(
        CandidateGraphNode(
            s.step_id,
            s.stage,
            s.capability_key,
            s.capability_version,
            registry.resolve(s.capability_key, s.capability_version).implementation_sha256,
            s.fingerprint(),
            s.obligation_ids,
            s.requested_outcome_access,
            s.visibility_ceiling,
            s.resource_budget,
        )
        for s in protocol.steps
    )
    graph = CandidateScientificGraph(
        f"{prefix}.graph",
        tuple(external[k] for k in sorted(external)),
        nodes,
        tuple(sorted(edges, key=lambda e: e.edge_id)),
    )
    owners = {o: s.step_id for s in protocol.steps for o in s.obligation_ids}
    coverage = tuple(
        ObligationCoverageBinding(
            o,
            owners.get(o, terminal),
            "report",
            tuple(
                sorted(
                    e.edge_id for e in graph.edges if e.consumer_node_id == owners.get(o, terminal)
                )
            ),
        )
        for o in required_candidate_obligation_ids(experiment, protocol)
    )
    template = StudyTemplate(
        f"{prefix}.template",
        "1.0.0",
        protocol,
        graph,
        ObligationCoverage(
            f"{prefix}.coverage", tuple(sorted(coverage, key=lambda b: b.obligation_id))
        ),
    )
    declaration = ObjectIdentity.from_record(continuation.config_id, continuation)
    science = next(v for v in original_draft.design_inputs if v.role is DesignInputRole.MOTIVATION)
    added = tuple(
        DesignInputRecord(
            f"{prefix}.{label}",
            identity,
            identity.object_fingerprint,
            science.information_cutoff,
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            science.operator_id,
            parent_input_ids=(science.input_id,),
        )
        for label, identity in (
            ("retained-inputs", declaration),
            ("owner-amendment", continuation.owner_amendment),
        )
    )
    design_inputs = tuple(sorted((*original_draft.design_inputs, *added), key=lambda v: v.input_id))
    qualification = replace(
        original.standard_context.base.qualifications[0],
        receipt_id=f"{prefix}.declaration-qualification",
        source_id=continuation.config_id,
        materialization=ObjectIdentity.from_record(
            f"config-artifact.{continuation.config_id}", declaration
        ),
        content_sha256=declaration.fingerprint(),
    )
    source = SourceMaterializationRef(
        continuation.config_id,
        SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        qualification.evidence_world_id,
        qualification.materialization,
        declaration.fingerprint(),
        declaration.fingerprint(),
        qualification.observation_operator,
        qualification.numerical_view_ids,
        ObjectIdentity.from_record(qualification.receipt_id, qualification),
        qualification.access_disposition,
    )
    system = replace(
        original_draft.system,
        authority_policy=replace(
            original_draft.system.authority_policy,
            policy_id=f"{prefix}.authority-policy",
            scope_ids=(prefix,),
        ),
    )
    experiment = replace(experiment, authority_policy_id=system.authority_policy.policy_id)
    draft = replace(
        original_draft,
        draft_id=f"{prefix}.draft",
        system=system,
        experiment=experiment,
        campaign=_campaign(
            system,
            experiment,
            prefix=prefix,
            budget=DEVELOPMENT_BUDGET,
            objective="Complete unchanged development methods over the exact retained panel.",
            actions=(("reveal", AuthorityAction.EVALUATOR_REVEAL), ("simulation", AuthorityAction.SIMULATION_EXECUTION)),
        ),
        design_inputs=design_inputs,
        design_origin=replace(
            original_draft.design_origin,
            origin_id=f"{prefix}.origin",
            declared_input_ids=tuple(
                sorted(
                    (*original_draft.design_origin.declared_input_ids, *(a.input_id for a in added))
                )
            ),
        ),
        dag_template_key=template.template_key,
        capability_selections=tuple(
            CapabilitySelection(m.capability_key, m.capability_version, m.implementation_sha256)
            for m in sorted(manifests, key=lambda m: m.capability_key)
        ),
        source_materializations=(source,),
    )
    context = CandidateCompilationContext(
        f"{prefix}.context",
        registry,
        (template,),
        (qualification,),
        design_inputs,
        implementation_sha256,
    )
    base, standard = preparation_entry(draft, context, science.object_identity, prefix=prefix)
    imports = tuple(
        kind(
            f"{prefix}.retained-{role}",
            declaration,
            ObjectIdentity.from_record(str(getattr(configs[role], "config_id")), configs[role]),
        )
        for role, kind in METHOD_IMPORT_TYPES.items()
    )
    payloads = tuple(sorted((*configs.values(), continuation, *imports), key=lambda r: r.SCHEMA))
    decoder_by_schema = {
        d.payload_schema: d for b in METHOD_BINDINGS for d in b.issued_decoder_registrations
    }
    decoders = tuple(decoder_by_schema[r.SCHEMA] for r in payloads)
    proposed = tuple(
        ProposedStudyExtension(
            f"{prefix}.extension.{i:02d}",
            f"{prefix}.namespace.{i:02d}",
            ObjectIdentity.from_record(str(getattr(r, "config_id")), r),
            len(r.canonical_bytes()),
            d.decoder_key,
            d.decoder_version,
            d.config_sha256,
            True,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        for i, (r, d) in enumerate(zip(payloads, decoders, strict=True))
    )
    extensions = ProposedStudyExtensionSet(
        f"{prefix}.proposed-extensions",
        ObjectIdentity.from_record(base.package_id, base),
        tuple(p.namespace_id for p in proposed),
        proposed,
    )
    selected = {m.capability_key for m in manifests}
    evidence = replace(
        original.evidence_profile,
        selection_id=f"{prefix}.evidence-profile",
        draft_id=draft.draft_id,
    )
    return PreparationAuthoringBundle(
        ExecutableStudyDefinition(f"{prefix}.authoring", base, extensions),
        standard,
        CandidateCapabilityCatalog(
            f"{prefix}.catalog",
            tuple(
                r
                for r in GENERATED_EXTENSION_BUNDLE_AGGREGATE.candidate_capability_catalog.registrations
                if r.manifest.capability_key in selected
            ),
            (template,),
        ),
        payloads,
        decoders,
        evidence,
    )


def build_preparation_method_resources(
    continuation: PreparationMethodContinuation,
    protocol: ProtocolTemplate,
    issued_extensions: ObjectIdentity,
) -> ExecutionResourceEnvelopeSpec:
    if {s.step_id for s in protocol.steps} != {
        t.task_id for t in unfinished_method_tasks()
    } or any(s.maximum_attempts != 1 for s in protocol.steps):
        raise ValueError("method continuation resources require exactly nine unchanged methods")
    cells = tuple(
        ExecutionResourceTaskCellSpec(
            f"cell.{s.step_id}",
            s.step_id,
            continuation.run_id,
            None,
            None,
            1,
            0,
            0,
            False,
            NonTimeResourceBudget(
                f"budget.{s.step_id}",
                s.resource_budget.cpu_cores,
                s.resource_budget.memory_bytes,
                s.resource_budget.output_bytes,
                0,
                s.resource_budget.source_scan_bytes,
            ),
            None,
            None,
            None,
        )
        for s in protocol.steps
    )
    return ExecutionResourceEnvelopeSpec(
        f"{continuation.run_id}.resources",
        issued_extensions,
        cells,
        (ChildResourceTokenLimit(continuation.run_id, 0, 0),),
        (),
        (),
        tuple(sorted(f.value for f in OperationalFailureClass)),
        None,
        None,
        2,
        16 * 1024**3,
        (),
        (),
        sha256(
            canonical_json_bytes(
                {
                    "continuation": continuation.fingerprint(),
                    "protocol": protocol.fingerprint(),
                    "native_execution_tokens": 0,
                    "retry_tokens": 0,
                }
            )
        ).hexdigest(),
        True,
        True,
        True,
        True,
    )
