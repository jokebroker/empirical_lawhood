# SPDX-License-Identifier: MPL-2.0
"""Neutral bounded provider proof shared by current integration families."""

import sys

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.execution import validate_runner_resource_interface
from empirical_lawhood.runtime.providers import capability_semantic_validation_registry


def prove_provider_projection(*, provider, registry, projection, resources, proof_id: str) -> dict[str, object]:
    """Read bounded provider inputs and validate the complete frozen projection.

    Sources are sequential and always closed. Load a fresh handoff for a later
    operation; this proof issues no study, executes no task and writes no input.
    """
    if projection.registry_sha256 != registry.fingerprint() or provider.registry_sha256 != projection.registry_sha256:
        raise ValueError('Integration provider registry differs from its frozen projection')
    resource_identity = getattr(projection, 'execution_resource_envelope_spec', None)
    if resource_identity is not None and resource_identity != ObjectIdentity.from_record(resources.envelope_spec_id, resources):
        raise ValueError('Integration projection differs from its exact resource envelope')
    runners = provider.runners(registry)
    by_capability = {(runner.manifest.capability_key, runner.manifest.capability_version): runner for runner in runners}
    manifests = {(value.capability_key, value.capability_version): value for value in registry.capabilities}
    if len(by_capability) != len(runners) or set(by_capability) != set(manifests) or any(by_capability[key].manifest != manifest for key, manifest in manifests.items()):
        raise ValueError('Integration runners do not bind the exact installed registry')
    cells = {cell.task_id: cell for cell in resources.task_cells}
    if len(cells) != len(resources.task_cells) or set(cells) != {task.task_id for task in projection.tasks}:
        raise ValueError('Integration resources do not bind the full task census')
    for task in projection.tasks:
        runner = by_capability.get((task.capability.capability_key, task.capability.capability_version))
        if runner is None or runner.manifest.implementation_sha256 != task.capability_implementation_sha256:
            raise ValueError('Integration task differs from its installed runner identity')
        validate_runner_resource_interface(runner, cells[task.task_id])
    outputs = provider.output_semantic_contracts(registry, projection)
    semantics = capability_semantic_validation_registry(registry_id=proof_id, plan=projection, contracts=outputs)
    if len(semantics.registrations) != sum(len(task.outputs) for task in projection.tasks):
        raise ValueError('Integration semantic validation omits an assigned output')
    adjudication = provider.scientific_adjudication_contract(registry, projection)
    if adjudication is None or adjudication.plumbing_only or adjudication.payload_schema != ScientificAdjudicationRecord.SCHEMA:
        raise ValueError('Integration lacks an exact scientific terminal adjudication locator')
    matches = tuple((task, output) for task in projection.tasks
                    if (task.capability.capability_key, task.capability.capability_version) == (adjudication.capability_key, adjudication.capability_version)
                    for output in task.outputs if (output.output_id, output.payload_schema) == (adjudication.output_id, adjudication.payload_schema))
    depended_on = {dependency for task in projection.tasks for dependency in task.dependency_task_ids}
    if len(matches) != 1 or {task.task_id for task in projection.tasks if task.task_id not in depended_on} != {matches[0][0].task_id}:
        raise ValueError('Integration adjudication does not identify its single terminal task/output')

    expected = {}
    scientific = {}
    for task in projection.tasks:
        for operand in task.external_inputs:
            if expected.setdefault(operand.logical_artifact_id, operand) != operand:
                raise ValueError('Integration has conflicting exact external operand descriptors')
        for edge in task.scientific_inputs:
            if edge.external_input_id is not None:
                scientific.setdefault(edge.operational_logical_artifact_id, []).append(edge)
    external = provider.external_inputs(projection)
    try:
        identifiers = tuple(operand.logical_artifact_id for operand in external)
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(expected):
            raise ValueError('Integration external inputs do not bind the complete exact census')
        for operand in external:
            required = expected[operand.logical_artifact_id]
            # The current artifact plane permits no unproved physical/logical
            # equivalence. Apply its same identity rule before streamed reads.
            if operand.logical_content_sha256 is not None and operand.logical_content_sha256 != operand.source_sha256:
                raise ValueError('Integration external operand lacks verified physical/logical equivalence')
            fields = ((required.expected_payload_schema, operand.payload_schema),
                      (required.expected_media_type, operand.media_type),
                      (required.expected_size_bytes, operand.size_bytes),
                      (required.expected_visibility_ceiling, operand.visibility_ceiling),
                      (required.expected_outcome_access, operand.outcome_access),
                      (required.expected_content_sha256, operand.logical_content_sha256 or operand.source_sha256))
            if any(expected_value is not None and expected_value != observed for expected_value, observed in fields):
                raise ValueError('Integration external operand differs from its frozen descriptor')
            if required.identity_scope_sha256 is not None and not any(parent.identity.object_fingerprint == required.identity_scope_sha256 for parent in operand.lineage_parents):
                raise ValueError('Integration external operand omits its exact enclosing identity')
            for edge in scientific.get(operand.logical_artifact_id, ()):
                if (operand.payload_schema != edge.payload_schema or operand.media_type != edge.media_type
                    or operand.visibility_ceiling != edge.visibility_ceiling or operand.outcome_access != edge.outcome_access
                    or operand.size_bytes > edge.maximum_size_bytes):
                    raise ValueError('Integration external operand differs from its declared scientific edge')
            for _ in operand.chunks():
                pass
    finally:
        # Close every returned source, including unentered siblings on refusal.
        # Continue cleanup if one source's close method itself fails.
        failure_pending = sys.exc_info()[0] is not None
        first_error = None
        for operand in external:
            try:
                operand.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None and not failure_pending:
            raise first_error
    return {'task_count': len(projection.tasks), 'runner_count': len(runners),
            'output_contract_count': len(outputs), 'external_input_count': len(external),
            'adjudication_output_id': adjudication.output_id, 'scientific_execution_performed': False}
