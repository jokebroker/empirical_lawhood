"""Bounded software provider proofs reject corrupt inputs and close all sources."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from empirical_lawhood.api.integration_proof import prove_provider_projection
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationOutputContract, ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.plans import ExternalInputSpec
from empirical_lawhood.runtime.providers import CapabilityOutputSemanticContract, ExternalInputPayload


class TrackingSource:
    def __init__(self, payload, *, read_error=None, close_error=None):
        self.payload = payload
        self.read_error = read_error
        self.close_error = close_error
        self.reads = self.closes = 0

    def chunks(self, maximum_chunk_bytes):
        self.reads += 1
        if self.read_error is not None:
            raise self.read_error
        for offset in range(0, len(self.payload), maximum_chunk_bytes):
            yield self.payload[offset:offset + maximum_chunk_bytes]

    def close(self):
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def proof():
    """Small port doubles supply no signer, issue, native source or qualification."""
    registry_sha, implementation = 'a' * 64, 'b' * 64
    manifest = SimpleNamespace(capability_key='test.proof', capability_version='1.0.0', implementation_sha256=implementation)
    registry = SimpleNamespace(capabilities=(manifest,), fingerprint=lambda: registry_sha)
    output = SimpleNamespace(output_id='test.terminal', payload_schema=ScientificAdjudicationRecord.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON, logical_artifact_id='test.terminal-artifact')
    operands, specs, edges, sources = [], [], [], []
    for index in range(2):
        logical = f'test.input-{index}'
        payload = ObjectIdentity(f'test.operand-{index}', ObjectIdentity.SCHEMA, '1.0.0', 'c' * 64).canonical_bytes()
        digest = sha256(payload).hexdigest()
        source = TrackingSource(payload)
        sources.append(source)
        operands.append(ExternalInputPayload(logical, ObjectIdentity.SCHEMA, ArtifactProfile.CANONICAL_JSON,
            'application/json', source, len(payload), digest, 1024, 32, VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND, (), ()))
        specs.append(ExternalInputSpec(f'test.port-{index}', logical, digest, ObjectIdentity.SCHEMA,
            'application/json', len(payload), VisibilityCeiling.PROSPECTIVE, OutcomeAccess.OUTCOME_BLIND, None))
        edges.append(SimpleNamespace(external_input_id=logical, operational_logical_artifact_id=logical,
            payload_schema=ObjectIdentity.SCHEMA, media_type='application/json',
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE, outcome_access=OutcomeAccess.OUTCOME_BLIND,
            maximum_size_bytes=1024))
    task = SimpleNamespace(task_id='test.task', capability=manifest, capability_implementation_sha256=implementation,
        external_inputs=tuple(specs), scientific_inputs=tuple(edges), outputs=(output,), dependency_task_ids=())
    projection = SimpleNamespace(registry_sha256=registry_sha, tasks=(task,))
    resources = SimpleNamespace(task_cells=(SimpleNamespace(task_id=task.task_id, jit_cell_id=None, progress_liveness=None),))
    contracts = (CapabilityOutputSemanticContract(manifest.capability_key, manifest.capability_version,
        implementation, output.payload_schema, output.profile, ('schema', 'value', 'version')),)
    terminal = ScientificAdjudicationOutputContract(manifest.capability_key, manifest.capability_version,
        output.output_id, output.payload_schema)
    provider = SimpleNamespace(registry_sha256=registry_sha, runners=lambda _: (SimpleNamespace(manifest=manifest),),
        output_semantic_contracts=lambda *_: provider.contracts, external_inputs=lambda _: provider.operands,
        scientific_adjudication_contract=lambda *_: provider.terminal,
        contracts=contracts, operands=tuple(operands), terminal=terminal)
    return SimpleNamespace(provider=provider, registry=registry, projection=projection, resources=resources, sources=sources)


def run(proof):
    return prove_provider_projection(provider=proof.provider, registry=proof.registry,
        projection=proof.projection, resources=proof.resources, proof_id='test.bounded-proof')


def test_complete_proof_reads_actual_bounded_bytes_and_closes_sources(proof):
    result = run(proof)
    assert result['task_count'] == result['output_contract_count'] == 1
    assert result['external_input_count'] == 2
    assert result['scientific_execution_performed'] is False
    assert [(source.reads, source.closes) for source in proof.sources] == [(1, 1), (1, 1)]


@pytest.mark.parametrize('failure', ('missing-output-contract', 'wrong-terminal', 'duplicate-resource', 'wrong-registry'))
def test_projection_refusals_precede_opening_provider_inputs(proof, failure):
    if failure == 'missing-output-contract':
        proof.provider.contracts = ()
    elif failure == 'wrong-terminal':
        proof.provider.terminal = replace(proof.provider.terminal, output_id='test.foreign-terminal')
    elif failure == 'duplicate-resource':
        proof.resources.task_cells *= 2
    else:
        proof.provider.registry_sha256 = 'd' * 64
    with pytest.raises(ValueError):
        run(proof)
    assert [(source.reads, source.closes) for source in proof.sources] == [(0, 0), (0, 0)]


@pytest.mark.parametrize('failure', ('missing-input', 'duplicate-input', 'wrong-access', 'wrong-hash', 'logical-hash-alias', 'scientific-byte-bound', 'missing-scope'))
def test_frozen_input_refusals_close_all_returned_sources_before_read(proof, failure):
    first, second = proof.provider.operands
    if failure == 'missing-input':
        proof.provider.operands = (second,)
    elif failure == 'duplicate-input':
        proof.provider.operands = (first, replace(second, logical_artifact_id=first.logical_artifact_id))
    elif failure == 'wrong-access':
        proof.provider.operands = (replace(first, outcome_access=OutcomeAccess.EVALUATOR_REVEAL), second)
    elif failure == 'wrong-hash':
        proof.provider.operands = (replace(first, source_sha256='d' * 64), second)
    elif failure == 'logical-hash-alias':
        proof.sources[0].payload = b'x' * first.size_bytes
        proof.provider.operands = (replace(first, source_sha256=sha256(proof.sources[0].payload).hexdigest(),
            logical_content_sha256=first.source_sha256), second)
    elif failure == 'scientific-byte-bound':
        proof.projection.tasks[0].scientific_inputs[0].maximum_size_bytes = first.size_bytes - 1
    else:
        task = proof.projection.tasks[0]
        task.external_inputs = (replace(task.external_inputs[0], identity_scope_sha256='e' * 64), task.external_inputs[1])
    with pytest.raises(ValueError):
        run(proof)
    expected = [(0, 0), (0, 1)] if failure == 'missing-input' else [(0, 1), (0, 1)]
    assert [(source.reads, source.closes) for source in proof.sources] == expected


@pytest.mark.parametrize('failure', ('corrupt-bytes', 'read-error', 'interrupt', 'chunk-bound'))
def test_failed_or_interrupted_read_closes_unentered_siblings(proof, failure):
    first, _ = proof.provider.operands
    if failure == 'corrupt-bytes':
        proof.sources[0].payload = b'x' * first.size_bytes
        expected = ValueError
    elif failure == 'chunk-bound':
        proof.sources[0].chunks = lambda maximum_chunk_bytes: iter((b'x' * (maximum_chunk_bytes + 1),))
        expected = ValueError
    else:
        expected = KeyboardInterrupt if failure == 'interrupt' else OSError
        proof.sources[0].read_error = expected('synthetic input interruption')
    with pytest.raises(expected):
        run(proof)
    assert [source.closes for source in proof.sources] == [1, 1]
    assert proof.sources[1].reads == 0


def test_close_failure_keeps_primary_read_error_and_closes_other_sources(proof):
    proof.sources[0].read_error = ValueError('primary corrupt input')
    proof.sources[0].close_error = OSError('secondary close failure')
    with pytest.raises(ValueError, match='primary corrupt input'):
        run(proof)
    assert [source.closes for source in proof.sources] == [1, 1]


def test_closed_family_dispatch_requires_the_actual_selected_payload_type():
    from empirical_lawhood.api.integration_handoffs import IntegrationApplicationHandoff
    from empirical_lawhood.api.rc_challenge_authoring import RCChallengeApplicationHandoff
    from empirical_lawhood.api.finite_rerun_authoring import FiniteResponseLawApplicationHandoff
    from empirical_lawhood.api.preparation_applicability import PreparationApplicabilityApplicationHandoff
    from empirical_lawhood.api.constructed_preparation_applicability import PreparationApplicabilityApplicationHandoff as ConstructedHandoff

    # Type admission only: no provider, source, qualification or issue is
    # supplied. Separate public tests prove the actual full-size payloads.
    placeholder = SimpleNamespace()
    directory = Path('nonexecuting-type-fixture')
    payloads = (
        ('rc-challenges', RCChallengeApplicationHandoff(directory, placeholder, placeholder, placeholder, ())),
        ('finite-response-law', FiniteResponseLawApplicationHandoff(directory, placeholder, placeholder, placeholder, placeholder, (), False)),
        ('preparation-applicability', PreparationApplicabilityApplicationHandoff(directory, placeholder, placeholder, placeholder, ())),
        ('constructed-preparation-applicability', ConstructedHandoff(directory, placeholder, placeholder, placeholder, ())),
    )
    for family, selected in payloads:
        assert IntegrationApplicationHandoff(family, selected).selected is selected
        for other_family, other_selected in payloads:
            if family != other_family:
                with pytest.raises(TypeError, match='exact typed handoff'):
                    IntegrationApplicationHandoff(family, other_selected)
    with pytest.raises(ValueError, match='Unsupported integration family'):
        IntegrationApplicationHandoff('unregistered-family', payloads[0][1])
    with pytest.raises(TypeError, match='exact typed handoff'):
        IntegrationApplicationHandoff('rc-challenges', placeholder)


def test_neutral_proof_and_profile_owners_preserve_the_exact_public_contract():
    from empirical_lawhood.api.integration_handoffs import prove_provider_projection as public_proof
    from empirical_lawhood.api.integration_artifact_profiles import integration_artifact_profile_validators
    from empirical_lawhood.api.rc_challenge_authoring import RC_CHALLENGE_ARTIFACT_PROFILE_VALIDATORS
    from empirical_lawhood.adapters.simulator_morphism_challenges.authoring import ARRAY_PAYLOAD_SCHEMA
    from empirical_lawhood.infrastructure.artifact_validation import ArtifactProfileValidatorRegistry, NumpyArrayContract, ArtifactIdentityConflict

    assert public_proof is prove_provider_projection
    registry = integration_artifact_profile_validators()
    assert registry is RC_CHALLENGE_ARTIFACT_PROFILE_VALIDATORS
    defaults = ArtifactProfileValidatorRegistry()
    assert set(registry.enabled_profiles) == set(defaults.enabled_profiles) | {ArtifactProfile.NUMPY_NO_PICKLE}
    assert registry.numpy_contract(ARRAY_PAYLOAD_SCHEMA) == NumpyArrayContract(ARRAY_PAYLOAD_SCHEMA, 'float64', (None,))
    for profile in (ArtifactProfile.ARROW_IPC, ArtifactProfile.PARQUET, ArtifactProfile.AUDITED_HDF5):
        assert profile not in registry.enabled_profiles
    with pytest.raises(ArtifactIdentityConflict, match='no registered structural contract'):
        registry.numpy_contract(ObjectIdentity.SCHEMA)
