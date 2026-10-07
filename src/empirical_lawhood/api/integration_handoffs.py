# SPDX-License-Identifier: MPL-2.0
"""Exact integration selection for the existing public campaign API.

Only code-owned current family loaders are selectable. This small composition
seam supplies the same context/catalog/registry/ports used by existing workflows;
it adds no scheduler, module locator in user input or per-study execution stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from empirical_lawhood.adapters._strict_json import loads_external_json
from empirical_lawhood.api.authoring_handoff import resolve_authoring_directory
from empirical_lawhood.api.integration_artifact_profiles import integration_artifact_profile_validators
from empirical_lawhood.api.integration_proof import prove_provider_projection
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.runtime.executable_bindings import LayeredCampaignRuntimeProviderResolver
from empirical_lawhood.runtime.providers import CampaignRuntimeProviderRegistry

if TYPE_CHECKING:
    from empirical_lawhood.api.rc_challenge_authoring import RCChallengeApplicationHandoff
    from empirical_lawhood.api.finite_rerun_authoring import FiniteResponseLawApplicationHandoff
    from empirical_lawhood.api.preparation_applicability import PreparationApplicabilityApplicationHandoff
    from empirical_lawhood.api.constructed_preparation_applicability import PreparationApplicabilityApplicationHandoff as ConstructedPreparationApplicabilityApplicationHandoff

IntegrationFamily = Literal['rc-challenges', 'finite-response-law', 'preparation-applicability', 'constructed-preparation-applicability']


@dataclass(frozen=True, slots=True)
class IntegrationApplicationHandoff:
    family: IntegrationFamily
    selected: RCChallengeApplicationHandoff | FiniteResponseLawApplicationHandoff | PreparationApplicabilityApplicationHandoff | ConstructedPreparationApplicabilityApplicationHandoff

    def __post_init__(self) -> None:
        # Import only the selected code-owned family. No user value supplies a
        # module path or executable type, and no provider is built here.
        if self.family == 'rc-challenges':
            from empirical_lawhood.api.rc_challenge_authoring import RCChallengeApplicationHandoff
            expected = RCChallengeApplicationHandoff
        elif self.family == 'finite-response-law':
            from empirical_lawhood.api.finite_rerun_authoring import FiniteResponseLawApplicationHandoff
            expected = FiniteResponseLawApplicationHandoff
        elif self.family == 'preparation-applicability':
            from empirical_lawhood.api.preparation_applicability import PreparationApplicabilityApplicationHandoff
            expected = PreparationApplicabilityApplicationHandoff
        elif self.family == 'constructed-preparation-applicability':
            from empirical_lawhood.api.constructed_preparation_applicability import PreparationApplicabilityApplicationHandoff as ConstructedPreparationApplicabilityApplicationHandoff
            expected = ConstructedPreparationApplicabilityApplicationHandoff
        else:
            raise ValueError('Unsupported integration family')
        if type(self.selected) is not expected:
            raise TypeError('Integration family differs from its exact typed handoff')


def load_integration_handoff(*, directory, root, storage_profile, artifact_writer) -> IntegrationApplicationHandoff:
    """Load one exact current handoff; never search for a latest result."""
    directory, _ = resolve_authoring_directory(directory, repo_root=root, storage_profile=storage_profile, escape_message='Integration handoff lies outside the explicitly selected guarded external root')
    markers = tuple(name for name in ('issued-inputs.json', 'rerun-input.json', 'selection.json') if (directory / name).is_file())
    if len(markers) != 1:
        raise ValueError('Select exactly one unambiguous current integration authoring directory')
    marker = markers[0]
    document = loads_external_json(read_bounded_bytes(directory / marker, maximum_bytes=16 * 1024**2))
    if not isinstance(document, dict):
        raise ValueError('Integration selector must be a typed object')
    schema = document.get('schema')
    if schema == 'empirical-lawhood/simulator-morphism-challenges/issued-inputs':
        from empirical_lawhood.api.rc_challenge_authoring import load_rc_challenge_handoff
        family, loader = 'rc-challenges', load_rc_challenge_handoff
    elif schema == 'empirical-lawhood/composition/finite-response-law/rerun-input':
        from empirical_lawhood.api.finite_rerun_authoring import load_finite_response_rerun_handoff
        family, loader = 'finite-response-law', load_finite_response_rerun_handoff
    elif schema == 'empirical-lawhood/composition/preparation-applicability/selection':
        from empirical_lawhood.api.preparation_applicability import load_preparation_applicability_handoff
        family, loader = 'preparation-applicability', load_preparation_applicability_handoff
    elif schema == 'empirical-lawhood/composition/constructed-preparation-applicability/selection':
        from empirical_lawhood.api.constructed_preparation_applicability import load_constructed_preparation_applicability_handoff
        family, loader = 'constructed-preparation-applicability', load_constructed_preparation_applicability_handoff
    else:
        raise ValueError('Unsupported integration selector; user input cannot select arbitrary modules')
    return IntegrationApplicationHandoff(family, loader(directory=directory, root=root, storage_profile=storage_profile, artifact_writer=artifact_writer))


def integration_api_inputs(handoff: IntegrationApplicationHandoff) -> dict[str, object]:
    """Supply only the existing generic API composition parameters."""
    if handoff.family == 'rc-challenges':
        from empirical_lawhood.adapters.simulator_morphism_challenges.composition import RCChallengeCandidateContextProvider
        selected = cast('RCChallengeApplicationHandoff', handoff.selected)
        composition = selected.composition
        return {'candidate_context_provider': RCChallengeCandidateContextProvider(composition),
                'candidate_capability_catalog': composition.catalog,
                'study_bundle_registry': composition.bundle.base.registry,
                'artifact_profile_validators': integration_artifact_profile_validators(),
                'executable_platform_ports': selected.platform_ports}
    if handoff.family not in {'finite-response-law', 'preparation-applicability', 'constructed-preparation-applicability'}:
        raise ValueError('Unsupported integration family')
    if handoff.family == 'finite-response-law':
        from empirical_lawhood.adapters.composition.response_geometry_prospective.design import ResponseGeometryAssayCandidateContextProvider
        selected = cast('FiniteResponseLawApplicationHandoff', handoff.selected)
        bundle = selected.bundle
        context = ResponseGeometryAssayCandidateContextProvider(bundle)
    else:
        from empirical_lawhood.adapters.composition.phase_authoring import PhaseContextProvider
        selected = cast('PreparationApplicabilityApplicationHandoff | ConstructedPreparationApplicabilityApplicationHandoff', handoff.selected)
        bundle = selected.bundle
        context = PhaseContextProvider(bundle)
    return {'candidate_context_provider': context,
            'candidate_capability_catalog': bundle.catalog,
            'study_bundle_registry': bundle.standard_context.base.registry,
            'executable_platform_ports': selected.platform_ports}


def prove_integration_handoff(handoff: IntegrationApplicationHandoff) -> dict[str, object]:
    """Validate installed providers and full task/resource/output census with no execute."""
    if handoff.family == 'rc-challenges':
        from empirical_lawhood.api.rc_challenge_authoring import prove_rc_challenge_handoff
        return {'family': handoff.family, **prove_rc_challenge_handoff(handoff.selected)}
    if handoff.family == 'finite-response-law':
        from empirical_lawhood.api.finite_rerun_authoring import prove_finite_response_rerun_handoff
        return {'family': handoff.family, **prove_finite_response_rerun_handoff(handoff.selected)}
    if handoff.family not in {'preparation-applicability', 'constructed-preparation-applicability'}:
        raise ValueError('Unsupported integration family')
    from empirical_lawhood.adapters.composition.generated_executable_bindings import EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    selected = cast('PreparationApplicabilityApplicationHandoff | ConstructedPreparationApplicabilityApplicationHandoff', handoff.selected)
    bundle, projection = selected.bundle, selected.projection
    registry = bundle.standard_context.base.registry
    resolver = LayeredCampaignRuntimeProviderResolver(precomposed=CampaignRuntimeProviderRegistry(()), factories=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY)
    provider = resolver.resolve(registry, decoded_records=tuple(sorted(bundle.payloads, key=lambda value: value.SCHEMA)), platform_ports=selected.platform_ports)
    return {'family': handoff.family, **prove_provider_projection(provider=provider, registry=registry,
            projection=projection, resources=selected.resources, proof_id=f'{projection.source_plan.object_id}.integration-proof')}
