"""Pure composition of issued parameterised roots into candidate topology."""

from __future__ import annotations

from dataclasses import replace
import hashlib

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition
from empirical_lawhood.planning.observation_order import ObservationOrderExperimentExtension
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.planning.native_source import NativeLawQualificationExperiment
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.planning.retrospective_prediction import RetrospectivePredictionExperiment

from .candidate_compiler import (
    CandidateCompilationContext,
    StandardCandidateCompilationContext,
)
from .observation_order import ObservationOrderSubstrateBinding
from .executable_bindings import ExecutableCapabilityProviderFactoryRegistry
from .linked_campaigns import rebind_study_template_to_expanded_protocol
from .response_experiment_ports import ResponseSubstrateBinding
from .plans import ScientificStage
from .source_qualification import FreshSourceQualificationSubstrateBinding
from .retrospective_prediction import RetrospectivePredictionBinding


def compose_parameterised_candidate_context(
    *,
    package: ExecutableStudyDefinition,
    context: CandidateCompilationContext | StandardCandidateCompilationContext,
    records: tuple[CanonicalRecord, ...],
    factories: ExecutableCapabilityProviderFactoryRegistry | None,
) -> CandidateCompilationContext | StandardCandidateCompilationContext:
    "Return the deterministic topology selected by one exact substrate root.\n\n    The function is deliberately public and effect-free so the same context can\n    compile the current preview, issue its immutable base candidate, and replay the\n    extension publication.  No adapter is imported through configuration and no source\n    or simulator is contacted.\n    "

    return compose_parameterised_candidate_context_for_template(
        template_key=package.base.draft.dag_template_key,
        context=context,
        records=records,
        factories=factories,
    )


def compose_parameterised_candidate_context_for_template(
    *,
    template_key: str,
    context: CandidateCompilationContext | StandardCandidateCompilationContext,
    records: tuple[CanonicalRecord, ...],
    factories: ExecutableCapabilityProviderFactoryRegistry | None,
) -> CandidateCompilationContext | StandardCandidateCompilationContext:
    """Compose before standard authoring when formal coverage needs expanded owners."""

    carriers = tuple(
        value
        for value in records
        if isinstance(
            value,
            (
                ResponseExperimentExtensionSet,
                ObservationOrderExperimentExtension,
                FreshSourceQualificationExperiment,
                RetrospectivePredictionExperiment,
            ),
        )
    )
    raw_substrates = tuple(value for value in records if isinstance(value, ResponseSubstrateBinding))
    observation_bindings = tuple(
        value for value in records if isinstance(value, ObservationOrderSubstrateBinding)
    )
    qualification_bindings = tuple(
        value for value in records if isinstance(value, FreshSourceQualificationSubstrateBinding)
    )
    historical_bindings = tuple(
        value for value in records if isinstance(value, RetrospectivePredictionBinding)
    )
    if (
        not carriers
        and not raw_substrates
        and not observation_bindings
        and not qualification_bindings
        and not historical_bindings
    ):
        return context
    if len(carriers) != 1:
        raise ValueError("parameterised candidate requires exactly one supported carrier")
    carrier = carriers[0]
    substrate: ResponseSubstrateBinding | RetrospectivePredictionBinding
    if historical_bindings and not isinstance(carrier, RetrospectivePredictionExperiment):
        raise ValueError("another carrier received a historical prediction binding")
    if qualification_bindings and not isinstance(carrier, FreshSourceQualificationExperiment):
        raise ValueError("another carrier received a qualification substrate association")
    if isinstance(carrier, RetrospectivePredictionExperiment):
        if (
            raw_substrates
            or observation_bindings
            or qualification_bindings
            or len(historical_bindings) != 1
        ):
            raise ValueError("historical prediction requires one held-source binding")
        historical = historical_bindings[0]
        if historical.prediction_carrier != ObjectIdentity.from_record(
            carrier.extension_set_id, carrier
        ):
            raise ValueError("historical binding selects another carrier")
        substrate = historical
    elif isinstance(carrier, ObservationOrderExperimentExtension):
        if raw_substrates or len(observation_bindings) != 1:
            raise ValueError("observation candidate requires one typed substrate association")
        association = observation_bindings[0]
        if association.observation_carrier != ObjectIdentity.from_record(
            carrier.extension_set_id,
            carrier,
        ):
            raise ValueError("observation substrate association selects another carrier")
        substrate = association.substrate
    elif isinstance(carrier, FreshSourceQualificationExperiment):
        if raw_substrates or observation_bindings or len(qualification_bindings) != 1:
            raise ValueError("qualification candidate requires one typed substrate association")
        qualification = qualification_bindings[0]
        if (
            qualification.qualification_carrier
            != ObjectIdentity.from_record(carrier.extension_set_id, carrier)
            or qualification.source_config != carrier.source_config
        ):
            raise ValueError(
                "qualification substrate association selects another carrier or config"
            )
        substrate = qualification.substrate
    else:
        if observation_bindings or len(raw_substrates) != 1:
            raise ValueError("Measurement through controller use candidate requires one substrate binding")
        substrate = raw_substrates[0]
    if factories is None:
        raise ValueError("parameterised candidate expansion is not composed")
    factory = factories.factory(substrate.installed_executable_binding.object_id)
    expand = getattr(factory, "expand_parameterised_protocol", None)
    if not callable(expand):
        raise ValueError("selected substrate lacks installed protocol expansion")
    base_context = (
        context.base if isinstance(context, StandardCandidateCompilationContext) else context
    )
    selected = base_context.template(template_key)
    if selected is None:
        raise ValueError("parameterised candidate lacks its selected programme template")
    expanded_protocol = expand(records=records, template=selected.protocol)
    if isinstance(carrier, ResponseExperimentExtensionSet) and carrier.admission_config is None:
        retained = tuple(
            value
            for value in expanded_protocol.steps
            if value.stage not in {ScientificStage.ADMISSION, ScientificStage.CONTROLLER}
        )
        retained_ids = {value.step_id for value in retained}
        expanded_protocol = replace(
            expanded_protocol,
            steps=tuple(
                replace(
                    value,
                    dependency_step_ids=tuple(
                        item for item in value.dependency_step_ids if item in retained_ids
                    ),
                )
                for value in retained
            ),
            requests_controller=False,
            nonactuating=True,
        )
    if isinstance(carrier, ResponseExperimentExtensionSet) and not isinstance(
        carrier, NativeLawQualificationExperiment
    ):
        # Preserve the established measurement-through-controller-use execution bound. Observation-order
        # acquisition instead keeps its frozen single-attempt contract.
        expanded_protocol = replace(
            expanded_protocol,
            steps=tuple(replace(step, maximum_attempts=2) for step in expanded_protocol.steps),
        )
    expanded_template = rebind_study_template_to_expanded_protocol(
        study_template=selected,
        expanded_protocol=expanded_protocol,
        capability_registry=base_context.registry,
    )
    templates = tuple(
        sorted(
            (
                expanded_template if value.template_key == selected.template_key else value
                for value in base_context.templates
            ),
            key=lambda value: value.template_key,
        )
    )
    composition_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "base_context": base_context.fingerprint(),
                "factory_binding": factory.binding,
                "expanded_template": expanded_template,
            }
        )
    ).hexdigest()
    expanded_base = replace(
        base_context,
        context_id=f"{base_context.context_id}.parameterised.{composition_sha256[:16]}",
        templates=templates,
    )
    if isinstance(context, StandardCandidateCompilationContext):
        return replace(
            context,
            context_id=f"{context.context_id}.parameterised.{composition_sha256[:16]}",
            base=expanded_base,
        )
    return expanded_base


__all__ = [
    'compose_parameterised_candidate_context_for_template',
    'compose_parameterised_candidate_context',
]
