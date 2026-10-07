'Registered calibration factories over a synthetic disjoint assignment.'

from tests.finite_response_seed_fixtures import ASSIGNED_SEEDS
import pytest

from empirical_lawhood.adapters.composition.finite_response_law.assignment import FiniteResponseLawCohortAssignment
from empirical_lawhood.adapters.composition.finite_response_law.authoring import build_native_authoring
from empirical_lawhood.adapters.composition.finite_response_law.exposure import native_seed_ids
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedCalibrationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.calibration import discovery, executable_binding
from empirical_lawhood.adapters.simulators.finite_response_law.executable_binding import native_bindings
from empirical_lawhood.adapters.simulators.finite_response_law.fresh_contracts import FiniteResponseLawCalibrationConfig
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding


def test_registered_calibration_factory_selects_assigned_source_and_methods() -> None:
    science = FiniteResponseLawScienceSpec()
    assignment = FiniteResponseLawCohortAssignment(
        "calibration",
        "empirical-lawhood.finite-response-law.calibration.development",
        science.plan_sha256,
        32,
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    scientific_seeds=ASSIGNED_SEEDS["empirical-lawhood.finite-response-law.calibration.development"],
    )
    source = FiniteResponseLawAssignedCalibrationConfig(
        "calibration",
        science,
        ObjectIdentity.from_record(
            discovery.SOURCE_CAPABILITY.capability_key,
            discovery.SOURCE_CAPABILITY,
        ),
        None,
        (),
        assignment.cohort_namespace,
        assignment.fingerprint(),
    scientific_seeds=assignment.scientific_seeds,
    )
    # The fabricated prior identifiers prove only the exact binding path.
    bundle = build_native_authoring(
        source=source,
        implementation_sha256="0" * 64,
        exposure=ObjectIdentity.from_record(
            "synthetic-development-assignment", assignment
        ),
        exposed_unit_ids=("synthetic-exposed-root",),
        exposed_seed_ids=("synthetic-exposed-seed",),
        new_seed_ids=native_seed_ids(source),
    )
    registry = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    source_factory = registry.provider_factory(
        executable_binding.SOURCE_BINDING.binding_id
    )
    source_records = tuple(
        value
        for value in bundle.payloads
        if type(value)
        in (
            FiniteResponseLawAssignedCalibrationConfig,
            PredecessorBoundSourceQualificationExperiment,
            PredecessorBoundSourceQualificationSubstrateBinding,
        )
    )
    provider = source_factory.build_provider(
        registry=CapabilityRegistry(
            "synthetic-calibration-source-registry", (discovery.SOURCE_CAPABILITY,)
        ),
        records=source_records,
        platform_ports=(),
    )
    assert type(source_factory).__name__ == 'FiniteResponseLawCalibrationSourceFactory'
    assert len(provider.invocations) == 640
    assert len(provider.runners(provider.registry)) == 1
    assert {
        task.root.physical_unit_id for task in provider.invocations.values()
    } == set(assignment.physical_unit_ids)
    for binding, manifest in (
        (executable_binding.PROJECTION_BINDING, discovery.PROJECTION_CAPABILITY),
        (executable_binding.EVALUATION_BINDING, discovery.EVALUATION_CAPABILITY),
    ):
        config = next(
            value for value in bundle.payloads if value.SCHEMA == manifest.config_schema
        )
        factory = registry.provider_factory(binding.binding_id)
        method = factory.build_provider(
            registry=CapabilityRegistry(
                f"synthetic-registry.{manifest.capability_key}", (manifest,)
            ),
            records=(config,),
            platform_ports=(),
        )
        assert len(method.runners(method.registry)) == 1
    fixed = FiniteResponseLawCalibrationConfig("calibration", science, source.native_owner, None, ())
    with pytest.raises(ValueError, match="Exposed fixed calibration roots"):
        native_bindings(fixed)
