"""Public finite-stage selections with exact precontact parent refusals."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters.composition.finite_response_law.assignment import FiniteResponseLawCohortAssignment, assigned_native_seed_ids
from empirical_lawhood.adapters.composition.finite_response_law.authoring import build_native_authoring
from empirical_lawhood.adapters.composition.finite_response_law.exposure import native_seed_ids, native_seed_collisions
from empirical_lawhood.adapters.composition.finite_response_law.native_input import _census
from empirical_lawhood.adapters.composition.prepared_response.native_authoring import _read_external, _read_plan
from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedCalibrationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.calibration import discovery as calibration_discovery
from empirical_lawhood.adapters.simulators.finite_response_law.calibration import executable_binding as calibration_binding
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding

STAGE_BINDINGS = {
    "informative-composition": (
        "retained-informative-composition-consumer",
        "RetainedInformativeCompositionConsumer",
        ("AUTHENTICATED_PARENT",),
    ),
    "calibration": (
        "binding.finite-response-law.calibration.source",
        'FiniteResponseLawCalibrationSourceFactory',
        (),
    ),
    "supplemental-development": (
        "binding.finite-response-law.source",
        'FiniteResponseLawSourceFactory',
        ("RETAINED_SOURCE_PORT",),
    ),
    "calibration-method": (
        "binding.finite-response-law.calibration-qualification",
        'FiniteResponseLawCalibrationQualificationFactory',
        ("CANDIDATE_PAYLOAD_PORT", "INPUT_RESOLVER_PORT"),
    ),
    "prospective-evaluation": (
        "binding.finite-response-law.evaluation.source",
        'FiniteResponseLawEvaluationSourceFactory',
        (
            "SOURCE_CONTROL_PORT",
            "CONTROL_RUNTIME_PORT",
            "CANDIDATE_PAYLOAD_PORT",
            "INPUT_RESOLVER_PORT",
            "SEALED_CONTROL_PORT",
            "REVEAL_CONTROL_PORT",
        ),
    ),
    "prospective-continuation": (
        "binding.finite-response-law.evaluation-continuation.source",
        'FiniteResponseLawContinuationSourceFactory',
        ("SOURCE_CONTROL_PORT", "IMPORT_RESOLVER_PORT", "INPUT_RESOLVER_PORT"),
    ),
    "preparation-screening": (
        "binding.finite-response-law.preparation-policy.source",
        'FiniteResponseLawPreparationPolicySourceFactory',
        ("PREPARATION_POLICY_RETAINED_SOURCE_PORT", "PREPARATION_POLICY_EVIDENCE_PORT"),
    ),
}


@dataclass(frozen=True, slots=True)
class FiniteResponseLawStageInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/finite-response-law-stage-input'
    config_id: str
    stage: str
    target_prefix: str
    evidence_role: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.target_prefix, field_name="target_prefix")
        if (
            self.stage not in STAGE_BINDINGS
            or not self.config_id.startswith(
                f"empirical-lawhood-finite-response-law-{self.stage}-"
            )
            or not self.target_prefix.startswith(
                f"empirical-lawhood.finite-response-law.{self.stage}."
            )
            or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        ):
            raise ValueError(
                "FINITE_RESPONSE_LAW_STAGE_INPUT_INVALID: selection or development role"
            )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPriorCensus(CanonicalRecord):
    """Exact development inspection, without receipt or issue authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/finite-response-law-prior-census'
    census_id: str
    plan_sha256: str
    source_file_sha256: str
    assignment_sha256: str
    excluded_unit_ids: tuple[str, ...]
    excluded_seed_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.census_id, field_name="census_id")
        for name in ("plan_sha256", "source_file_sha256", "assignment_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        for name in ("excluded_unit_ids", "excluded_seed_ids"):
            values = getattr(self, name)
            require_sorted_unique_strings(values, field_name=name)
            if not values:
                raise ValueError(
                    "prepared response finite prior census lacks its complete inspected set"
                )
            for value in values:
                validate_stable_id(value, field_name=name)


def _build_assigned_calibration(
    config: FiniteResponseLawStageInput,
    assignment: FiniteResponseLawCohortAssignment,
    plan_sha: str,
    prior_sha: str,
    excluded_units: tuple[str, ...],
    excluded_seeds: tuple[str, ...],
) -> dict[str, object]:
    """Build only the registered development providers, with no native task entry."""

    manifest = calibration_discovery.SOURCE_CAPABILITY
    source = FiniteResponseLawAssignedCalibrationConfig(
        "calibration",
        FiniteResponseLawScienceSpec(),
        ObjectIdentity.from_record(manifest.capability_key, manifest),
        None,
        (),
        assignment.cohort_namespace,
        assignment.fingerprint(),
        assignment.scientific_seeds,
    )
    if tuple(
        root.physical_unit_id for root in source.roots
    ) != assignment.physical_unit_ids or native_seed_ids(
        source
    ) != assigned_native_seed_ids(assignment):
        raise ValueError("FINITE_RESPONSE_LAW_ASSIGNED_NATIVE_CENSUS_MISMATCH")
    census = FiniteResponseLawPriorCensus(
        f"{config.target_prefix}.prior-census",
        plan_sha,
        prior_sha,
        assignment.fingerprint(),
        excluded_units,
        excluded_seeds,
    )
    bundle = build_native_authoring(
        source=source,
        implementation_sha256=manifest.implementation_sha256,
        exposure=ObjectIdentity.from_record(census.census_id, census),
        exposed_unit_ids=excluded_units,
        exposed_seed_ids=excluded_seeds,
        new_seed_ids=native_seed_ids(source),
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
    registry = GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry
    factories = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    provider = factories.provider_factory(
        calibration_binding.SOURCE_BINDING.binding_id
    ).build_provider(registry=registry, records=source_records, platform_ports=())
    runners = provider.runners(registry)
    if (
        len(source_records) != 3
        or len(provider.invocations) != 640
        or len(runners) != 1
        or runners[0].manifest != manifest
    ):
        raise ValueError("FINITE_RESPONSE_LAW_ASSIGNED_NATIVE_PROVIDER_MISMATCH")
    for binding, method_manifest in (
        (
            calibration_binding.PROJECTION_BINDING,
            calibration_discovery.PROJECTION_CAPABILITY,
        ),
        (
            calibration_binding.EVALUATION_BINDING,
            calibration_discovery.EVALUATION_CAPABILITY,
        ),
    ):
        method_config = next(
            value
            for value in bundle.payloads
            if value.SCHEMA == method_manifest.config_schema
        )
        method = factories.provider_factory(binding.binding_id).build_provider(
            registry=registry, records=(method_config,), platform_ports=()
        )
        selected = method.runners(registry)
        if len(selected) != 1 or selected[0].manifest != method_manifest:
            raise ValueError("FINITE_RESPONSE_LAW_ASSIGNED_METHOD_PROVIDER_MISMATCH")
    return {
        "status": "FINITE_RESPONSE_LAW_DEVELOPMENT_BINDING_READY",
        "source_contract_compatible": True,
        "provider_built": True,
        "projection_provider_built": True,
        "evaluation_provider_built": True,
        "native_runner_selected": type(runners[0]).__name__,
        "prior_census_sha256": census.fingerprint(),
        "native_source_sha256": source.fingerprint(),
        "native_tasks_planned": len(provider.invocations),
        "future_streams_per_unit": 2,
        "native_task_budget": 640,
        "native_updates_budget": 751104,
    }


def check_finite_stage_input(
    config: FiniteResponseLawStageInput,
    *,
    source_root: Path | None,
    plan: Path | None,
    prior_exposure: Path | None,
    assignment: FiniteResponseLawCohortAssignment | None,
    parent_manifest: Path | None,
    custody: Path | None,
    reveal_record: Path | None,
    analysis_record: Path | None,
    authority_store=None,
    authoring_input: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Select the real factory and stop at the first missing safe input gate."""

    if source_root is None:
        raise ValueError("FINITE_RESPONSE_LAW_SOURCE_ROOT_REQUIRED")
    if plan is None:
        raise ValueError("FINITE_RESPONSE_LAW_PLAN_REQUIRED")
    if prior_exposure is None:
        raise ValueError("FINITE_RESPONSE_LAW_PRIOR_EXPOSURE_REQUIRED")
    plan_sha = sha256(_read_plan(plan)).hexdigest()
    if plan_sha != FiniteResponseLawScienceSpec().plan_sha256:
        raise ValueError("FINITE_RESPONSE_LAW_PLAN_MISMATCH: plan differs from frozen science")
    prior_raw, _ = _read_external(source_root, prior_exposure, 128 * 1024**2)
    excluded_units, excluded_seeds = _census(prior_raw)
    binding_id, factory_name, ports = STAGE_BINDINGS[config.stage]
    factory = (
        None
        if config.stage == "informative-composition"
        else EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.factory(binding_id)
    )
    if factory is not None and type(factory).__name__ != factory_name:
        raise ValueError("FINITE_RESPONSE_LAW_REGISTERED_BINDING_MISMATCH")
    report: dict[str, object] = {
        "config_id": config.config_id,
        "stage": config.stage,
        "target_prefix": config.target_prefix,
        "evidence_role": config.evidence_role,
        "binding_id": binding_id,
        "binding_selected": factory_name,
        "missing_port_keys": list(ports),
        "provider_built": False,
        "native_plan_sha256": plan_sha,
        "prior_exposure_sha256": sha256(prior_raw).hexdigest(),
        "prior_custody_authenticated": False,
        "parent_custody_authenticated": False,
        "target_reveal_authorized": False,
        "target_analysis_authorized": False,
        "independent_units": 0,
        "nested_views_per_unit": 2,
        "native_tasks_planned": 0,
        "native_tasks_executed": 0,
        "analysis_tasks_executed": 0,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }
    assigned_stage = (
        "prospective-evaluation" if config.stage in ("prospective-evaluation", "prospective-continuation") else "calibration"
    )
    if config.stage in ("calibration", "prospective-evaluation", "prospective-continuation"):
        if assignment is None:
            raise ValueError("FINITE_RESPONSE_LAW_ASSIGNMENT_REQUIRED")
        if (
            assignment.stage != assigned_stage
            or assignment.science_plan_sha256 != plan_sha
        ):
            raise ValueError("FINITE_RESPONSE_LAW_ASSIGNMENT_MISMATCH: stage or frozen plan")
        units = assignment.physical_unit_ids
        seeds = assigned_native_seed_ids(assignment)
        unit_collisions = set(units) & set(excluded_units)
        seed_collisions = (
            set(seeds) & set(excluded_seeds)
            if config.stage == "prospective-continuation"
            else native_seed_collisions(seeds, excluded_seeds)
        )
        if config.stage == "prospective-continuation":
            if unit_collisions != set(units) or seed_collisions != set(seeds):
                raise ValueError(
                    "FINITE_RESPONSE_LAW_RETAINED_ROSTER_REQUIRED: original independent evaluation units and streams"
                )
        elif unit_collisions or seed_collisions:
            raise ValueError(
                "FINITE_RESPONSE_LAW_EXPOSED_ASSIGNED_ROSTER: "
                f"{len(unit_collisions)} units, {len(seed_collisions)} streams"
            )
        report.update(
            assignment_sha256=assignment.fingerprint(),
            cohort_namespace=assignment.cohort_namespace,
            independent_units=len(units),
        )
        report[
            "streams_reused"
            if config.stage == "prospective-continuation"
            else "streams_reserved"
        ] = len(seeds)
        if config.stage == "calibration":
            report.update(
                _build_assigned_calibration(
                    config,
                    assignment,
                    plan_sha,
                    sha256(prior_raw).hexdigest(),
                    excluded_units,
                    excluded_seeds,
                )
            )
            return report
    if parent_manifest is None:
        raise ValueError("FINITE_RESPONSE_LAW_PARENT_MANIFEST_REQUIRED")
    if custody is None:
        raise ValueError("FINITE_RESPONSE_LAW_TARGET_CUSTODY_REQUIRED")
    if reveal_record is None:
        raise ValueError("FINITE_RESPONSE_LAW_TARGET_REVEAL_REQUIRED")
    if analysis_record is None:
        raise ValueError("FINITE_RESPONSE_LAW_TARGET_ANALYSIS_REQUIRED")
    if authority_store is None:
        raise ValueError("FINITE_RESPONSE_LAW_TARGET_CUSTODY_STORE_REQUIRED")
    from empirical_lawhood.adapters.composition.response_parent_reader import read_authenticated_parent
    from empirical_lawhood.adapters.composition.prepared_response.native_authoring import _implementation_digest
    from empirical_lawhood.kernel.decoding import decode_canonical_bytes

    from .consumer_input import FiniteResponseLawAuthoringInput, build_consumer_authoring, validate_parent_join
    from .consumer_ports import consumer_ports

    route = {
        "supplemental-development": "finite-response-law-supplemental-development",
        "informative-composition": "finite-response-law-informative-composition",
        "calibration-method": "finite-response-law-calibration-method",
        "prospective-evaluation": "finite-response-law-prospective-parent",
        "prospective-continuation": "finite-response-law-prospective-parent",
        "preparation-screening": "finite-response-law-preparation-screening",
    }[config.stage]
    primary = read_authenticated_parent(
        route=route,
        source_root=source_root,
        parent_manifest=parent_manifest,
        custody=custody,
        reveal_record=reveal_record,
        analysis_record=analysis_record,
        authority_store=authority_store,
    )
    if config.stage == "informative-composition":
        from .informative_composition_input import check_informative_composition_parent

        report.update(check_informative_composition_parent(primary, expected_plan_sha256=plan_sha))
        return report
    if authoring_input is None:
        raise ValueError("FINITE_RESPONSE_LAW_CONSUMER_INPUT_REQUIRED")
    if repo_root is None:
        raise ValueError("FINITE_RESPONSE_LAW_PROJECT_ROOT_REQUIRED")
    raw, _ = _read_external(source_root, authoring_input, 64 * 1024**2)
    packet = decode_canonical_bytes(
        raw, FiniteResponseLawAuthoringInput, maximum_bytes=64 * 1024**2
    )
    if packet.stage != config.stage or packet.source.science.plan_sha256 != plan_sha:
        raise ValueError("FINITE_RESPONSE_LAW_CONSUMER_SELECTION_MISMATCH")
    if assignment is not None and (
        getattr(packet.source, "cohort_namespace", None) != assignment.cohort_namespace
        or getattr(packet.source, "assignment_sha256", None) != assignment.fingerprint()
        or getattr(packet.source, "scientific_seeds", None) != assignment.scientific_seeds
        or tuple(root.physical_unit_id for root in packet.source.roots)
        != assignment.physical_unit_ids
    ):
        raise ValueError("FINITE_RESPONSE_LAW_CONSUMER_ASSIGNMENT_MISMATCH")
    parents = [primary]
    manifests = [parent_manifest]
    for additional in packet.additional_parents:
        parents.append(
            read_authenticated_parent(
                route=route,
                source_root=Path(additional.source_root),
                parent_manifest=Path(additional.parent_manifest),
                custody=Path(additional.custody),
                reveal_record=Path(additional.reveal_record),
                analysis_record=Path(additional.analysis_record),
                authority_store=authority_store,
            )
        )
        manifests.append(Path(additional.parent_manifest))
    validate_parent_join(packet, tuple(parents))
    exposure = ObjectIdentity.from_record(
        f"{config.target_prefix}.consumer-input", packet
    )
    bundle = build_consumer_authoring(
        packet,
        implementation_sha256=_implementation_digest(repo_root),
        exposure=exposure,
        current_units=excluded_units,
        current_seeds=excluded_seeds,
    )
    plane = getattr(authority_store, "plane", None)
    if plane is None:
        raise ValueError("FINITE_RESPONSE_LAW_TARGET_ARTIFACT_PLANE_REQUIRED")
    available = consumer_ports(
        packet=packet, bundle=bundle, plane=plane, manifests=tuple(manifests)
    )
    owners = inspect_consumer_providers(bundle, available)
    if packet.completion is not None:
        from empirical_lawhood.adapters.methods.finite_response_law.completion.executable_binding import ASSIGNED_BINDING

        binding_id = ASSIGNED_BINDING.binding_id
    selected = next((row for row in owners if row["binding_id"] == binding_id), None)
    if selected is None:
        raise ValueError("FINITE_RESPONSE_LAW_SELECTED_STAGE_PROVIDER_MISSING")
    report.update(
        status="FINITE_RESPONSE_LAW_DEVELOPMENT_BINDING_READY",
        binding_id=selected["binding_id"],
        binding_selected=selected["factory"],
        provider_built=True,
        missing_port_keys=[],
        parent_custody_authenticated=True,
        target_reveal_authorized=True,
        target_analysis_authorized=True,
        primary_parent_sha256=primary.grant.parent.physical_sha256,
        parent_custody_sha256s=[parent.custody.fingerprint() for parent in parents],
        authoring_input_sha256=packet.fingerprint(),
        native_source_sha256=packet.source.fingerprint(),
        independent_units=len(packet.source.roots),
        native_tasks_planned=selected["native_tasks_planned"],
        selected_providers=owners,
        prospective_issue_eligible=False,
        nonacquiring_completion=packet.completion is not None,
        protocol_steps_planned=len(
            bundle.standard_context.base.templates[0].protocol.steps
        ),
    )
    return report


def inspect_consumer_providers(bundle, platform_ports) -> list[dict[str, object]]:
    """Resolve every retained owner selected by the builder's actual graph."""
    from dataclasses import asdict

    registry = bundle.standard_context.base.registry
    factories = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    capabilities = {
        step.capability_key
        for step in bundle.standard_context.base.templates[0].protocol.steps
    }
    owners = []
    for capability in sorted(capabilities):
        matching = [
            binding
            for binding in factories.aggregate.bindings
            if binding.capability_key == capability
        ]
        if len(matching) != 1:
            raise ValueError("FINITE_RESPONSE_LAW_STAGE_CAPABILITY_BINDING_AMBIGUOUS")
        binding = matching[0]
        factory = factories.provider_factory(binding.binding_id)
        schemas = {value.record_schema for value in binding.accepted_config_types}
        records = tuple(value for value in bundle.payloads if value.SCHEMA in schemas)
        ports = tuple(
            port
            for port in platform_ports
            if port.port_key in binding.required_platform_port_keys
        )
        provider = factory.build_provider(
            registry=registry, records=records, platform_ports=ports
        )
        runners = provider.runners(registry)
        if len(runners) != 1 or runners[0].manifest.capability_key != capability:
            raise ValueError("FINITE_RESPONSE_LAW_STAGE_RUNNER_MISMATCH")
        semantics = provider.output_semantic_contracts(registry)
        if {contract.payload_schema for contract in semantics} != set(
            runners[0].manifest.output_schema_ids
        ):
            raise ValueError("FINITE_RESPONSE_LAW_STAGE_OUTPUT_CONTRACT_MISMATCH")
        owners.append(
            {
                "binding_id": binding.binding_id,
                "factory": type(factory).__name__,
                "runner": type(runners[0]).__name__,
                "output_schema_ids": list(runners[0].manifest.output_schema_ids),
                "output_semantic_contracts": [asdict(contract) for contract in semantics],
                "provider_built": True,
                "native_tasks_planned": len(getattr(provider, "invocations", ())),
            }
        )
    return owners


__all__ = ["STAGE_BINDINGS", 'FiniteResponseLawStageInput', "check_finite_stage_input"]
