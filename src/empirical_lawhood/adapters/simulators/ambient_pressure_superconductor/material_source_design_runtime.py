'Receipt-first ambient pressure superconductor material source design source qualification and design-basis audit.'

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
from typing import Any, TypeVar, cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
    encode_scientific_adjudication,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .material_source_design_contracts import MATERIAL_SOURCE_DESIGN_ADJUDICATION_SCHEMA, MATERIAL_SOURCE_DESIGN_CAPABILITY_VERSION, MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA, MATERIAL_SOURCE_DESIGN_CONFIG_VERSION, MATERIAL_SOURCE_DESIGN_DESIGN_CAPABILITY_KEY, MATERIAL_SOURCE_DESIGN_EVALUATOR_CAPABILITY_KEY, MATERIAL_SOURCE_DESIGN_SOURCE_CAPABILITY_KEY, MaterialSourceDesignConfig, MaterialSourceDesignConfigLifecycle, MaterialSourceDesignDesignBasisAudit, MaterialSourceDesignDisposition, MaterialSourceDesignResult, MaterialSourceDesignSourceQualification, ExplorationDesignFreeze, MaterialRosterFreeze, ScienceDesignFreeze, SplitPartition, decode_material_source_design_config_bytes
from .material_source_design_design import build_exploration_design, build_material_roster, build_science_design
from .material_source_design_source import expected_material_source_design_source_qualification, inspect_material_source_design_external_sources
from .material_source_design_system import task_budget


CONFIG_ARTIFACT_ID = 'config-artifact.ambient-pressure-superconductor-material-source-design'
SOURCE_CONTRACT_ARTIFACT_ID = 'source-contract.ambient-pressure-superconductor-material-source-design-public-roster'
SOURCE_STEP = 'material-source-design-source-qualification'
DESIGN_STEP = 'material-source-design-design-basis-audit'
EVALUATOR_STEP = 'material-source-design-result-evaluator'


def material_source_design_config_ref(config: MaterialSourceDesignConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id='config.ambient-pressure-superconductor-material-source-design-source-design-audit',
        config_schema=MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA,
        config_schema_sha256=sha256(MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA.encode()).hexdigest(),
        content_sha256=config.payload_sha256,
        artifact_id=CONFIG_ARTIFACT_ID,
    )


def material_source_design_protocol(*, registry: CapabilityRegistry, config: MaterialSourceDesignConfig) -> ProtocolTemplate:
    reference = material_source_design_config_ref(config)

    def step(
        *,
        step_id: str,
        stage: ScientificStage,
        capability_key: str,
        dependencies: tuple[str, ...],
        outputs: tuple[tuple[str, str], ...],
        outcome_access: OutcomeAccess,
        visibility: VisibilityCeiling,
        barrier: BarrierKind,
        obligations: tuple[str, ...],
    ) -> ProtocolStepTemplate:
        manifest = registry.resolve(capability_key, MATERIAL_SOURCE_DESIGN_CAPABILITY_VERSION)
        return ProtocolStepTemplate(
            step_id=step_id,
            stage=stage,
            capability_key=capability_key,
            capability_version=MATERIAL_SOURCE_DESIGN_CAPABILITY_VERSION,
            config=reference,
            dependency_step_ids=tuple(sorted(dependencies)),
            outputs=tuple(
                sorted(
                    (
                        OutputTemplate(
                            output_id=output_id,
                            payload_schema=schema,
                            profile=ArtifactProfile.CANONICAL_JSON,
                            media_type="application/json",
                            filename_suffix=".json",
                        )
                        for output_id, schema in outputs
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=manifest.permissions,
            requested_outcome_access=outcome_access,
            visibility_ceiling=visibility,
            resource_budget=task_budget(config),
            resource_lock_ids=('ambient-pressure-superconductor-material-source-design-external-source-scan',),
            barrier=barrier,
            maximum_attempts=1,
            obligation_ids=tuple(sorted(obligations)),
        )

    steps = (
        step(
            step_id=SOURCE_STEP,
            stage=ScientificStage.QUALIFY,
            capability_key=MATERIAL_SOURCE_DESIGN_SOURCE_CAPABILITY_KEY,
            dependencies=(),
            outputs=(("source-qualification", MaterialSourceDesignSourceQualification.SCHEMA),),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
            obligations=('material-source-design-exact-public-source-qualification',),
        ),
        step(
            step_id=DESIGN_STEP,
            stage=ScientificStage.FREEZE,
            capability_key=MATERIAL_SOURCE_DESIGN_DESIGN_CAPABILITY_KEY,
            dependencies=(SOURCE_STEP,),
            outputs=(
                ("design-basis-audit", MaterialSourceDesignDesignBasisAudit.SCHEMA),
                ("exploration-design-candidate", ExplorationDesignFreeze.SCHEMA),
                ("material-roster-candidate", MaterialRosterFreeze.SCHEMA),
                ("science-design-candidate", ScienceDesignFreeze.SCHEMA),
            ),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.FREEZE,
            obligations=(
                'material-source-design-candidate-design-components-closed',
                'material-source-design-executable-development-provider-audit',
                'material-source-design-target-contact-count-zero',
            ),
        ),
        step(
            step_id=EVALUATOR_STEP,
            stage=ScientificStage.EVALUATE,
            capability_key=MATERIAL_SOURCE_DESIGN_EVALUATOR_CAPABILITY_KEY,
            dependencies=(SOURCE_STEP, DESIGN_STEP),
            outputs=(
                ('material-source-design-result', MaterialSourceDesignResult.SCHEMA),
                ("scientific-adjudication", MATERIAL_SOURCE_DESIGN_ADJUDICATION_SCHEMA),
            ),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.REVEAL,
            obligations=('material-source-design-source-and-design-intersection-adjudication',),
        ),
    )
    return ProtocolTemplate(
        template_id='protocol.ambient-pressure-superconductor-material-source-design-source-design-audit-audit',
        template_version=MATERIAL_SOURCE_DESIGN_CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=True,
        requests_controller=False,
        nonactuating=True,
    )


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _one(records: tuple[CanonicalRecord, ...], record_type: type[_RecordT]) -> _RecordT:
    values = tuple(value for value in records if isinstance(value, record_type))
    if len(values) != 1:
        raise ValueError(f'material source design task requires exactly one {record_type.__name__}')
    return values[0]


_INPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    MaterialSourceDesignSourceQualification.SCHEMA: MaterialSourceDesignSourceQualification,
    MaterialSourceDesignDesignBasisAudit.SCHEMA: MaterialSourceDesignDesignBasisAudit,
    ExplorationDesignFreeze.SCHEMA: ExplorationDesignFreeze,
    MaterialRosterFreeze.SCHEMA: MaterialRosterFreeze,
    ScienceDesignFreeze.SCHEMA: ScienceDesignFreeze,
}


def _build_audit(
    *,
    config: MaterialSourceDesignConfig,
    source: MaterialSourceDesignSourceQualification,
    roster: MaterialRosterFreeze,
    exploration: ExplorationDesignFreeze,
    science: ScienceDesignFreeze,
) -> MaterialSourceDesignDesignBasisAudit:
    if source.fingerprint() != config.source_qualification_sha256:
        raise ValueError('material source design source qualification differs from config')
    expected = (
        (roster.fingerprint(), config.material_roster_sha256, "material roster"),
        (exploration.fingerprint(), config.exploration_design_sha256, "exploration design"),
        (science.fingerprint(), config.science_design_sha256, "science design"),
    )
    for observed, configured, label in expected:
        if observed != configured:
            raise ValueError(f'material source design {label} differs from config')
    return MaterialSourceDesignDesignBasisAudit(
        audit_id='audit.ambient-pressure-superconductor-material-source-design-design-basis',
        config_sha256=config.payload_sha256,
        source_qualification_sha256=source.fingerprint(),
        material_roster_sha256=roster.fingerprint(),
        exploration_design_sha256=exploration.fingerprint(),
        science_design_sha256=science.fingerprint(),
        intended_development_protocol_id=config.development_protocol_id,
        target_contact_count=0,
        source_qualified=True,
        candidate_components_closed=True,
        full_development_template_present=False,
        exact_provider_set_present=False,
        missing_capability_ids=tuple(
            sorted(
                (
                    'capability.ambient-pressure-superconductor-material-control-control-and-science-freeze',
                    'capability.ambient-pressure-superconductor-response-guided-exploration-matched-wave-executor',
                    'capability.ambient-pressure-superconductor-transport-nomination-transport-admission-compiler',
                    'capability.full-material-source-design-transport-nomination-development-template',
                    "capability.material-specific-gauge-closed-transverse-producer",
                )
            )
        ),
        reason_codes=tuple(
            sorted(
                (
                    'material-source-design-full-development-template-not-implemented',
                    'material-source-design-material-gauge-covariant-response-transverse-producer-not-registered',
                    'material-source-design-provider-set-incomplete',
                    "design-basis-freeze-prohibited",
                )
            )
        ),
    )


def _adjudicate(
    *,
    config: MaterialSourceDesignConfig,
    source: MaterialSourceDesignSourceQualification,
    audit: MaterialSourceDesignDesignBasisAudit,
    roster: MaterialRosterFreeze,
    exploration: ExplorationDesignFreeze,
    science: ScienceDesignFreeze,
) -> MaterialSourceDesignResult:
    if audit.config_sha256 != config.payload_sha256:
        raise ValueError('material source design audit config differs')
    if audit.source_qualification_sha256 != source.fingerprint():
        raise ValueError('material source design audit source differs')
    if audit.material_roster_sha256 != roster.fingerprint():
        raise ValueError('material source design audit roster differs')
    if audit.exploration_design_sha256 != exploration.fingerprint():
        raise ValueError('material source design audit exploration design differs')
    if audit.science_design_sha256 != science.fingerprint():
        raise ValueError('material source design audit science design differs')
    development_families = sum(family.partition is SplitPartition.DEVELOPMENT_ATLAS for family in roster.families)
    prospective_families = sum(
        family.partition is SplitPartition.SEALED_PROSPECTIVE for family in roster.families
    )
    return MaterialSourceDesignResult(
        result_id='result.ambient-pressure-superconductor-material-source-design-source-design-audit',
        config_sha256=config.payload_sha256,
        source_qualification_sha256=source.fingerprint(),
        design_basis_record_sha256=audit.fingerprint(),
        disposition=MaterialSourceDesignDisposition.DESIGN_NOT_FROZEN,
        source_qualified=True,
        design_basis_frozen=False,
        target_contact_count=0,
        calibration_structure_count=len(roster.calibration_structure_ids),
        development_family_count=development_families,
        development_initial_action_count=len(roster.development_initial_action_ids),
        development_eligible_wave_action_count=len(roster.development_eligible_wave_action_ids),
        prospective_family_count=prospective_families,
        prospective_action_count=len(roster.prospective_action_ids),
        truth_world_count=len(exploration.truth_worlds),
        policy_count=len(exploration.policy_specs),
        calibration_derived_field_count=len(science.only_calibration_derived_ids),
        reason_codes=audit.reason_codes,
    )


class MaterialSourceDesignRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: MaterialSourceDesignConfig,
        payload: bytes,
        source_contract: MaterialSourceDesignSourceQualification,
    ) -> None:
        if decode_material_source_design_config_bytes(payload) != config:
            raise ValueError('material source design runner config differs from registered bytes')
        self.manifest = manifest
        self.config = config
        self.payload = payload
        self.source_contract = source_contract
        self.execution_count = 0

    def _read(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        records: list[CanonicalRecord] = []
        observed_config = False
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError('material source design task input size differs')
            if port.payload_schema == MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA:
                if payload != self.payload or decode_material_source_design_config_bytes(payload) != self.config:
                    raise ValueError('material source design task config materialization differs')
                observed_config = True
                continue
            try:
                record_type = _INPUT_TYPES[port.payload_schema]
            except KeyError as error:
                raise ValueError('material source design task received an unknown input schema') from error
            records.append(
                decode_canonical_bytes(payload, record_type, maximum_bytes=port.size_bytes)
            )
        if not observed_config or context.config.content_sha256 != self.config.payload_sha256:
            raise ValueError('material source design task lacks its exact config')
        return tuple(records)

    @staticmethod
    def _scientific_adjudication(
        context: TaskContext, result: MaterialSourceDesignResult
    ) -> ScientificAdjudicationRecord:
        adjudication_context = context.scientific_adjudication_context
        if adjudication_context is None:
            raise ValueError('material source design evaluator lacks scientific adjudication context')
        output_ids = tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
            )
        )
        if len(output_ids) != len(context.output_ports):
            raise ValueError('material source design evaluator output lacks logical identity')
        return ScientificAdjudicationRecord(
            adjudication_id=f'adjudication.{context.run_id}.{context.task_id}',
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication_context.execution_plan,
            input_materialization_ids=tuple(sorted(context.input_materialization_ids)),
            output_logical_artifact_ids=output_ids,
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication_context.evidence_world_id,
            evidence_world_kind=adjudication_context.evidence_world_kind,
            relation=adjudication_context.relation,
            independent_unit_id=adjudication_context.independent_unit_id,
            information_cutoffs=adjudication_context.information_cutoffs,
            visibility_ceiling=adjudication_context.visibility_ceiling,
            outcome_access=adjudication_context.outcome_access,
            evaluability=AdjudicationEvaluability.EVALUABLE,
            scientific_status=ScientificStatus.NOT_SUPPORTED,
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=result.reason_codes,
            fixture_scope_id=None,
            plumbing_only=False,
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        if self.config.lifecycle is not MaterialSourceDesignConfigLifecycle.FROZEN:
            raise ValueError('material source design execution requires a frozen source design audit audit config')
        self.execution_count += 1
        records = self._read(context)
        outputs: dict[str, CanonicalRecord | bytes]
        if context.task_id == SOURCE_STEP:
            contract = _one(records, MaterialSourceDesignSourceQualification)
            if contract != self.source_contract:
                raise ValueError('material source design source contract differs')
            observed = inspect_material_source_design_external_sources()
            if observed != contract:
                raise ValueError('material source design physical source qualification differs')
            outputs = {"source-qualification": observed}
        elif context.task_id == DESIGN_STEP:
            source = _one(records, MaterialSourceDesignSourceQualification)
            roster = build_material_roster()
            exploration = build_exploration_design(roster)
            science = build_science_design()
            audit = _build_audit(
                config=self.config,
                source=source,
                roster=roster,
                exploration=exploration,
                science=science,
            )
            outputs = {
                "design-basis-audit": audit,
                "exploration-design-candidate": exploration,
                "material-roster-candidate": roster,
                "science-design-candidate": science,
            }
        elif context.task_id == EVALUATOR_STEP:
            result = _adjudicate(
                config=self.config,
                source=_one(records, MaterialSourceDesignSourceQualification),
                audit=_one(records, MaterialSourceDesignDesignBasisAudit),
                roster=_one(records, MaterialRosterFreeze),
                exploration=_one(records, ExplorationDesignFreeze),
                science=_one(records, ScienceDesignFreeze),
            )
            outputs = {
                'material-source-design-result': result,
                "scientific-adjudication": encode_scientific_adjudication(
                    self._scientific_adjudication(context, result),
                    payload_schema=MATERIAL_SOURCE_DESIGN_ADJUDICATION_SCHEMA,
                ),
            }
        else:
            raise ValueError('unknown material source design task identity')
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=(
                        value
                        if isinstance(
                            value := outputs[port.output_id.removeprefix(f'{context.task_id}.')],
                            bytes,
                        )
                        else value.canonical_bytes()
                    ),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("exact-config-decoded", True, ()),
                ReceiptCheck("external-source-root-only", True, ()),
                ReceiptCheck("target-contact-count-zero", True, ()),
                ReceiptCheck("typed-output-produced", True, ()),
            ),
        )


_OUTPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    MaterialSourceDesignDesignBasisAudit.SCHEMA: MaterialSourceDesignDesignBasisAudit,
    MaterialSourceDesignResult.SCHEMA: MaterialSourceDesignResult,
    MaterialSourceDesignSourceQualification.SCHEMA: MaterialSourceDesignSourceQualification,
    ExplorationDesignFreeze.SCHEMA: ExplorationDesignFreeze,
    MaterialRosterFreeze.SCHEMA: MaterialRosterFreeze,
    ScienceDesignFreeze.SCHEMA: ScienceDesignFreeze,
}


class MaterialSourceDesignRuntimeProvider(CampaignRuntimeProvider):
    issued_source_schema_ids: tuple[str, ...] = (StudyOperationAuthority.SCHEMA,)

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: MaterialSourceDesignConfig,
        config_payload: bytes,
        source_contract: MaterialSourceDesignSourceQualification,
    ) -> None:
        if any(
            manifest.implementation_sha256 != registry.capabilities[0].implementation_sha256
            for manifest in registry.capabilities
        ):
            raise ValueError('material source design registry mixes implementation identities')
        if decode_material_source_design_config_bytes(config_payload) != config:
            raise ValueError('material source design provider config differs from registered bytes')
        if source_contract != expected_material_source_design_source_qualification():
            raise ValueError('material source design provider source-contract binding differs')
        self.registry = registry
        self.config = config
        self.config_payload = config_payload
        self.source_contract = source_contract
        self.registry_sha256 = registry.fingerprint()
        self._runners: tuple[MaterialSourceDesignRunner, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        del source_records
        if registry != self.registry:
            raise ValueError('material source design runner registry differs')
        self._runners = tuple(
            MaterialSourceDesignRunner(manifest, self.config, self.config_payload, self.source_contract)
            for manifest in registry.capabilities
        )
        return cast(tuple[TaskRunner, ...], self._runners)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        del source_records
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError('material source design execution plan registry differs')
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        expected = {CONFIG_ARTIFACT_ID, SOURCE_CONTRACT_ARTIFACT_ID}
        if set(specs) != expected:
            raise ValueError('material source design plan external input set differs')

        def payload(
            logical_artifact_id: str,
            *,
            payload_schema: str,
            content: bytes,
            object_id: str,
            object_version: str,
        ) -> ExternalInputPayload:
            spec = specs[logical_artifact_id]
            parent = ArtifactLineageParent(
                identity=ObjectIdentity(
                    object_id=object_id,
                    object_schema=payload_schema,
                    object_version=object_version,
                    object_fingerprint=sha256(content).hexdigest(),
                ),
                visibility_ceiling=(
                    spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                ),
                outcome_access=spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND,
            )
            return ExternalInputPayload.from_bytes(
                logical_artifact_id=logical_artifact_id,
                payload_schema=payload_schema,
                profile=ArtifactProfile.TEXT_PARAMETERS,
                media_type="application/json",
                payload=content,
                visibility_ceiling=parent.visibility_ceiling,
                outcome_access=parent.outcome_access,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=sha256(content).hexdigest(),
            )

        return (
            payload(
                CONFIG_ARTIFACT_ID,
                payload_schema=MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA,
                content=self.config_payload,
                object_id=CONFIG_ARTIFACT_ID,
                object_version=MATERIAL_SOURCE_DESIGN_CONFIG_VERSION,
            ),
            payload(
                SOURCE_CONTRACT_ARTIFACT_ID,
                payload_schema=MaterialSourceDesignSourceQualification.SCHEMA,
                content=self.source_contract.canonical_bytes(),
                object_id=SOURCE_CONTRACT_ARTIFACT_ID,
                object_version=MaterialSourceDesignSourceQualification.VERSION,
            ),
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        del execution_plan
        if registry != self.registry:
            raise ValueError('material source design semantic registry differs')
        contracts = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                record_type = (
                    ScientificAdjudicationRecord
                    if schema == MATERIAL_SOURCE_DESIGN_ADJUDICATION_SCHEMA
                    else _OUTPUT_TYPES[schema]
                )
                contracts.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(
                            sorted(value.name for value in fields(cast(Any, record_type)))
                        ),
                    )
                )
        return tuple(sorted(contracts, key=lambda value: value.key))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract:
        del execution_plan
        if registry != self.registry:
            raise ValueError('material source design adjudication registry differs')
        return ScientificAdjudicationOutputContract(
            capability_key=MATERIAL_SOURCE_DESIGN_EVALUATOR_CAPABILITY_KEY,
            capability_version=MATERIAL_SOURCE_DESIGN_CAPABILITY_VERSION,
            output_id=f'{EVALUATOR_STEP}.scientific-adjudication',
            payload_schema=MATERIAL_SOURCE_DESIGN_ADJUDICATION_SCHEMA,
            maximum_bytes=128 * 1024,
            fixture_scope_id=None,
            plumbing_only=False,
        )


__all__ = [
    'MaterialSourceDesignRunner',
    'MaterialSourceDesignRuntimeProvider',
    "CONFIG_ARTIFACT_ID",
    "DESIGN_STEP",
    "EVALUATOR_STEP",
    "SOURCE_CONTRACT_ARTIFACT_ID",
    "SOURCE_STEP",
    'material_source_design_config_ref',
    'material_source_design_protocol',
]
