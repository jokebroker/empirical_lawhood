"Current standard authoring from measurement through controller use for the three open simulator campaigns.\n\nThis module is the shared adapter.  It owns substrate facts and\nbuilds ordinary platform records; it does not own scheduling, persistence,\nauthority, or a substrate-specific CLI.\n"

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
import hashlib
from types import MappingProxyType
from typing import ClassVar, Final, Mapping

from empirical_lawhood.adapters.methods.formal_analysis import (
    FORMAL_DOMAIN_CAPABILITY_KEYS,
    FORMAL_PANEL_CAPABILITY_KEY,
    FormalCapabilityConfigDecoder,
    formal_method_capability_manifests,
    standard_formal_method_catalog,
    standard_formal_method_configs,
)
from empirical_lawhood.adapters.methods.formal_protocol import (
    add_standard_formal_analysis,
    formal_gap_obligation_id,
)
from empirical_lawhood.kernel.admission import (
    AdmissionGateKind,
    AdmissionGateResult,
    GateStatus,
)
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import (
    ClaimSpec,
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import (
    AssignmentKind,
    AssignmentSpec,
    ControlKind,
    ControlSpec,
    ExperimentSpec,
    PrecisionGoal,
    RevealBarrierSpec,
)
from empirical_lawhood.kernel.obligations import (
    ClosureSpec,
    ComputabilityEvidence,
    FalsifierKind,
    FalsifierSpec,
    ObligationStatus,
    ScientificObligations,
    StructuralConvergenceSpec,
    SupportSpec,
    UncertaintySpec,
    ValiditySpec,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.quantities import (
    QuantityKind,
    QuantitySpec,
    ResponseDirection,
)
from empirical_lawhood.kernel.references import QuantityBound
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import (
    AdmissionStatus,
    LifecycleStatus,
    ReadinessStatus,
    ScientificStatus,
)
from empirical_lawhood.kernel.systems import (
    BalanceRole,
    IndependentUnitSpec,
    PortDirection,
    PortSpec,
    RelationalIdentity,
    SystemBoundaryKind,
    SystemSpec,
)
from empirical_lawhood.kernel.time import (
    AvailabilitySpec,
    CausalPhase,
    ClockLabelSemantics,
    ClockSpec,
    HoldSemantics,
    HorizonSpec,
    InformationCutoff,
    SamplingSemantics,
)
from empirical_lawhood.kernel.worlds import (
    ComputabilityEnvelope,
    EvidenceUnitScope,
    NumericalCoordinateKind,
    NumericalCoordinateSpec,
    NumericalViewSpec,
    RandomnessSemantics,
    WorldKind,
    WorldSpec,
)
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodCatalog,
    derive_formal_gap_applicability,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalDomain,
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.planning.formal_results import (
    FormalAnalysisInputManifest,
    FormalAnalysisSample,
    FormalDomainMethodConfig,
    FormalGapAdjudicationPanel,
    FormalPanelEvaluatorConfig,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, ConditionalGateKind, ConditionalChildRequest, ConditionalTerminalDisposition, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationReport, CandidateCompilationContext, CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole, StudyCompilationReport, required_candidate_obligation_ids
from empirical_lawhood.runtime.conditional_children import ConditionalChildInstantiation, ConditionalChildResolution, bind_frozen_parent_input
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.source_resolution import (
    CandidateCapabilityConfigDecoder,
    SourceMaterializationConfig,
    SourceReadMode,
)


class OpenSimulatorKind(StrEnum):
    PYBAMM = "PYBAMM"
    GYM_TORAX = "GYM_TORAX"


@dataclass(frozen=True, slots=True)
class OpenSimulatorActionDeliveryBinding(CanonicalRecord):
    """One abstract action coordinate bound to exact native delivery points."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/open-simulator-action-delivery-binding'

    coordinate_id: str
    native_point_ids: tuple[str, ...]
    native_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        require_sorted_unique_strings(
            self.native_point_ids,
            field_name="native_point_ids",
            allow_empty=False,
        )
        if not self.native_unit:
            raise ValueError("native action delivery unit must be nonempty")


@dataclass(frozen=True, slots=True)
class OpenSimulatorSourceManifest(CanonicalRecord):
    """Compact exact source closure; software payload bytes remain external."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/open-simulator-source-manifest'

    source_id: str
    substrate: OpenSimulatorKind
    release_ids: tuple[str, ...]
    version_bindings: tuple[str, ...]
    held_payload_sha256: tuple[str, ...]
    repository_commit: str | None
    runtime_identity_ids: tuple[str, ...]
    runtime_payload_sha256: tuple[str, ...]
    source_qualification_check_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.source_id, field_name="source_id")
        for name, values in (
            ("release_ids", self.release_ids),
            ("version_bindings", self.version_bindings),
            ("held_payload_sha256", self.held_payload_sha256),
            ("runtime_identity_ids", self.runtime_identity_ids),
            ("runtime_payload_sha256", self.runtime_payload_sha256),
            ("source_qualification_check_ids", self.source_qualification_check_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        for value in self.held_payload_sha256:
            validate_sha256(value, field_name="held_payload_sha256")
        for value in self.runtime_payload_sha256:
            validate_sha256(value, field_name="runtime_payload_sha256")
        if self.repository_commit is not None and (
            len(self.repository_commit) != 40
            or any(value not in "0123456789abcdef" for value in self.repository_commit)
        ):
            raise ValueError("repository_commit must be a lowercase Git SHA-1")


@dataclass(frozen=True, slots=True)
class OpenSimulatorStudyConfig(CanonicalRecord):
    """Closed substrate and protocol configuration consumed by all stage runners."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/open-simulator-study-config'

    config_id: str
    substrate: OpenSimulatorKind
    campaign_definition_id: str
    source_manifest: ObjectIdentity
    simulator_versions: tuple[str, ...]
    model_id: str
    parameterization_ids: tuple[str, ...]
    runtime_identity_ids: tuple[str, ...]
    action_coordinate_ids: tuple[str, ...]
    action_delivery_bindings: tuple[OpenSimulatorActionDeliveryBinding, ...]
    action_stage_ids: tuple[str, ...]
    receiver_coordinate_ids: tuple[str, ...]
    state_coordinate_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    development_unit_ids: tuple[str, ...]
    evaluation_unit_ids: tuple[str, ...]
    prospective_unit_ids: tuple[str, ...]
    horizon_steps: int
    horizon_seconds: Decimal
    assurance_profile_id: str
    loopback_service_id: str | None
    exact_publication_reproduction: bool
    controller_use_requires_fresh_issue: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("config_id", self.config_id),
            ("campaign_definition_id", self.campaign_definition_id),
            ("model_id", self.model_id),
            ("assurance_profile_id", self.assurance_profile_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.source_manifest.object_schema != OpenSimulatorSourceManifest.SCHEMA:
            raise ValueError("open simulator config binds another source-manifest schema")
        for name, values in (
            ("simulator_versions", self.simulator_versions),
            ("parameterization_ids", self.parameterization_ids),
            ("runtime_identity_ids", self.runtime_identity_ids),
            ("action_coordinate_ids", self.action_coordinate_ids),
            ("action_stage_ids", self.action_stage_ids),
            ("receiver_coordinate_ids", self.receiver_coordinate_ids),
            ("state_coordinate_ids", self.state_coordinate_ids),
            ("clock_ids", self.clock_ids),
            ("numerical_view_ids", self.numerical_view_ids),
            ("development_unit_ids", self.development_unit_ids),
            ("evaluation_unit_ids", self.evaluation_unit_ids),
            ("prospective_unit_ids", self.prospective_unit_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_ids(
            self.action_delivery_bindings,
            attribute="coordinate_id",
            field_name="action_delivery_bindings",
        )
        if tuple(value.coordinate_id for value in self.action_delivery_bindings) != (
            self.action_coordinate_ids
        ):
            raise ValueError(
                "every action coordinate must bind exactly one native delivery definition"
            )
        if self.horizon_steps <= 0 or self.horizon_seconds <= 0:
            raise ValueError("open simulator horizon must be positive")
        if self.loopback_service_id is not None:
            validate_stable_id(self.loopback_service_id, field_name="loopback_service_id")
        if self.loopback_service_id is not None:
            raise ValueError("retained open simulators do not bind a loopback service")
        if self.exact_publication_reproduction:
            raise ValueError("current open campaigns cannot claim exact publication reproduction")
        if not self.controller_use_requires_fresh_issue:
            raise ValueError("Controller use must remain a separately issued fresh act")


@dataclass(frozen=True, slots=True)
class OpenSimulatorStageRecord(CanonicalRecord):
    """Compact stage envelope; scientific arrays remain in formal input records."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/open-simulator-stage-record'

    record_id: str
    substrate: OpenSimulatorKind
    stage_id: str
    config: ObjectIdentity
    input_materialization_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    admission_noncompensating_gate_ids: tuple[str, ...]
    mandatory_hold_outside_admission: bool
    controller_use_executed: bool

    def __post_init__(self) -> None:
        for name, value in (("record_id", self.record_id), ("stage_id", self.stage_id)):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.input_materialization_ids,
            field_name="input_materialization_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.admission_noncompensating_gate_ids,
            field_name="admission_noncompensating_gate_ids",
        )
        if self.controller_use_executed and not self.stage_id.startswith("prospective-controller-"):
            raise ValueError("only an exact conditional controller use stage may report controller use execution")


@dataclass(frozen=True, slots=True)
class OpenSimulatorQuantityBand(CanonicalRecord):
    "One native-unit, pre-outcome preservation constraint for admission and controller use."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/open-simulator-quantity-band'

    constraint_id: str
    quantity_id: str
    native_unit: str
    lower_bound: Decimal | None
    upper_bound: Decimal | None

    def __post_init__(self) -> None:
        validate_stable_id(self.constraint_id, field_name="constraint_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("quantity band requires at least one native bound")
        if self.lower_bound is not None:
            validate_decimal(self.lower_bound, field_name="lower_bound")
        if self.upper_bound is not None:
            validate_decimal(self.upper_bound, field_name="upper_bound")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("quantity band lower bound exceeds upper bound")


@dataclass(frozen=True, slots=True)
class OpenSimulatorAdmissionPolicy(CanonicalRecord):
    "Frozen, substrate-local policy for the noncompensating admission intersection."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/open-simulator-admission-policy'

    policy_id: str
    substrate: OpenSimulatorKind
    candidate_action_word_ids: tuple[str, ...]
    primary_target_quantity_id: str
    primary_target_direction: ResponseDirection
    minimum_target_improvement: Decimal
    target_native_unit: str
    preservation_bands: tuple[OpenSimulatorQuantityBand, ...]
    maximum_sink_relative_increase: Decimal
    maximum_effort_relative_increase: Decimal
    maximum_normalized_standard_error: Decimal
    maximum_normalized_view_disagreement: Decimal
    minimum_response_rank: int
    required_dynamics_gap_ids: tuple[str, ...]
    required_preservation_gap_ids: tuple[str, ...]
    required_reachability_gap_ids: tuple[str, ...]
    authority_policy_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.policy_id, field_name="policy_id")
        validate_stable_id(
            self.primary_target_quantity_id,
            field_name="primary_target_quantity_id",
        )
        validate_stable_id(self.authority_policy_id, field_name="authority_policy_id")
        validate_nonempty(self.target_native_unit, field_name="target_native_unit")
        require_sorted_unique_strings(
            self.candidate_action_word_ids,
            field_name="candidate_action_word_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.preservation_bands,
            attribute="constraint_id",
            field_name="preservation_bands",
        )
        for name, values in (
            ("required_dynamics_gap_ids", self.required_dynamics_gap_ids),
            ("required_preservation_gap_ids", self.required_preservation_gap_ids),
            ("required_reachability_gap_ids", self.required_reachability_gap_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if self.primary_target_direction not in {
            ResponseDirection.HIGHER_IS_BETTER,
            ResponseDirection.LOWER_IS_BETTER,
        }:
            raise ValueError("Admission primary target requires an ordered response direction")
        for name, value in (
            ("minimum_target_improvement", self.minimum_target_improvement),
            ("maximum_sink_relative_increase", self.maximum_sink_relative_increase),
            ("maximum_effort_relative_increase", self.maximum_effort_relative_increase),
            (
                "maximum_normalized_standard_error",
                self.maximum_normalized_standard_error,
            ),
            (
                "maximum_normalized_view_disagreement",
                self.maximum_normalized_view_disagreement,
            ),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if self.minimum_response_rank <= 0:
            raise ValueError("Admission response rank must be positive")


@dataclass(frozen=True, slots=True)
class OpenSimulatorRungResult(CanonicalRecord):
    "One noncompensating rung disposition in the revealed ladder from measurement through local law."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/open-simulator-rung-result'

    result_id: str
    rung: EvidenceRung
    status: ScientificStatus
    formal_gap_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_strings(self.formal_gap_ids, field_name="formal_gap_ids")
        require_sorted_unique_strings(
            self.evidence_ids,
            field_name="evidence_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is ScientificStatus.SUPPORTED and self.reason_codes:
            raise ValueError("supported rung cannot retain stop reasons")
        if self.status is not ScientificStatus.SUPPORTED and not self.reason_codes:
            raise ValueError("non-supported rung requires typed reasons")


@dataclass(frozen=True, slots=True)
class OpenSimulatorRungAdjudication(CanonicalRecord):
    "Receipt-bound, revealed results from measurement through local law; formal exclusions remain neutral."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/open-simulator-rung-adjudication'

    adjudication_id: str
    substrate: OpenSimulatorKind
    config: ObjectIdentity
    formal_panel: ObjectIdentity
    results: tuple[OpenSimulatorRungResult, ...]
    maximum_supported_rung: EvidenceRung | None
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(self.results, attribute="result_id", field_name="results")
        expected = {
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
            EvidenceRung.LOCAL_LAW,
        }
        if len(self.results) != 4 or {value.rung for value in self.results} != expected:
            raise ValueError("rung adjudication requires exactly measurement, order relation, response, and local law")
        supported: list[EvidenceRung] = []
        predecessor_supported = True
        for rung in (
            EvidenceRung.MEASUREMENT,
            EvidenceRung.ORDER_RELATION,
            EvidenceRung.RESPONSE,
            EvidenceRung.LOCAL_LAW,
        ):
            result = next(value for value in self.results if value.rung is rung)
            if result.status is ScientificStatus.SUPPORTED and predecessor_supported:
                supported.append(rung)
            elif result.status is ScientificStatus.SUPPORTED:
                raise ValueError("a rung cannot be supported above an unsupported predecessor")
            else:
                predecessor_supported = False
        derived = supported[-1] if supported else None
        if self.maximum_supported_rung is not derived:
            raise ValueError("maximum supported rung differs from the exact ladder")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("rung adjudication must be evaluator-revealed")


@dataclass(frozen=True, slots=True)
class OpenSimulatorAdmissionAdjudication(CanonicalRecord):
    "Outcome-visible nine-gate admission result and sole controller use eligibility source."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/open-simulator-admission-adjudication'

    adjudication_id: str
    substrate: OpenSimulatorKind
    policy: ObjectIdentity
    rung_adjudication: ObjectIdentity
    formal_panel: ObjectIdentity
    selected_action_word_id: str | None
    gates: tuple[AdmissionGateResult, ...]
    admission_status: AdmissionStatus
    response_rank: int
    mandatory_hold_outside_admission: bool
    controller_use_eligible: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        if self.selected_action_word_id is not None:
            validate_stable_id(
                self.selected_action_word_id,
                field_name="selected_action_word_id",
            )
        require_sorted_unique_ids(self.gates, attribute="gate_id", field_name="gates")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if len(self.gates) != len(AdmissionGateKind) or {value.kind for value in self.gates} != set(
            AdmissionGateKind
        ):
            raise ValueError("Admission adjudication requires all nine admission gates")
        admitted = all(value.status is GateStatus.PASS for value in self.gates)
        expected_status = (
            AdmissionStatus.ADMITTED
            if admitted
            else (
                AdmissionStatus.UNEVALUABLE
                if any(value.status is GateStatus.UNEVALUABLE for value in self.gates)
                else AdmissionStatus.EMPTY
            )
        )
        if self.admission_status is not expected_status:
            raise ValueError("Admission status differs from the noncompensating gate intersection")
        if self.response_rank < 0:
            raise ValueError("Admission response rank cannot be negative")
        if self.mandatory_hold_outside_admission is not True:
            raise ValueError("Admission must retain mandatory hold outside admission")
        if self.controller_use_eligible != admitted:
            raise ValueError("Controller use eligibility must equal the complete admission intersection")
        if admitted and (self.selected_action_word_id is None or self.reason_codes):
            raise ValueError("admitted result requires one selected direction and no stop reason")
        if not admitted and not self.reason_codes:
            raise ValueError("non-admitted result requires typed reasons")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("Admission adjudication must be evaluation-revealed")


@dataclass(frozen=True, slots=True)
class OpenSimulatorProspectiveEvaluationInput(CanonicalRecord):
    "Sealed fresh-unit controller/comparator observations for controller use evaluation."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/open-simulator-prospective-evaluation-input'

    input_id: str
    substrate: OpenSimulatorKind
    config: ObjectIdentity
    policy: ObjectIdentity
    parent_admission: ObjectIdentity
    selected_action_word_id: str
    comparator_action_word_id: str
    independent_unit_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    samples: tuple[FormalAnalysisSample, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("input_id", self.input_id),
            ("selected_action_word_id", self.selected_action_word_id),
            ("comparator_action_word_id", self.comparator_action_word_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("independent_unit_ids", self.independent_unit_ids),
            ("numerical_view_ids", self.numerical_view_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_ids(self.samples, attribute="sample_id", field_name="samples")
        if not self.samples:
            raise ValueError("Controller use evaluation input requires fresh observations")
        if any(
            value.independent_unit_id not in self.independent_unit_ids
            or value.numerical_view_id not in self.numerical_view_ids
            or value.action_word_id
            not in {self.selected_action_word_id, self.comparator_action_word_id}
            or value.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            for value in self.samples
        ):
            raise ValueError("Controller use sample differs from its frozen fresh-unit envelope")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("Controller/comparator input for controller use must remain sealed")


@dataclass(frozen=True, slots=True)
class _Definition:
    kind: OpenSimulatorKind
    slug: str
    label: str
    source_manifest: OpenSimulatorSourceManifest
    model_id: str
    versions: tuple[str, ...]
    parameterization_ids: tuple[str, ...]
    runtime_ids: tuple[str, ...]
    actions: tuple[tuple[str, str, str], ...]
    receivers: tuple[tuple[str, str, str, ResponseDirection], ...]
    state_ids: tuple[str, ...]
    denominator_ids: tuple[str, ...]
    history_ids: tuple[str, ...]
    view_rows: tuple[tuple[str, Decimal, int, str], ...]
    nominal_period: Decimal
    horizon_steps: int
    horizon_seconds: Decimal
    development_units: tuple[str, ...]
    evaluation_units: tuple[str, ...]
    prospective_units: tuple[str, ...]
    represented_physics: tuple[str, ...]
    unrepresented_physics: tuple[str, ...]
    loopback_service_id: str | None = None


_PYBAMM_SDIST_SHA256: Final = "9675b427c39a97a925785603c4e7a74ba6348459001c2a6b449abe6997471c13"
_GYM_TORAX_SHA256: Final = "c6a449bcec0ed0c44304ab3d5d88a318f1c866cfce69e089680c4f13b722f84a"
_TORAX_SHA256: Final = "2a7d110dc46eff4137fd62e29605f952deee91dac02e7fb0135144180fb14927"


def _source_manifest(kind: OpenSimulatorKind) -> OpenSimulatorSourceManifest:
    if kind is OpenSimulatorKind.PYBAMM:
        return OpenSimulatorSourceManifest(
            source_id="source.pybamm.v26.6.2.0-held",
            substrate=kind,
            release_ids=("release.pybamm.v26.6.2.0",),
            version_bindings=("pybamm=26.6.2.0",),
            held_payload_sha256=(_PYBAMM_SDIST_SHA256,),
            repository_commit=None,
            runtime_identity_ids=(
                "runtime.casadi-coarse",
                "runtime.idaklu-refined",
                "runtime.pybamm-cpython-3.11",
            ),
            runtime_payload_sha256=(
                "f669809e3cd9488f5f0a02b025f84961b0bfc294a6ec8788f1c4241b1e0329e9",
            ),
            source_qualification_check_ids=tuple(
                sorted(
                    (
                        "pybamm-held-sdist-sha256",
                        "pybamm-installed-distribution-version",
                        "pybamm-spme-chen2020-conformance",
                    )
                )
            ),
        )
    return OpenSimulatorSourceManifest(
        source_id="source.gym-torax.v1.1.1-torax.v1.4.2-held",
        substrate=kind,
        release_ids=(
            "release.gym-torax.v1.1.1",
            "release.torax.v1.4.2",
        ),
        version_bindings=("gymtorax=1.1.1", "torax=1.4.2"),
        held_payload_sha256=tuple(sorted((_GYM_TORAX_SHA256, _TORAX_SHA256))),
        repository_commit="8e6a62d7c78ab6ebcdc5ffa16e1499921ca5a6ac",
        runtime_identity_ids=(
            "runtime.cpu-jax-float64",
            "runtime.gymtorax-iterhybrid-v0",
            "runtime.torax-v1.4.2",
        ),
        runtime_payload_sha256=tuple(
            sorted(
                (
                    "cc9a4c898a3d207ebf5e53a84d5df46028471d9751dbee9065dbb33b3460b20c",
                    "7677c9dbe58e45f0b730f1d3a893bc2ddf4f4975a5d0593b8bb179a4fd899487",
                )
            )
        ),
        source_qualification_check_ids=tuple(
            sorted(
                (
                    "gym-torax-held-archive-sha256",
                    "gym-torax-full-150-transition-contract",
                    "torax-held-archive-sha256",
                )
            )
        ),
    )


def open_simulator_definitions() -> tuple[_Definition, ...]:
    "Return frozen denominator-local defaults for the open simulator authoring bundle."

    return (
        _Definition(
            kind=OpenSimulatorKind.GYM_TORAX,
            slug="gym-torax-full-iter",
            label="Gym--TORAX current-version full ITER campaign",
            source_manifest=_source_manifest(OpenSimulatorKind.GYM_TORAX),
            model_id="gymtorax.iterhybrid-v0-full-episode",
            versions=("gymtorax=1.1.1", "torax=1.4.2"),
            parameterization_ids=(
                "gymtorax.fixed-public-pi-kd-0",
                "gymtorax.fixed-public-pi-ki-34.257",
                "gymtorax.fixed-public-pi-kp-0.700",
                "gymtorax.iter-cell-grid-zeff-nbar-current-profile",
                "gymtorax.open-loop-reference",
                "gymtorax.strict-delivery-comparator",
            ),
            runtime_ids=(
                "runtime.cpu-jax-float64",
                "runtime.gymtorax-iterhybrid-v0",
                "runtime.torax-v1.4.2",
            ),
            actions=(
                ("action.ecrh-power", "W", "torax-actuator"),
                ("action.ip-target", "A", "torax-actuator"),
                ("action.nbi-power", "W", "torax-actuator"),
            ),
            receivers=(
                (
                    "receiver.normalized-beta",
                    "1",
                    "iter-plasma",
                    ResponseDirection.TARGET_BAND,
                ),
                (
                    "receiver.confinement-h98",
                    "1",
                    "iter-plasma",
                    ResponseDirection.HIGHER_IS_BETTER,
                ),
                (
                    "receiver.fusion-gain",
                    "1",
                    "iter-plasma",
                    ResponseDirection.HIGHER_IS_BETTER,
                ),
                (
                    "receiver.greenwald-line-fraction",
                    "1",
                    "iter-plasma",
                    ResponseDirection.LOWER_IS_BETTER,
                ),
                (
                    "receiver.greenwald-volume-fraction",
                    "1",
                    "iter-plasma",
                    ResponseDirection.LOWER_IS_BETTER,
                ),
                (
                    "receiver.q-minimum",
                    "1",
                    "iter-plasma",
                    ResponseDirection.HIGHER_IS_BETTER,
                ),
                (
                    "sink.radiated-power",
                    "W",
                    "iter-plasma",
                    ResponseDirection.LOWER_IS_BETTER,
                ),
                (
                    "effort.auxiliary-energy",
                    "J",
                    "torax-actuator",
                    ResponseDirection.LOWER_IS_BETTER,
                ),
            ),
            state_ids=(
                "state.current-profile",
                "state.density-profile",
                "state.ion-temperature-profile",
            ),
            denominator_ids=(
                "denominator.iter-preparation-cell",
                "denominator.torax-closure",
            ),
            history_ids=("history.transport-state",),
            view_rows=(
                ("view.gym-torax.primary", Decimal("1"), 0, "torax-primary-step"),
                ("view.gym-torax.refined", Decimal("0.5"), 1, "torax-refined-step"),
            ),
            nominal_period=Decimal("1"),
            horizon_steps=150,
            horizon_seconds=Decimal("150"),
            development_units=tuple(f"iter-parameter-cell-dev-{i:02d}" for i in range(1, 7)),
            evaluation_units=tuple(f"iter-parameter-cell-eval-{i:02d}" for i in range(1, 5)),
            prospective_units=tuple(f"iter-parameter-cell-prospective-{i:02d}" for i in range(1, 5)),
            represented_physics=(
                "one-dimensional-core-transport",
                "plasma-current-evolution",
                "source-and-sink-closures",
            ),
            unrepresented_physics=(
                "facility-actuator-discrepancy",
                "full-mhd-and-edge-physics",
            ),
        ),
        _Definition(
            kind=OpenSimulatorKind.PYBAMM,
            slug="pybamm",
            label="PyBaMM SPMe electrothermal campaign",
            source_manifest=_source_manifest(OpenSimulatorKind.PYBAMM),
            model_id="pybamm.spme-lumped-thermal-chen2020",
            versions=("pybamm=26.6.2.0",),
            parameterization_ids=(
                "pybamm.chen2020",
                "pybamm.current-controller-for-admission-and-controller-use",
                "pybamm.spme-lumped-thermal",
            ),
            runtime_ids=(
                "runtime.casadi-coarse",
                "runtime.idaklu-refined",
                "runtime.pybamm-cpython-3.11",
            ),
            actions=(
                ("action.ambient-temperature", "K", "battery-boundary"),
                ("action.applied-current", "A", "battery-terminal"),
            ),
            receivers=(
                (
                    "receiver.capacity",
                    "A.h",
                    "battery-terminal",
                    ResponseDirection.HIGHER_IS_BETTER,
                ),
                (
                    "receiver.cell-temperature",
                    "K",
                    "battery-cell",
                    ResponseDirection.TARGET_BAND,
                ),
                (
                    "receiver.terminal-voltage",
                    "V",
                    "battery-terminal",
                    ResponseDirection.TARGET_BAND,
                ),
                (
                    "sink.volumetric-heating",
                    "W.m-3",
                    "battery-cell",
                    ResponseDirection.LOWER_IS_BETTER,
                ),
                (
                    "effort.current-throughput",
                    "A.s",
                    "battery-terminal",
                    ResponseDirection.LOWER_IS_BETTER,
                ),
            ),
            state_ids=(
                "state.electrochemical",
                "state.soc",
                "state.temperature",
            ),
            denominator_ids=(
                "denominator.initial-soc",
                "denominator.initial-temperature",
                "denominator.model-parameterization",
            ),
            history_ids=("history.electrothermal-state",),
            view_rows=(
                ("view.pybamm.casadi-coarse", Decimal("10"), 0, "casadi"),
                ("view.pybamm.idaklu-refined", Decimal("1"), 1, "idaklu"),
            ),
            nominal_period=Decimal("1"),
            horizon_steps=900,
            horizon_seconds=Decimal("900"),
            development_units=tuple(f"pybamm-soc-temp-dev-{i:02d}" for i in range(1, 7)),
            evaluation_units=tuple(f"pybamm-soc-temp-eval-{i:02d}" for i in range(1, 5)),
            prospective_units=tuple(f"pybamm-soc-temp-prospective-{i:02d}" for i in range(1, 5)),
            represented_physics=(
                "electrochemical-spme-dynamics",
                "lumped-cell-thermal-dynamics",
            ),
            unrepresented_physics=(
                "manufacturing-population-variability",
                "physical-cell-model-discrepancy",
            ),
        ),
    )


def _definition(kind: OpenSimulatorKind) -> _Definition:
    return next(value for value in open_simulator_definitions() if value.kind is kind)


def _admission_policy(kind: OpenSimulatorKind) -> OpenSimulatorAdmissionPolicy:
    "Return the frozen native-unit admission policy for one open simulator family."

    bands: tuple[OpenSimulatorQuantityBand, ...]
    candidate_words: tuple[str, ...]
    if kind is OpenSimulatorKind.PYBAMM:
        target_id = "receiver.capacity"
        target_unit = "A.h"
        minimum_improvement = Decimal("0.001")
        bands = (
            OpenSimulatorQuantityBand(
                constraint_id="constraint.pybamm.capacity-nonnegative",
                quantity_id="receiver.capacity",
                native_unit="A.h",
                lower_bound=Decimal("0"),
                upper_bound=None,
            ),
            OpenSimulatorQuantityBand(
                constraint_id="constraint.pybamm.cell-temperature",
                quantity_id="receiver.cell-temperature",
                native_unit="K",
                lower_bound=Decimal("273.15"),
                upper_bound=Decimal("333.15"),
            ),
            OpenSimulatorQuantityBand(
                constraint_id="constraint.pybamm.current-throughput",
                quantity_id="effort.current-throughput",
                native_unit="A.s",
                lower_bound=Decimal("0"),
                upper_bound=Decimal("1000"),
            ),
            OpenSimulatorQuantityBand(
                constraint_id="constraint.pybamm.terminal-voltage",
                quantity_id="receiver.terminal-voltage",
                native_unit="V",
                lower_bound=Decimal("2.5"),
                upper_bound=Decimal("4.2"),
            ),
        )
        candidate_words = (
            "action.negative.coordinate.applied-current",
            "action.positive.coordinate.applied-current",
        )
    else:
        target_id = "receiver.fusion-gain"
        target_unit = "1"
        minimum_improvement = Decimal("0.0017")
        bands = (
            OpenSimulatorQuantityBand(
                constraint_id="constraint.gym-torax.confinement-h98",
                quantity_id="receiver.confinement-h98",
                native_unit="1",
                lower_bound=Decimal("1"),
                upper_bound=None,
            ),
            OpenSimulatorQuantityBand(
                constraint_id="constraint.gym-torax.fusion-gain",
                quantity_id="receiver.fusion-gain",
                native_unit="1",
                lower_bound=Decimal("0"),
                upper_bound=None,
            ),
            OpenSimulatorQuantityBand(
                constraint_id="constraint.gym-torax.greenwald-line",
                quantity_id="receiver.greenwald-line-fraction",
                native_unit="1",
                lower_bound=Decimal("0"),
                upper_bound=Decimal("1"),
            ),
            OpenSimulatorQuantityBand(
                constraint_id="constraint.gym-torax.greenwald-volume",
                quantity_id="receiver.greenwald-volume-fraction",
                native_unit="1",
                lower_bound=Decimal("0"),
                upper_bound=Decimal("1"),
            ),
            OpenSimulatorQuantityBand(
                constraint_id="constraint.gym-torax.normalized-beta",
                quantity_id="receiver.normalized-beta",
                native_unit="1",
                lower_bound=Decimal("0"),
                upper_bound=Decimal("3.5"),
            ),
            OpenSimulatorQuantityBand(
                constraint_id="constraint.gym-torax.q-minimum",
                quantity_id="receiver.q-minimum",
                native_unit="1",
                lower_bound=Decimal("1"),
                upper_bound=None,
            ),
        )
        candidate_words = tuple(
            sorted(
                f"action.{sign}.coordinate.{coordinate}"
                for sign in ("negative", "positive")
                for coordinate in ("ecrh-power", "ip-target", "nbi-power")
            )
        )
    return OpenSimulatorAdmissionPolicy(
        policy_id=f"policy.open-sim.{kind.value.lower()}.admission",
        substrate=kind,
        candidate_action_word_ids=candidate_words,
        primary_target_quantity_id=target_id,
        primary_target_direction=ResponseDirection.HIGHER_IS_BETTER,
        minimum_target_improvement=minimum_improvement,
        target_native_unit=target_unit,
        preservation_bands=tuple(sorted(bands, key=lambda value: value.constraint_id)),
        maximum_sink_relative_increase=Decimal("0.10"),
        maximum_effort_relative_increase=Decimal("0.10"),
        maximum_normalized_standard_error=Decimal("0.10"),
        maximum_normalized_view_disagreement=Decimal("0.10"),
        minimum_response_rank=1,
        required_dynamics_gap_ids=(
            "gap.dynamics.causal-cones-clock-transport",
            "gap.dynamics.local-evolution-operator",
            "gap.dynamics.state-closure-memory",
        ),
        required_preservation_gap_ids=(
            "gap.dynamics.hysteresis-return",
            "gap.dynamics.stability-transient",
        ),
        required_reachability_gap_ids=(
            "gap.calculus.local-jacobian-rank",
            "gap.dynamics.controllability-observability",
            "gap.geometry.tangent-rank",
        ),
        authority_policy_id=f"authority-policy.open-sim.{kind.value.lower()}.fresh-controller-use",
    )


_TASK_BUDGET = ResourceBudget(
    cpu_cores=4,
    memory_bytes=12 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=4 * 3_600,
    source_scan_bytes=2 * 1024**3,
    output_bytes=512 * 1024**2,
)
_SIMULATOR_ACQUISITION_TASK_BUDGET = ResourceBudget(
    cpu_cores=_TASK_BUDGET.cpu_cores,
    memory_bytes=_TASK_BUDGET.memory_bytes,
    gpu_devices=_TASK_BUDGET.gpu_devices,
    wall_time_seconds=12 * 3_600,
    source_scan_bytes=_TASK_BUDGET.source_scan_bytes,
    output_bytes=2 * 1024**3,
)
_PROGRAMME_BUDGET = ResourceBudget(
    cpu_cores=4,
    memory_bytes=12 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=48 * 3_600,
    source_scan_bytes=16 * 1024**3,
    output_bytes=4 * 1024**3,
)


def _task_budget(
    definition: _Definition,
    stage_id: str,
) -> ResourceBudget:
    del definition
    if stage_id in {"development", "sealed-evaluation", "prospective-controller-execution"}:
        return _SIMULATOR_ACQUISITION_TASK_BUDGET
    return _TASK_BUDGET


_READ_WRITE = tuple(
    sorted(
        (
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        key=lambda value: value.value,
    )
)
_DEVELOP = tuple(
    sorted(
        (*_READ_WRITE, CapabilityPermission.READ_DEVELOPMENT),
        key=lambda value: value.value,
    )
)
_EVALUATE = tuple(
    sorted(
        (
            *_READ_WRITE,
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        ),
        key=lambda value: value.value,
    )
)
_REPORT = tuple(
    sorted(
        (*_READ_WRITE, CapabilityPermission.READ_OUTCOME_VISIBLE),
        key=lambda value: value.value,
    )
)


def _clock(definition: _Definition) -> ClockSpec:
    return ClockSpec(
        clock_id=f"clock.{definition.slug}.simulator",
        label=f"{definition.label} simulator clock",
        time_unit="s",
        coordinate_frame=f"{definition.slug}-simulator-time",
        sampling=SamplingSemantics.REGULAR,
        hold=HoldSemantics.ZERO_ORDER,
        label_semantics=ClockLabelSemantics.INSTANT,
        nominal_period=definition.nominal_period,
        alignment_tolerance=Decimal("1e-9"),
    )


def _quantity(
    *,
    quantity_id: str,
    label: str,
    kind: QuantityKind,
    unit: str,
    frame: str,
    clock_id: str,
    phase: CausalPhase,
    access: OutcomeAccess,
    direction: ResponseDirection = ResponseDirection.NOT_APPLICABLE,
) -> QuantitySpec:
    return QuantitySpec(
        quantity_id=quantity_id,
        label=label,
        kind=kind,
        dimension=quantity_id.split(".", 1)[0],
        native_unit=unit,
        coordinate_frame=frame,
        clock_id=clock_id,
        availability=AvailabilitySpec(
            clock_id=clock_id,
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def open_simulator_system(kind: OpenSimulatorKind) -> SystemSpec:
    definition = _definition(kind)
    clock = _clock(definition)
    unit_id = f"unit.{definition.slug}.complete-preparation"
    quantities: list[QuantitySpec] = []
    for quantity_id in definition.denominator_ids:
        quantities.append(
            _quantity(
                quantity_id=quantity_id,
                label=quantity_id.replace(".", " "),
                kind=QuantityKind.DENOMINATOR,
                unit="1",
                frame=f"{definition.slug}-preparation",
                clock_id=clock.clock_id,
                phase=CausalPhase.PRE_ACTION,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
    for quantity_id in definition.history_ids:
        quantities.append(
            _quantity(
                quantity_id=quantity_id,
                label=quantity_id.replace(".", " "),
                kind=QuantityKind.HISTORY,
                unit="1",
                frame=f"{definition.slug}-state",
                clock_id=clock.clock_id,
                phase=CausalPhase.PRE_ACTION,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
    for quantity_id in definition.state_ids:
        state_unit = {
            "state.hvac-mode": "1",
            "state.thermal-storage": "K",
            "state.current-profile": "Wb",
            "state.density-profile": "m-3",
            "state.ion-temperature-profile": "keV",
            "state.electrochemical": "V",
            "state.soc": "1",
            "state.temperature": "K",
        }[quantity_id]
        quantities.append(
            _quantity(
                quantity_id=quantity_id,
                label=quantity_id.replace(".", " "),
                kind=QuantityKind.STATE,
                unit=state_unit,
                frame=f"{definition.slug}-state",
                clock_id=clock.clock_id,
                phase=CausalPhase.PRE_ACTION,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
    for quantity_id, unit, frame in definition.actions:
        quantities.append(
            _quantity(
                quantity_id=quantity_id,
                label=quantity_id.replace(".", " "),
                kind=QuantityKind.ACTION,
                unit=unit,
                frame=frame,
                clock_id=clock.clock_id,
                phase=CausalPhase.ACTION_APPLIED,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
    for quantity_id, unit, frame, direction in definition.receivers:
        kind_value = (
            QuantityKind.SINK
            if quantity_id.startswith("sink.")
            else QuantityKind.EFFORT
            if quantity_id.startswith("effort.")
            else QuantityKind.RECEIVER
        )
        quantities.append(
            _quantity(
                quantity_id=quantity_id,
                label=quantity_id.replace(".", " "),
                kind=kind_value,
                unit=unit,
                frame=frame,
                clock_id=clock.clock_id,
                phase=CausalPhase.RECEIVER,
                access=OutcomeAccess.EVALUATOR_REVEAL,
                direction=direction,
            )
        )
    quantities_tuple = tuple(sorted(quantities, key=lambda value: value.quantity_id))
    receiver_ids = tuple(
        sorted(
            value.quantity_id for value in quantities_tuple if value.kind is QuantityKind.RECEIVER
        )
    )
    action_ids = tuple(sorted(value[0] for value in definition.actions))
    relation = RelationalIdentity(
        relation_id=f"relation.{definition.slug}.denominator-local",
        denominator_quantity_ids=tuple(sorted(definition.denominator_ids)),
        history_quantity_ids=tuple(sorted(definition.history_ids)),
        memoryless=False,
        action_quantity_ids=action_ids,
        receiver_quantity_ids=receiver_ids,
        horizon=HorizonSpec(
            horizon_id=f"horizon.{definition.slug}.campaign",
            clock_id=clock.clock_id,
            duration=definition.horizon_seconds,
            time_unit="s",
        ),
    )
    world = WorldSpec(
        world_id=f"world.{definition.slug}.numerical-simulator",
        label=definition.label,
        kind=WorldKind.NUMERICAL_SIMULATOR,
        represented_physics=tuple(sorted(definition.represented_physics)),
        unrepresented_physics=tuple(sorted(definition.unrepresented_physics)),
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.CONTROLLER_USE,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.EVALUATION_REVEALED,
            }
        ),
    )
    envelope = ComputabilityEnvelope(
        envelope_id=f"compute.{definition.slug}.trusted-local",
        represented_effect_ids=tuple(sorted(definition.represented_physics)),
        unresolved_effect_ids=(),
        required_structure_ids=(
            "formal-48-row-panel",
            "observation-to-law-adjudication",
            "noncompensating-admission",
        ),
        computable_structure_ids=(
            "formal-48-row-panel",
            "observation-to-law-adjudication",
            "noncompensating-admission",
        ),
        max_cpu_cores=_PROGRAMME_BUDGET.cpu_cores,
        max_memory_bytes=_PROGRAMME_BUDGET.memory_bytes,
        max_gpu_devices=0,
        max_wall_time_seconds=_PROGRAMME_BUDGET.wall_time_seconds,
        max_output_bytes=_PROGRAMME_BUDGET.output_bytes,
        worst_case_latency_seconds=Decimal(_PROGRAMME_BUDGET.wall_time_seconds),
        deadline_seconds=None,
    )
    views = tuple(
        NumericalViewSpec(
            view_id=view_id,
            world_id=world.world_id,
            physical_preparation_id=unit_id,
            equations_id=definition.model_id,
            closure_ids=tuple(sorted(definition.parameterization_ids)),
            boundary_condition_ids=(f"boundary.{definition.slug}.frozen",),
            coordinates=(
                NumericalCoordinateSpec(
                    coordinate_id=f"coordinate.{view_id}.step",
                    kind=NumericalCoordinateKind.TIMESTEP,
                    value=step,
                    unit="s",
                    refinement_level=level,
                ),
            ),
            solver_id=solver,
            solver_version=definition.versions[-1].split("=", 1)[1],
            precision="float64",
            device_class="cpu",
            runtime_id=definition.runtime_ids[0],
            randomness=RandomnessSemantics.DETERMINISTIC,
            observation_operator_id=f"observer.{definition.slug}.native",
            computability_envelope_id=envelope.envelope_id,
        )
        for view_id, step, level, solver in definition.view_rows
    )
    authority = AuthorityPolicy(
        policy_id=f"authority-policy.{definition.slug}.simulation",
        delegator_id="human.project-owner",
        delegate_id="gate.open-simulator-campaign",
        scope_ids=(f"campaign.{definition.slug}.observation-to-controller-use",),
        allowed_world_kinds=frozenset({WorldKind.NUMERICAL_SIMULATOR}),
        allowed_actions=frozenset(
            {
                AuthorityAction.SIMULATION_EXECUTION,
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.OFFICIAL_OPEN_PUBLIC}),
        required_gate_ids=(
            "clean-implementation",
            "exact-source-closure",
            "separate-execution-and-reveal-authority",
        ),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
        ),
        budget_ceiling=_PROGRAMME_BUDGET,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    ports = tuple(
        sorted(
            (
                *(
                    PortSpec(
                        port_id=f"port.{quantity_id}.input",
                        quantity_id=quantity_id,
                        clock_id=clock.clock_id,
                        direction=PortDirection.INPUT,
                        balance_role=BalanceRole.COMMAND,
                        authority_action=AuthorityAction.SIMULATION_EXECUTION,
                    )
                    for quantity_id in action_ids
                ),
                *(
                    PortSpec(
                        port_id=f"port.{value.quantity_id}.output",
                        quantity_id=value.quantity_id,
                        clock_id=clock.clock_id,
                        direction=PortDirection.OUTPUT,
                        balance_role=(
                            BalanceRole.ENERGY
                            if value.kind in {QuantityKind.SINK, QuantityKind.EFFORT}
                            else BalanceRole.OBSERVATION
                        ),
                    )
                    for value in quantities_tuple
                    if value.kind in {QuantityKind.RECEIVER, QuantityKind.SINK, QuantityKind.EFFORT}
                ),
            ),
            key=lambda value: value.port_id,
        )
    )
    return SystemSpec(
        system_id=f"system.{definition.slug}.observation-to-controller-use",
        label=definition.label,
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities_tuple,
        independent_unit=IndependentUnitSpec(
            unit_id=unit_id,
            label=f"{definition.label} complete preparation",
            grouping_key=f"group.{definition.slug}.complete-preparation",
            scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        ),
        authority_policy=authority,
        ports=ports,
        computability_envelopes=(envelope,),
        numerical_views=tuple(sorted(views, key=lambda value: value.view_id)),
    )


def _obligations(
    definition: _Definition,
    system: SystemSpec,
    evaluation_units: tuple[str, ...],
) -> ScientificObligations:
    cutoff_id = f"cutoff.{definition.slug}.pre-action"
    action_bounds = tuple(
        QuantityBound(
            bound_id=f"bound.{quantity_id}",
            quantity_id=quantity_id,
            native_unit=unit,
            lower=Decimal("-1"),
            upper=Decimal("1"),
        )
        for quantity_id, unit, _ in definition.actions
    )
    return ScientificObligations(
        obligations_id=f"obligations.{definition.slug}.observation-to-controller-use",
        support=SupportSpec(
            support_id=f"support.{definition.slug}.mode-local",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=len(evaluation_units),
            nested_numerical_view_count=len(system.numerical_views),
            information_cutoff_id=cutoff_id,
            chart_ids=tuple(f"chart.{value[0]}" for value in definition.actions),
            denominator_cell_ids=(f"cell.{definition.slug}.prepared",),
            action_bounds=tuple(sorted(action_bounds, key=lambda value: value.bound_id)),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id=f"validity.{definition.slug}.native",
            validity_domain_ids=(f"domain.{definition.slug}.native-support",),
            assumption_ids=(
                "simulator-denominator-is-response-medium",
                "source-and-runtime-closure-exact",
            ),
            exclusion_reason_codes=(),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id=f"uncertainty.{definition.slug}.preparation-resampling",
            method_key="independent-preparation-unit-bootstrap",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.95"),
            interval_quantity_ids=system.relation.receiver_quantity_ids,
            limitation_codes=("NESTED_VIEWS_NOT_REPLICATION",),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id=f"falsifier.{definition.slug}.wrong-action-time-history",
                kind=FalsifierKind.WRONG_ACTION,
                capability_key=f"open-sim.{definition.slug}.development",
                description="Wrong action, time, history and delivery controls.",
                decisive_rule="Any predeclared counterfeit that survives invalidates promotion.",
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id=f"closure.{definition.slug}.retained-history",
            recurrence_cell_ids=(f"cell.{definition.slug}.prepared",),
            exchange_factor_ids=system.relation.denominator_quantity_ids,
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id=f"convergence.{definition.slug}.views",
            required_structure_ids=(
                "formal-response-structure",
                "mode-local-law",
                "admission-boundary",
            ),
            numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id=f"computability.{definition.slug}.trusted-local",
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=(f"readiness.{definition.slug}.source-runtime",),
        ),
    )


def open_simulator_experiment(
    kind: OpenSimulatorKind,
    system: SystemSpec,
) -> ExperimentSpec:
    definition = _definition(kind)
    cutoff = InformationCutoff(
        cutoff_id=f"cutoff.{definition.slug}.pre-action",
        clock_id=system.clocks[0].clock_id,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal("0"),
    )
    claim = ClaimSpec(
        claim_id=f"claim.{definition.slug}.observation-to-local-law",
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition=(
            "A denominator-local, native-action response law is supported on "
            "held-out complete simulator preparations and can be intersected "
            "noncompensatingly for a conditional controller branch."
        ),
        estimand=(
            "Preparation-local structure from measurement through local law, native receiver response, formal-gap disposition, and conditional admission."
        ),
        physical_independent_unit_id=system.independent_unit.unit_id,
        requested_rung=EvidenceRung.LOCAL_LAW,
        evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        promotion_rule=(
            "Admission requires supported local law plus every noncompensating gate; controller use is a new separately issued fresh-preparation act."
        ),
        assumption_ids=(
            "exact-held-source-and-runtime",
            "nested-numerical-views-not-independent-units",
        ),
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
    )
    controls = (
        ControlSpec(
            control_id="control.negative-action",
            kind=ControlKind.NEGATIVE_ACTION,
            capability_key=f"open-sim.{definition.slug}.development",
            target_quantity_ids=system.relation.receiver_quantity_ids,
            decisive_rule="Signed or feasible inverse probes must obey native support.",
        ),
        ControlSpec(
            control_id="control.support-matched-comparator",
            kind=ControlKind.BASELINE_COMPARATOR,
            capability_key=f"open-sim.{definition.slug}.development",
            target_quantity_ids=system.relation.receiver_quantity_ids,
            decisive_rule="Compare only support-, preparation-, and clock-matched actions.",
        ),
    )
    return ExperimentSpec(
        experiment_id=f"experiment.{definition.slug}.observation-to-controller-use",
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(claim,),
        assignment=AssignmentSpec(
            assignment_id=f"assignment.{definition.slug}.simulator-intervention",
            kind=AssignmentKind.SIMULATOR_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism="Frozen source-native deterministic intervention roster.",
            support_restriction_ids=(f"support.{definition.slug}.native-actions",),
        ),
        measurement_quantity_ids=tuple(
            sorted(
                value.quantity_id
                for value in system.quantities
                if value.kind
                in {
                    QuantityKind.RECEIVER,
                    QuantityKind.SINK,
                    QuantityKind.EFFORT,
                    QuantityKind.STATE,
                }
            )
        ),
        controls=tuple(sorted(controls, key=lambda value: value.control_id)),
        precision_goals=(
            PrecisionGoal(
                goal_id=f"precision.{definition.slug}.frozen-roster",
                metric_id="preparation-unit-standard-error",
                target_width=Decimal("0.1"),
                native_unit="1",
                maximum_independent_units=len(definition.evaluation_units),
                stopping_rule="Stop after the frozen finite roster; never outcome-retune.",
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id=f"reveal.{definition.slug}.evaluation",
            development_unit_ids=tuple(sorted(definition.development_units)),
            evaluation_cohort_id=f"cohort.{definition.slug}.evaluation",
            evaluation_manifest_sha256=hashlib.sha256(
                "\n".join(definition.evaluation_units).encode("ascii")
            ).hexdigest(),
            sealed_outcome_artifact_ids=tuple(
                sorted(
                    f"artifact.formal-evaluation-{domain.value.lower()}" for domain in FormalDomain
                )
            ),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_obligations(definition, system, definition.evaluation_units),
        design_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


def open_simulator_campaign(
    kind: OpenSimulatorKind,
    system: SystemSpec,
    experiment: ExperimentSpec,
) -> CampaignSpec:
    definition = _definition(kind)
    node = CampaignNode(
        node_id=f"campaign-node.{definition.slug}.observation-to-admission",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(
            experiment.experiment_id,
            experiment,
        ),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id=f"campaign.{definition.slug}.observation-to-controller-use",
        objective=(
            f'Run the complete standard measurement through local law and formal panel for {definition.label}, then permit admission/controller use only through frozen noncompensating prerequisites.'
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=_PROGRAMME_BUDGET,
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id=f"decision-right.{definition.slug}.simulation",
                action=AuthorityAction.SIMULATION_EXECUTION,
                decision_maker_id=system.authority_policy.delegate_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=True,
            ),
        ),
        nodes=(node,),
        root_node_ids=(node.node_id,),
        active_node_ids=(node.node_id,),
        evidence_state=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def open_simulator_config(kind: OpenSimulatorKind) -> OpenSimulatorStudyConfig:
    definition = _definition(kind)
    source = definition.source_manifest
    action_delivery_bindings = {
        OpenSimulatorKind.GYM_TORAX: (
            OpenSimulatorActionDeliveryBinding(
                coordinate_id="action.ecrh-power",
                native_point_ids=("ECRH",),
                native_unit="W",
            ),
            OpenSimulatorActionDeliveryBinding(
                coordinate_id="action.ip-target",
                native_point_ids=("Ip",),
                native_unit="A",
            ),
            OpenSimulatorActionDeliveryBinding(
                coordinate_id="action.nbi-power",
                native_point_ids=("NBI",),
                native_unit="W",
            ),
        ),
        OpenSimulatorKind.PYBAMM: (
            OpenSimulatorActionDeliveryBinding(
                coordinate_id="action.ambient-temperature",
                native_point_ids=("Ambient temperature [K]",),
                native_unit="K",
            ),
            OpenSimulatorActionDeliveryBinding(
                coordinate_id="action.applied-current",
                native_point_ids=("Current function [A]",),
                native_unit="A",
            ),
        ),
    }[kind]
    return OpenSimulatorStudyConfig(
        config_id=f"config.open-sim.{definition.slug}.observation-to-controller-use",
        substrate=kind,
        campaign_definition_id=f"definition.{definition.slug}.observation-to-controller-use",
        source_manifest=ObjectIdentity.from_record(source.source_id, source),
        simulator_versions=definition.versions,
        model_id=definition.model_id,
        parameterization_ids=definition.parameterization_ids,
        runtime_identity_ids=definition.runtime_ids,
        action_coordinate_ids=tuple(sorted(value[0] for value in definition.actions)),
        action_delivery_bindings=action_delivery_bindings,
        action_stage_ids=(
            "action-stage.accepted",
            "action-stage.applied",
            "action-stage.realized",
            "action-stage.requested",
        ),
        receiver_coordinate_ids=tuple(sorted(value[0] for value in definition.receivers)),
        state_coordinate_ids=tuple(sorted(definition.state_ids)),
        clock_ids=(f"clock.{definition.slug}.simulator",),
        numerical_view_ids=tuple(sorted(value[0] for value in definition.view_rows)),
        development_unit_ids=tuple(sorted(definition.development_units)),
        evaluation_unit_ids=tuple(sorted(definition.evaluation_units)),
        prospective_unit_ids=tuple(sorted(definition.prospective_units)),
        horizon_steps=definition.horizon_steps,
        horizon_seconds=definition.horizon_seconds,
        assurance_profile_id="assurance.trusted-local",
        loopback_service_id=definition.loopback_service_id,
        exact_publication_reproduction=False,
        controller_use_requires_fresh_issue=True,
    )


def _schema_sha256(schema: str) -> str:
    return hashlib.sha256(schema.encode("ascii")).hexdigest()


def _config_ref(config: CanonicalRecord, config_id: str) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=config_id,
        config_schema=config.SCHEMA,
        config_schema_sha256=_schema_sha256(config.SCHEMA),
        content_sha256=config.fingerprint(),
        artifact_id=f"config-artifact.{config_id}",
    )


def _base_capability_manifests(
    definition: _Definition,
    config: OpenSimulatorStudyConfig,
    admission_policy: OpenSimulatorAdmissionPolicy,
    implementation_sha256: str,
) -> tuple[CapabilityManifest, ...]:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    schemas = {
        "source": (
            CapabilityKind.SOURCE,
            (OpenSimulatorSourceManifest.SCHEMA,),
            (OpenSimulatorStageRecord.SCHEMA,),
            _READ_WRITE,
            OutcomeAccess.OUTCOME_BLIND,
            EvidenceCeiling.MEASUREMENT,
            False,
        ),
        "development": (
            CapabilityKind.SIMULATOR,
            (OpenSimulatorStageRecord.SCHEMA,),
            (
                FormalAnalysisInputManifest.SCHEMA,
                OpenSimulatorStageRecord.SCHEMA,
            ),
            _DEVELOP,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            EvidenceCeiling.LOCAL_LAW,
            False,
        ),
        "freeze": (
            CapabilityKind.REPORTER,
            (OpenSimulatorStageRecord.SCHEMA,),
            (OpenSimulatorStageRecord.SCHEMA,),
            _DEVELOP,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            EvidenceCeiling.LOCAL_LAW,
            False,
        ),
        "sealed-evaluation": (
            CapabilityKind.SIMULATOR,
            (OpenSimulatorStageRecord.SCHEMA,),
            (FormalAnalysisInputManifest.SCHEMA,),
            _DEVELOP,
            OutcomeAccess.EVALUATION_SEALED,
            EvidenceCeiling.LOCAL_LAW,
            False,
        ),
        "rung-adjudicate": (
            CapabilityKind.EVALUATOR,
            (
                FormalAnalysisInputManifest.SCHEMA,
                FormalGapAdjudicationPanel.SCHEMA,
                OpenSimulatorStageRecord.SCHEMA,
            ),
            (OpenSimulatorRungAdjudication.SCHEMA,),
            _EVALUATE,
            OutcomeAccess.EVALUATOR_REVEAL,
            EvidenceCeiling.LOCAL_LAW,
            False,
        ),
        "admission": (
            CapabilityKind.ADMISSION_EVALUATOR,
            (
                FormalAnalysisInputManifest.SCHEMA,
                FormalGapAdjudicationPanel.SCHEMA,
                OpenSimulatorRungAdjudication.SCHEMA,
            ),
            (OpenSimulatorAdmissionAdjudication.SCHEMA,),
            _REPORT,
            OutcomeAccess.EVALUATION_REVEALED,
            EvidenceCeiling.ADMISSION,
            False,
        ),
        "report": (
            CapabilityKind.REPORTER,
            (
                FormalGapAdjudicationPanel.SCHEMA,
                OpenSimulatorAdmissionAdjudication.SCHEMA,
            ),
            (ScientificAdjudicationRecord.SCHEMA,),
            _REPORT,
            OutcomeAccess.EVALUATION_REVEALED,
            EvidenceCeiling.ADMISSION,
            False,
        ),
        "prospective-controller-execution": (
            CapabilityKind.CONTROLLER_SYNTHESIZER,
            (OpenSimulatorAdmissionAdjudication.SCHEMA,),
            (OpenSimulatorProspectiveEvaluationInput.SCHEMA,),
            _DEVELOP,
            OutcomeAccess.EVALUATION_SEALED,
            EvidenceCeiling.CONTROLLER_USE,
            False,
        ),
        "prospective-controller-evaluation": (
            CapabilityKind.EVALUATOR,
            (OpenSimulatorProspectiveEvaluationInput.SCHEMA,),
            (ScientificAdjudicationRecord.SCHEMA,),
            _EVALUATE,
            OutcomeAccess.EVALUATOR_REVEAL,
            EvidenceCeiling.CONTROLLER_USE,
            False,
        ),
    }
    values = tuple(
        CapabilityManifest(
            capability_key=f"open-sim.{definition.slug}.{stage}",
            capability_version="1.0.0",
            kind=value[0],
            config_schema=(
                OpenSimulatorAdmissionPolicy.SCHEMA
                if stage == "admission"
                else OpenSimulatorStudyConfig.SCHEMA
            ),
            config_schema_sha256=_schema_sha256(
                OpenSimulatorAdmissionPolicy.SCHEMA
                if stage == "admission"
                else OpenSimulatorStudyConfig.SCHEMA
            ),
            input_schema_ids=tuple(sorted(value[1])),
            output_schema_ids=tuple(sorted(value[2])),
            permissions=tuple(sorted(value[3], key=lambda item: item.value)),
            maximum_evidence_ceiling=value[5],
            maximum_outcome_access=value[4],
            resource_ceiling=_task_budget(definition, stage),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id=definition.runtime_ids[0],
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=value[6],
            conformance_check_ids=(
                f"{definition.slug}-{stage}-path-free",
                f"{definition.slug}-{stage}-source-version-bound",
                f"{definition.slug}-{stage}-typed-action-ledger",
            ),
            implementation_sha256=implementation_sha256,
        )
        for stage, value in schemas.items()
    )
    del config, admission_policy
    return tuple(sorted(values, key=lambda value: value.registry_id))


def _output(output_id: str, schema: str) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/json",
        filename_suffix=".json",
    )


def _step(
    *,
    definition: _Definition,
    stage_id: str,
    stage: ScientificStage,
    config_ref: CapabilityConfigRef,
    dependencies: tuple[str, ...],
    outputs: tuple[OutputTemplate, ...],
    permissions: tuple[CapabilityPermission, ...],
    outcome_access: OutcomeAccess,
    visibility: VisibilityCeiling,
    barrier: BarrierKind,
    obligations: tuple[str, ...],
) -> ProtocolStepTemplate:
    return ProtocolStepTemplate(
        step_id=stage_id,
        stage=stage,
        capability_key=f"open-sim.{definition.slug}.{stage_id}",
        capability_version="1.0.0",
        config=config_ref,
        dependency_step_ids=tuple(sorted(dependencies)),
        outputs=tuple(sorted(outputs, key=lambda value: value.output_id)),
        required_permissions=tuple(sorted(permissions, key=lambda value: value.value)),
        requested_outcome_access=outcome_access,
        visibility_ceiling=visibility,
        resource_budget=_task_budget(definition, stage_id),
        resource_lock_ids=(f"lock.open-sim.{definition.slug}",),
        barrier=barrier,
        maximum_attempts=2
        if stage not in {ScientificStage.EVALUATE, ScientificStage.REVEAL}
        else 1,
        obligation_ids=tuple(sorted(obligations)),
    )


def _base_protocol(
    definition: _Definition,
    config: OpenSimulatorStudyConfig,
    admission_policy: OpenSimulatorAdmissionPolicy,
) -> ProtocolTemplate:
    ref = _config_ref(config, config.config_id)
    admission_policy_reference = _config_ref(admission_policy, admission_policy.policy_id)
    domain_outputs = tuple(
        _output(
            f"formal-input-{domain.value.lower()}",
            FormalAnalysisInputManifest.SCHEMA,
        )
        for domain in FormalDomain
    )
    sealed_outputs = tuple(
        _output(
            f"formal-evaluation-{domain.value.lower()}",
            FormalAnalysisInputManifest.SCHEMA,
        )
        for domain in FormalDomain
    )
    steps = (
        _step(
            definition=definition,
            stage_id="source",
            stage=ScientificStage.PREPARE,
            config_ref=ref,
            dependencies=(),
            outputs=(_output("source-ready", OpenSimulatorStageRecord.SCHEMA),),
            permissions=_READ_WRITE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
            obligations=("source-runtime-units-valid",),
        ),
        _step(
            definition=definition,
            stage_id="development",
            stage=ScientificStage.ACQUIRE,
            config_ref=ref,
            dependencies=("source",),
            outputs=(
                *domain_outputs,
                _output("development-ledger", OpenSimulatorStageRecord.SCHEMA),
            ),
            permissions=_DEVELOP,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.NONE,
            obligations=(
                "order-recurrence",
                "native-action-response",
                "local-law-development",
            ),
        ),
        _step(
            definition=definition,
            stage_id="freeze",
            stage=ScientificStage.FREEZE,
            config_ref=ref,
            dependencies=("development",),
            outputs=(_output("frozen-methods", OpenSimulatorStageRecord.SCHEMA),),
            permissions=_DEVELOP,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.FREEZE,
            obligations=("development-freeze-before-evaluation",),
        ),
        _step(
            definition=definition,
            stage_id="sealed-evaluation",
            stage=ScientificStage.ACQUIRE,
            config_ref=ref,
            dependencies=("freeze", "source"),
            outputs=sealed_outputs,
            permissions=_DEVELOP,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
            obligations=("sealed-held-out-evaluation-acquisition",),
        ),
        _step(
            definition=definition,
            stage_id="rung-adjudicate",
            stage=ScientificStage.EVALUATE,
            config_ref=ref,
            dependencies=("freeze", "sealed-evaluation"),
            outputs=(
                _output(
                    "observation-to-law-adjudication",
                    OpenSimulatorRungAdjudication.SCHEMA,
                ),
            ),
            permissions=_EVALUATE,
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.REVEAL,
            obligations=("observation-to-law-terminal-adjudication",),
        ),
        _step(
            definition=definition,
            stage_id="admission",
            stage=ScientificStage.ADMISSION,
            config_ref=admission_policy_reference,
            dependencies=("rung-adjudicate", "sealed-evaluation"),
            outputs=(
                _output(
                    "admission-disposition",
                    OpenSimulatorAdmissionAdjudication.SCHEMA,
                ),
            ),
            permissions=_REPORT,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.NONE,
            obligations=(
                "admission-all-gates-noncompensating",
                "hold-outside-admission",
            ),
        ),
        _step(
            definition=definition,
            stage_id="report",
            stage=ScientificStage.REPORT,
            config_ref=ref,
            dependencies=("admission",),
            outputs=(
                _output(
                    "campaign-adjudication",
                    ScientificAdjudicationRecord.SCHEMA,
                ),
            ),
            permissions=_REPORT,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.NONE,
            obligations=("terminal-report-complete",),
        ),
    )
    return ProtocolTemplate(
        template_id=f"protocol.{definition.slug}.standard-observation-to-admission",
        template_version="1.0.0",
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _graph(
    *,
    definition: _Definition,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source: OpenSimulatorSourceManifest,
) -> CandidateScientificGraph:
    has_source_step = any(value.step_id == "source" for value in protocol.steps)
    source_input = CandidateGraphExternalInput(
        input_id=(source.source_id if has_source_step else f"parent-receipt.{definition.slug}.admission"),
        scientific_role=(
            ScientificInputRole.PREPARED_MEDIUM
            if has_source_step
            else ScientificInputRole.PARENT_RECEIPT
        ),
        logical_artifact_id=(
            f"artifact.{source.source_id}"
            if has_source_step
            else f"artifact.parent-receipt.{definition.slug}.admission"
        ),
        content_identity_policy=(
            ContentIdentityPolicy.EXACT_SHA256
            if has_source_step
            else ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
        ),
        expected_content_sha256=source.fingerprint() if has_source_step else None,
        payload_schema=(
            OpenSimulatorSourceManifest.SCHEMA
            if has_source_step
            else OpenSimulatorAdmissionAdjudication.SCHEMA
        ),
        media_type="application/json",
        maximum_size_bytes=1024 * 1024,
        outcome_access=(
            OutcomeAccess.OUTCOME_BLIND if has_source_step else OutcomeAccess.EVALUATION_REVEALED
        ),
        visibility_ceiling=(
            VisibilityCeiling.PROSPECTIVE if has_source_step else VisibilityCeiling.OUTCOME_VISIBLE
        ),
    )
    manifests = {value.registry_id: value for value in registry.capabilities}
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=manifests[
                f"{step.capability_key}@{step.capability_version}"
            ].implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    steps = {value.step_id: value for value in protocol.steps}
    external_consumer = "source" if has_source_step else "prospective-controller-execution"
    edges: list[CandidateGraphEdge] = [
        CandidateGraphEdge(
            edge_id=f"edge.external.{source_input.input_id}.{external_consumer}",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=source_input.input_id,
            consumer_node_id=external_consumer,
            consumer_input_id=("held-source-manifest" if has_source_step else "admission-parent-receipt"),
            scientific_role=source_input.scientific_role,
            logical_artifact_id=source_input.logical_artifact_id,
            payload_schema=source_input.payload_schema,
            media_type=source_input.media_type,
            maximum_size_bytes=source_input.maximum_size_bytes,
            outcome_access=source_input.outcome_access,
            visibility_ceiling=source_input.visibility_ceiling,
            barrier=BarrierKind.NONE,
        )
    ]
    for step in protocol.steps:
        for parent_id in step.dependency_step_ids:
            parent = steps[parent_id]
            output = parent.outputs[0]
            edges.append(
                CandidateGraphEdge(
                    edge_id=f"edge.{parent_id}.{step.step_id}",
                    producer_node_id=parent_id,
                    producer_output_id=output.output_id,
                    external_input_id=None,
                    consumer_node_id=step.step_id,
                    consumer_input_id=f"input-{parent_id}",
                    scientific_role=(
                        ScientificInputRole.QUALIFICATION
                        if parent.stage is ScientificStage.FREEZE
                        else ScientificInputRole.OUTCOME
                    ),
                    logical_artifact_id=f"artifact.{parent_id}.{output.output_id}",
                    payload_schema=output.payload_schema,
                    media_type=output.media_type,
                    maximum_size_bytes=parent.resource_budget.output_bytes,
                    outcome_access=parent.requested_outcome_access,
                    visibility_ceiling=parent.visibility_ceiling,
                    barrier=step.barrier,
                )
            )
    return CandidateScientificGraph(
        graph_id=f"graph.{definition.slug}.standard-observation-to-admission",
        external_inputs=(source_input,),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _coverage(
    *,
    coverage_id: str,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> ObligationCoverage:
    incoming = {
        node.node_id: tuple(
            sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id)
        )
        for node in graph.nodes
    }
    owners = {
        obligation_id: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation_id in step.obligation_ids
    }
    fallback_owner = (
        "report",
        next(value for value in protocol.steps if value.step_id == "report").outputs[0].output_id,
    )
    bindings = tuple(
        ObligationCoverageBinding(
            obligation_id=obligation_id,
            proof_owner_node_id=owners.get(obligation_id, fallback_owner)[0],
            required_output_id=owners.get(obligation_id, fallback_owner)[1],
            contributor_edge_ids=incoming[owners.get(obligation_id, fallback_owner)[0]],
        )
        for obligation_id in required_candidate_obligation_ids(experiment, protocol)
    )
    return ObligationCoverage(
        coverage_id=coverage_id,
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def _formal_coverage(
    *,
    definition: _Definition,
    register: FormalGapRegister,
    draft_id: str,
    inventory: FormalGapSourceCapabilityInventory,
) -> FormalGapCoverage:
    applicability = derive_formal_gap_applicability(register, inventory)
    inapplicable = set(inventory.denominator_inapplicable_gap_ids)
    assignments = tuple(
        (
            FormalGapCoverageAssignment(
                gap_id=gap.gap_id,
                disposition=FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR,
                readiness_reason=None,
                reason_codes=("DENOMINATOR_MATHEMATICAL_OBJECT_UNDEFINED",),
                selected_estimator_family_id=None,
                selected_control_ids=(),
                selected_multiplicity_family_id=None,
                obligation_ids=(),
                output_ids=(),
                adjudication_owner_ids=(),
            )
            if gap.gap_id in inapplicable
            else FormalGapCoverageAssignment(
                gap_id=gap.gap_id,
                disposition=FormalGapCoverageDisposition.TEST_IN_THIS_ACT,
                readiness_reason=None,
                reason_codes=(),
                selected_estimator_family_id=gap.estimator_family_ids[0],
                selected_control_ids=gap.control_ids,
                selected_multiplicity_family_id=gap.multiplicity_family_id,
                obligation_ids=(formal_gap_obligation_id(gap.gap_id),),
                output_ids=("formal-gap-panel",),
                adjudication_owner_ids=("formal-panel-evaluator",),
            )
        )
        for gap in register.gaps
    )
    return FormalGapCoverage(
        coverage_id=f"formal-gap-coverage.{definition.slug}.observation-to-controller-use",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=inventory.denominator_id,
        candidate_act_id=draft_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _entry_package(
    *,
    definition: _Definition,
    draft: StudyDraft,
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
) -> ExperimentEntryPackage:
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(
                f"binding.{definition.slug}.{requirement.value.lower().replace('_', '-')}",
            ),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("EXECUTION_AND_REVEAL_AUTHORITY_SEPARATE",)
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(
            ExperimentEntryRequirement,
            key=lambda value: value.value,
        )
    )
    checklist = ExperimentEntryChecklist(
        checklist_id=f"entry-checklist.{definition.slug}.observation-to-controller-use",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        execution_route_id=f"route.open-sim.{definition.slug}.standard",
        durability_disposition_id="durability.limited-sole-copy-no-replica",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id=f"experiment-entry-package.{definition.slug}.observation-to-controller-use",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


@dataclass(frozen=True, slots=True)
class OpenSimulatorAuthoringBundle:
    definition: _Definition
    source_manifest: OpenSimulatorSourceManifest
    source_config: SourceMaterializationConfig
    capability_config: OpenSimulatorStudyConfig
    admission_policy: OpenSimulatorAdmissionPolicy
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    qualification: MaterializationQualificationReceipt
    inventory: FormalGapSourceCapabilityInventory
    formal_configs: tuple[FormalDomainMethodConfig, ...]
    panel_config: FormalPanelEvaluatorConfig
    registry: CapabilityRegistry
    primary_template: StudyTemplate
    conditional_template: StudyTemplate
    draft: StudyDraft
    package: StudyDefinition


def _conditional_template(
    *,
    definition: _Definition,
    config: OpenSimulatorStudyConfig,
    registry: CapabilityRegistry,
) -> StudyTemplate:
    ref = _config_ref(config, config.config_id)
    controller = _step(
        definition=definition,
        stage_id="prospective-controller-execution",
        stage=ScientificStage.CONTROLLER,
        config_ref=ref,
        dependencies=(),
        outputs=(
            _output(
                "prospective-controller-execution-record",
                OpenSimulatorProspectiveEvaluationInput.SCHEMA,
            ),
        ),
        permissions=_DEVELOP,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility=VisibilityCeiling.PROSPECTIVE,
        barrier=BarrierKind.AUTHORITY,
        obligations=("fresh-prospective-controller-execution",),
    )
    evaluator = _step(
        definition=definition,
        stage_id="prospective-controller-evaluation",
        stage=ScientificStage.EVALUATE,
        config_ref=ref,
        dependencies=("prospective-controller-execution",),
        outputs=(_output("controller-use-adjudication", ScientificAdjudicationRecord.SCHEMA),),
        permissions=_EVALUATE,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility=VisibilityCeiling.OUTCOME_VISIBLE,
        barrier=BarrierKind.REVEAL,
        obligations=("fresh-controller-use-adjudication",),
    )
    protocol = ProtocolTemplate(
        template_id=f"protocol.{definition.slug}.conditional-controller-use",
        template_version="1.0.0",
        steps=tuple(sorted((controller, evaluator), key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=True,
        nonactuating=True,
    )
    graph = _graph(
        definition=definition,
        protocol=protocol,
        registry=registry,
        source=definition.source_manifest,
    )
    coverage = ObligationCoverage(
        coverage_id=f"obligation-coverage.{definition.slug}.conditional-controller-use",
        bindings=tuple(
            ObligationCoverageBinding(
                obligation_id=obligation,
                proof_owner_node_id=step.step_id,
                required_output_id=step.outputs[0].output_id,
                contributor_edge_ids=tuple(
                    sorted(
                        value.edge_id
                        for value in graph.edges
                        if value.consumer_node_id == step.step_id
                    )
                ),
            )
            for step in protocol.steps
            for obligation in step.obligation_ids
        ),
    )
    return StudyTemplate(
        template_key=f"open-sim.{definition.slug}.conditional-controller-use",
        template_version="1.0.0",
        protocol=protocol,
        graph=graph,
        coverage=coverage,
    )


def build_open_simulator_authoring_bundle(
    *,
    kind: OpenSimulatorKind,
    register: FormalGapRegister,
    implementation_sha256: str,
) -> OpenSimulatorAuthoringBundle:
    """Build one exact outcome-blind standard authoring/candidate family."""

    definition = _definition(kind)
    admission_policy = _admission_policy(kind)
    source_manifest = definition.source_manifest
    config = open_simulator_config(kind)
    system = open_simulator_system(kind)
    experiment = open_simulator_experiment(kind, system)
    campaign = open_simulator_campaign(kind, system, experiment)
    source_config = SourceMaterializationConfig(
        config_id=f"source-config.{definition.slug}.held-manifest",
        source_id=source_manifest.source_id,
        role=SourceMaterializationRole.PREPARED_MEDIUM,
        content_sha256=source_manifest.fingerprint(),
        expected_size_bytes=len(source_manifest.canonical_bytes()),
        maximum_bytes=1024 * 1024,
        payload_schema=OpenSimulatorSourceManifest.SCHEMA,
        media_type="application/json",
        read_mode=SourceReadMode.ORDINARY_BOUNDED,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    observation_operator = ObjectIdentity(
        object_id=f"observer.{definition.slug}.native",
        object_schema='empirical-lawhood/simulators/open-source-response/observation-operator',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id=f"qualification.{definition.slug}.held-source",
        source_id=source_manifest.source_id,
        materialization=ObjectIdentity.from_record(
            source_manifest.source_id,
            source_manifest,
        ),
        content_sha256=source_manifest.fingerprint(),
        evidence_world_id=system.world.world_id,
        observation_operator=observation_operator,
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
        native_unit_ids=tuple(sorted({value.native_unit for value in system.quantities})),
        frame_ids=tuple(sorted({value.coordinate_frame for value in system.quantities})),
        clock_ids=tuple(value.clock_id for value in system.clocks),
        receiver_semantics_id=system.relation.relation_id,
        validity_contract_id=experiment.obligations.validity.validity_id,
        uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    all_operands = tuple(
        sorted({item for gap in register.gaps for item in gap.required_operand_ids})
    )
    all_prerequisites = tuple(
        sorted({item for gap in register.gaps for item in gap.support_prerequisite_ids})
    )
    estimator_ids = tuple(
        sorted({item for gap in register.gaps for item in gap.estimator_family_ids})
    )
    multiplicity_ids = tuple(sorted({gap.multiplicity_family_id for gap in register.gaps}))
    inapplicable = (
        "gap.geometry.cohomology",
        "gap.geometry.information-geometry",
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id=f"formal-source-inventory.{definition.slug}.observation-to-controller-use",
        denominator_id=system.system_id,
        evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        source_materializations=(
            ObjectIdentity.from_record(source_manifest.source_id, source_manifest),
        ),
        present_operand_ids=all_operands,
        satisfied_prerequisite_ids=all_prerequisites,
        independent_unit_ids=tuple(
            sorted((*definition.development_units, *definition.evaluation_units))
        ),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
        available_estimator_family_ids=estimator_ids,
        available_control_ids=tuple(sorted(value.control_id for value in experiment.controls)),
        multiplicity_family_ids=multiplicity_ids,
        denominator_inapplicable_gap_ids=inapplicable,
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    design_input = DesignInputRecord(
        input_id=f"design-input.{definition.slug}.frozen-default",
        object_identity=ObjectIdentity.from_record(system.system_id, system),
        materialization_sha256=system.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    base_manifests = _base_capability_manifests(
        definition,
        config,
        admission_policy,
        implementation_sha256,
    )
    formal_manifests = formal_method_capability_manifests()
    registry = CapabilityRegistry(
        registry_id=f"registry.open-sim.{definition.slug}.observation-to-controller-use",
        capabilities=tuple(
            sorted((*base_manifests, *formal_manifests), key=lambda value: value.registry_id)
        ),
    )
    draft_id = f"draft.{definition.slug}.observation-to-controller-use"
    draft_stub = StudyDraft(
        draft_id=draft_id,
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            f"What denominator-local response laws and admissible controller "
            f"directions are supported in {definition.label}?"
        ),
        alternative_ids=tuple(
            sorted(
                (
                    f"alternative.{definition.slug}.local-law-supported",
                    f"alternative.{definition.slug}.local-law-not-supported",
                    f"alternative.{definition.slug}.unevaluable",
                )
            )
        ),
        design_origin=DesignOrigin(
            origin_id=f"origin.{definition.slug}.owner-predeclared",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=(design_input.input_id,),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=(design_input,),
        development_unit_ids=tuple(sorted(definition.development_units)),
        evaluation_unit_ids=tuple(
            sorted((*definition.evaluation_units, *definition.prospective_units))
        ),
        development_seed_ids=(f"seed.{definition.slug}.development-fixed",),
        evaluation_seed_ids=(f"seed.{definition.slug}.evaluation-fixed",),
        unresolved_decisions=(),
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=f"open-sim.{definition.slug}.standard-observation-to-admission.formal-standard",
        capability_selections=tuple(
            CapabilitySelection(
                capability_key=value.capability_key,
                capability_version=value.capability_version,
                implementation_sha256=value.implementation_sha256,
            )
            for value in registry.capabilities
        ),
        source_materializations=(
            SourceMaterializationRef(
                source_id=source_manifest.source_id,
                role=SourceMaterializationRole.PREPARED_MEDIUM,
                evidence_world_id=system.world.world_id,
                materialization=ObjectIdentity.from_record(
                    source_manifest.source_id,
                    source_manifest,
                ),
                content_sha256=source_manifest.fingerprint(),
                source_config_sha256=source_config.fingerprint(),
                observation_operator=observation_operator,
                numerical_view_ids=qualification.numerical_view_ids,
                qualification_receipt=ObjectIdentity.from_record(
                    qualification.receipt_id,
                    qualification,
                ),
                access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            ),
        ),
        resource_ceiling=_PROGRAMME_BUDGET,
        conditional_successor=ConditionalChildRequest(
            request_id=f"conditional.{definition.slug}.controller-use",
            parent_node_id="admission",
            parent_receipt_input_id=f"parent-receipt.{definition.slug}.admission",
            gate_kind=ConditionalGateKind.ADMISSION_AND_REACHABILITY_PASSED,
            template_key=f"open-sim.{definition.slug}.conditional-controller-use",
            evaluation_unit_ids=tuple(sorted(definition.prospective_units)),
            eligible_disposition=ConditionalTerminalDisposition.EXECUTION_ELIGIBLE,
            ineligible_disposition=(
                ConditionalTerminalDisposition.NOT_ATTEMPTED_PREREQUISITE_NOT_MET
            ),
        ),
    )
    formal_coverage = _formal_coverage(
        definition=definition,
        register=register,
        draft_id=draft_id,
        inventory=inventory,
    )
    formal_configs, panel_config = standard_formal_method_configs(
        register,
        formal_coverage,
    )
    base_protocol = _base_protocol(definition, config, admission_policy)
    base_graph = _graph(
        definition=definition,
        protocol=base_protocol,
        registry=registry,
        source=source_manifest,
    )
    base_template = StudyTemplate(
        template_key=f"open-sim.{definition.slug}.standard-observation-to-admission",
        template_version="1.0.0",
        protocol=base_protocol,
        graph=base_graph,
        coverage=_coverage(
            coverage_id=f"obligation-coverage.{definition.slug}.observation-to-admission",
            experiment=experiment,
            protocol=base_protocol,
            graph=base_graph,
        ),
    )
    formal_template = add_standard_formal_analysis(
        base=base_template,
        register=register,
        manifests=formal_manifests,
        domain_configs=tuple(
            (
                value.domain,
                _config_ref(value, value.config_id),
            )
            for value in formal_configs
        ),
        panel_config=_config_ref(panel_config, panel_config.config_id),
        formal_input_step_id="development",
        report_step_id="report",
        sealed_evaluation_step_id="sealed-evaluation",
        sealed_evaluation_output_ids=tuple(
            (
                domain,
                f"formal-evaluation-{domain.value.lower()}",
            )
            for domain in FormalDomain
        ),
        formal_panel_consumer_step_ids=("admission", "rung-adjudicate"),
    )
    conditional_template = _conditional_template(
        definition=definition,
        config=config,
        registry=registry,
    )
    draft = draft_stub
    entry = _entry_package(
        definition=definition,
        draft=draft,
        register=register,
        coverage=formal_coverage,
    )
    package = StudyDefinition(
        package_id=f"programme-authoring-package.{definition.slug}.observation-to-controller-use",
        draft=draft,
        entry_package=entry,
    )
    return OpenSimulatorAuthoringBundle(
        definition=definition,
        source_manifest=source_manifest,
        source_config=source_config,
        capability_config=config,
        admission_policy=admission_policy,
        system=system,
        experiment=experiment,
        campaign=campaign,
        qualification=qualification,
        inventory=inventory,
        formal_configs=formal_configs,
        panel_config=panel_config,
        registry=registry,
        primary_template=formal_template,
        conditional_template=conditional_template,
        draft=draft,
        package=package,
    )


class OpenSimulatorConfigDecoder:
    """Strict static config decoder; no import, URL, path, SQL, or callable."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        config: OpenSimulatorStudyConfig | OpenSimulatorAdmissionPolicy,
    ) -> None:
        self.manifest = manifest
        self.config = config

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        from empirical_lawhood.kernel.decoding import decode_canonical_bytes

        if expected_schema != self.config.SCHEMA:
            raise ValueError("open simulator config schema differs")
        observed = decode_canonical_bytes(
            payload,
            type(self.config),
            maximum_bytes=1024 * 1024,
        )
        if observed != self.config:
            raise ValueError("open simulator config differs from static registration")


def open_simulator_candidate_catalog(
    bundles: tuple[OpenSimulatorAuthoringBundle, ...],
) -> CandidateCapabilityCatalog:
    registrations: dict[str, CandidateCapabilityRegistration] = {}
    templates: list[StudyTemplate] = []
    for bundle in bundles:
        for manifest in bundle.registry.capabilities:
            config_media_type = "application/json"
            registrations.setdefault(
                manifest.registry_id,
                CandidateCapabilityRegistration(
                    manifest=manifest,
                    provider_key=manifest.capability_key,
                    provider_version=manifest.capability_version,
                    config_media_type=config_media_type,
                    maximum_config_bytes=1024 * 1024,
                ),
            )
        templates.extend((bundle.primary_template, bundle.conditional_template))
    return CandidateCapabilityCatalog(
        catalog_id="catalog.open-simulator-observation-to-controller-use",
        registrations=tuple(
            sorted(registrations.values(), key=lambda value: value.registration_id)
        ),
        templates=tuple(sorted(templates, key=lambda value: value.template_key)),
    )


def open_simulator_config_decoders(
    bundles: tuple[OpenSimulatorAuthoringBundle, ...],
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    values: dict[str, CandidateCapabilityConfigDecoder] = {}
    formal_manifests: dict[str, CapabilityManifest] = {}
    formal_configs_by_registration: dict[
        str,
        list[FormalDomainMethodConfig | FormalPanelEvaluatorConfig],
    ] = {}
    for bundle in bundles:
        formal_configs = {value.domain: value for value in bundle.formal_configs}
        for manifest in bundle.registry.capabilities:
            if manifest.capability_key in FORMAL_DOMAIN_CAPABILITY_KEYS.values():
                domain = next(
                    key
                    for key, value in FORMAL_DOMAIN_CAPABILITY_KEYS.items()
                    if value == manifest.capability_key
                )
                formal_manifests[manifest.registry_id] = manifest
                formal_configs_by_registration.setdefault(manifest.registry_id, []).append(
                    formal_configs[domain]
                )
            elif manifest.capability_key == FORMAL_PANEL_CAPABILITY_KEY:
                formal_manifests[manifest.registry_id] = manifest
                formal_configs_by_registration.setdefault(manifest.registry_id, []).append(
                    bundle.panel_config
                )
            else:
                config: OpenSimulatorStudyConfig | OpenSimulatorAdmissionPolicy = (
                    bundle.admission_policy
                    if manifest.capability_key.endswith(".admission")
                    else bundle.capability_config
                )
                values.setdefault(
                    manifest.registry_id,
                    OpenSimulatorConfigDecoder(
                        manifest,
                        config,
                    ),
                )
    for registration_id, configs in formal_configs_by_registration.items():
        values[registration_id] = FormalCapabilityConfigDecoder(
            formal_manifests[registration_id],
            tuple(dict.fromkeys(configs)),
        )
    return tuple(values[key] for key in sorted(values))


def open_simulator_formal_catalog(
    register: FormalGapRegister,
) -> FormalMethodCatalog:
    return standard_formal_method_catalog(register)


def open_simulator_compilation_context(
    bundle: OpenSimulatorAuthoringBundle,
    *,
    implementation_sha256: str,
) -> CandidateCompilationContext:
    return CandidateCompilationContext(
        context_id=f"context.{bundle.definition.slug}.observation-to-controller-use",
        registry=bundle.registry,
        templates=tuple(
            sorted(
                (bundle.primary_template, bundle.conditional_template),
                key=lambda value: value.template_key,
            )
        ),
        qualifications=(bundle.qualification,),
        known_design_inputs=bundle.draft.design_inputs,
        implementation_sha256=implementation_sha256,
    )


def instantiate_open_simulator_prospective(
    *,
    parent_compilation: CandidateCompilationReport | StudyCompilationReport,
    admission_adjudication: OpenSimulatorAdmissionAdjudication,
) -> tuple[
    ConditionalChildInstantiation,
    CandidateCompilationReport | StudyCompilationReport | None,
]:
    "Derive the frozen controller use branch solely from the validated nine-gate record."

    from empirical_lawhood.runtime.conditional_children import instantiate_conditional_child

    if isinstance(parent_compilation, StudyCompilationReport):
        base_compilation = parent_compilation.base_report
    elif isinstance(parent_compilation, CandidateCompilationReport):
        base_compilation = parent_compilation
    else:
        raise TypeError("open simulator controller use requires a candidate compilation report")
    if base_compilation.candidate is None:
        raise ValueError("open simulator controller use parent has no compiled candidate")
    admission_step = next(
        (
            value
            for value in base_compilation.candidate.protocol.steps
            if value.step_id == "admission"
        ),
        None,
    )
    if (
        admission_step is None
        or admission_step.config.config_id != admission_adjudication.policy.object_id
        or admission_step.config.config_schema != admission_adjudication.policy.object_schema
        or admission_step.config.content_sha256 != admission_adjudication.policy.object_fingerprint
    ):
        raise ValueError("Admission result differs from the parent candidate's frozen policy")
    reasons = (
        ()
        if admission_adjudication.controller_use_eligible
        else (admission_adjudication.reason_codes or ("CONTROLLER_USE_NOT_ATTEMPTED_ADMISSION_PREREQUISITE_NOT_MET",))
    )
    instantiation, child = instantiate_conditional_child(
        parent_compilation=base_compilation,
        parent_record_id=admission_adjudication.adjudication_id,
        parent_record=admission_adjudication,
        eligible=admission_adjudication.controller_use_eligible,
        reason_codes=reasons,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    if child is None or not isinstance(
        parent_compilation,
        StudyCompilationReport,
    ):
        return instantiation, child
    if parent_compilation.candidate is None or child.candidate is None:
        raise ValueError("standard controller use child lost its formal authoring root")
    parent_standard = parent_compilation.candidate
    candidate_seed = {
        "base_candidate": ObjectIdentity.from_record(
            child.candidate.candidate_id,
            child.candidate,
        ),
        "parent_standard_candidate": ObjectIdentity.from_record(
            parent_standard.candidate_id,
            parent_standard,
        ),
        "parent_admission": ObjectIdentity.from_record(
            admission_adjudication.adjudication_id,
            admission_adjudication,
        ),
    }
    candidate_digest = hashlib.sha256(canonical_json_bytes(candidate_seed)).hexdigest()
    standard_child = replace(
        parent_standard,
        candidate_id=f"standard-candidate.conditional.{candidate_digest[:24]}",
        base_candidate=child.candidate,
    )
    report_seed = {
        "base_report": child,
        "candidate": standard_child,
        "parent_report": parent_compilation.report_id,
    }
    report_digest = hashlib.sha256(canonical_json_bytes(report_seed)).hexdigest()
    return (
        instantiation,
        replace(
            parent_compilation,
            report_id=f"standard-report.conditional.{report_digest[:24]}",
            base_report=child,
            candidate=standard_child,
        ),
    )


@dataclass(frozen=True, slots=True)
class OpenSimulatorConditionalChildResolver:
    "Public-composition adapter for exact admission-derived controller-use instantiation."

    @property
    def parent_record_schemas(self) -> Mapping[str, type[CanonicalRecord]]:
        return MappingProxyType({OpenSimulatorAdmissionAdjudication.SCHEMA: OpenSimulatorAdmissionAdjudication})

    def resolve(
        self,
        *,
        parent_compilation: StudyCompilationReport,
        parent_record: CanonicalRecord,
    ) -> ConditionalChildResolution:
        if not isinstance(parent_record, OpenSimulatorAdmissionAdjudication):
            raise TypeError("open simulator conditional child requires an exact admission adjudication")
        parent_base = parent_compilation.base_report.candidate
        if parent_base is None or parent_base.conditional_successor is None:
            raise ValueError("open simulator parent lacks its frozen controller-use child")
        external_input_id = parent_base.conditional_successor.request.parent_receipt_input_id
        instantiation, child = instantiate_open_simulator_prospective(
            parent_compilation=parent_compilation,
            admission_adjudication=parent_record,
        )
        if child is None:
            return ConditionalChildResolution(
                instantiation=instantiation,
                compilation=None,
                parent_input_binding=None,
            )
        if not isinstance(child, StudyCompilationReport):
            raise TypeError("open simulator conditional child lost its standard compilation root")
        if child.candidate is None:
            raise ValueError("open simulator conditional child lacks its exact child candidate")
        binding = bind_frozen_parent_input(
            candidate=child.candidate.base_candidate,
            external_input_id=external_input_id,
            parent_record_id=parent_record.adjudication_id,
            parent_record=parent_record,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        )
        return ConditionalChildResolution(
            instantiation=instantiation,
            compilation=child,
            parent_input_binding=binding,
        )


__all__ = [
    "OpenSimulatorActionDeliveryBinding",
    "OpenSimulatorAuthoringBundle",
    'OpenSimulatorStudyConfig',
    'OpenSimulatorConditionalChildResolver',
    "OpenSimulatorConfigDecoder",
    "OpenSimulatorKind",
    'OpenSimulatorAdmissionAdjudication',
    'OpenSimulatorAdmissionPolicy',
    'OpenSimulatorProspectiveEvaluationInput',
    "OpenSimulatorQuantityBand",
    "OpenSimulatorRungAdjudication",
    "OpenSimulatorRungResult",
    "OpenSimulatorSourceManifest",
    "OpenSimulatorStageRecord",
    "build_open_simulator_authoring_bundle",
    "open_simulator_candidate_catalog",
    "open_simulator_compilation_context",
    "open_simulator_config",
    "open_simulator_config_decoders",
    "open_simulator_definitions",
    "open_simulator_experiment",
    "open_simulator_formal_catalog",
    "open_simulator_system",
    'instantiate_open_simulator_prospective',
]
