"""Bounded truth-known executable-metatheory conformance campaign."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryEvidenceCeiling
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignStageRole
from empirical_lawhood.planning.obstruction_atlas import ObstructionOperationalStatus
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateGraphEdge,
    CandidateGraphExternalInput,
    CandidateGraphNode,
    CandidateScientificGraph,
    ContentIdentityPolicy,
    ScientificInputRole,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_metatheory_release import MetatheoryConformanceCase
from empirical_lawhood.runtime.metatheory_campaigns import MetatheoryCampaignCompilation
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolTemplate, ScientificStage


EXECUTABLE_METATHEORY_ACQUISITION_KEY = (
    'executable-metatheory.truth-known-acquisition-conformance'
)
EXECUTABLE_METATHEORY_DEVELOPMENT_KEY = (
    'executable-metatheory.truth-known-development-conformance'
)
EXECUTABLE_METATHEORY_QUALIFICATION_KEY = (
    'executable-metatheory.truth-known-qualification-conformance'
)
EXECUTABLE_METATHEORY_EVALUATOR_KEY = 'executable-metatheory.truth-known-evaluator-conformance'
EXECUTABLE_METATHEORY_PREDICTION_KEY = 'executable-metatheory.truth-known-prediction-conformance'
EXECUTABLE_METATHEORY_REPORTER_KEY = 'executable-metatheory.truth-known-reporter-conformance'
EXECUTABLE_METATHEORY_CONFORMANCE_VERSION = "1.0.0"
EXECUTABLE_METATHEORY_CONFORMANCE_IMPLEMENTATION_SHA256 = sha256(
    b'empirical-lawhood/executable-metatheory/truth-known-stage-conformance'
).hexdigest()
EXECUTABLE_METATHEORY_CONFORMANCE_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"


@dataclass(frozen=True, slots=True)
class MetatheoryConformanceFixture(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-fixture'

    fixture_id: str
    case_kind: MetatheoryConformanceCase
    physical_unit_ids: tuple[str, ...]
    companion_product_refs: tuple[ObjectIdentity, ...]
    truth_known: bool
    target_outcomes_embedded: bool
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.fixture_id, field_name="fixture_id")
        require_sorted_unique_strings(
            self.physical_unit_ids,
            field_name="physical_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.companion_product_refs,
            attribute="object_id",
            field_name="companion_product_refs",
        )
        if (
            not self.truth_known
            or self.target_outcomes_embedded
            or self.maximum_structural_evidence_ceiling
            is not MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE
            or self.grants_authority
        ):
            raise ValueError("metatheory conformance fixture exceeds truth-known scope")


@dataclass(frozen=True, slots=True)
class MetatheoryConformanceCampaignConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-campaign-config'

    config_id: str
    applicable_roles: tuple[MetatheoryCampaignStageRole, ...]
    fixture: ObjectIdentity
    truth_known_only: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(
            self.applicable_roles,
            attribute="value",
            field_name="applicable_roles",
        )
        if not self.applicable_roles or MetatheoryCampaignStageRole.REPORT not in set(
            self.applicable_roles
        ):
            raise ValueError("metatheory conformance config lacks an applicable report")
        if self.fixture.object_schema != MetatheoryConformanceFixture.SCHEMA:
            raise ValueError("metatheory stage config binds another fixture schema")
        if not self.truth_known_only or self.grants_authority:
            raise ValueError("metatheory campaign config exceeds truth-known scope")


_ROLES_BY_CAPABILITY_KEY = {
    EXECUTABLE_METATHEORY_ACQUISITION_KEY: frozenset(
        {
            MetatheoryCampaignStageRole.SOURCE_PIPELINE,
            MetatheoryCampaignStageRole.TARGET_ACQUISITION,
        }
    ),
    EXECUTABLE_METATHEORY_DEVELOPMENT_KEY: frozenset(
        {
            MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION,
            MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT,
        }
    ),
    EXECUTABLE_METATHEORY_QUALIFICATION_KEY: frozenset(
        {
            MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION,
            MetatheoryCampaignStageRole.COORDINATE_EVALUATION,
            MetatheoryCampaignStageRole.LAW_QUALIFICATION_OPTIONAL,
            MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL,
            MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL,
            MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL,
        }
    ),
    EXECUTABLE_METATHEORY_EVALUATOR_KEY: frozenset(
        {MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION}
    ),
    EXECUTABLE_METATHEORY_PREDICTION_KEY: frozenset(
        {MetatheoryCampaignStageRole.PREDICTION_ISSUE}
    ),
    EXECUTABLE_METATHEORY_REPORTER_KEY: frozenset({MetatheoryCampaignStageRole.REPORT}),
}


@dataclass(frozen=True, slots=True)
class MetatheoryConformanceExecutableConfigPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-executable-config-payload'

    payload_id: str
    campaign: MetatheoryConformanceCampaignConfig
    fixture: MetatheoryConformanceFixture
    owned_roles: tuple[MetatheoryCampaignStageRole, ...]
    capability_key: str
    capability_version: str
    implementation_sha256: str
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.payload_id, field_name="payload_id")
        require_sorted_unique_ids(
            self.owned_roles,
            attribute="value",
            field_name="owned_roles",
        )
        expected_roles = _ROLES_BY_CAPABILITY_KEY.get(self.capability_key)
        if expected_roles is None or set(self.owned_roles) != expected_roles:
            raise ValueError("metatheory executable config role ownership differs")
        if not self.owned_roles:
            raise ValueError("metatheory executable config owns no applicable role")
        if self.campaign.fixture != ObjectIdentity.from_record(
            self.fixture.fixture_id,
            self.fixture,
        ):
            raise ValueError("metatheory executable config crosses fixtures")
        if (
            self.capability_version != EXECUTABLE_METATHEORY_CONFORMANCE_VERSION
            or self.implementation_sha256 != EXECUTABLE_METATHEORY_CONFORMANCE_IMPLEMENTATION_SHA256
            or self.grants_authority
        ):
            raise ValueError("metatheory executable config names another implementation")


def _validate_executable_config_wrapper(
    *,
    config_id: str,
    payload: MetatheoryConformanceExecutableConfigPayload,
    expected_capability_key: str,
) -> None:
    validate_stable_id(config_id, field_name="config_id")
    if payload.capability_key != expected_capability_key:
        raise ValueError("metatheory executable config wrapper names another capability")


@dataclass(frozen=True, slots=True)
class MetatheoryConformanceAcquisitionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-acquisition-config'

    config_id: str
    payload: MetatheoryConformanceExecutableConfigPayload

    def __post_init__(self) -> None:
        _validate_executable_config_wrapper(
            config_id=self.config_id,
            payload=self.payload,
            expected_capability_key=EXECUTABLE_METATHEORY_ACQUISITION_KEY,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryConformanceDevelopmentConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-development-config'

    config_id: str
    payload: MetatheoryConformanceExecutableConfigPayload

    def __post_init__(self) -> None:
        _validate_executable_config_wrapper(
            config_id=self.config_id,
            payload=self.payload,
            expected_capability_key=EXECUTABLE_METATHEORY_DEVELOPMENT_KEY,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryConformanceQualificationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-qualification-config'

    config_id: str
    payload: MetatheoryConformanceExecutableConfigPayload

    def __post_init__(self) -> None:
        _validate_executable_config_wrapper(
            config_id=self.config_id,
            payload=self.payload,
            expected_capability_key=EXECUTABLE_METATHEORY_QUALIFICATION_KEY,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryConformanceEvaluatorConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-evaluator-config'

    config_id: str
    payload: MetatheoryConformanceExecutableConfigPayload

    def __post_init__(self) -> None:
        _validate_executable_config_wrapper(
            config_id=self.config_id,
            payload=self.payload,
            expected_capability_key=EXECUTABLE_METATHEORY_EVALUATOR_KEY,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryConformancePredictionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-prediction-config'

    config_id: str
    payload: MetatheoryConformanceExecutableConfigPayload

    def __post_init__(self) -> None:
        _validate_executable_config_wrapper(
            config_id=self.config_id,
            payload=self.payload,
            expected_capability_key=EXECUTABLE_METATHEORY_PREDICTION_KEY,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryConformanceReporterConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-reporter-config'

    config_id: str
    payload: MetatheoryConformanceExecutableConfigPayload

    def __post_init__(self) -> None:
        _validate_executable_config_wrapper(
            config_id=self.config_id,
            payload=self.payload,
            expected_capability_key=EXECUTABLE_METATHEORY_REPORTER_KEY,
        )


MetatheoryConformanceExecutableConfig = (
    MetatheoryConformanceAcquisitionConfig
    | MetatheoryConformanceDevelopmentConfig
    | MetatheoryConformanceQualificationConfig
    | MetatheoryConformanceEvaluatorConfig
    | MetatheoryConformancePredictionConfig
    | MetatheoryConformanceReporterConfig
)
METATHEORY_CONFORMANCE_EXECUTABLE_CONFIG_TYPES = (
    MetatheoryConformanceAcquisitionConfig,
    MetatheoryConformanceDevelopmentConfig,
    MetatheoryConformanceEvaluatorConfig,
    MetatheoryConformancePredictionConfig,
    MetatheoryConformanceQualificationConfig,
    MetatheoryConformanceReporterConfig,
)


@dataclass(frozen=True, slots=True)
class MetatheoryConformanceStageTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-stage-terminal'

    terminal_id: str
    config: ObjectIdentity
    fixture: ObjectIdentity
    role: MetatheoryCampaignStageRole
    case_kind: MetatheoryConformanceCase
    predecessor_terminal: ObjectIdentity | None
    physical_unit_ids: tuple[str, ...]
    scientific_disposition: MetatheoryAggregateDisposition
    operational_status: ObstructionOperationalStatus
    target_contact_count: int
    reveal_count: int
    reason_codes: tuple[str, ...]
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    parent_promotion_permitted: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.terminal_id, field_name="terminal_id")
        require_sorted_unique_strings(
            self.physical_unit_ids,
            field_name="physical_unit_ids",
            allow_empty=False,
        )
        for name in ("target_contact_count", "reveal_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > 1:
                raise ValueError("conformance contact/reveal count must be zero or one")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.maximum_ordinary_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.maximum_structural_evidence_ceiling
            is not MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE
            or self.parent_promotion_permitted
            or self.grants_authority
        ):
            raise ValueError("metatheory conformance terminal exceeds its ceiling")


def _validate_terminal_envelope(
    *,
    envelope_id: str,
    terminal: MetatheoryConformanceStageTerminal,
    expected_role: MetatheoryCampaignStageRole,
) -> None:
    validate_stable_id(envelope_id, field_name="envelope_id")
    if terminal.role is not expected_role:
        raise ValueError("metatheory terminal envelope role differs")


@dataclass(frozen=True, slots=True)
class MetatheorySourcePipelineTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-source-pipeline-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.SOURCE_PIPELINE,
        )


@dataclass(frozen=True, slots=True)
class MetatheorySourceQualificationTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-source-qualification-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=(MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION),
        )


@dataclass(frozen=True, slots=True)
class MetatheoryCoordinateConstructionTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-coordinate-construction-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryCoordinateEvaluationTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-coordinate-evaluation-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.COORDINATE_EVALUATION,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryLawQualificationTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-law-qualification-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.LAW_QUALIFICATION_OPTIONAL,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryAtlasQualificationTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-atlas-qualification-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryDecisionAssuranceTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-decision-assurance-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryPropertySurvivalTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-property-survival-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryPredictionIssueTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-prediction-issue-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.PREDICTION_ISSUE,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryTargetAcquisitionTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-target-acquisition-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.TARGET_ACQUISITION,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryRevealAdjudicationTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-reveal-adjudication-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryObstructionCloseoutTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/metatheory-obstruction-closeout-terminal'

    envelope_id: str
    terminal: MetatheoryConformanceStageTerminal

    def __post_init__(self) -> None:
        _validate_terminal_envelope(
            envelope_id=self.envelope_id,
            terminal=self.terminal,
            expected_role=MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT,
        )


MetatheoryConformanceTerminalEnvelope = (
    MetatheorySourcePipelineTerminal
    | MetatheorySourceQualificationTerminal
    | MetatheoryCoordinateConstructionTerminal
    | MetatheoryCoordinateEvaluationTerminal
    | MetatheoryLawQualificationTerminal
    | MetatheoryAtlasQualificationTerminal
    | MetatheoryDecisionAssuranceTerminal
    | MetatheoryPropertySurvivalTerminal
    | MetatheoryPredictionIssueTerminal
    | MetatheoryTargetAcquisitionTerminal
    | MetatheoryRevealAdjudicationTerminal
    | MetatheoryObstructionCloseoutTerminal
)
METATHEORY_CONFORMANCE_TERMINAL_TYPES = (
    MetatheoryAtlasQualificationTerminal,
    MetatheoryCoordinateConstructionTerminal,
    MetatheoryCoordinateEvaluationTerminal,
    MetatheoryDecisionAssuranceTerminal,
    MetatheoryLawQualificationTerminal,
    MetatheoryObstructionCloseoutTerminal,
    MetatheoryPredictionIssueTerminal,
    MetatheoryPropertySurvivalTerminal,
    MetatheoryRevealAdjudicationTerminal,
    MetatheorySourcePipelineTerminal,
    MetatheorySourceQualificationTerminal,
    MetatheoryTargetAcquisitionTerminal,
)
METATHEORY_CONFORMANCE_TERMINAL_TYPE_BY_ROLE: dict[
    MetatheoryCampaignStageRole,
    type[MetatheoryConformanceTerminalEnvelope],
] = {
    MetatheoryCampaignStageRole.SOURCE_PIPELINE: MetatheorySourcePipelineTerminal,
    MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION: (
        MetatheorySourceQualificationTerminal
    ),
    MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION: (
        MetatheoryCoordinateConstructionTerminal
    ),
    MetatheoryCampaignStageRole.COORDINATE_EVALUATION: (MetatheoryCoordinateEvaluationTerminal),
    MetatheoryCampaignStageRole.LAW_QUALIFICATION_OPTIONAL: (
        MetatheoryLawQualificationTerminal
    ),
    MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL: (
        MetatheoryAtlasQualificationTerminal
    ),
    MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL: (
        MetatheoryDecisionAssuranceTerminal
    ),
    MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL: (
        MetatheoryPropertySurvivalTerminal
    ),
    MetatheoryCampaignStageRole.PREDICTION_ISSUE: MetatheoryPredictionIssueTerminal,
    MetatheoryCampaignStageRole.TARGET_ACQUISITION: (MetatheoryTargetAcquisitionTerminal),
    MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION: (MetatheoryRevealAdjudicationTerminal),
    MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT: (MetatheoryObstructionCloseoutTerminal),
}


def metatheory_conformance_study_config(
    *,
    config_id: str,
    applicable_roles: tuple[MetatheoryCampaignStageRole, ...],
    fixture: MetatheoryConformanceFixture,
) -> MetatheoryConformanceCampaignConfig:
    """Bind one exact issued campaign config to every applicable protocol role."""

    expected_roles = set(MetatheoryCampaignStageRole) - {
        MetatheoryCampaignStageRole.LAW_QUALIFICATION_OPTIONAL
    }
    if set(applicable_roles) != expected_roles:
        raise ValueError("truth-known conformance requires its exact closed role topology")
    fixture_identity = ObjectIdentity.from_record(fixture.fixture_id, fixture)
    return MetatheoryConformanceCampaignConfig(
        config_id=config_id,
        applicable_roles=tuple(sorted(applicable_roles, key=lambda value: value.value)),
        fixture=fixture_identity,
        truth_known_only=True,
        grants_authority=False,
    )


def metatheory_conformance_executable_configs(
    campaign: MetatheoryConformanceCampaignConfig,
    fixture: MetatheoryConformanceFixture,
) -> tuple[MetatheoryConformanceExecutableConfig, ...]:
    """Build the six stage-kind-correct issued configs for one campaign."""

    wrappers: tuple[
        tuple[
            str,
            type[
                MetatheoryConformanceAcquisitionConfig
                | MetatheoryConformanceDevelopmentConfig
                | MetatheoryConformanceQualificationConfig
                | MetatheoryConformanceEvaluatorConfig
                | MetatheoryConformancePredictionConfig
                | MetatheoryConformanceReporterConfig
            ],
        ],
        ...,
    ] = (
        (EXECUTABLE_METATHEORY_ACQUISITION_KEY, MetatheoryConformanceAcquisitionConfig),
        (EXECUTABLE_METATHEORY_DEVELOPMENT_KEY, MetatheoryConformanceDevelopmentConfig),
        (EXECUTABLE_METATHEORY_EVALUATOR_KEY, MetatheoryConformanceEvaluatorConfig),
        (EXECUTABLE_METATHEORY_PREDICTION_KEY, MetatheoryConformancePredictionConfig),
        (EXECUTABLE_METATHEORY_QUALIFICATION_KEY, MetatheoryConformanceQualificationConfig),
        (EXECUTABLE_METATHEORY_REPORTER_KEY, MetatheoryConformanceReporterConfig),
    )
    values: list[MetatheoryConformanceExecutableConfig] = []
    for capability_key, wrapper_type in wrappers:
        token = capability_key.rsplit("-", 1)[0].rsplit("-", 1)[-1]
        owned_roles = tuple(
            sorted(
                _ROLES_BY_CAPABILITY_KEY[capability_key],
                key=lambda value: value.value,
            )
        )
        payload = MetatheoryConformanceExecutableConfigPayload(
            payload_id=f"payload.{campaign.config_id}.{token}",
            campaign=campaign,
            fixture=fixture,
            owned_roles=owned_roles,
            capability_key=capability_key,
            capability_version=EXECUTABLE_METATHEORY_CONFORMANCE_VERSION,
            implementation_sha256=(EXECUTABLE_METATHEORY_CONFORMANCE_IMPLEMENTATION_SHA256),
            grants_authority=False,
        )
        values.append(
            wrapper_type(
                config_id=f"{campaign.config_id}.{token}",
                payload=payload,
            )
        )
    return tuple(sorted(values, key=lambda value: value.SCHEMA))


def metatheory_conformance_executable_config_for_role(
    configs: tuple[MetatheoryConformanceExecutableConfig, ...],
    role: MetatheoryCampaignStageRole,
) -> MetatheoryConformanceExecutableConfig:
    values = tuple(value for value in configs if role in set(value.payload.owned_roles))
    if len(values) != 1:
        raise ValueError("metatheory role lacks one executable config owner")
    return values[0]


def build_metatheory_conformance_graph(
    *,
    compilation: MetatheoryCampaignCompilation,
    registry: CapabilityRegistry,
    executable_configs: tuple[MetatheoryConformanceExecutableConfig, ...],
    fixture: MetatheoryConformanceFixture,
    fixture_input_id: str,
) -> CandidateScientificGraph:
    """Build a single closed chain over the already-compiled protocol."""

    validate_stable_id(fixture_input_id, field_name="fixture_input_id")
    applicable_roles = {value.role for value in compilation.stage_applicability if value.applicable}
    campaigns = tuple(value.payload.campaign for value in executable_configs)
    if not campaigns or any(value != campaigns[0] for value in campaigns[1:]):
        raise ValueError("metatheory executable configs cross campaign contracts")
    campaign = campaigns[0]
    if set(campaign.applicable_roles) != applicable_roles:
        raise ValueError("metatheory config and compiled applicability differ")
    if campaign.fixture != ObjectIdentity.from_record(fixture.fixture_id, fixture):
        raise ValueError("metatheory config and graph fixture differ")
    protocol: ProtocolTemplate = compilation.protocol
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=registry.resolve(
                step.capability_key,
                step.capability_version,
            ).implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    for role in applicable_roles:
        step = next(
            value
            for value in protocol.steps
            if value.step_id == f"metatheory-{role.value.lower().replace('_', '-')}"
        )
        executable_config = metatheory_conformance_executable_config_for_role(
            executable_configs,
            role,
        )
        if (
            step.config.config_id != executable_config.config_id
            or step.config.config_schema != executable_config.SCHEMA
            or step.config.content_sha256 != executable_config.fingerprint()
        ):
            raise ValueError("metatheory protocol and executable config differ")
    fixture_input = CandidateGraphExternalInput(
        input_id=fixture_input_id,
        scientific_role=ScientificInputRole.PREPARED_MEDIUM,
        logical_artifact_id=f"artifact.{fixture.fixture_id}",
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=fixture.fingerprint(),
        payload_schema=MetatheoryConformanceFixture.SCHEMA,
        media_type=EXECUTABLE_METATHEORY_CONFORMANCE_MEDIA_TYPE,
        maximum_size_bytes=4_000_000,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    role_to_step = {
        role: next(
            value
            for value in protocol.steps
            if value.step_id == f"metatheory-{role.value.lower().replace('_', '-')}"
        )
        for role in MetatheoryCampaignStageRole
        if any(value.role is role and value.applicable for value in compilation.stage_applicability)
    }
    ordered_roles = tuple(role for role in MetatheoryCampaignStageRole if role in role_to_step)
    first = role_to_step[ordered_roles[0]]
    edges = [
        CandidateGraphEdge(
            edge_id="edge.metatheory-conformance.fixture",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=fixture_input_id,
            consumer_node_id=first.step_id,
            consumer_input_id="truth-known-fixture",
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id=fixture_input.logical_artifact_id,
            payload_schema=fixture_input.payload_schema,
            media_type=fixture_input.media_type,
            maximum_size_bytes=fixture_input.maximum_size_bytes,
            outcome_access=fixture_input.outcome_access,
            visibility_ceiling=fixture_input.visibility_ceiling,
            barrier=BarrierKind.NONE,
        )
    ]
    for upstream_role, downstream_role in zip(
        ordered_roles,
        ordered_roles[1:],
    ):
        upstream = role_to_step[upstream_role]
        downstream = role_to_step[downstream_role]
        output = upstream.outputs[0]
        if (
            output.payload_schema
            != METATHEORY_CONFORMANCE_TERMINAL_TYPE_BY_ROLE[upstream_role].SCHEMA
        ):
            raise ValueError("nonterminal metatheory output appears before report")
        edges.append(
            CandidateGraphEdge(
                edge_id=(
                    f"edge.metatheory-conformance.{upstream_role.value.lower()}"
                    f".{downstream_role.value.lower()}"
                ),
                producer_node_id=upstream.step_id,
                producer_output_id=output.output_id,
                external_input_id=None,
                consumer_node_id=downstream.step_id,
                consumer_input_id="predecessor-terminal",
                scientific_role=ScientificInputRole.PARENT_RECEIPT,
                logical_artifact_id=f"artifact.{upstream.step_id}.{output.output_id}",
                payload_schema=output.payload_schema,
                media_type=output.media_type,
                maximum_size_bytes=upstream.resource_budget.output_bytes,
                outcome_access=(
                    OutcomeAccess.EVALUATION_REVEALED
                    if upstream.stage in {ScientificStage.EVALUATE, ScientificStage.REVEAL}
                    else upstream.requested_outcome_access
                ),
                visibility_ceiling=upstream.visibility_ceiling,
                barrier=downstream.barrier,
            )
        )
    if (
        role_to_step[ordered_roles[-1]].outputs[0].payload_schema
        != ScientificAdjudicationRecord.SCHEMA
    ):
        raise ValueError("metatheory conformance report lacks scientific adjudication output")
    return CandidateScientificGraph(
        graph_id=f"graph.{compilation.compilation_id}",
        external_inputs=(fixture_input,),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


__all__ = [
    "EXECUTABLE_METATHEORY_ACQUISITION_KEY",
    "EXECUTABLE_METATHEORY_CONFORMANCE_IMPLEMENTATION_SHA256",
    "EXECUTABLE_METATHEORY_CONFORMANCE_MEDIA_TYPE",
    "EXECUTABLE_METATHEORY_CONFORMANCE_VERSION",
    "EXECUTABLE_METATHEORY_DEVELOPMENT_KEY",
    "EXECUTABLE_METATHEORY_EVALUATOR_KEY",
    "EXECUTABLE_METATHEORY_PREDICTION_KEY",
    "EXECUTABLE_METATHEORY_QUALIFICATION_KEY",
    "EXECUTABLE_METATHEORY_REPORTER_KEY",
    "METATHEORY_CONFORMANCE_EXECUTABLE_CONFIG_TYPES",
    "METATHEORY_CONFORMANCE_TERMINAL_TYPES",
    "METATHEORY_CONFORMANCE_TERMINAL_TYPE_BY_ROLE",
    'MetatheoryAtlasQualificationTerminal',
    'MetatheoryConformanceAcquisitionConfig',
    'MetatheoryConformanceCase',
    'MetatheoryConformanceFixture',
    'MetatheoryConformanceCampaignConfig',
    'MetatheoryConformanceDevelopmentConfig',
    'MetatheoryConformanceEvaluatorConfig',
    'MetatheoryConformanceExecutableConfigPayload',
    'MetatheoryConformanceExecutableConfig',
    'MetatheoryConformancePredictionConfig',
    'MetatheoryConformanceQualificationConfig',
    'MetatheoryConformanceReporterConfig',
    'MetatheoryConformanceStageTerminal',
    'MetatheoryConformanceTerminalEnvelope',
    'MetatheoryCoordinateConstructionTerminal',
    'MetatheoryCoordinateEvaluationTerminal',
    'MetatheoryDecisionAssuranceTerminal',
    'MetatheoryLawQualificationTerminal',
    'MetatheoryObstructionCloseoutTerminal',
    'MetatheoryPredictionIssueTerminal',
    'MetatheoryPropertySurvivalTerminal',
    'MetatheoryRevealAdjudicationTerminal',
    'MetatheorySourcePipelineTerminal',
    'MetatheorySourceQualificationTerminal',
    'MetatheoryTargetAcquisitionTerminal',
    'build_metatheory_conformance_graph',
    'metatheory_conformance_study_config',
    'metatheory_conformance_executable_config_for_role',
    'metatheory_conformance_executable_configs',
]
