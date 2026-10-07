"Prospective Gym-TORAX selected-parent projection and measurement-through-law-qualification adjudication.\n\nThis module is the narrow bridge between the immutable Gym--TORAX metadata-complete episode\nartifacts and the existing generic confirmatory finite-action law owner.  The\nscience freeze is constructed from the frozen BT roster and artifact sidecars\nonly: no episode payload is opened.  A later authority-gated evaluator reveals\nthe exact 144 primary member/action views, builds the generic projection, and\ninvokes the ordinary G1/G1R producer and sole response-law finalizer once.\n\nThe remaining C/B/T cells are retained external evidence for optional,\nnon-gating interaction analysis.  They cannot enter the primary law roster.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, ClassVar, Mapping

import numpy as np

from empirical_lawhood.adapters.methods.confirmatory_finite_action import CONFIRMATORY_DESIGN_CONFIG_ID_RULE, CONFIRMATORY_FINITE_ACTION_KEY, CONFIRMATORY_FINITE_ACTION_VERSION, ConfirmatoryActionRole, ConfirmatoryActionRoleBinding, ConfirmatoryFiniteActionBindingReceipt, ConfirmatoryFiniteActionConfig, ConfirmatoryFiniteActionDesignTemplateBindingReceipt, ConfirmatoryFiniteActionDesign, ConfirmatoryFiniteActionDesignTemplate, ConfirmatoryFiniteActionProspectiveCore, ConfirmatoryPrefixSpec, ConfirmatoryResponseQuantity, ConfirmatoryUnitTierBinding, bind_confirmatory_finite_action_design_template, bind_confirmatory_finite_action_design
from empirical_lawhood.adapters.methods.confirmatory_finite_action_region_support import CONFIRMATORY_LOCAL_REGION_UNIT_COUNT, CONFIRMATORY_LOCAL_REGION_CRITICAL_VALUE, CONFIRMATORY_LOCAL_REGION_DEGREES_OF_FREEDOM, CONFIRMATORY_LOCAL_REGION_FAMILY_ALPHA, CONFIRMATORY_LOCAL_REGION_MATERIALITY, CONFIRMATORY_LOCAL_REGION_MINIMUM_POSITIVE_UNITS, CONFIRMATORY_LOCAL_REGION_PER_TEST_TAIL_PROBABILITY, CONFIRMATORY_LOCAL_REGION_TEST_COUNT, ConfirmatoryFiniteActionRegionSupportSpec, ConfirmatoryFiniteActionTierRegionBinding, ProspectiveConfirmatoryFiniteActionEvaluatorBinding, ProspectiveConfirmatoryFiniteActionEvaluation, ProspectiveConfirmatoryFiniteActionEvaluator, prospective_confirmatory_finite_action_evaluator_binding
from empirical_lawhood.adapters.methods.confirmatory_finite_action_registration import (
    confirmatory_finite_action_local_support_profile_evaluator,
)
from empirical_lawhood.adapters.methods.contracts import (
    CandidateClaimTemplate,
    FalsifierObligationTemplate,
    LawCandidateAxisBinding,
    LawCandidateAxisMap,
    LawObligationTemplate,
)
from empirical_lawhood.adapters.methods.evidence_projection import (
    ObservationValuePartition,
)
from empirical_lawhood.adapters.methods.evidence_projection import IdentificationEvidenceProjector, TaggedIdentificationProjectionPlan
from empirical_lawhood.adapters.methods.evidence_projection_extensions import build_identification_projection_extension
from empirical_lawhood.adapters.methods.evidence_projection_templates import IdentificationProjectionBindingStage, IdentificationProjectionObservationTemplate, IdentificationProjectionPayloadTemplate, IdentificationProjectionRole, IdentificationProjectionTemplateBindingReceipt, IdentificationProjectionTemplate, bind_identification_projection_template
from empirical_lawhood.adapters.methods.finite_action_identification import FiniteActionCandidateScaffold
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidateFamilyAssembler,
    CandidatePayloadDecoderRegistry,
    CanonicalFiniteActionCompatibilitySetDecoder,
    LawAssessmentAssembler,
    QualificationProfileEvaluatorRegistry,
    ResponseLawQualificationService,
)
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.laws import CausalStrength
from empirical_lawhood.kernel.obligations import FalsifierKind
from empirical_lawhood.kernel.provenance import (
    EvidenceLink,
    EvidenceRelation,
    ObjectIdentity,
)
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    NamedDecimal,
    QuantityBound,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import (
    ClaimUnitBinding,
    ExternalEvidencePayload,
    IdentificationEvidenceManifest,
    IdentificationEvidenceDomain,
    IdentificationManifestObservation,
    NestedCoordinateKind,
    NestedEvidenceCoordinate,
    ObservationActionDeliveryBinding,
    ObservationDisposition,
    QualificationScopeSpec,
    IdentificationEvidenceProjection,
)
from empirical_lawhood.planning.identification_evidence_extensions import ActionOccurrenceBinding, AuthorityAxis, ComputabilityEffectCoordinate, ComputabilityEffectEntry, ComputabilityEffectStatus, EffectReasonCode, EpisodeTerminalDisposition, EpisodeTerminalReasonCode, IdentificationEvidenceProjectionExtension, NumericalValidityAxis, ObservationValidityAxis, PhysicalSinkAxis, ScientificTerminalAxis
from empirical_lawhood.planning.finite_action_occurrence import (
    FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS,
)
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure
from empirical_lawhood.runtime.artifacts import ArtifactManifest
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent

from .action_word import GYM_TORAX_FUTURE_IP_ACTION_WORD_ID, GYM_TORAX_LOWER_IP_ACTION_WORD_ID, GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID, GYM_TORAX_WRONG_SIGN_IP_ACTION_WORD_ID, build_gym_torax_action_word_chart
from .diagnostic_contracts import GymToraxDeliveryDisposition, GymToraxNativeAction, GymToraxNumericalDisposition, GymToraxObservationDisposition, GymToraxSourceDisposition
from .field_metadata_contracts import GymToraxFieldMetadataFloat64Block, GymToraxFieldMetadataNativeEpisode
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest
from .selected_parent_execution import GymToraxSelectedParentTerminalAcquisitionDisposition, GymToraxSelectedParentTerminalAcquisitionReceipt
from .selected_parent_protocol import GymToraxSelectedParentCell, GymToraxSelectedParentFreeze, materialize_gym_torax_response_experiment_request
from .source_assessment_execution import GymToraxArtifactPublicationItem, GymToraxBoundedArtifactStore
from .system import GYM_TORAX_ACTION_QUANTITY_ID, GYM_TORAX_RECEIVER_QUANTITY_ID, GYM_TORAX_STATE_CLOCK_ID, GYM_TORAX_SYSTEM_ID, build_gym_torax_gym_torax_system


GYM_TORAX_MATCHED_EVALUATION_SCIENCE_FREEZE_ID = 'freeze.tokamak-control.matched-evaluation-science'
GYM_TORAX_MATCHED_EVALUATION_SCIENCE_RELATIVE_ROOT = (
    'tokamak-control/publication/runs/run.tokamak-control.matched-evaluation/science'
)
GYM_TORAX_MATCHED_EVALUATION_REVEAL_AUTHORITY_ID = (
    'authority.tokamak-control.matched-evaluation.outcome-reveal'
)
GYM_TORAX_MATCHED_EVALUATION_SCIENCE_EVALUATED_AT_UTC = "2026-08-25T15:00:00Z"
GYM_TORAX_MATCHED_EVALUATION_SCIENCE_FREEZE_LOGICAL_ID = (
    'artifact.freeze.tokamak-control.matched-evaluation-science'
)
GYM_TORAX_MATCHED_EVALUATION_SCIENCE_EVALUATION_LOGICAL_ID = (
    'artifact.evaluation-bundle.tokamak-control.matched-evaluation'
)

_PREPARED_DENOMINATOR_ID = 'denominator.tokamak-control.gym-torax-prepared'
_CHART_ID = 'chart.tokamak-control.native-ip-finite-action'
_RETAINED_HISTORY_ID = 'history.tokamak-control.preaction-through-state-0104'
_INFORMATION_CUTOFF_ID = 'cutoff.tokamak-control.state-0104'
_EVALUATION_SPLIT_ID = 'split.tokamak-control.fresh-confirmatory-evaluation'
_SOURCE_PAYLOAD_ROLE = "identification-observations"
_RESPONSE_FRAME_ID = 'frame.tokamak-control.q-fusion-phase-mean-states-0105-0110'
_PRIMARY_CLAIM_ID = 'claim.tokamak-control.joint-face-finite-action-response'
# This exact identity set is consumed by the prospective HFR occurrence binder.
# Do not mint adapter-local aliases for the same three J3 x tier support cells.
_LOCAL_SUPPORT_IDS = FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS
_TIER_IDS = (
    'tier.tokamak-control.joint-depth-lower',
    'tier.tokamak-control.joint-depth-middle',
    'tier.tokamak-control.joint-depth-upper',
)
_SOURCE_TIER_TO_TIER = dict(
    zip(("TIER_01", "TIER_02", "TIER_03"), _TIER_IDS, strict=True)
)
_PREFIX_ROWS = (
    ('prefix.tokamak-control.joint-depth-lower', 1, (_TIER_IDS[0],), 5, CONFIRMATORY_LOCAL_REGION_CRITICAL_VALUE),
    (
        'prefix.tokamak-control.joint-depth-middle',
        2,
        (_TIER_IDS[0], _TIER_IDS[1]),
        11,
        Decimal("3.2081223331681157"),
    ),
    (
        'prefix.tokamak-control.joint-depth-upper',
        3,
        _TIER_IDS,
        17,
        Decimal("2.9840416584797573"),
    ),
)
_DENOMINATOR_VALUE_IDS = tuple(
    sorted(
        (
            'tokamak-control.preparation.bootstrap-multiplier',
            'tokamak-control.preparation.initial-density-nbar',
            'tokamak-control.preparation.initial-temperature-scale',
            'tokamak-control.preparation.inner-transport-scale',
        )
    )
)
_HISTORY_VALUE_IDS = ('tokamak-control.history.preaction-last-state-clock',)
_ACTION_VALUE_IDS = ('tokamak-control.action.controlled-ip-phase-mean',)
_RESPONSE_VALUE_ID = 'tokamak-control.response.q-fusion-phase-mean'
_PRECUTOFF_VALUE_ID = 'tokamak-control.response.precutoff-difference-flag'
_ONSET_VALUE_ID = 'tokamak-control.response.onset-state-clock'
_RECEIVER_VALUE_IDS = tuple(
    sorted((_RESPONSE_VALUE_ID, _PRECUTOFF_VALUE_ID, _ONSET_VALUE_ID))
)


def _identity(identifier: str, record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(identifier, record)


def _member_slug(member_id: str) -> str:
    value = member_id.removeprefix('member.tokamak-control.')
    if value not in {"primary", "refined"}:
        raise ValueError("Gym-TORAX science roster names another numerical member")
    return value


def _candidate_version_id(member_id: str) -> str:
    return f'candidate-version.tokamak-control.{_member_slug(member_id)}'


def _view_id(member_id: str) -> str:
    return f'view.tokamak-control.{_member_slug(member_id)}'


def _role_for_word(word_id: str) -> ConfirmatoryActionRole:
    return {
        GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID: ConfirmatoryActionRole.MEASURED_HOLD,
        GYM_TORAX_LOWER_IP_ACTION_WORD_ID: ConfirmatoryActionRole.TARGET_ACTIVE,
        GYM_TORAX_WRONG_SIGN_IP_ACTION_WORD_ID: ConfirmatoryActionRole.WRONG_SIGN_CONTROL,
        GYM_TORAX_FUTURE_IP_ACTION_WORD_ID: ConfirmatoryActionRole.FUTURE_NULL_CONTROL,
    }[word_id]


def _role_id(word_id: str) -> str:
    return f"role.tokamak-control.{_role_for_word(word_id).value.lower().replace('_', '-')}"


def _episode_logical_id(episode_id: str) -> str:
    return f"artifact.{episode_id}"


def _episode_object_identity(
    manifest: ArtifactManifest, episode_id: str
) -> ObjectIdentity:
    if (
        manifest.logical.logical_artifact_id != _episode_logical_id(episode_id)
        or manifest.logical.payload_schema != GymToraxFieldMetadataNativeEpisode.SCHEMA
        or manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        or manifest.logical.outcome_access is not OutcomeAccess.EVALUATION_SEALED
    ):
        raise ValueError(
            "Gym-TORAX primary episode sidecar changes its sealed scientific identity"
        )
    return ObjectIdentity(
        object_id=episode_id,
        object_schema=GymToraxFieldMetadataNativeEpisode.SCHEMA,
        object_version="1.0.0",
        object_fingerprint=manifest.logical.content_sha256,
    )


def _payload_from_manifest(
    manifest: ArtifactManifest,
) -> IdentificationProjectionPayloadTemplate:
    publication = manifest.publication
    if publication is None:
        raise ValueError("Gym-TORAX primary episode lacks an atomic publication binding")
    logical = manifest.logical
    materialization = manifest.materialization
    artifact = ArtifactIdentity(
        artifact_id=logical.logical_artifact_id,
        role=_SOURCE_PAYLOAD_ROLE,
        payload_schema=logical.payload_schema,
        sha256=logical.content_sha256,
        media_type=logical.media_type,
        size_bytes=materialization.size_bytes,
    )
    return IdentificationProjectionPayloadTemplate(
        payload_id=artifact.artifact_id,
        artifact_role=artifact.role,
        artifact_payload_schema=artifact.payload_schema,
        artifact_media_type=artifact.media_type,
        artifact_extensions=(),
        external_root_contract_id=materialization.storage_root_id,
        relative_locator=materialization.relative_path,
        publication_receipt_id=publication.publication_batch_id,
        publication_receipt_schema=publication.SCHEMA,
        publication_receipt_version=publication.VERSION,
        recovery_identity_id=materialization.materialization_id,
        recovery_identity_schema=materialization.SCHEMA,
        recovery_identity_version=materialization.VERSION,
    )


@dataclass(frozen=True, slots=True)
class GymToraxSelectedParentScienceFreeze(CanonicalRecord):
    """Outcome-sealed science design built without opening an episode payload."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-selected-parent-science-freeze'

    freeze_id: str
    relative_root: str
    prospective_protocol_parent: ObjectIdentity
    source_campaign_roster: ObjectIdentity
    source_qualification: ObjectIdentity
    field_metadata_manifest: ObjectIdentity
    extraction_manifest: ObjectIdentity
    implementation_source_closure: ImplementationSourceClosure
    system: SystemSpec
    projection_template: IdentificationProjectionTemplate
    scaffold: FiniteActionCandidateScaffold
    region_support_spec: ConfirmatoryFiniteActionRegionSupportSpec
    confirmatory_design_template: ConfirmatoryFiniteActionDesignTemplate
    evaluator_binding: ProspectiveConfirmatoryFiniteActionEvaluatorBinding
    expected_primary_cell_ids: tuple[str, ...]
    expected_primary_episode_ids: tuple[str, ...]
    required_reveal_authority_id: str
    scientific_values_read: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_relative_locator(self.relative_root)
        validate_stable_id(
            self.required_reveal_authority_id,
            field_name="required_reveal_authority_id",
        )
        require_sorted_unique_strings(
            self.expected_primary_cell_ids,
            field_name="expected_primary_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.expected_primary_episode_ids,
            field_name="expected_primary_episode_ids",
            allow_empty=False,
        )
        if (
            self.freeze_id != GYM_TORAX_MATCHED_EVALUATION_SCIENCE_FREEZE_ID
            or self.relative_root != GYM_TORAX_MATCHED_EVALUATION_SCIENCE_RELATIVE_ROOT
            or self.required_reveal_authority_id != GYM_TORAX_MATCHED_EVALUATION_REVEAL_AUTHORITY_ID
        ):
            raise ValueError("Gym-TORAX measurement-through-law-qualification science identity changed")
        if (
            len(self.expected_primary_cell_ids) != 144
            or len(self.expected_primary_episode_ids) != 144
        ):
            raise ValueError(
                "Gym-TORAX primary science freeze requires 144 BT member/action views"
            )
        if (
            len(self.projection_template.observations) != 144
            or len(self.projection_template.payloads) != 144
        ):
            raise ValueError(
                "Gym-TORAX primary projection template changes its 144-view roster"
            )
        if self.scaffold.system != _identity(self.system.system_id, self.system):
            raise ValueError("Gym-TORAX science scaffold binds another system")
        if self.scaffold.claim_template.compatibility_result_id is not None:
            raise ValueError(
                "Gym-TORAX current finite-action route cannot request an R6 projection"
            )
        if self.confirmatory_design_template.projection_template != _identity(
            self.projection_template.template_id,
            self.projection_template,
        ):
            raise ValueError("Gym-TORAX science design binds another projection template")
        if self.confirmatory_design_template.core.scaffold_template != _identity(
            self.scaffold.scaffold_id,
            self.scaffold,
        ):
            raise ValueError("Gym-TORAX science design binds another candidate scaffold")
        if self.confirmatory_design_template.core.region_support_spec != _identity(
            self.region_support_spec.spec_id,
            self.region_support_spec,
        ):
            raise ValueError("Gym-TORAX science design binds another regional support spec")
        if self.evaluator_binding.qualification_profile.object_id != (
            "profile.confirmatory-finite-action-local-support.method-equivalent"
        ):
            raise ValueError("Gym-TORAX evaluator/profile identity changed")
        if self.scientific_values_read:
            raise ValueError(
                "Gym-TORAX science freeze cannot read protected scientific values"
            )
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
        ):
            raise ValueError(
                "Gym-TORAX science freeze changed its prospective evidence lane"
            )


def _authority_policy(freeze: GymToraxSelectedParentFreeze) -> AuthorityPolicy:
    return AuthorityPolicy(
        policy_id='authority-policy.tokamak-control.matched-evaluation-science',
        delegator_id='owner.tokamak-control',
        delegate_id="evaluator.prospective-confirmatory-finite-action",
        scope_ids=(GYM_TORAX_SYSTEM_ID,),
        allowed_world_kinds=frozenset({WorldKind.NUMERICAL_SIMULATOR}),
        allowed_actions=frozenset(
            {
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.SIMULATION_EXECUTION,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.NONE}),
        required_gate_ids=(freeze.source_qualification.object_id,),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.HUMAN_OR_ANIMAL_INTERVENTION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.PAID_OR_EXTERNALLY_BILLED_RESOURCE,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
        ),
        budget_ceiling=freeze.campaign_resource_budget,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        expires_at_utc=None,
    )


def _axis_map(freeze: GymToraxSelectedParentFreeze) -> LawCandidateAxisMap:
    return LawCandidateAxisMap(
        axis_map_id='axis-map.tokamak-control.hfr-bt-primary-refined',
        bindings=tuple(
            sorted(
                (
                    LawCandidateAxisBinding(
                        binding_id=f'axis-binding.tokamak-control.{_member_slug(member.member_id)}',
                        candidate_version_member_id=_candidate_version_id(
                            member.member_id
                        ),
                        denominator_member_id=member.member_id,
                        qualification_view_ids=(_view_id(member.member_id),),
                        claimed_property_ids=(
                            'property.tokamak-control.native-finite-action-response',
                        ),
                        nontransported_property_ids=(
                            'property.tokamak-control.cross-member-magnitude-transport',
                        ),
                    )
                    for member in freeze.numerical_members
                ),
                key=lambda value: value.candidate_version_member_id,
            )
        ),
    )


def _prefixes() -> tuple[ConfirmatoryPrefixSpec, ...]:
    return tuple(
        sorted(
            (
                ConfirmatoryPrefixSpec(
                    prefix_id=prefix_id,
                    order_index=order,
                    constituent_tier_ids=tiers,
                    degrees_of_freedom=df,
                    one_sided_tail_probability=CONFIRMATORY_LOCAL_REGION_PER_TEST_TAIL_PROBABILITY,
                    one_sided_critical_value=critical,
                    critical_value_source_id=f'critical.tokamak-control.student-t-j{order}',
                )
                for prefix_id, order, tiers, df, critical in _PREFIX_ROWS
            ),
            key=lambda value: value.prefix_id,
        )
    )


def _claim_template() -> CandidateClaimTemplate:
    return CandidateClaimTemplate(
        template_id='claim-template.tokamak-control.hfr-bt-finite-action-response',
        terminal_result_id='qualification-result.tokamak-control.matched-evaluation',
        # The current finite-action evaluator owns the exact action/sink/
        # computability result.  R6 is an optional non-authoritative projection
        # and this direct follow-up has no R6 consumer.
        compatibility_result_id=None,
        claim_id=_PRIMARY_CLAIM_ID,
        law_id='response-law.tokamak-control.joint-face-finite-action-response',
        proposition=(
            "On the three declared fresh BT support regions of the prepared Gym--TORAX "
            "denominator, the exact four-word Ip chart has a reproducible member-local "
            "Q_fusion phase-mean response satisfying the frozen G1 and G1R intersection."
        ),
        estimand=(
            "For each numerical member and BT preparation, mean Q_fusion over states "
            "105--110 under each exact action word minus its paired native HOLD episode."
        ),
        promotion_rule=(
            "All six Bonferroni G1 prefix tests, all six tier-local G1R tests, exact "
            "delivery/history/causal controls, and structural member agreement pass."
        ),
        causal_strength=CausalStrength.SIMULATOR_INTERVENTION,
        requested_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        interface_input_quantity_ids=(GYM_TORAX_ACTION_QUANTITY_ID,),
        interface_output_quantity_ids=(GYM_TORAX_RECEIVER_QUANTITY_ID,),
        mapping_assumption_ids=(
            'assumption.tokamak-control.simulator-local-prepared-denominator',
        ),
        joint_response_sink_effort_distribution_identified=False,
    )


def _obligation_template(
    *,
    system: SystemSpec,
    axis_map: LawCandidateAxisMap,
) -> LawObligationTemplate:
    return LawObligationTemplate(
        template_id='obligation-template.tokamak-control.hfr-bt-finite-action-response',
        obligations_id='obligations.tokamak-control.hfr-bt-finite-action-response',
        support_id='support.tokamak-control.hfr-bt-finite-action-response',
        validity_id='validity.tokamak-control.hfr-bt-finite-action-response',
        uncertainty_id='uncertainty.tokamak-control.hfr-bt-finite-action-response',
        closure_id='closure.tokamak-control.hfr-bt-finite-action-response',
        structural_convergence_id='structural.tokamak-control.hfr-bt-member-agreement',
        computability_id='computability.tokamak-control.hfr-bt-finite-action-response',
        independent_unit_id=system.independent_unit.unit_id,
        physical_unit_count=18,
        nested_numerical_view_count=2,
        information_cutoff_id=_INFORMATION_CUTOFF_ID,
        chart_ids=(_CHART_ID,),
        denominator_cell_ids=tuple(
            sorted((_PREPARED_DENOMINATOR_ID, *_LOCAL_SUPPORT_IDS))
        ),
        action_bounds=(
            QuantityBound(
                bound_id='bound.tokamak-control.ip-target-chart',
                quantity_id=GYM_TORAX_ACTION_QUANTITY_ID,
                native_unit="A",
                lower=Decimal("12400000"),
                upper=Decimal("12600000"),
            ),
        ),
        validity_domain_ids=('validity-domain.tokamak-control.gym-torax-1-1-1-torax-1-4-2',),
        assumption_ids=('assumption.tokamak-control.simulator-local-prepared-denominator',),
        uncertainty_method_key='uncertainty.tokamak-control.bonferroni-one-sided-student-t',
        uncertainty_confidence_level=Decimal("0.975"),
        interval_quantity_ids=(GYM_TORAX_RECEIVER_QUANTITY_ID,),
        uncertainty_limitation_codes=(
            "MEMBER_MAGNITUDES_NOT_POOLED",
            "SIMULATOR_LOCAL_ONLY",
        ),
        falsifiers=tuple(
            sorted(
                (
                    FalsifierObligationTemplate(
                        falsifier_id='falsifier.tokamak-control.future-null',
                        kind=FalsifierKind.TEMPORAL_SUPPORT,
                        capability_key="finite-action.confirmatory-prefix",
                        description="Future word equals HOLD through state 110 and begins no earlier than 112.",
                        decisive_rule="maximum absolute pre-112 Q_fusion difference is at most 1e-12",
                    ),
                    FalsifierObligationTemplate(
                        falsifier_id='falsifier.tokamak-control.preaction-identity',
                        kind=FalsifierKind.BASELINE_COMPARATOR,
                        capability_key="finite-action.confirmatory-prefix",
                        description="Paired registered pre-action source surfaces are byte-identical.",
                        decisive_rule="all registered source operands through state 104 match exactly",
                    ),
                    FalsifierObligationTemplate(
                        falsifier_id='falsifier.tokamak-control.recurrence',
                        kind=FalsifierKind.WITHIN_CELL_RECURRENCE,
                        capability_key="finite-action.confirmatory-prefix",
                        description="Every tier and nested member retains six complete BT preparations.",
                        decisive_rule="complete fixed independent-unit roster with no substitution",
                    ),
                    FalsifierObligationTemplate(
                        falsifier_id='falsifier.tokamak-control.wrong-sign',
                        kind=FalsifierKind.WRONG_ACTION,
                        capability_key="finite-action.confirmatory-prefix",
                        description="Wrong-sign response is retained with its measured sign and causal clock.",
                        decisive_rule="exact delivery and response onset in states 105 through 110",
                    ),
                ),
                key=lambda value: value.falsifier_id,
            )
        ),
        recurrence_cell_ids=tuple(sorted(value[0] for value in _PREFIX_ROWS)),
        exchange_factor_ids=('exchange.tokamak-control.exact-action-word-within-preparation',),
        retained_history_ids=(_RETAINED_HISTORY_ID,),
        required_structure_ids=("finite-action-response",),
        numerical_view_ids=axis_map.qualification_view_ids,
        structural_tolerances=(
            NamedDecimal(
                value_id='tolerance.tokamak-control.member-sign-support-causal-agreement',
                value=Decimal(0),
                unit="1",
            ),
        ),
        computability_envelope_id=system.computability_envelopes[0].envelope_id,
    )


def build_gym_torax_response_experiment_science_freeze(
    *,
    acquisition_freeze: GymToraxSelectedParentFreeze,
    episode_manifests: tuple[ArtifactManifest, ...],
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    implementation_source_closure: ImplementationSourceClosure,
) -> GymToraxSelectedParentScienceFreeze:
    """Build the exact primary science design from sidecars only."""

    manifests = {
        value.logical.logical_artifact_id: value for value in episode_manifests
    }
    if len(manifests) != len(episode_manifests):
        raise ValueError(
            "Gym-TORAX science sidecar roster contains duplicate logical identities"
        )
    primary_units = tuple(
        sorted(
            value.unit_id
            for value in acquisition_freeze.units
            if value.cell_kind == "bt"
        )
    )
    if len(primary_units) != 18:
        raise ValueError("Gym-TORAX primary law requires exactly 18 BT preparations")
    unit_set = set(primary_units)
    primary_cells = tuple(
        sorted(
            (
                value
                for value in acquisition_freeze.cells
                if value.physical_unit_instance_id in unit_set
            ),
            key=lambda value: value.cell_id,
        )
    )
    if len(primary_cells) != 144:
        raise ValueError("Gym-TORAX primary law requires the exact 18 x 2 x 4 cell product")
    if set(manifests) != {
        _episode_logical_id(value.episode_id) for value in primary_cells
    }:
        raise ValueError(
            "Gym-TORAX science sidecars differ from the frozen BT episode roster"
        )
    for cell in primary_cells:
        _episode_object_identity(
            manifests[_episode_logical_id(cell.episode_id)], cell.episode_id
        )

    system = build_gym_torax_gym_torax_system(
        authority_policy=_authority_policy(acquisition_freeze)
    )
    system_identity = _identity(system.system_id, system)
    extraction_identity = _identity(
        extraction_manifest.manifest_id, extraction_manifest
    )
    metadata_identity = _identity(
        field_metadata_manifest.manifest_id, field_metadata_manifest
    )
    closure_identity = _identity(
        implementation_source_closure.source_closure_id,
        implementation_source_closure,
    )
    axis_map = _axis_map(acquisition_freeze)
    action_words = build_gym_torax_action_word_chart()
    action_roles = tuple(
        sorted(
            (
                ConfirmatoryActionRoleBinding(
                    binding_id=f"action-role.tokamak-control.{_role_for_word(word.word_id).value.lower().replace('_', '-')}",
                    action_word_id=word.word_id,
                    role=_role_for_word(word.word_id),
                )
                for word in action_words
            ),
            key=lambda value: value.binding_id,
        )
    )
    units_by_id = {value.unit_id: value for value in acquisition_freeze.units}
    unit_tiers = tuple(
        sorted(
            (
                ConfirmatoryUnitTierBinding(
                    binding_id=f'unit-tier.tokamak-control.{unit_id}',
                    physical_independent_unit_id=unit_id,
                    tier_id=_SOURCE_TIER_TO_TIER[units_by_id[unit_id].tier_id],
                    denominator_cell_id=_PREPARED_DENOMINATOR_ID,
                )
                for unit_id in primary_units
            ),
            key=lambda value: value.binding_id,
        )
    )
    tier_regions = tuple(
        ConfirmatoryFiniteActionTierRegionBinding(
            binding_id=f"tier-region.tokamak-control.{tier_id.rsplit('.', 1)[1]}",
            tier_id=tier_id,
            local_support_id=local_id,
            admission_support_cell_id=local_id,
            physical_independent_unit_ids=tuple(
                sorted(
                    value.physical_independent_unit_id
                    for value in unit_tiers
                    if value.tier_id == tier_id
                )
            ),
        )
        for tier_id, local_id in zip(_TIER_IDS, _LOCAL_SUPPORT_IDS, strict=True)
    )
    prefixes = _prefixes()
    response = ConfirmatoryResponseQuantity(
        binding_id='response-binding.tokamak-control.q-fusion-phase-mean',
        response_projection_value_id=_RESPONSE_VALUE_ID,
        precutoff_projection_value_id=_PRECUTOFF_VALUE_ID,
        onset_clock_projection_value_id=_ONSET_VALUE_ID,
        quantity_id=GYM_TORAX_RECEIVER_QUANTITY_ID,
        native_unit="1",
        native_frame_id=_RESPONSE_FRAME_ID,
        response_clock_id=GYM_TORAX_STATE_CLOCK_ID,
        maximum_null_absolute_effect=Decimal("1e-12"),
        maximum_precutoff_absolute_effect=Decimal(0),
        earliest_active_response_clock=Decimal(105),
        latest_active_response_clock=Decimal(110),
        earliest_future_response_clock=Decimal(112),
    )
    region_spec = ConfirmatoryFiniteActionRegionSupportSpec(
        spec_id='spec.tokamak-control.hfr-bt-g1-g1r',
        prepared_denominator_id=_PREPARED_DENOMINATOR_ID,
        chart_id=_CHART_ID,
        horizon=_identity(system.relation.horizon.horizon_id, system.relation.horizon),
        retained_history_ids=(_RETAINED_HISTORY_ID,),
        action_words=action_words,
        action_role_bindings=action_roles,
        axis_map=axis_map,
        tier_region_bindings=tier_regions,
        prefixes=prefixes,
        response_quantity=response,
        information_cutoff_id=_INFORMATION_CUTOFF_ID,
        region_family_id='multiplicity.tokamak-control.g1r-six-tests',
        family_alpha=CONFIRMATORY_LOCAL_REGION_FAMILY_ALPHA,
        simultaneous_test_count=CONFIRMATORY_LOCAL_REGION_TEST_COUNT,
        per_test_tail_probability=CONFIRMATORY_LOCAL_REGION_PER_TEST_TAIL_PROBABILITY,
        degrees_of_freedom=CONFIRMATORY_LOCAL_REGION_DEGREES_OF_FREEDOM,
        critical_value=CONFIRMATORY_LOCAL_REGION_CRITICAL_VALUE,
        materiality=CONFIRMATORY_LOCAL_REGION_MATERIALITY,
        minimum_complete_units=CONFIRMATORY_LOCAL_REGION_UNIT_COUNT,
        minimum_positive_units=CONFIRMATORY_LOCAL_REGION_MINIMUM_POSITIVE_UNITS,
        median_materiality=CONFIRMATORY_LOCAL_REGION_MATERIALITY,
        producer_implementation_id='implementation.tokamak-control.hfr-bt-g1-g1r',
        producer_implementation_sha256=extraction_manifest.fingerprint(),
        interpolation_allowed=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )

    coordinate_rows: list[NestedEvidenceCoordinate] = []
    observation_rows: list[IdentificationProjectionObservationTemplate] = []
    for unit_id in primary_units:
        unit_coordinate = f'coordinate.tokamak-control.preparation.{unit_id}'
        coordinate_rows.append(
            NestedEvidenceCoordinate(
                unit_coordinate, NestedCoordinateKind.PREPARATION, None
            )
        )
        for member in acquisition_freeze.numerical_members:
            member_coordinate = (
                f'coordinate.tokamak-control.member.{unit_id}.{_member_slug(member.member_id)}'
            )
            coordinate_rows.append(
                NestedEvidenceCoordinate(
                    member_coordinate,
                    NestedCoordinateKind.DENOMINATOR_MEMBER,
                    unit_coordinate,
                )
            )
            member_cells = tuple(
                value
                for value in primary_cells
                if value.physical_unit_instance_id == unit_id
                and value.numerical_member_id == member.member_id
            )
            for cell in member_cells:
                action_slug = cell.action_word_id.removeprefix('action-word.tokamak-control.')
                action_coordinate = f'coordinate.tokamak-control.action.{unit_id}.{_member_slug(member.member_id)}.{action_slug}'
                coordinate_rows.append(
                    NestedEvidenceCoordinate(
                        action_coordinate,
                        NestedCoordinateKind.REPEATED_DELIVERY,
                        member_coordinate,
                    )
                )
                word = next(
                    value
                    for value in action_words
                    if value.word_id == cell.action_word_id
                )
                observation_rows.append(
                    IdentificationProjectionObservationTemplate(
                        observation_id=f"observation.{cell.episode_id.removeprefix('episode.')}",
                        physical_unit_instance_id=unit_id,
                        nested_coordinate_ids=tuple(
                            sorted(
                                (unit_coordinate, member_coordinate, action_coordinate)
                            )
                        ),
                        denominator_cell_id=_PREPARED_DENOMINATOR_ID,
                        chart_id=_CHART_ID,
                        split_id=_EVALUATION_SPLIT_ID,
                        role_id=_role_id(cell.action_word_id),
                        member_id=member.member_id,
                        candidate_version_id=_candidate_version_id(member.member_id),
                        qualification_view_id=_view_id(member.member_id),
                        receiver_id=GYM_TORAX_RECEIVER_QUANTITY_ID,
                        receiver_clock_id=GYM_TORAX_STATE_CLOCK_ID,
                        native_frame_id=_RESPONSE_FRAME_ID,
                        payload_id=_episode_logical_id(cell.episode_id),
                        payload_member_locator="blocks/source-scalar/Q_fusion",
                        action_word_id=cell.action_word_id,
                        occurrence_ids=tuple(sorted(word.chronological_occurrence_ids)),
                        tags=tuple(
                            sorted(
                                (
                                    f"tag.tokamak-control.{units_by_id[unit_id].tier_id.lower().replace('_', '-')}",
                                    'tag.tokamak-control.primary-law-bt',
                                    f'tag.tokamak-control.{action_slug}',
                                )
                            )
                        ),
                    )
                )
    coordinates = tuple(sorted(coordinate_rows, key=lambda value: value.coordinate_id))
    observations = tuple(
        sorted(observation_rows, key=lambda value: value.observation_id)
    )
    scope = QualificationScopeSpec(
        scope_id='scope.tokamak-control.hfr-bt-primary-law',
        claim_id=_PRIMARY_CLAIM_ID,
        population_id='population.tokamak-control.fresh-bt-preparations',
        physical_unit_type_id=system.independent_unit.unit_id,
        independent_unit_instance_ids=primary_units,
        nested_coordinate_ids=tuple(value.coordinate_id for value in coordinates),
        aggregation_level_id=system.independent_unit.unit_id,
        uncertainty_unit_id=system.independent_unit.unit_id,
        locality_scope_id='locality.tokamak-control.three-bt-support-regions',
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
    )
    claim_binding_id = 'claim-unit.tokamak-control.hfr-bt-primary-law'
    claim_binding = ClaimUnitBinding(
        binding_id=claim_binding_id,
        claim_id=scope.claim_id,
        scope=_identity(scope.scope_id, scope),
        independent_unit_instance_ids=scope.independent_unit_instance_ids,
        aggregation_level_id=scope.aggregation_level_id,
    )
    projection_template = IdentificationProjectionTemplate(
        template_id='projection-template.tokamak-control.hfr-bt-primary-law',
        role=IdentificationProjectionRole.PRIMARY_LAW,
        expected_manifest_id='manifest.tokamak-control.hfr-bt-primary-law',
        system=system_identity,
        relation=system.relation,
        source_qualification=acquisition_freeze.source_qualification,
        runtime_qualification=extraction_identity,
        observation_operator_qualification=metadata_identity,
        evidence_world_id=system.world.world_id,
        evidence_domain=IdentificationEvidenceDomain.EVALUATOR_REVEAL,
        expected_allowed_consumer_ids=(CONFIRMATORY_FINITE_ACTION_KEY,),
        information_cutoff_id=_INFORMATION_CUTOFF_ID,
        payloads=tuple(
            sorted(
                (_payload_from_manifest(value) for value in episode_manifests),
                key=lambda value: value.payload_id,
            )
        ),
        coordinates=coordinates,
        action_words=action_words,
        qualification_scopes=(scope,),
        qualification_scope_id=scope.scope_id,
        claim_unit_binding_id=claim_binding_id,
        allowed_consumer_id=CONFIRMATORY_FINITE_ACTION_KEY,
        projection_capability=_identity(
            'capability.tokamak-control.hfr-bt-projector', extraction_manifest
        ),
        projection_config=_identity(
            'config.tokamak-control.hfr-bt-projector', extraction_manifest
        ),
        projection_implementation=closure_identity,
        value_partition=ObservationValuePartition(
            partition_id='partition.tokamak-control.hfr-bt-primary-law',
            denominator_value_ids=_DENOMINATOR_VALUE_IDS,
            history_value_ids=_HISTORY_VALUE_IDS,
            action_value_ids=_ACTION_VALUE_IDS,
            receiver_value_ids=_RECEIVER_VALUE_IDS,
            tags=(),
        ),
        observations=observations,
        binder_implementation_id='implementation.tokamak-control.hfr-bt-projector-binder',
        binder_implementation_sha256=extraction_manifest.fingerprint(),
        binding_stage=IdentificationProjectionBindingStage.EVALUATOR_REVEAL,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        expected_manifest_parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        expected_manifest_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    evidence_link = EvidenceLink(
        link_id='evidence-link.tokamak-control.hfr-bt-primary-law',
        relation=EvidenceRelation.DERIVED_FROM,
        source=_identity(projection_template.template_id, projection_template),
        target=_identity(system.relation.relation_id, system.relation),
        artifact_ids=tuple(value.payload_id for value in projection_template.payloads),
        world_id=system.world.world_id,
        information_cutoff_id=_INFORMATION_CUTOFF_ID,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        reason="Exact sealed BT episode artifacts support the predeclared confirmatory projection.",
    )
    scaffold = FiniteActionCandidateScaffold(
        scaffold_id='scaffold.tokamak-control.hfr-bt-primary-law',
        evidence_id='evidence.tokamak-control.hfr-bt-primary-law',
        candidate_id='candidate.tokamak-control.hfr-bt-primary-law',
        system=system_identity,
        axis_map=axis_map,
        claim_unit_binding=_identity(claim_binding.binding_id, claim_binding),
        claim_template=_claim_template(),
        obligation_template=_obligation_template(system=system, axis_map=axis_map),
        evidence_links=(evidence_link,),
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    core = ConfirmatoryFiniteActionProspectiveCore(
        method_key=CONFIRMATORY_FINITE_ACTION_KEY,
        method_version=CONFIRMATORY_FINITE_ACTION_VERSION,
        implementation_sha256=extraction_manifest.fingerprint(),
        allowed_consumer_id=CONFIRMATORY_FINITE_ACTION_KEY,
        scaffold_template=_identity(scaffold.scaffold_id, scaffold),
        expected_physical_unit_ids=primary_units,
        expected_source_payload_role=_SOURCE_PAYLOAD_ROLE,
        config_id_derivation_rule=CONFIRMATORY_DESIGN_CONFIG_ID_RULE,
        region_support_spec=_identity(region_spec.spec_id, region_spec),
        horizon=region_spec.horizon,
        prepared_denominator_id=_PREPARED_DENOMINATOR_ID,
        chart_id=_CHART_ID,
        retained_history_ids=(_RETAINED_HISTORY_ID,),
        action_words=action_words,
        action_role_bindings=action_roles,
        axis_map=axis_map,
        unit_tier_bindings=unit_tiers,
        prefixes=prefixes,
        evaluation_split_ids=(_EVALUATION_SPLIT_ID,),
        response_quantity=response,
        minimum_complete_units_per_tier=CONFIRMATORY_LOCAL_REGION_UNIT_COUNT,
        minimum_positive_units_per_tier=CONFIRMATORY_LOCAL_REGION_MINIMUM_POSITIVE_UNITS,
        materiality=CONFIRMATORY_LOCAL_REGION_MATERIALITY,
        family_alpha=CONFIRMATORY_LOCAL_REGION_FAMILY_ALPHA,
        simultaneous_test_count=CONFIRMATORY_LOCAL_REGION_TEST_COUNT,
        multiplicity_family_id='multiplicity.tokamak-control.g1-g1r',
        information_cutoff_id=_INFORMATION_CUTOFF_ID,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    design_template = ConfirmatoryFiniteActionDesignTemplate(
        template_id='design-template.tokamak-control.hfr-bt-primary-law',
        expected_design_id='design.tokamak-control.hfr-bt-primary-law',
        expected_projection_plan_id=f"plan.{projection_template.template_id}",
        projection_template=_identity(
            projection_template.template_id, projection_template
        ),
        core=core,
        binder_implementation_id="implementation.confirmatory-design-binder",
        binder_implementation_sha256=extraction_manifest.fingerprint(),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    profile_evaluator = confirmatory_finite_action_local_support_profile_evaluator(
        implementation_sha256=extraction_manifest.fingerprint()
    )
    evaluator_binding = prospective_confirmatory_finite_action_evaluator_binding(
        implementation_sha256=extraction_manifest.fingerprint(),
        binding_implementation_sha256=extraction_manifest.fingerprint(),
        qualification_profile=_identity(
            profile_evaluator.profile.profile_id,
            profile_evaluator.profile,
        ),
    )
    return GymToraxSelectedParentScienceFreeze(
        freeze_id=GYM_TORAX_MATCHED_EVALUATION_SCIENCE_FREEZE_ID,
        relative_root=GYM_TORAX_MATCHED_EVALUATION_SCIENCE_RELATIVE_ROOT,
        prospective_protocol_parent=acquisition_freeze.prospective_protocol_parent,
        source_campaign_roster=acquisition_freeze.source_campaign_roster,
        source_qualification=acquisition_freeze.source_qualification,
        field_metadata_manifest=metadata_identity,
        extraction_manifest=extraction_identity,
        implementation_source_closure=implementation_source_closure,
        system=system,
        projection_template=projection_template,
        scaffold=scaffold,
        region_support_spec=region_spec,
        confirmatory_design_template=design_template,
        evaluator_binding=evaluator_binding,
        expected_primary_cell_ids=tuple(value.cell_id for value in primary_cells),
        expected_primary_episode_ids=tuple(
            sorted(value.episode_id for value in primary_cells)
        ),
        required_reveal_authority_id=GYM_TORAX_MATCHED_EVALUATION_REVEAL_AUTHORITY_ID,
        scientific_values_read=False,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
    )


def publish_gym_torax_response_experiment_science_freeze(
    *,
    science_freeze: GymToraxSelectedParentScienceFreeze,
    acquisition_freeze: GymToraxSelectedParentFreeze,
    acquisition_terminal: GymToraxSelectedParentTerminalAcquisitionReceipt,
    episode_manifests: tuple[ArtifactManifest, ...],
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> ObjectIdentity:
    """Publish the topology-only science freeze without revealing episode bytes."""

    if (
        acquisition_terminal.freeze
        != _identity(acquisition_freeze.freeze_id, acquisition_freeze)
        or acquisition_terminal.disposition
        is not GymToraxSelectedParentTerminalAcquisitionDisposition.COMPLETE
        or acquisition_terminal.completed_episode_count != 576
        or acquisition_terminal.failed_cell_count
    ):
        raise ValueError(
            "Gym-TORAX science freeze requires complete terminal acquisition accounting"
        )
    require_study_authority(
        custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=_identity(acquisition_freeze.freeze_id, acquisition_freeze),
        prerequisite_authority=None,
        grantee_id="operator.execution-service",
        storage_root_id=store.storage_root_id,
        relative_root=acquisition_freeze.relative_root,
        at_utc=at_utc,
    )
    manifests = {
        value.logical.logical_artifact_id: value for value in episode_manifests
    }
    expected = {
        _episode_logical_id(value)
        for value in science_freeze.expected_primary_episode_ids
    }
    if set(manifests) != expected:
        raise ValueError(
            "Gym-TORAX science freeze publication changes its 144 sidecar roster"
        )
    parents = [
        ArtifactLineageParent(
            identity=science_freeze.prospective_protocol_parent,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        ArtifactLineageParent(
            identity=science_freeze.source_campaign_roster,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        ArtifactLineageParent(
            identity=science_freeze.source_qualification,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
    ]
    parents.extend(
        ArtifactLineageParent(
            identity=_episode_object_identity(
                manifests[_episode_logical_id(episode_id)], episode_id
            ),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
        for episode_id in science_freeze.expected_primary_episode_ids
    )
    return store.publish_atomic(
        publication_scope_id='publication.tokamak-control.matched-evaluation-science',
        publication_scope_relative_root=acquisition_freeze.relative_root,
        implementation_sha256=(
            science_freeze.implementation_source_closure.implementation_sha256
        ),
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id=GYM_TORAX_MATCHED_EVALUATION_SCIENCE_FREEZE_LOGICAL_ID,
                relative_path=f"{science_freeze.relative_root}/science-freeze.json",
                record=science_freeze,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                lineage_parents=tuple(
                    sorted(parents, key=lambda value: value.identity.object_id)
                ),
            ),
        ),
    )[0]


@dataclass(frozen=True, slots=True)
class GymToraxSelectedParentScienceEvaluationBundle(CanonicalRecord):
    """Complete compact evaluator output and its sole generic finalization."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-selected-parent-science-evaluation-bundle'
    )

    bundle_id: str
    science_freeze: ObjectIdentity
    acquisition_terminal: ObjectIdentity
    manifest: IdentificationEvidenceManifest
    projection_plan: TaggedIdentificationProjectionPlan
    projection_template_binding: IdentificationProjectionTemplateBindingReceipt
    projection: IdentificationEvidenceProjection
    projection_extension: IdentificationEvidenceProjectionExtension
    design: ConfirmatoryFiniteActionDesign
    design_template_binding: ConfirmatoryFiniteActionDesignTemplateBindingReceipt
    config: ConfirmatoryFiniteActionConfig
    config_binding: ConfirmatoryFiniteActionBindingReceipt
    evaluation: ProspectiveConfirmatoryFiniteActionEvaluation
    revealed_episode_count: int
    nonprimary_episode_count_read: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if self.science_freeze.object_schema != GymToraxSelectedParentScienceFreeze.SCHEMA:
            raise ValueError("Gym-TORAX science bundle binds another science freeze")
        if (
            self.acquisition_terminal.object_schema
            != GymToraxSelectedParentTerminalAcquisitionReceipt.SCHEMA
        ):
            raise ValueError("Gym-TORAX science bundle binds another acquisition terminal")
        if (
            self.revealed_episode_count != 144
            or self.nonprimary_episode_count_read != 0
        ):
            raise ValueError(
                "Gym-TORAX science evaluator must reveal exactly the primary 144 views"
            )
        if self.projection.manifest != _identity(
            self.manifest.manifest_id, self.manifest
        ):
            raise ValueError("Gym-TORAX science bundle projection names another manifest")
        if self.config.projection != _identity(
            self.projection.projection_id, self.projection
        ):
            raise ValueError("Gym-TORAX science bundle config names another projection")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError(
                "Gym-TORAX science evaluation changed its prospective reveal lane"
            )


GymToraxFieldMetadataEpisodeLoader = Callable[[str], GymToraxFieldMetadataNativeEpisode]


@dataclass(frozen=True, slots=True)
class _GymToraxCompactObservationSource:
    values_by_observation_id: Mapping[str, tuple[NamedDecimal, ...]]

    def read_compact_values(
        self,
        manifest: IdentificationEvidenceManifest,
        observation: IdentificationManifestObservation,
        required_value_ids: tuple[str, ...],
    ) -> tuple[NamedDecimal, ...]:
        del manifest
        values = self.values_by_observation_id[observation.observation_id]
        if tuple(value.value_id for value in values) != required_value_ids:
            raise ValueError("Gym-TORAX compact projection changed its frozen value roster")
        return values


def _external_payload(manifest: ArtifactManifest) -> ExternalEvidencePayload:
    publication = manifest.publication
    if publication is None:
        raise ValueError("Gym-TORAX episode lacks an atomic publication identity")
    logical = manifest.logical
    materialization = manifest.materialization
    return ExternalEvidencePayload(
        payload_id=logical.logical_artifact_id,
        artifact=ArtifactIdentity(
            artifact_id=logical.logical_artifact_id,
            role=_SOURCE_PAYLOAD_ROLE,
            payload_schema=logical.payload_schema,
            sha256=logical.content_sha256,
            media_type=logical.media_type,
            size_bytes=materialization.size_bytes,
        ),
        external_root_contract_id=materialization.storage_root_id,
        relative_locator=materialization.relative_path,
        publication_receipt=_identity(publication.publication_batch_id, publication),
        recovery_identity=_identity(
            materialization.materialization_id, materialization
        ),
    )


def _action_close(
    observed: GymToraxNativeAction,
    expected: GymToraxNativeAction,
) -> bool:
    return bool(
        abs(observed.ip_a - expected.ip_a) <= Decimal("0.5")
        and abs(observed.nbi_power_w - expected.nbi_power_w) <= Decimal("0.5")
        and abs(observed.ecrh_power_w - expected.ecrh_power_w) <= Decimal("0.5")
        and abs(observed.nbi_location - expected.nbi_location) <= Decimal("1e-12")
        and abs(observed.nbi_width - expected.nbi_width) <= Decimal("1e-12")
        and abs(observed.ecrh_location - expected.ecrh_location) <= Decimal("1e-12")
        and abs(observed.ecrh_width - expected.ecrh_width) <= Decimal("1e-12")
    )


def _validate_episode_and_q(
    *,
    acquisition_freeze: GymToraxSelectedParentFreeze,
    cell: GymToraxSelectedParentCell,
    episode: GymToraxFieldMetadataNativeEpisode,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
) -> np.ndarray:
    request = materialize_gym_torax_response_experiment_request(acquisition_freeze, cell)
    if (
        episode.episode_id != cell.episode_id
        or episode.request != _identity(request.request_id, request)
        or episode.preparation
        != _identity(request.preparation.preparation_id, request.preparation)
        or episode.numerical_member
        != _identity(request.numerical_member.member_id, request.numerical_member)
        or episode.action_word
        != _identity(request.schedule.action_word.word_id, request.schedule.action_word)
    ):
        raise ValueError(f"Gym-TORAX episode lineage differs:{cell.cell_id}")
    if (
        episode.state_clocks != tuple(range(121))
        or episode.missing_required_state_clocks
        or episode.last_valid_state_clock != 120
        or episode.source_disposition is not GymToraxSourceDisposition.AVAILABLE
        or episode.delivery_disposition is not GymToraxDeliveryDisposition.COMPLETE
        or episode.numerical_disposition is not GymToraxNumericalDisposition.VALID
        or episode.observation_disposition
        is not GymToraxObservationDisposition.COMPLETE
        or episode.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        or episode.evidence_ceiling is not EvidenceCeiling.MEASUREMENT
    ):
        raise ValueError(
            f"Gym-TORAX episode is not complete valid measurement evidence:{cell.cell_id}"
        )
    deliveries = {value.request_clock: value for value in episode.deliveries}
    if set(deliveries) != set(range(120)):
        raise ValueError(f"Gym-TORAX delivery clock roster differs:{cell.cell_id}")
    for expected_row in request.schedule.rows:
        delivery = deliveries[expected_row.request_clock]
        if (
            delivery.receiver_clock != expected_row.request_clock + 1
            or delivery.occurrence_id != expected_row.controlled_occurrence_id
            or delivery.disposition is not GymToraxDeliveryDisposition.COMPLETE
            or delivery.requested != expected_row.action
            or delivery.accepted is None
            or delivery.applied is None
            or delivery.realized is None
            or not _action_close(delivery.accepted, expected_row.action)
            or not _action_close(delivery.applied, expected_row.action)
            or not _action_close(delivery.realized, expected_row.action)
        ):
            raise ValueError(f"Gym-TORAX four-stage delivery differs:{cell.cell_id}")
    q_blocks = tuple(
        value
        for value in episode.blocks
        if value.category == "source-scalar" and value.native_field_id == "Q_fusion"
    )
    if len(q_blocks) != 1:
        raise ValueError(f"Gym-TORAX episode lacks one Q_fusion block:{cell.cell_id}")
    q_block = q_blocks[0]
    q_metadata = field_metadata_manifest.by_key()[("scalars", "Q_fusion")]
    q = q_block.array()
    if (
        q_block.native_unit != q_metadata.native_unit
        or q_block.native_frame_id != q_metadata.native_frame_id
        or q_block.field_metadata_id != q_metadata.field_metadata_id
        or q_block.dimension_ids != ("state-clock", *q_metadata.native_dimension_ids)
        or q_block.clock_values != tuple(range(121))
        or q.shape != (121, 1)
        or not np.isfinite(q).all()
    ):
        raise ValueError(f"Gym-TORAX source assessment_fusion metadata/shape differs:{cell.cell_id}")
    return np.asarray(q[:, 0], dtype=np.float64)


def _precutoff_equal(
    episode: GymToraxFieldMetadataNativeEpisode,
    hold: GymToraxFieldMetadataNativeEpisode,
) -> bool:
    def retained(value: GymToraxFieldMetadataNativeEpisode) -> dict[str, GymToraxFieldMetadataFloat64Block]:
        return {
            block.block_id: block
            for block in value.blocks
            if block.category == "coordinate" or block.category.startswith("source-")
        }

    left = retained(episode)
    right = retained(hold)
    if set(left) != set(right):
        return False
    for block_id, left_raw in left.items():
        left_block = left_raw
        right_block = right[block_id]
        left_topology = (
            left_block.category,
            left_block.native_field_id,
            left_block.native_unit,
            left_block.native_frame_id,
            left_block.field_metadata_id,
            left_block.dimension_ids,
            left_block.shape,
            left_block.clock_values,
        )
        right_topology = (
            right_block.category,
            right_block.native_field_id,
            right_block.native_unit,
            right_block.native_frame_id,
            right_block.field_metadata_id,
            right_block.dimension_ids,
            right_block.shape,
            right_block.clock_values,
        )
        if left_topology != right_topology:
            return False
        left_array = left_block.array()
        right_array = right_block.array()
        if left_block.clock_values:
            clocks = np.asarray(left_block.clock_values)
            mask = clocks <= 104
            left_array = left_array[mask]
            right_array = right_array[mask]
        if left_array.tobytes(order="C") != right_array.tobytes(order="C"):
            return False
    return True


def _manifest_delivery(
    cell: GymToraxSelectedParentCell,
    words: Mapping[str, OccurrenceActionWord],
) -> ObservationActionDeliveryBinding:
    word = words[cell.action_word_id]
    occurrence_ids = tuple(sorted(value.occurrence_id for value in word.occurrences))

    def events(stage: str) -> tuple[str, ...]:
        return tuple(
            sorted(f"event.{occurrence_id}.{stage}" for occurrence_id in occurrence_ids)
        )

    return ObservationActionDeliveryBinding(
        binding_id=f"delivery-binding.{cell.episode_id}",
        action_word=_identity(word.word_id, word),
        occurrence_ids=occurrence_ids,
        requested_event_ids=events("requested"),
        accepted_event_ids=events("accepted"),
        applied_event_ids=events("applied"),
        realized_event_ids=events("realized"),
    )


def _compact_values(
    *,
    cell: GymToraxSelectedParentCell,
    acquisition_freeze: GymToraxSelectedParentFreeze,
    episode: GymToraxFieldMetadataNativeEpisode,
    q: np.ndarray,
    hold_episode: GymToraxFieldMetadataNativeEpisode,
    hold_q: np.ndarray,
) -> tuple[NamedDecimal, ...]:
    unit = next(
        value
        for value in acquisition_freeze.units
        if value.unit_id == cell.physical_unit_instance_id
    )
    preparation = unit.preparation
    delta = np.abs(q - hold_q)
    differing = np.flatnonzero(delta > 1e-12)
    onset = int(differing[0]) if differing.size else 121
    realized = tuple(
        value.realized.ip_a
        for value in episode.deliveries
        if value.realized is not None and 104 <= value.request_clock <= 109
    )
    if len(realized) != 6:
        raise ValueError(
            f"Gym-TORAX controlled-window delivery is incomplete:{cell.cell_id}"
        )
    values = {
        'tokamak-control.preparation.bootstrap-multiplier': preparation.bootstrap_multiplier,
        'tokamak-control.preparation.initial-density-nbar': preparation.initial_density_nbar,
        'tokamak-control.preparation.initial-temperature-scale': preparation.initial_temperature_scale,
        'tokamak-control.preparation.inner-transport-scale': preparation.inner_transport_scale,
        _HISTORY_VALUE_IDS[0]: Decimal(104),
        _ACTION_VALUE_IDS[0]: sum(realized, Decimal(0)) / Decimal(6),
        _RESPONSE_VALUE_ID: Decimal(str(float(np.mean(q[105:111])))),
        _PRECUTOFF_VALUE_ID: Decimal(0)
        if _precutoff_equal(episode, hold_episode)
        else Decimal(1),
        _ONSET_VALUE_ID: Decimal(onset),
    }
    return tuple(
        NamedDecimal(
            value_id=value_id,
            value=values[value_id],
            unit="A" if value_id in _ACTION_VALUE_IDS else "1",
        )
        for value_id in sorted(values)
    )


def evaluate_gym_torax_response_experiment_science(
    *,
    science_freeze: GymToraxSelectedParentScienceFreeze,
    acquisition_freeze: GymToraxSelectedParentFreeze,
    acquisition_terminal: GymToraxSelectedParentTerminalAcquisitionReceipt,
    episode_manifests: tuple[ArtifactManifest, ...],
    episode_loader: GymToraxFieldMetadataEpisodeLoader,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    reveal_authority: StudyOperationAuthority,
    execution_authority: StudyOperationAuthority,
    candidate_payload_plane: CandidatePayloadPlane,
    evaluated_at_utc: str,
) -> GymToraxSelectedParentScienceEvaluationBundle:
    """Reveal exactly 144 frozen BT views and invoke the sole finalizer once."""

    terminal_identity = _identity(acquisition_terminal.receipt_id, acquisition_terminal)
    execution_identity = _identity(
        execution_authority.authority_id, execution_authority
    )
    require_study_authority(
        reveal_authority,
        kind=StudyAuthorityKind.OUTCOME_REVEAL,
        subject=terminal_identity,
        prerequisite_authority=execution_identity,
        grantee_id=science_freeze.evaluator_binding.evaluator_id,
        at_utc=evaluated_at_utc,
    )
    if (
        reveal_authority.authority_id != science_freeze.required_reveal_authority_id
        or not reveal_authority.allows_reveal
        or reveal_authority.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
    ):
        raise PermissionError(
            "Gym-TORAX evaluator lacks its exact prospective reveal authority"
        )
    if (
        acquisition_terminal.freeze
        != _identity(acquisition_freeze.freeze_id, acquisition_freeze)
        or acquisition_terminal.disposition
        is not GymToraxSelectedParentTerminalAcquisitionDisposition.COMPLETE
        or acquisition_terminal.completed_episode_count != 576
        or acquisition_terminal.failed_cell_count != 0
    ):
        raise ValueError(
            "Gym-TORAX science evaluation requires the exact complete acquisition terminal"
        )

    manifests = {
        value.logical.logical_artifact_id: value for value in episode_manifests
    }
    expected_payload_ids = {
        _episode_logical_id(value)
        for value in science_freeze.expected_primary_episode_ids
    }
    if set(manifests) != expected_payload_ids:
        raise ValueError("Gym-TORAX reveal sidecars differ from the frozen 144-view roster")
    cells = {
        value.episode_id: value
        for value in acquisition_freeze.cells
        if value.episode_id in set(science_freeze.expected_primary_episode_ids)
    }
    if len(cells) != 144:
        raise ValueError("Gym-TORAX reveal cannot reconstruct its primary cell roster")

    episodes: dict[str, GymToraxFieldMetadataNativeEpisode] = {}
    q_values: dict[str, np.ndarray] = {}
    for episode_id in science_freeze.expected_primary_episode_ids:
        episode_manifest = manifests[_episode_logical_id(episode_id)]
        episode = episode_loader(episode_id)
        if episode.fingerprint() != episode_manifest.logical.content_sha256:
            raise ValueError(
                f"Gym-TORAX episode bytes differ from their sidecar:{episode_id}"
            )
        episodes[episode_id] = episode
        q_values[episode_id] = _validate_episode_and_q(
            acquisition_freeze=acquisition_freeze,
            cell=cells[episode_id],
            episode=episode,
            field_metadata_manifest=field_metadata_manifest,
        )

    words = {
        value.word_id: value
        for value in science_freeze.projection_template.action_words
    }
    hold_by_unit_member = {
        (cell.physical_unit_instance_id, cell.numerical_member_id): episodes[
            cell.episode_id
        ]
        for cell in cells.values()
        if cell.action_word_id == GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID
    }
    if len(hold_by_unit_member) != 36:
        raise ValueError("Gym-TORAX reveal lacks one paired HOLD per unit/member")
    q_by_episode = q_values
    template_observations = {
        value.payload_id: value
        for value in science_freeze.projection_template.observations
    }
    manifest_observations: list[IdentificationManifestObservation] = []
    compact_by_observation: dict[str, tuple[NamedDecimal, ...]] = {}
    for episode_id in science_freeze.expected_primary_episode_ids:
        cell = cells[episode_id]
        template = template_observations[_episode_logical_id(episode_id)]
        delivery = _manifest_delivery(cell, words)
        manifest_observations.append(
            IdentificationManifestObservation(
                observation_id=template.observation_id,
                physical_unit_instance_id=template.physical_unit_instance_id,
                nested_coordinate_ids=template.nested_coordinate_ids,
                denominator_cell_id=template.denominator_cell_id,
                chart_id=template.chart_id,
                split_id=template.split_id,
                role_id=template.role_id,
                member_id=template.member_id,
                candidate_version_id=template.candidate_version_id,
                qualification_view_id=template.qualification_view_id,
                receiver_id=template.receiver_id,
                receiver_clock_id=template.receiver_clock_id,
                native_frame_id=template.native_frame_id,
                payload_id=template.payload_id,
                payload_member_locator=template.payload_member_locator,
                action_delivery=delivery,
                disposition=ObservationDisposition.COMPLETE,
                reason_codes=(),
            )
        )
        hold = hold_by_unit_member[
            (cell.physical_unit_instance_id, cell.numerical_member_id)
        ]
        compact_by_observation[template.observation_id] = _compact_values(
            cell=cell,
            acquisition_freeze=acquisition_freeze,
            episode=episodes[episode_id],
            q=q_by_episode[episode_id],
            hold_episode=hold,
            hold_q=q_by_episode[hold.episode_id],
        )

    projection_template = science_freeze.projection_template
    evidence_manifest = IdentificationEvidenceManifest(
        manifest_id=projection_template.expected_manifest_id,
        system=projection_template.system,
        relation=projection_template.relation,
        source_qualification=projection_template.source_qualification,
        runtime_qualification=projection_template.runtime_qualification,
        observation_operator_qualification=projection_template.observation_operator_qualification,
        evidence_world_id=projection_template.evidence_world_id,
        evidence_domain=projection_template.evidence_domain,
        allowed_consumer_ids=projection_template.expected_allowed_consumer_ids,
        information_cutoff_id=projection_template.information_cutoff_id,
        payloads=tuple(
            sorted(
                (_external_payload(value) for value in episode_manifests),
                key=lambda value: value.payload_id,
            )
        ),
        coordinates=projection_template.coordinates,
        action_words=projection_template.action_words,
        observations=tuple(
            sorted(manifest_observations, key=lambda value: value.observation_id)
        ),
        qualification_scopes=projection_template.qualification_scopes,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    plan, projection_template_binding = bind_identification_projection_template(
        binding_id='binding.tokamak-control.hfr-bt-primary-projection',
        template=projection_template,
        manifest=evidence_manifest,
        binder_implementation_id=projection_template.binder_implementation_id,
        binder_implementation_sha256=projection_template.binder_implementation_sha256,
    )
    projection = IdentificationEvidenceProjector().project(
        science_freeze.system,
        evidence_manifest,
        plan,
        _GymToraxCompactObservationSource(compact_by_observation),
    )

    terminals: list[EpisodeTerminalDisposition] = []
    occurrences: list[ActionOccurrenceBinding] = []
    for observation in projection.observations:
        episode_id = f"episode.{observation.manifest_observation_id.removeprefix('observation.')}"
        cell = cells[episode_id]
        word = words[cell.action_word_id]
        response = next(
            value
            for value in observation.compact_values
            if value.value_id == _RESPONSE_VALUE_ID
        )
        terminals.append(
            EpisodeTerminalDisposition(
                episode_id=episode_id,
                projected_observation_id=observation.projected_observation_id,
                physical_independent_unit_id=observation.physical_unit_instance_id,
                physical_sink=PhysicalSinkAxis.NONE,
                observation_validity=ObservationValidityAxis.VALID,
                numerical_validity=NumericalValidityAxis.VALID,
                authority=AuthorityAxis.AUTHORIZED,
                scientific_status=ScientificTerminalAxis.OBSERVED,
                sink_operands=(),
                score_operands=(response,),
                reason_codes=(EpisodeTerminalReasonCode.EPISODE_OBSERVED,),
            )
        )
        for occurrence in word.occurrences:
            stem = f"{episode_id}.{occurrence.occurrence_id}"
            occurrences.append(
                ActionOccurrenceBinding(
                    binding_id=f"occurrence-binding.{stem}",
                    episode_id=episode_id,
                    projected_observation_id=observation.projected_observation_id,
                    action_word=_identity(word.word_id, word),
                    occurrence=occurrence,
                    stage_receipt_ids=(
                        f"receipt.{stem}.requested",
                        f"receipt.{stem}.accepted",
                        f"receipt.{stem}.applied",
                        f"receipt.{stem}.realized",
                    ),
                    stage_clock_binding_ids=(
                        f"clock-binding.{stem}.requested",
                        f"clock-binding.{stem}.accepted",
                        f"clock-binding.{stem}.applied",
                        f"clock-binding.{stem}.realized",
                    ),
                )
            )
    effect_coordinates = tuple(
        ComputabilityEffectCoordinate(
            coordinate_id=f'effect-coordinate.tokamak-control.{_member_slug(member.member_id)}',
            effect_id='effect.tokamak-control.finite-action-response',
            denominator_member_id=member.member_id,
            numerical_view_id=_view_id(member.member_id),
        )
        for member in acquisition_freeze.numerical_members
    )
    extension = build_identification_projection_extension(
        extension_id='projection-extension.tokamak-control.hfr-bt-primary-law',
        manifest=evidence_manifest,
        projection=projection,
        action_words=projection_template.action_words,
        action_occurrences=tuple(occurrences),
        terminal_dispositions=tuple(terminals),
        declared_effect_coordinates=effect_coordinates,
        computability_entries=tuple(
            ComputabilityEffectEntry(
                entry_id=f"effect-entry.{value.coordinate_id}",
                coordinate=value,
                status=ComputabilityEffectStatus.REPRESENTED,
                assumption_ids=(),
                reason_code=EffectReasonCode.EFFECT_REPRESENTED,
            )
            for value in effect_coordinates
        ),
    )
    design, design_binding = bind_confirmatory_finite_action_design_template(
        receipt_id='binding.tokamak-control.hfr-bt-primary-design',
        template=science_freeze.confirmatory_design_template,
        projection_template=projection_template,
        projection_template_binding=projection_template_binding,
        projection_plan=plan,
        binder_implementation_id=science_freeze.confirmatory_design_template.binder_implementation_id,
        binder_implementation_sha256=science_freeze.confirmatory_design_template.binder_implementation_sha256,
    )
    config, config_binding = bind_confirmatory_finite_action_design(
        design=design,
        manifest=evidence_manifest,
        projection_plan=plan,
        projection=projection,
        projection_extension=extension,
        scaffold=science_freeze.scaffold,
        binding_implementation_id=science_freeze.evaluator_binding.binding_implementation_id,
        binding_implementation_sha256=science_freeze.evaluator_binding.binding_implementation_sha256,
    )
    profile_evaluator = confirmatory_finite_action_local_support_profile_evaluator(
        implementation_sha256=science_freeze.extraction_manifest.object_fingerprint
    )
    decoders = CandidatePayloadDecoderRegistry(
        decoders=(CanonicalFiniteActionCompatibilitySetDecoder(),)
    )
    evaluator = ProspectiveConfirmatoryFiniteActionEvaluator(
        binding=science_freeze.evaluator_binding,
        payload_publisher=candidate_payload_plane,
        assessment_assembler=LawAssessmentAssembler(
            payload_reader=candidate_payload_plane,
            decoder_registry=decoders,
            profile_registry=QualificationProfileEvaluatorRegistry(
                evaluators=(profile_evaluator,)
            ),
        ),
        family_assembler=CandidateFamilyAssembler(),
        qualification_service=ResponseLawQualificationService(
            payload_reader=candidate_payload_plane,
            decoder_registry=decoders,
        ),
    )
    evaluation = evaluator.evaluate(
        design=design,
        manifest=evidence_manifest,
        projection_plan=plan,
        projection=projection,
        projection_extension=extension,
        scaffold=science_freeze.scaffold,
        region_support_spec=science_freeze.region_support_spec,
        config=config,
        binding_receipt=config_binding,
        system=science_freeze.system,
        reveal_authority=reveal_authority,
        authority_subject=terminal_identity,
        prerequisite_authority=execution_identity,
        evaluated_at_utc=evaluated_at_utc,
    )
    return GymToraxSelectedParentScienceEvaluationBundle(
        bundle_id='evaluation-bundle.tokamak-control.matched-evaluation',
        science_freeze=_identity(science_freeze.freeze_id, science_freeze),
        acquisition_terminal=terminal_identity,
        manifest=evidence_manifest,
        projection_plan=plan,
        projection_template_binding=projection_template_binding,
        projection=projection,
        projection_extension=extension,
        design=design,
        design_template_binding=design_binding,
        config=config,
        config_binding=config_binding,
        evaluation=evaluation,
        revealed_episode_count=144,
        nonprimary_episode_count_read=0,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def publish_gym_torax_response_experiment_science_evaluation(
    *,
    science_freeze: GymToraxSelectedParentScienceFreeze,
    acquisition_freeze: GymToraxSelectedParentFreeze,
    bundle: GymToraxSelectedParentScienceEvaluationBundle,
    episode_manifests: tuple[ArtifactManifest, ...],
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> ObjectIdentity:
    """Publish the exact prospective evaluator bundle after generic finalization."""

    require_study_authority(
        custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=_identity(acquisition_freeze.freeze_id, acquisition_freeze),
        prerequisite_authority=None,
        grantee_id="operator.execution-service",
        storage_root_id=store.storage_root_id,
        relative_root=acquisition_freeze.relative_root,
        at_utc=at_utc,
    )
    if bundle.science_freeze != _identity(science_freeze.freeze_id, science_freeze):
        raise ValueError("Gym-TORAX evaluation publication names another science freeze")
    manifests = {
        value.logical.logical_artifact_id: value for value in episode_manifests
    }
    if set(manifests) != {
        _episode_logical_id(value)
        for value in science_freeze.expected_primary_episode_ids
    }:
        raise ValueError("Gym-TORAX evaluation publication changes its 144 sidecar roster")
    parents = [
        ArtifactLineageParent(
            identity=_identity(science_freeze.freeze_id, science_freeze),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
    ]
    parents.extend(
        ArtifactLineageParent(
            identity=_episode_object_identity(
                manifests[_episode_logical_id(episode_id)], episode_id
            ),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
        for episode_id in science_freeze.expected_primary_episode_ids
    )
    return store.publish_atomic(
        publication_scope_id='publication.tokamak-control.matched-evaluation-evaluation',
        publication_scope_relative_root=acquisition_freeze.relative_root,
        implementation_sha256=(
            science_freeze.implementation_source_closure.implementation_sha256
        ),
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id=GYM_TORAX_MATCHED_EVALUATION_SCIENCE_EVALUATION_LOGICAL_ID,
                relative_path=f"{science_freeze.relative_root}/evaluation-bundle.json",
                record=bundle,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                lineage_parents=tuple(
                    sorted(parents, key=lambda value: value.identity.object_id)
                ),
            ),
        ),
    )[0]


__all__ = [
    'GymToraxSelectedParentScienceEvaluationBundle',
    'GymToraxSelectedParentScienceFreeze',
    'GYM_TORAX_MATCHED_EVALUATION_REVEAL_AUTHORITY_ID',
    'GYM_TORAX_MATCHED_EVALUATION_SCIENCE_EVALUATED_AT_UTC',
    'GYM_TORAX_MATCHED_EVALUATION_SCIENCE_FREEZE_ID',
    'GYM_TORAX_MATCHED_EVALUATION_SCIENCE_RELATIVE_ROOT',
    'build_gym_torax_response_experiment_science_freeze',
    'evaluate_gym_torax_response_experiment_science',
    'publish_gym_torax_response_experiment_science_evaluation',
    'publish_gym_torax_response_experiment_science_freeze',
]
