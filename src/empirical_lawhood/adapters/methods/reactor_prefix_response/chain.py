"""Finite-action owner composition; no alternate estimator or law finalizer."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.contracts import CandidateFamilyLedger
from empirical_lawhood.adapters.methods.finite_action_identification import FiniteActionCandidateScaffold, FiniteActionIdentificationConfig, FiniteActionIdentificationResult
from empirical_lawhood.adapters.methods.finite_action_registration import (
    compose_finite_action_identification,
)
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidateFamilyAssembler,
    CandidatePayloadPublisher,
    CandidatePayloadReader,
    LawAssessmentAssembler,
    ResponseLawQualificationService,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import IdentificationEvidenceProjection
from empirical_lawhood.planning.identification_evidence_extensions import IdentificationEvidenceProjectionExtension
from empirical_lawhood.adapters.methods.qualification_profiles import (
    MethodEquivalentQualificationProfileEvaluator,
)


@dataclass(frozen=True, slots=True)
class FiniteChainConfig(CanonicalRecord):
    """Frozen method/scaffold/roster after exact projection-parent binding.

    Acquisition design and thresholds must precede the native parent. This
    binding authenticates its resulting identities; it grants no acquisition,
    source readiness, issue, or outcome access.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/finite-chain-config'
    config_id: str
    system: SystemSpec
    method: FiniteActionIdentificationConfig
    scaffold: FiniteActionCandidateScaffold
    family: CandidateFamilyLedger

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        identity = ObjectIdentity.from_record(self.system.system_id, self.system)
        if (
            self.scaffold.system != identity
            or self.family.system != identity
            or self.family.dataset_or_projection != self.method.projection
            or self.family.axis_map != self.scaffold.axis_map
            or self.method.axis_map != self.scaffold.axis_map
            or self.family.claim_template != self.scaffold.claim_template
            or self.family.obligation_template != self.scaffold.obligation_template
            or self.family.claim_unit_binding != self.scaffold.claim_unit_binding
            or self.family.method_key != self.method.method_key
            or self.family.method_version != self.method.method_version
        ):
            raise ValueError("finite chain system, projection, scientific templates or axes differ")
        if (
            len(self.family.members) != 1
            or self.family.members[0].candidate_id != self.scaffold.candidate_id
            or self.family.members[0].config
            != ObjectIdentity.from_record(self.method.config_id, self.method)
            or self.scaffold.outcome_access
            not in {OutcomeAccess.OUTCOME_BLIND, OutcomeAccess.EVALUATOR_REVEAL}
            or self.family.outcome_access is not self.scaffold.outcome_access
        ):
            raise ValueError(
                "finite chain requires one frozen candidate under matching declared outcome access"
            )


def identify_and_qualify(
    *,
    config: FiniteChainConfig,
    projection: IdentificationEvidenceProjection,
    extension: IdentificationEvidenceProjectionExtension,
    publisher: CandidatePayloadPublisher,
    reader: CandidatePayloadReader,
) -> tuple[FiniteActionIdentificationResult, LawQualificationResult]:
    """Invoke the installed producer, proof profile and sole finalizer in order."""
    if (
        ObjectIdentity.from_record(projection.projection_id, projection) != config.method.projection
        or ObjectIdentity.from_record(extension.extension_id, extension)
        != config.method.projection_extension
    ):
        raise ValueError("finite chain projection parent identity differs")
    components = compose_finite_action_identification(
        config=config.method, payload_publisher=publisher
    )
    profile = components.profile_registry.evaluators[0]
    if not isinstance(profile, MethodEquivalentQualificationProfileEvaluator):
        raise TypeError("finite chain requires the installed method-equivalent proof owner")
    profile_identity = ObjectIdentity.from_record(profile.profile.profile_id, profile.profile)
    owner_identity = ObjectIdentity.from_record(profile.owner.owner_id, profile.owner)
    if (
        config.family.selector_capability != owner_identity
        or config.family.selector_implementation != owner_identity
        or config.family.selector_config != profile_identity
    ):
        raise ValueError("finite chain qualification owner differs from its frozen ledger")
    result = components.producer.identify(
        projection=projection, extension=extension, config=config.method, scaffold=config.scaffold
    )
    assessment = LawAssessmentAssembler(
        payload_reader=reader,
        decoder_registry=components.decoder_registry,
        profile_registry=components.profile_registry,
    ).assemble(
        system=config.system,
        candidate=result.candidate_evidence,
        qualification_profile=profile_identity,
    )
    family = CandidateFamilyAssembler().assemble(config.family, (assessment,))
    qualification = ResponseLawQualificationService(
        payload_reader=reader, decoder_registry=components.decoder_registry
    ).qualify(config.system, config.method.projection, family)
    return result, qualification
