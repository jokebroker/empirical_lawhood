"""Assembly of existing finite-action proof owners from explicit adapter operands."""

from dataclasses import fields, replace
from decimal import Decimal as D

from empirical_lawhood.adapters.methods.contracts import LawCandidateEvidence, JointQualificationAssessment
from empirical_lawhood.adapters.methods.qualification_profiles import MethodEquivalentProfileKind, ComponentQualificationProfile, JointQualificationProfile, QualificationProofOwner, assess_joint_predictive_candidate, build_method_equivalent_qualification_profile
from empirical_lawhood.kernel.identification import (
    AdequacyCheckKind as K,
    AdequacyCheckResult,
    UncertaintyClass,
    UncertaintyComponent,
    UncertaintyDecomposition,
)
from empirical_lawhood.kernel.obligations import ObligationStatus as O
from empirical_lawhood.kernel.predictive_uncertainty import JointPredictiveUncertainty
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal


def finite_profile(
    *, owner: QualificationProofOwner, stem: str, method: str, payload_schema: str, semantics: str
) -> JointQualificationProfile:
    base = build_method_equivalent_qualification_profile(
        owner, MethodEquivalentProfileKind.FINITE_ACTION
    )
    values = {f.name: getattr(base, f.name) for f in fields(ComponentQualificationProfile)}

    def rename(text: str) -> str:
        return text.replace("finite-action", stem)

    values.update(
        profile_id=f"profile.{stem}",
        applicable_method_keys=(method,),
        applicable_payload_schemas=(payload_schema,),
        applicable_extension_schemas=(payload_schema,),
        proof_owner_bindings=tuple(
            sorted(
                (
                    replace(
                        b,
                        binding_id=rename(b.binding_id),
                        output_id=rename(b.output_id),
                        rule_id=rename(b.rule_id),
                        input_schema=payload_schema
                        if b.input_schema == base.applicable_payload_schemas[0]
                        else b.input_schema,
                        rule_semantics=semantics,
                    )
                    for b in base.proof_owner_bindings
                ),
                key=lambda b: b.binding_id,
            )
        ),
        facet_ids=tuple(sorted(rename(v) for v in base.facet_ids)),
        conformance_case_ids=tuple(sorted(rename(v) for v in base.conformance_case_ids)),
        allowed_not_applicable_output_ids=("uncertainty.transport",),
    )
    return JointQualificationProfile(**values)


def finite_assessment(
    *,
    owner: QualificationProofOwner,
    profile: JointQualificationProfile,
    candidate: LawCandidateEvidence,
    stem: str,
    custody: bool,
    evaluable: bool,
    qualifies: bool,
    assigned_roots: tuple[str, ...],
    qualification: ObjectIdentity,
    design_artifact: ArtifactIdentity,
    metrics: tuple[NamedDecimal, ...],
    widths: tuple[NamedDecimal, ...],
    numerical_floors: tuple[NamedDecimal, ...],
    failure_reason: str,
    unknown_reason: str,
    probability: D = D(".90"),
) -> JointQualificationAssessment:
    status = (
        O.UNEVALUABLE if not custody or not evaluable else O.SATISFIED if qualifies else O.FAILED
    )
    reasons = (
        () if status is O.SATISFIED else (failure_reason if status is O.FAILED else unknown_reason,)
    )
    links = tuple(e.link_id for e in candidate.evidence_links) if custody else ()
    checks = tuple(
        sorted(
            (
                AdequacyCheckResult(
                    f"{stem}.{k.value.lower().replace('_', '-')}",
                    k,
                    status
                    if k in (K.HELD_OUT_CALIBRATION, K.DECISIVE_FALSIFIER)
                    else O.SATISFIED
                    if custody and evaluable
                    else O.UNEVALUABLE,
                    True,
                    metrics if custody else (),
                    reasons
                    if k in (K.HELD_OUT_CALIBRATION, K.DECISIVE_FALSIFIER)
                    or not custody
                    or not evaluable
                    else (),
                    links,
                )
                for k in K
            ),
            key=lambda v: v.check_id,
        )
    )
    diagnostics = UncertaintyDecomposition(
        f"{stem}.diagnostics",
        tuple(
            sorted(
                (
                    UncertaintyComponent(
                        f"{stem}.{k.value.lower()}",
                        k,
                        O.NOT_APPLICABLE if k is UncertaintyClass.TRANSPORT else O.UNEVALUABLE,
                        (),
                        ()
                        if k is UncertaintyClass.TRANSPORT
                        else ("FINITE_BOUND_DOES_NOT_IDENTIFY_LATENT_COMPONENTS",),
                        (),
                    )
                    for k in UncertaintyClass
                ),
                key=lambda v: v.component_id,
            )
        ),
    )
    template = candidate.obligation_template
    joint = JointPredictiveUncertainty(
        f"{stem}.coverage",
        template.uncertainty_method_key,
        qualification if custody else None,
        design_artifact,
        template.independent_unit_id,
        assigned_roots,
        "all-assigned-word-both-views-no-refit",
        template.interval_quantity_ids,
        template.support_id,
        probability,
        tuple(sorted(widths, key=lambda v: v.value_id)) if custody else (),
        tuple(sorted(numerical_floors, key=lambda v: v.value_id)),
        template.assumption_ids,
        status,
        reasons,
        links,
    )
    return assess_joint_predictive_candidate(
        candidate=candidate,
        profile=profile,
        owner=owner,
        prefix=stem,
        checks=checks,
        component_diagnostics=diagnostics,
        joint_uncertainty=joint,
    )
