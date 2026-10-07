"Pure tokamak control replication package topology, authority requirements, and terminal rules."

from __future__ import annotations

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.planning.prospective_config import ProspectiveAuthorityAccess, ProspectiveAuthorityOperation, ProspectiveAuthorityRequirementSet, ProspectiveAuthorityRequirement, ProspectivePackageKind, ProspectivePackageLineageNode, ProspectivePackageLineage, ProspectiveSelectorTransition, ProspectiveTerminalMatrix, ProspectiveTerminalRule


GYM_TORAX_PACKAGE_IDS = (
    'tokamak-control.matched-evaluation-parent',
    'tokamak-control.prospective-response-child',
    'tokamak-control.prospective-route-qualification',
    'tokamak-control.source-qualification-excluded',
)


def build_gym_torax_package_lineage() -> ProspectivePackageLineage:
    """Freeze all four package identities and conditions before outcomes."""

    nodes = (
        ProspectivePackageLineageNode(
            package_id='tokamak-control.matched-evaluation-parent',
            package_label="fresh sealed parent from measurement through admission",
            kind=ProspectivePackageKind.PROTECTED_PARENT,
            parent_package_ids=('tokamak-control.source-qualification-excluded',),
            issue_condition_ids=('condition.tokamak-control.source-and-controlled-io-qualified',),
            maximum_evidence=EvidenceCeiling.ADMISSION,
            current_linked_campaign_role="role.protected-measurement-through-admission-parent",
            contains_action_selection_study=True,
            contains_controller=True,
            contains_compiler=True,
            contains_current_admission_or_controller_evaluation_record=True,
        ),
        ProspectivePackageLineageNode(
            package_id='tokamak-control.prospective-response-child',
            package_label="conditional prospective matched ACTION/HOLD child",
            kind=ProspectivePackageKind.PROSPECTIVE_CHILD,
            parent_package_ids=('tokamak-control.matched-evaluation-parent', 'tokamak-control.prospective-route-qualification'),
            issue_condition_ids=('condition.tokamak-control.prospective-evaluation-route-qualified',),
            maximum_evidence=EvidenceCeiling.CONTROLLER_USE,
            current_linked_campaign_role="role.conditional-controller-use-child",
            contains_action_selection_study=False,
            contains_controller=False,
            contains_compiler=False,
            contains_current_admission_or_controller_evaluation_record=True,
        ),
        ProspectivePackageLineageNode(
            package_id='tokamak-control.prospective-route-qualification',
            package_label="excluded controller use route qualification",
            kind=ProspectivePackageKind.ROUTE_QUALIFICATION,
            parent_package_ids=('tokamak-control.matched-evaluation-parent',),
            issue_condition_ids=('condition.tokamak-control.exact-controller-admission-eligible',),
            maximum_evidence=EvidenceCeiling.NON_PROMOTABLE,
            current_linked_campaign_role="role.conditional-controller-use-route-qualification",
            contains_action_selection_study=False,
            contains_controller=False,
            contains_compiler=False,
            contains_current_admission_or_controller_evaluation_record=False,
        ),
        ProspectivePackageLineageNode(
            package_id='tokamak-control.source-qualification-excluded',
            package_label="excluded source runtime and resource qualification",
            kind=ProspectivePackageKind.EXCLUDED_QUALIFICATION,
            parent_package_ids=(),
            issue_condition_ids=('condition.tokamak-control.no-effect-readiness-accepted',),
            maximum_evidence=EvidenceCeiling.NON_PROMOTABLE,
            current_linked_campaign_role="role.excluded-qualification",
            contains_action_selection_study=False,
            contains_controller=False,
            contains_compiler=False,
            contains_current_admission_or_controller_evaluation_record=False,
        ),
    )
    return ProspectivePackageLineage(
        lineage_id='package-lineage.tokamak-control.four-package',
        shared_qualification_package_id='tokamak-control.source-qualification-excluded',
        conditional_branch_id='branch.tokamak-control.exact-prospective-evaluation-eligibility',
        nodes=nodes,
        mutually_exclusive_with_branch_ids=('branch.tokamak-control.no-prospective-evaluation-nonattempt',),
        route_predeclared_before_parent_issue=True,
        outcome_dependent_package_construction=False,
        grants_issue_or_execution_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


_AUTHORITY_CONTRACT = {
    ProspectiveAuthorityOperation.ISSUE: (
        StudyAuthorityKind.CUSTODY_PUBLICATION,
        ProspectiveAuthorityAccess.OUTCOME_BLIND,
        'role.tokamak-control.base-custodian',
    ),
    ProspectiveAuthorityOperation.EXTENSION_ISSUE: (
        StudyAuthorityKind.CUSTODY_PUBLICATION,
        ProspectiveAuthorityAccess.OUTCOME_BLIND,
        'role.tokamak-control.extension-custodian',
    ),
    ProspectiveAuthorityOperation.BASE_EXECUTE: (
        StudyAuthorityKind.EXPERIMENT_EXECUTION,
        ProspectiveAuthorityAccess.SEALED_EXECUTION,
        'role.tokamak-control.base-executor',
    ),
    ProspectiveAuthorityOperation.EXECUTE: (
        StudyAuthorityKind.EXPERIMENT_EXECUTION,
        ProspectiveAuthorityAccess.SEALED_EXECUTION,
        'role.tokamak-control.extension-executor',
    ),
    ProspectiveAuthorityOperation.RECOVER: (
        StudyAuthorityKind.EXPERIMENT_EXECUTION,
        ProspectiveAuthorityAccess.SEALED_EXECUTION,
        'role.tokamak-control.recovery-operator',
    ),
    ProspectiveAuthorityOperation.REVEAL: (
        StudyAuthorityKind.OUTCOME_REVEAL,
        ProspectiveAuthorityAccess.EVALUATOR_ONLY_REVEAL,
        'role.tokamak-control.evaluator-revealer',
    ),
}


def build_gym_torax_authority_requirements() -> ProspectiveAuthorityRequirementSet:
    """Describe exact required grants without embedding or manufacturing one."""

    requirements: list[ProspectiveAuthorityRequirement] = []
    for package_id in GYM_TORAX_PACKAGE_IDS:
        slug = package_id.lower()
        prior_id: str | None = None
        execute_id: str | None = None
        for operation in ProspectiveAuthorityOperation:
            kind, access, role_id = _AUTHORITY_CONTRACT[operation]
            requirement_id = f"authority-requirement.{slug}.{operation.value.lower()}"
            if operation is ProspectiveAuthorityOperation.ISSUE:
                prerequisites: tuple[str, ...] = ()
            elif operation is ProspectiveAuthorityOperation.EXTENSION_ISSUE:
                prerequisites = (prior_id or "",)
            elif operation is ProspectiveAuthorityOperation.BASE_EXECUTE:
                prerequisites = (prior_id or "",)
            elif operation is ProspectiveAuthorityOperation.EXECUTE:
                prerequisites = (prior_id or "",)
                execute_id = requirement_id
            elif operation in {
                ProspectiveAuthorityOperation.RECOVER,
                ProspectiveAuthorityOperation.REVEAL,
            }:
                prerequisites = (execute_id or "",)
            else:  # pragma: no cover - exhaustive enum contract
                raise AssertionError("unknown prospective authority operation")
            requirements.append(
                ProspectiveAuthorityRequirement(
                    requirement_id=requirement_id,
                    package_id=package_id,
                    operation=operation,
                    scope_id=f'scope.tokamak-control.{slug}.{operation.value.lower()}',
                    responsible_role_id=role_id,
                    expected_grantee_id=role_id.replace("role.", "operator."),
                    required_authority_schema=StudyOperationAuthority.SCHEMA,
                    required_authority_kind=kind,
                    required_access=access,
                    prerequisite_requirement_ids=prerequisites,
                    authority_record_embedded=False,
                    authority_granted=False,
                )
            )
            if operation not in {
                ProspectiveAuthorityOperation.RECOVER,
                ProspectiveAuthorityOperation.REVEAL,
            }:
                prior_id = requirement_id
    return ProspectiveAuthorityRequirementSet(
        requirement_set_id='authority-requirements.tokamak-control.four-package',
        package_ids=GYM_TORAX_PACKAGE_IDS,
        reveal_package_ids=GYM_TORAX_PACKAGE_IDS,
        requirements=tuple(sorted(requirements, key=lambda value: value.requirement_id)),
        distinct_operation_roles_required=True,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


_TERMINAL_ROWS = (
    (
        "generic owner/codec/binding cannot represent exact design",
        "GENERIC_CONSUMER_NOT_READY",
        "generic-extension audit",
        "no scientific issue",
    ),
    (
        "Gym source/runtime/field/action qualification fails",
        "SOURCE_OR_RUNTIME_NOT_QUALIFIED",
        "readiness receipts",
        "no scientific issue",
    ),
    (
        "source-backed controlled-IO contract unavailable",
        "CONTROLLED_IO_SOURCE_NOT_QUALIFIED",
        "source qualification receipts",
        "no scientific issue or finite-grid substitute",
    ),
    (
        "controlled-IO basis exceeds frozen bounds",
        "CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED",
        "sizing receipt",
        "no truncation or automatic follow-up",
    ),
    (
        "storage/resource/authority fails",
        "EXACT_OPERATIONAL_STOP",
        "readiness or authority record",
        "no worker or scientific write",
    ),
    (
        "Measurement incomplete or invalid",
        "MEASUREMENT_PARTIAL_NOT_SUPPORTED_OR_UNEVALUABLE",
        "independent measurement facets",
        "no dependent law or admission",
    ),
    (
        "Order-relation causal/delivery/recurrence falsifier",
        "ORDER_RELATION_NEGATIVE_MIXED_OR_UNEVALUABLE",
        "Measurement and exact order-relation facets",
        "no response or local law promotion",
    ),
    (
        "response submaterial or member mixed",
        "RESPONSE_NOT_SUPPORTED_OR_MIXED",
        "finite response products",
        "no local law",
    ),
    (
        "confirmation/prefix/support/structure/computability fails",
        "LOCAL_LAW_OBSTRUCTION",
        "terminal result and batch coordinate",
        "zero-law atlas obstruction",
    ),
    (
        "confirmatory design cannot bind revealed projection",
        "CONFIRMATORY_BINDING_REFUSED",
        "sealed projection and refusal",
        "no finalizer invocation",
    ),
    (
        "controlled-IO member unsupported or unevaluable",
        "CONTROLLED_IO_ADMISSION_OBSTRUCTION",
        "law atlas and member receipts",
        "no active controller or controller use",
    ),
    (
        "Local law supported but admission empty or unevaluable",
        "ADMISSION_OBSTRUCTION",
        "law atlas and admission corpus",
        "no active controller or controller use",
    ),
    (
        "active admitted but reachability empty",
        "REACHABILITY_OBSTRUCTION",
        "law and admission receipts",
        "HOLD or NONATTEMPT only",
    ),
    (
        "active admitted but measured HOLD absent",
        "MISSING_HOLD_OBSTRUCTION",
        "action commitment correctness",
        "Controller use efficacy unevaluable",
    ),
    (
        "robust active set differs from lower-Ip singleton",
        "REPLICATION_CONDITION_FALSE",
        "exact admission result",
        "no controller use child",
    ),
    ("Controller use route canary fails", "CONTROLLER_USE_ROUTE_NOT_QUALIFIED", "Admission controller parent", "no controller use issue"),
    (
        "Controller use acquisition/delivery/custody/reveal invalid",
        "CONTROLLER_USE_OPERATIONAL_OR_TECHNICAL_STOP",
        "sealed or partial records",
        "no retune",
    ),
    (
        "Controller use lower bound is at most 0.0017",
        "CONTROLLER_USE_NEGATIVE_OR_SUBMATERIAL",
        "full cohort result",
        "replication not positive",
    ),
    (
        "member or stratum mixed",
        "CONTROLLER_USE_MIXED",
        "member and stratum results",
        "replication not positive",
    ),
    (
        "outside-support HOLD control fails",
        "CONTROLLER_USE_HOLD_CONTROL_FAILED",
        "matched controller use adjudication",
        "no efficacy rewrite",
    ),
    (
        "all measurement through controller use rules pass",
        "POSITIVE_SIMULATOR_LOCAL_CONTROLLER_USE_REPLICATION",
        "law atlas admission controller and controller use chain",
        "terminal; no further action authorized",
    ),
)


def build_gym_torax_terminal_matrix() -> ProspectiveTerminalMatrix:
    """Return the complete first-applicable honest terminal precedence."""

    return ProspectiveTerminalMatrix(
        matrix_id='terminal-matrix.tokamak-control.complete',
        selector_transitions=(
            ProspectiveSelectorTransition(1, "NO_EFFECT_READINESS_ACCEPTED", 'SOURCE_ASSESSMENT'),
            ProspectiveSelectorTransition(2, "SOURCE_AND_CONTROLLED_IO_QUALIFIED", 'TOKAMAK_SOURCE_AND_CONTROLLER_QUALIFICATION'),
            ProspectiveSelectorTransition(3, "EXACT_CONTROLLER_ADMISSION_ELIGIBLE", 'TOKAMAK_CONTROLLER_ADMISSION'),
            ProspectiveSelectorTransition(4, "PROSPECTIVE_CONTROLLER_USE_ROUTE_QUALIFIED", 'TOKAMAK_PROSPECTIVE_EVALUATION'),
        ),
        terminal_rules=tuple(
            ProspectiveTerminalRule(index, *row)
            for index, row in enumerate(_TERMINAL_ROWS, start=1)
        ),
        first_applicable_terminal_wins=True,
        valid_negative_mixed_partial_stopped_or_unevaluable_is_complete=True,
        postissue_alternate_rescue_allowed=False,
        grants_authority_or_followon_action=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


__all__ = [
    'GYM_TORAX_PACKAGE_IDS',
    'build_gym_torax_authority_requirements',
    'build_gym_torax_package_lineage',
    'build_gym_torax_terminal_matrix',
]
