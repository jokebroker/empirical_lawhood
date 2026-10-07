"""Actual issued run and effective grant checks before protected output contact."""


from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore, ExternalIssuedStudyPublisher, PROGRAMME_ISSUE_GRANTEE_ID
from empirical_lawhood.adapters.methods.preparation_applicability.result_access import PREPARATION_PROTECTED_OUTCOMES
from empirical_lawhood.kernel.provenance import ObjectIdentity


def authenticate_preparation_result_access(*, context, receipt, logical, writer, include_package=False):
    """Read public custody metadata first; authorize before protected payload I/O."""
    if logical.outcome_access not in PREPARATION_PROTECTED_OUTCOMES:
        from empirical_lawhood.api.current_result_custody import authenticate_unprotected_current_result_if_issued
        authenticate_unprotected_current_result_if_issued(writer=writer, receipt=receipt,
            logical=logical, capability_prefixes=('preparation-applicability.', 'constructed-preparation-applicability.'))
        return
    if context is None or not isinstance(writer, ExternalArtifactPlane):
        raise PermissionError("PREPARATION_PROTECTED_RESULT_CURRENT_AUTHORITY_REQUIRED")
    if context.run_id != receipt.run_id:
        raise PermissionError("PREPARATION_RESULT_RUN_MISMATCH")
    from empirical_lawhood.api.current_result_custody import _authenticate_protected_current_result
    plan, package = _authenticate_protected_current_result(issued_study=context.issued_study,
        execution_authority=context.execution_authority, reveal_authority=context.reveal_authority,
        receipt=receipt, logical=logical, writer=writer, family='preparation')
    return (plan, package) if include_package else plan


def preparation_result_access(*, run_id, issued_study_id, execution_authority_id, reveal_authority_id, writer):
    """Resolve selected IDs from the actual publication and authority stores."""
    from empirical_lawhood.adapters.methods.preparation_applicability.result_access import PreparationApplicabilityResultAccess
    store = ExternalStudyOperationAuthorityStore(writer)
    issued = ExternalIssuedStudyPublisher(artifact_plane=writer, authority_store=store,
        grantee_id=PROGRAMME_ISSUE_GRANTEE_ID).load_executable_study(issued_study_id).manifest
    execution, reveal = store.load(execution_authority_id), store.load(reveal_authority_id)
    return PreparationApplicabilityResultAccess(run_id, ObjectIdentity.from_record(issued.issue_id, issued),
        ObjectIdentity.from_record(execution.authority_id, execution), ObjectIdentity.from_record(reveal.authority_id, reveal))
