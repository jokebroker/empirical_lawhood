"""Generate the flagship runbook from the implemented request and help contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields
import hashlib

from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.runtime.multi_world_study_runbook import FLAGSHIP_OPERATION_PATHS, MultiWorldStudyOperationHelpBinding, MultiWorldStudyOperationRequestContract, MultiWorldStudyOperationRunbook, build_multi_world_study_operation_runbook
from empirical_lawhood.runtime.multi_world_study import MultiWorldJointAdjudicationResult
from empirical_lawhood.runtime.multi_world_study_issue import MultiWorldAuthorityRequired, MultiWorldOutcomeBarrierPrefix, MultiWorldStudyBarrierStatus

from .models import ExecutableStudyCompilationSummary, StudyBundleCompilationSummary, MultiWorldStudyIssueSummary, ExecutableStudyIssueSummary
from .results import AdvanceStudyBundleRequest, CampaignStatusRequest, CloseStudyBundleRequest, CompileCandidateRequest, CompileStudyBundleRequest, IssueExtensionsRequest, IssueStudyBundleRequest, StudyBundleStatusRequest, ResumeCampaignRequest, RunCampaignRequest


FLAGSHIP_REQUEST_TYPES = {
    "campaign.bundle-candidate-compile": CompileStudyBundleRequest,
    "campaign.close-bundle": CloseStudyBundleRequest,
    "campaign.issue-bundle": IssueStudyBundleRequest,
    "campaign.bundle-status": StudyBundleStatusRequest,
    "campaign.advance-bundle": AdvanceStudyBundleRequest,
    "campaign.compile-candidate": CompileCandidateRequest,
    "campaign.issue-extensions": IssueExtensionsRequest,
    "campaign.resume": ResumeCampaignRequest,
    "campaign.run": RunCampaignRequest,
    "campaign.status": CampaignStatusRequest,
}


def _request_contracts() -> tuple[MultiWorldStudyOperationRequestContract, ...]:
    values = []
    for operation, request_type in FLAGSHIP_REQUEST_TYPES.items():
        contract = tuple(
            (
                field.name,
                str(field.type),
                "MISSING" if field.default is MISSING else repr(field.default),
                (
                    "MISSING"
                    if field.default_factory is MISSING
                    else getattr(field.default_factory, "__qualname__", repr(field.default_factory))
                ),
            )
            for field in fields(request_type)
        )
        values.append(
            MultiWorldStudyOperationRequestContract(
                contract_id=f"request-contract.{operation}",
                operation_id=operation,
                request_type=f"{request_type.__module__}.{request_type.__qualname__}",
                field_names=tuple(sorted(field.name for field in fields(request_type))),
                field_contract_sha256=hashlib.sha256(canonical_json_bytes(contract)).hexdigest(),
            )
        )
    return tuple(sorted(values, key=lambda value: value.contract_id))


def _payload_schemas() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    generic_result = 'empirical-lawhood/api/result'
    return {
        "campaign.bundle-candidate-compile": (
            (StudyBundleCompilationSummary.SCHEMA,),
            (generic_result,),
        ),
        "campaign.close-bundle": (
            (MultiWorldJointAdjudicationResult.SCHEMA,),
            (generic_result,),
        ),
        "campaign.issue-bundle": (
            (MultiWorldStudyIssueSummary.SCHEMA,),
            (generic_result,),
        ),
        "campaign.bundle-status": (
            (MultiWorldStudyBarrierStatus.SCHEMA,),
            (generic_result,),
        ),
        "campaign.advance-bundle": (
            (MultiWorldOutcomeBarrierPrefix.SCHEMA,),
            (MultiWorldAuthorityRequired.SCHEMA,),
        ),
        "campaign.compile-candidate": (
            (ExecutableStudyCompilationSummary.SCHEMA,),
            (generic_result,),
        ),
        "campaign.issue-extensions": ((ExecutableStudyIssueSummary.SCHEMA,), (generic_result,)),
        "campaign.resume": (
            ('empirical-lawhood/api/run-execution-summary',),
            ('empirical-lawhood/api/campaign-authority-required',),
        ),
        "campaign.run": (
            ('empirical-lawhood/api/run-execution-summary',),
            ('empirical-lawhood/api/campaign-authority-required',),
        ),
        "campaign.status": (('empirical-lawhood/api/run-status-summary',), (generic_result,)),
    }


def _prerequisites() -> dict[str, tuple[str, ...]]:
    return {
        "campaign.compile-candidate": (),
        "campaign.issue-extensions": ("campaign.compile-candidate",),
        "campaign.bundle-candidate-compile": ("campaign.compile-candidate",),
        "campaign.issue-bundle": (
            "campaign.bundle-candidate-compile",
            "campaign.issue-extensions",
        ),
        "campaign.run": ("campaign.issue-bundle",),
        "campaign.status": ("campaign.run",),
        "campaign.resume": ("campaign.run",),
        "campaign.advance-bundle": ("campaign.run",),
        "campaign.bundle-status": ("campaign.advance-bundle",),
        "campaign.close-bundle": (
            "campaign.bundle-status",
            "campaign.advance-bundle",
        ),
    }


def build_implemented_study_operation_runbook(
    help_payload_by_operation: Mapping[str, bytes],
    *,
    config_root: str,
) -> MultiWorldStudyOperationRunbook:
    """Bind actual help bytes to the current typed request/result surface."""

    paths = dict(FLAGSHIP_OPERATION_PATHS)
    if set(help_payload_by_operation) != set(paths):
        raise ValueError("flagship runbook help does not cover the implemented operation roster")
    help_bindings = tuple(
        sorted(
            (
                MultiWorldStudyOperationHelpBinding(
                    binding_id=f"help-binding.{operation}",
                    operation_id=operation,
                    command_path=paths[operation],
                    help_sha256=hashlib.sha256(payload).hexdigest(),
                    help_size_bytes=len(payload),
                )
                for operation, payload in help_payload_by_operation.items()
            ),
            key=lambda value: value.binding_id,
        )
    )
    return build_multi_world_study_operation_runbook(
        config_root=config_root,
        request_contracts=_request_contracts(),
        help_bindings=help_bindings,
        step_payload_schemas=_payload_schemas(),
        prerequisite_operations=_prerequisites(),
        write_confirmation_operations=frozenset(
            {
                "campaign.issue-bundle",
                "campaign.issue-extensions",
                "campaign.resume",
                "campaign.run",
            }
        ),
    )


__all__ = [
    "FLAGSHIP_REQUEST_TYPES",
    'build_implemented_study_operation_runbook',
]
