# SPDX-License-Identifier: MPL-2.0
"""Human readouts of existing, authorized diagnostic projections."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib

from empirical_lawhood.api.execution import RunStatusSummary
from empirical_lawhood.api.models import DoctorSummary
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes


def _source_version_hint(root: Path | None, installed: str) -> str | None:
    """Compare explicit local metadata only; never search cwd or reinstall."""
    if root is None:
        return None
    metadata = root / "pyproject.toml"
    if metadata.is_symlink() or not metadata.is_file():
        return None
    try:
        document = tomllib.loads(
            read_bounded_bytes(metadata, maximum_bytes=262_144).decode("utf-8")
        )
        source_version = document["project"]["version"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return None
    if (
        not isinstance(source_version, str) or source_version == installed
        or len(source_version) > 128
        or re.fullmatch(r"[A-Za-z0-9.+_-]+", source_version) is None
    ):
        return None
    return (
        f"Metadata hint: selected source declares {source_version}; installed distribution reports {installed}. "
        "Check the selected environment/install; this observation does not establish a wheel defect."
    )


def render_doctor(summary: DoctorSummary, *, project_root: Path | None = None) -> str:
    lines = [
        f"Environment diagnosis completed: empirical-lawhood {summary.package_version}, Python {summary.python_version}",
        f"Selected dependency route: {summary.environment_route or 'none'}",
    ]
    if summary.optional_dependencies:
        for dependency in summary.optional_dependencies:
            state = (
                "matches required version" if dependency.version_matches is True
                else "version mismatch" if dependency.version_matches is False
                else "available" if dependency.available else "unavailable"
            )
            lines.append(
                f"  {dependency.dependency_id}: {state}; installed={dependency.installed_version or 'unknown'}, "
                f"required={dependency.required_version or 'not specified'}"
            )
    else:
        lines.append("No optional dependency route inspected; select --route for its version checks.")
    lines.extend((
        "Dependency observations do not run a native task or qualify an experiment.",
        f"Operating project: {summary.repository_root or 'unconfigured'}; catalog: {summary.catalog_state}",
        f"External storage: read_ready={summary.storage_read_ready}, write_ready={summary.storage_write_ready}",
        "Issued execution readiness (separate from the public demonstration):",
    ))
    for action in summary.action_readiness:
        authority = ", ".join(action.authority_required) or "none"
        reasons = ", ".join(action.reason_codes) or "none"
        lines.append(
            f"  {action.action_id}: infrastructure_ready={action.infrastructure_ready}, "
            f"executable_now={action.executable_now}; authority_required={authority}; blockers={reasons}"
        )
    if not summary.repository_root:
        lines.append("The public example needs no project or operator profile. Use workflow show reactor-response for that path.")
    if summary.warnings:
        lines.append("Warnings: " + ", ".join(summary.warnings))
    hint = _source_version_hint(project_root, summary.package_version)
    if hint is not None:
        lines.append(hint)
    lines.append("Next: workflow list for task selection; docs/results-and-failures.md for result interpretation.")
    return "\n".join(lines)


def render_campaign_status(summary: RunStatusSummary) -> str:
    lines = [
        f"Run {summary.run_id}: operational_status={summary.operational_status}",
        f"Tasks: succeeded={len(summary.succeeded_task_ids)}, failed={len(summary.failed_task_ids)}, "
        f"blocked={len(summary.blocked_task_ids)}, running={len(summary.running_task_ids)}",
        f"Status source: {summary.status_source}",
        f"Implementation: {summary.implementation_commit or 'unreported'}; execution plan: {summary.execution_plan_id or 'unreported'}",
        "Scientific evaluability, verdict and admission: not read by status; inspect separately authorized adjudication records.",
        "Operational failure is distinct from a negative or unevaluable scientific result.",
    ]
    for kind, path in (
        ("Recovery index", summary.recovery_index_relative_path),
        ("Recovery terminal event", summary.recovery_terminal_event_relative_path),
    ):
        if path is not None:
            lines.append(f"{kind}: {path}")
    for path in summary.artifact_receipt_relative_paths:
        lines.append(f"Artifact receipt: {path}")
    for path in summary.adjudication_receipt_relative_paths:
        lines.append(f"Adjudication receipt: {path}")
    for attempt in summary.attempt_history:
        lines.append(
            f"Attempt {attempt.attempt_id} / task {attempt.task_id} / ordinal {attempt.ordinal}: "
            f"{attempt.disposition}; reason={attempt.reason_code or 'none'}; "
            f"failure_class={attempt.failure_class or 'none'}; retryable={attempt.retryable}"
        )
        for kind, path in (
            ("  Recovery event", attempt.recovery_event_relative_path),
            ("  Receipt", attempt.receipt_relative_path),
        ):
            if path is not None:
                lines.append(f"{kind}: {path}")
    if summary.attempt_history_has_more:
        lines.append(f"More attempts: --attempt-history-cursor {summary.attempt_history_next_cursor}")
    elif summary.attempt_history_limit is None:
        lines.append("Use --attempt-history for a bounded page of retained operational attempts.")
    lines.append("Next: verify the same run's receipts and recovery index before any permitted resume; status does not replay tasks or reveal outcomes.")
    return "\n".join(lines)
