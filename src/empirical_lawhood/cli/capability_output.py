"""Render static capability summaries for terminal selection.

Discovery describes installed code and missing inputs.
Use JSON output to inspect the complete canonical records.
"""

from __future__ import annotations

import re

from empirical_lawhood.api import CapabilityListSummary, CapabilityShowSummary
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.executable_bindings import (
    CapabilityExecutionAvailability,
)


def _words(value: str) -> str:
    return re.sub(r"[._-]+", " ", value).lower()


def _subject(registration: CandidateCapabilityRegistration) -> ObjectIdentity:
    manifest = registration.manifest
    return ObjectIdentity.from_record(
        f"capability-manifest.{manifest.capability_key}."
        f"{manifest.capability_version.replace('.', '-')}",
        manifest,
    )


def _entry(
    registration: CandidateCapabilityRegistration,
    availability: CapabilityExecutionAvailability | None,
) -> list[str]:
    manifest = registration.manifest
    lines = [
        _words(manifest.capability_key).capitalize(),
        f"  ID: {manifest.registry_id} | Kind: {_words(manifest.kind.value)}",
    ]
    if availability is None:
        lines.append(
            "  Binding: not reported | Limit: execution availability not composed"
        )
    else:
        installed = "installed" if availability.binding is not None else "absent"
        reasons = "; ".join(
            _words(reason.value) for reason in availability.reason_codes
        )
        lines.append(f"  Binding: {installed} | Limit: {reasons}")
    return lines


def render_capability_list(summary: CapabilityListSummary) -> str:
    """Keep each registration to three lines and retain the exact page cursor."""
    availability = {
        value.discovery_subject: value for value in summary.execution_availability
    }
    lines = [
        f"Capabilities: {summary.returned_count} returned (limit {summary.limit})."
    ]
    for registration in summary.registrations:
        lines.extend(_entry(registration, availability.get(_subject(registration))))
    if summary.next_cursor is not None:
        lines.extend(("More results. Continue with --cursor:", summary.next_cursor))
    else:
        lines.append("End of results.")
    lines.append(
        "Static discovery establishes neither readiness nor authority. Use --format json for complete records."
    )
    return "\n".join(lines)


def render_capability_show(summary: CapabilityShowSummary) -> str:
    """Show the selection contract without expanding nested canonical records."""
    manifest = summary.registration.manifest
    lines = _entry(summary.registration, summary.execution_availability)
    lines.extend(
        (
            f"  Evidence limit: {_words(manifest.maximum_evidence_ceiling.value)}",
            f"  Outcome access limit: {_words(manifest.maximum_outcome_access.value)}",
            "  Permissions: "
            + (
                ", ".join(_words(value.value) for value in manifest.permissions)
                or "none"
            ),
            f"  Configuration: {manifest.config_schema}",
        )
    )
    if summary.bundle_binding is not None:
        binding = summary.bundle_binding
        lines.append(f"  Bundle: {binding.bundle_id}@{binding.bundle_version}")
    lines.append(
        "Static discovery establishes neither readiness nor authority. Use --format json for complete records."
    )
    return "\n".join(lines)
