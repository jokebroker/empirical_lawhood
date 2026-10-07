# SPDX-License-Identifier: MPL-2.0
"""Existing manuscript links must not silently carry superseded citation data."""

from pathlib import Path

import pytest

from scripts.check_documentation import current_paper_citation_errors


METADATA = {"edition": "0.60", "title": "Empirical Lawhood"}


def citation_errors(tmp_path: Path, text: str, metadata: dict = METADATA) -> list[str]:
    return current_paper_citation_errors(
        tmp_path, "experiments/example/guide.md", text, metadata
    )


@pytest.mark.parametrize("edition", ["0.10", "**0.10**"])
def test_valid_manuscript_target_cannot_retain_old_edition(tmp_path, edition):
    manuscript = tmp_path / "paper/manuscript.md"
    manuscript.parent.mkdir()
    manuscript.touch()
    errors = citation_errors(
        tmp_path,
        f"Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition {edition}.",
    )
    assert len(errors) == 1
    assert "edition 0.10 differs from metadata 0.60" in errors[0]


def test_current_bibliography_title_must_match_metadata(tmp_path):
    errors = citation_errors(
        tmp_path,
        "Seneque (2026), [*Empirical Lawhood: Towards a civilisational response-law engine*]"
        "(../../paper/manuscript.md), edition 0.60.",
    )
    assert len(errors) == 1
    assert "current paper title" in errors[0]


@pytest.mark.parametrize("target", [
    "manuscript.md", "manuscript.pdf",
    "editions/v0.60/Empirical_Lawhood_Manuscript_v0.60.md",
    "editions/v0.60/Empirical_Lawhood_Manuscript_v0.60.pdf",
    "editions/v0.60-deposit/Empirical_Lawhood_Manuscript_v0.60.pdf",
])
def test_current_aliases_accept_correct_bibliography(tmp_path, target):
    assert citation_errors(
        tmp_path,
        f"Seneque (2026), [*Empirical Lawhood*](../../paper/{target}#context), edition 0.60.",
    ) == []


@pytest.mark.parametrize("text", [
    "See [published preprint PDF](../../paper/manuscript.pdf).",
    "See [manuscript](../../paper/manuscript.md), edition **0.60**.",
    "Seneque (2026), [preprint](https://example.org/archive), edition 0.10.",
    "Seneque (2026), [old title](../../paper/editions/v0.10/old.md), edition 0.10.",
    "Seneque (2026), [companion title](../../paper/revision-and-evidence-notes.md), edition 0.10.",
    "Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60. "
    "An earlier edition 0.10 is preserved externally.",
    "[history](https://example.org/archive), edition 0.10; "
    "Seneque (2026), [*Empirical Lawhood*](../../paper/manuscript.md), edition 0.60.",
])
def test_reader_labels_and_distinct_history_keep_their_scope(tmp_path, text):
    assert citation_errors(tmp_path, text) == []


def test_selected_metadata_is_the_citation_owner(tmp_path):
    metadata = {"edition": "0.61", "title": "Revised Empirical Lawhood"}
    text = (
        "Seneque (2026), [*Revised Empirical Lawhood*]"
        "(../../paper/editions/v0.61/Empirical_Lawhood_Manuscript_v0.61.pdf), edition 0.61."
    )
    assert citation_errors(tmp_path, text, metadata) == []
    errors = citation_errors(tmp_path, text.replace("edition 0.61", "edition 0.60"), metadata)
    assert len(errors) == 1
    assert "metadata 0.61" in errors[0]
