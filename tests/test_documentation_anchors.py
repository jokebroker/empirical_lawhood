# SPDX-License-Identifier: MPL-2.0
"""Maintained navigation must reach a real heading, not just an existing file."""

from pathlib import Path

import pytest

from scripts.check_documentation import (
    FINITE_SCIENCE_DOCUMENT,
    FINITE_SCIENCE_LEGACY_TARGET,
    frozen_scientific_link_targets,
    local_anchor_errors,
    markdown_anchors,
)


FROZEN_SCIENCE_TARGETS = {
    FINITE_SCIENCE_DOCUMENT: frozenset((FINITE_SCIENCE_LEGACY_TARGET,)),
    "src/empirical_lawhood/adapters/methods/preparation_applicability/frozen_specification.md": frozenset((
        "../../scripts/analyze_cc1_applicability_opportunity_v1.py",
        "../guides/scientific-experiment-design-standard.md",
        "../results/cc1-applicability-production-opportunity-v1-2026-10-01.md",
        "cc1-boundary-production-v1.md",
        "../results/cc1-boundary-production-v1-2026-10-01.md",
    )),
    "src/empirical_lawhood/adapters/methods/constructed_preparation_applicability/frozen_specification.md": frozenset((
        "../reference/project-owner-authority.md",
        "../results/cc1-applicability-production-opportunity-v1-2026-10-01.md",
    )),
}


def _copy_frozen_science_documents(root: Path, destination: Path) -> None:
    for document in FROZEN_SCIENCE_TARGETS:
        selected = destination / document
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes((root / document).read_bytes())


def test_heading_ids_skip_examples_and_preserve_duplicates_and_explicit_ids():
    text = "# First `run`\n# First run\n```md\n# Not a heading\n```\n## Edit input {#editable}\n<a id='legacy'></a>\n"
    assert markdown_anchors(text) == {"first-run", "first-run-1", "edit-input", "editable", "legacy"}


def test_link_to_existing_markdown_still_refuses_stale_fragment(tmp_path):
    (tmp_path / "guide.md").write_text("# Results and failures\n")
    errors = local_anchor_errors(tmp_path, "README.md", "[read](guide.md#old-results)")
    assert errors == ["README.md: missing local Markdown anchor guide.md#old-results"]
    assert local_anchor_errors(tmp_path, "README.md", "[read](guide.md#results-and-failures)") == []


def test_same_document_encoded_heading_and_setext_are_supported(tmp_path):
    text = "# Résultats\nNext step\n---\n[one](#r%C3%A9sultats) [two](#next-step)\n"
    (tmp_path / "guide.md").write_text(text)
    assert local_anchor_errors(tmp_path, "guide.md", text) == []


def test_remote_and_pdf_fragments_do_not_claim_local_anchor_validation(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"not a PDF parser input")
    assert local_anchor_errors(tmp_path, "guide.md", "[external](https://example.org/#missing) [paper](paper.pdf#page=2)") == []


@pytest.mark.parametrize("document", tuple(FROZEN_SCIENCE_TARGETS))
def test_frozen_science_citation_requires_exact_plan_and_exempts_only_legacy_target(tmp_path, document):
    root = Path(__file__).resolve().parents[1]
    _copy_frozen_science_documents(root, tmp_path)
    selected = tmp_path / document
    raw = (root / document).read_bytes()
    assert frozen_scientific_link_targets(tmp_path) == FROZEN_SCIENCE_TARGETS
    selected.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="(specification|provenance) bytes differ"):
        frozen_scientific_link_targets(tmp_path)
    selected.unlink()
    selected.symlink_to(root / document)
    with pytest.raises(ValueError, match="traverses a link"):
        frozen_scientific_link_targets(tmp_path)


@pytest.mark.parametrize("document", tuple(FROZEN_SCIENCE_TARGETS))
def test_frozen_scientific_citations_refuse_matching_bytes_through_ancestor_link(tmp_path, document):
    root = Path(__file__).resolve().parents[1]
    _copy_frozen_science_documents(root, tmp_path)
    selected = tmp_path / document
    external_owner = tmp_path / "linked-owner"
    selected.parent.rename(external_owner)
    selected.parent.symlink_to(external_owner, target_is_directory=True)
    assert selected.read_bytes() == (root / document).read_bytes()
    with pytest.raises(ValueError, match="traverses a link"):
        frozen_scientific_link_targets(tmp_path)
