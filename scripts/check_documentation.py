# SPDX-License-Identifier: MPL-2.0
"""Check shipped targets, current paper citations and unactivated-shell commands."""

from html.parser import HTMLParser
from html import unescape
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
import zipfile


# Publication Markdown keeps its supplied links only after authentication against
# the independently pinned current-edition export. Maintained aliases/guides
# receive normal target checks. The complete original archive stays external.
SELECTED_PAPER_ARCHIVE_SHA256 = "9e3ea52ae38cf9113f21dc4f479ac6f51f64e0a0d58a1f70fb977674c15df1b5"
FROZEN_EDITION_DOCUMENTS = frozenset((
    "editions/v0.60/Empirical_Lawhood_Manuscript_v0.60.md",
    "editions/v0.60/Empirical_Lawhood_Evidence_and_Provenance_v0.60.md",
    "editions/v0.60/Revision_Summary_v0.60.md",
))
# The finite science plan binds exact bytes, including its historical citation.
# Only this removed bibliography target is exempt; other navigation is checked.
FINITE_SCIENCE_DOCUMENT = "src/empirical_lawhood/adapters/methods/finite_response_law/specification.md"
FINITE_SCIENCE_SHA256 = "90baf62da60fd3f925f9212429e5b724184d6ea29da33a4da6d5bc42aa825d5f"
FINITE_SCIENCE_LEGACY_TARGET = "../../../../../paper/sources/Empirical_Lawhood_Methods_and_Evidence_v0.9.1.pdf"


def _paper_file(paper: Path, relative: str) -> Path:
    path = paper / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"frozen documentary path escapes paper: {relative}")
    cursor = path
    while cursor != paper:
        if cursor.is_symlink():
            raise ValueError(f"frozen documentary path traverses a link: {relative}")
        cursor = cursor.parent
    if not path.is_file():
        raise ValueError(f"frozen documentary source is missing: {relative}")
    return path


def frozen_edition_documents(root: Path) -> dict[str, tuple[int, str]]:
    """Authenticate exact current-edition Markdown before retaining its links.

    The independently pinned curated ZIP binds selected original member bytes.
    Older publication payloads are absent from this export and shipped tree.
    """
    paper = root / "paper"
    inputs = json.loads(_paper_file(paper, "source/canonical-inputs.json").read_bytes())
    if inputs["edition"] != "0.60":
        raise ValueError("frozen documentary canonical edition differs")
    archive = inputs["archive"]
    if (
        archive["path"] != "editions/Empirical_Lawhood_v0.60_Selected_Source.zip"
        or archive["bytes"] != 2128309
        or archive["sha256"] != SELECTED_PAPER_ARCHIVE_SHA256
        or archive["members"] != 17
        or archive["kind"] != "CURATED_CURRENT_EDITION_ONLY_EXPORT"
    ):
        raise ValueError("frozen documentary archive binding differs")
    archive_path = _paper_file(paper, archive["path"])
    if (
        archive_path.stat().st_size != archive["bytes"]
        or hashlib.sha256(archive_path.read_bytes()).hexdigest() != archive["sha256"]
    ):
        raise ValueError("frozen documentary archive bytes differ")

    frozen: dict[str, tuple[int, str]] = {}
    with zipfile.ZipFile(archive_path) as bundle:
        if len(bundle.infolist()) != archive["members"]:
            raise ValueError("frozen documentary archive member count differs")
        for record in inputs["files"]:
            relative = record["path"]
            if relative not in FROZEN_EDITION_DOCUMENTS:
                continue
            if record["member"] != "Empirical_Lawhood_v0.60/" + relative.removeprefix("editions/v0.60/"):
                raise ValueError(f"frozen documentary member mapping differs: {relative}")
            member = bundle.getinfo(record["member"])
            if (
                member.file_size != record["bytes"]
                or hashlib.sha256(bundle.read(member)).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"frozen documentary archive member differs: {relative}")
            name = "paper/" + relative
            if name in frozen:
                raise ValueError(f"duplicate frozen documentary path: {relative}")
            frozen[name] = (record["bytes"], record["sha256"])
    if set(frozen) != {"paper/" + name for name in FROZEN_EDITION_DOCUMENTS}:
        raise ValueError("frozen documentary edition document roster differs")
    for name, (expected_bytes, expected_sha256) in frozen.items():
        raw = _paper_file(paper, name.removeprefix("paper/")).read_bytes()
        if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError(f"{name}: frozen documentary source bytes differ")
    return frozen


def frozen_scientific_link_targets(root: Path) -> dict[str, frozenset[str]]:
    """Retain only authenticated frozen citations without changing scientific identities."""
    raw = _paper_file(root, FINITE_SCIENCE_DOCUMENT).read_bytes()
    if len(raw) != 84916 or hashlib.sha256(raw).hexdigest() != FINITE_SCIENCE_SHA256:
        raise ValueError("finite scientific specification bytes differ from frozen plan")
    bindings = {
        "src/empirical_lawhood/adapters/methods/preparation_applicability/frozen_specification.md":
            (7334, "be628f6d7869e2c8388fa13ae1ee4ab2784eda4471ee4580c5c5e3c78e8da66d", (
                "../../scripts/analyze_cc1_applicability_opportunity_v1.py",
                "../guides/scientific-experiment-design-standard.md",
                "../results/cc1-applicability-production-opportunity-v1-2026-10-01.md",
                "cc1-boundary-production-v1.md",
                "../results/cc1-boundary-production-v1-2026-10-01.md",
            )),
        "src/empirical_lawhood/adapters/methods/constructed_preparation_applicability/frozen_specification.md":
            (12145, "1b50614ffe258a3c0a5e8a7625ef8c820d1a328af439c0612b484fc1e78b3361", (
                "../reference/project-owner-authority.md",
                "../results/cc1-applicability-production-opportunity-v1-2026-10-01.md",
            )),
    }
    targets = {FINITE_SCIENCE_DOCUMENT: frozenset((FINITE_SCIENCE_LEGACY_TARGET,))}
    for relative, (size, checksum, historical_targets) in bindings.items():
        raw = _paper_file(root, relative).read_bytes()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError(f"frozen scientific provenance bytes differ: {relative}")
        targets[relative] = frozenset(historical_targets)
    return targets


class _HTMLTargets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"src", "href"} and value is not None:
                self.targets.append(value)


def markdown_anchors(content: str) -> frozenset[str]:
    """Collect ordinary GitHub heading slugs and explicit documentary IDs.

    Fenced examples do not declare headings. Duplicate headings receive the
    same numeric suffixes as the public renderer. Pandoc/HTML IDs also support
    the supplied documentary targets without rewriting their source.
    """
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    fence: str | None = None
    previous = ""
    for line in content.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            previous = ""
            continue
        if fence is not None:
            continue
        anchors.update(re.findall(r'\b(?:id|name)=["\']([^"\']+)["\']', line))
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        text = heading[1] if heading else None
        if text is None and previous.strip() and re.fullmatch(r"\s{0,3}(?:=+|-+)\s*", line):
            text = previous.strip()
        if text is not None:
            explicit = re.search(r"\{#([^}\s]+)[^}]*\}\s*$", text)
            if explicit:
                anchors.add(explicit[1])
                text = text[:explicit.start()]
            text = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", text)
            text = unescape(re.sub(r"<[^>]*>", "", text)).lower().strip()
            slug = re.sub(r"[^\w\- ]", "", text).replace(" ", "-")
            count = counts.get(slug, 0)
            candidate = slug if count == 0 else f"{slug}-{count}"
            while candidate in anchors:
                count += 1
                candidate = f"{slug}-{count}"
            counts[slug] = count + 1
            anchors.add(candidate)
        previous = line
    return frozenset(anchors)


def local_anchor_errors(root: Path, name: str, content: str) -> list[str]:
    """Check local Markdown fragments; external/PDF viewer fragments are excluded."""
    errors = []
    html = _HTMLTargets()
    html.feed(content)
    for raw in re.findall(r"\]\(([^)]+)\)", content) + html.targets:
        target = raw.strip().strip("<>")
        parts = urlsplit(target)
        if parts.scheme or not parts.fragment:
            continue
        selected = unquote(parts.path)
        path = (root / name).parent / selected if selected else root / name
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            continue  # Ordinary shipped-target checks diagnose containment.
        if path.suffix != ".md" or not path.is_file():
            continue
        anchor = unquote(parts.fragment)
        if anchor not in markdown_anchors(path.read_text(encoding="utf-8")):
            errors.append(f"{name}: missing local Markdown anchor {target}")
    return errors


def current_paper_citation_errors(
    root: Path, name: str, content: str, metadata: dict
) -> list[str]:
    """Check explicit citations to current manuscript aliases against their owner.

    Generic reader labels are allowed. An author/year prefix and explicit edition
    identify bibliography-style titles; unrelated history and companion documents
    keep their own identities. Authenticated original publication text is skipped
    by the caller, as it is for ordinary target checking.
    """
    edition, title = metadata["edition"], metadata["title"]
    manuscripts = {"paper/manuscript.md", "paper/manuscript.pdf"}
    manuscripts.update(
        f"paper/editions/v{edition}/Empirical_Lawhood_Manuscript_v{edition}.{suffix}"
        for suffix in ("md", "pdf")
    )
    manuscripts.add(
        f"paper/editions/v{edition}-deposit/Empirical_Lawhood_Manuscript_v{edition}.pdf"
    )
    errors = []
    for line in content.splitlines():
        for link in re.finditer(r"\[([^\]\n]+)\]\(([^)\n]+)\)", line):
            label, raw = link.groups()
            target = raw.strip().strip("<>")
            if target.startswith(("#", "mailto:")) or urlsplit(target).scheme:
                continue
            try:
                relative = (
                    (root / name).parent / unquote(urlsplit(target).path)
                ).resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue  # Ordinary target checking reports paths outside the tree.
            if relative not in manuscripts:
                continue
            cited = re.match(
                r"\s*,?\s*edition\s+\*{0,2}(\d+(?:\.\d+)+)", line[link.end():]
            )
            if cited is None:
                continue
            if cited[1] != edition:
                errors.append(
                    f"{name}: current paper edition {cited[1]} differs from metadata {edition}"
                )
            if re.search(r"\(\d{4}\),\s*$", line[:link.start()]):
                cited_title = label.strip("*_` ")
                if cited_title != title:
                    errors.append(
                        f"{name}: current paper title {cited_title!r} differs from metadata {title!r}"
                    )
    return errors


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    inventory = set(
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
        ).split("\0")
    ) - {""}
    names = sorted(name for name in inventory if name.endswith(".md"))
    errors = []
    frozen_sources = 0
    frozen_editions = frozen_edition_documents(root)
    frozen_scientific_links = frozen_scientific_link_targets(root)
    paper_metadata = json.loads((root / "paper/metadata.json").read_text(encoding="utf-8"))
    for name in names:
        path = root / name
        if name in frozen_editions:
            expected_bytes, expected_sha256 = frozen_editions[name]
            raw = _paper_file(root / "paper", name.removeprefix("paper/")).read_bytes()
            if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
                errors.append(f"{name}: frozen documentary source bytes differ")
            frozen_sources += 1
            continue
        content = path.read_text(encoding="utf-8")
        errors.extend(current_paper_citation_errors(root, name, content, paper_metadata))
        errors.extend(local_anchor_errors(root, name, content))
        html = _HTMLTargets()
        html.feed(content)
        for raw in re.findall(r"\]\(([^)]+)\)", content) + html.targets:
            target = raw.strip().strip("<>")
            if target.startswith(("#", "mailto:")) or urlsplit(target).scheme:
                continue
            selected = unquote(target.split("#", 1)[0])
            if not selected:
                continue
            if selected in frozen_scientific_links.get(name, ()):
                continue
            target_path = path.parent / selected
            if not target_path.exists():
                errors.append(f"{name}: missing local target {selected}")
                continue
            try:
                relative = target_path.resolve().relative_to(root).as_posix()
            except ValueError:
                errors.append(f"{name}: local target outside shipped tree {selected}")
                continue
            if (
                relative != "."
                and relative not in inventory
                and not any(item.startswith(relative + "/") for item in inventory)
            ):
                errors.append(f"{name}: local target is not shipped {selected}")
        if name.startswith("experiments/") and name.endswith("/guide.md"):
            if re.search(r"^empirical-lawhood\b", content, re.MULTILINE):
                errors.append(
                    f"{name}: bare CLI; use uv run --no-sync from the selected checkout"
                )
    if errors:
        raise SystemExit("\n".join(errors))
    guides = sum(
        name.startswith("experiments/") and name.endswith("/guide.md") for name in names
    )
    print(
        f"checked shipped Markdown/HTML targets in {len(names)} source Markdown files "
        "and current manuscript citation metadata; "
        f"and unactivated-shell rules in {guides} experiment guides; "
        f"{frozen_sources} authenticated historical documentary sources retain their original local links; "
        "authenticated frozen scientific historical targets retained; "
        "maintained local Markdown anchors checked; external URLs and PDF viewer fragments are not checked"
    )


if __name__ == "__main__":
    main()
