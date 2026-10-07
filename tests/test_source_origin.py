# SPDX-License-Identifier: MPL-2.0
"""Exercise actual imports and clean-source inspection in isolated Git checkouts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = "\nimport importlib.util\nfrom pathlib import Path\nimport sys\nfrom empirical_lawhood.infrastructure.study_issue import GitStudySourceClosureInspector\nfrom empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind\nroot, commit, tree, extra = sys.argv[1:]\nif extra:\n    spec = importlib.util.spec_from_file_location('empirical_lawhood.origin_probe', extra)\n    module = importlib.util.module_from_spec(spec)\n    sys.modules[spec.name] = module\n    spec.loader.exec_module(module)\nexpected = ImplementationSourceClosure(\n    source_closure_id='test.source-closure', kind=SourceClosureKind.CLEAN_GIT_COMMIT,\n    implementation_commit=commit, implementation_sha256='5' * 64,\n    source_tree_sha256=tree, clean_worktree=True,\n)\ntry:\n    GitStudySourceClosureInspector(Path(root)).observe(expected)\nexcept PermissionError as error:\n    print(str(error))\n    sys.exit(17)\nprint('verified')\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def build_source_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "selected"
    shutil.copytree(ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Synthetic source conformance")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "Isolated current source fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = hashlib.sha256(subprocess.check_output([
        "git", "-C", str(repo), "ls-tree", "-r", "-z", "--full-tree", "HEAD",
    ])).hexdigest()
    return repo, commit, tree


@pytest.fixture
def source_repo(tmp_path: Path) -> tuple[Path, str, str]:
    return build_source_repo(tmp_path)


def _probe(repo: Path, commit: str, tree: str, *, import_path: Path | None = None,
           extra: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ, PYTHONPATH=str(import_path or repo / "src"),
                       PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-c", PROBE, str(repo), commit, tree, "" if extra is None else str(extra)],
        cwd=repo.parent, env=environment, capture_output=True, text=True, timeout=30,
    )


def test_clean_source_and_same_commit_worktree_are_accepted(source_repo, tmp_path):
    repo, commit, tree = source_repo
    result = _probe(repo, commit, tree)
    assert result.returncode == 0, result.stdout + result.stderr
    worktree = tmp_path / "worktree"
    _git(repo, "worktree", "add", "--quiet", "--detach", str(worktree), commit)
    result = _probe(worktree, commit, tree)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mismatch", ("tracked-dirty", "untracked", "head", "tree"))
def test_joint_boundary_reobserves_each_source_identity(source_repo, mismatch):
    repo, commit, tree = source_repo
    expected = "dirty"
    if mismatch == "tracked-dirty":
        with (repo / "src/empirical_lawhood/__init__.py").open("a") as stream:
            stream.write("\n# synthetic tracked edit\n")
    elif mismatch == "untracked":
        (repo / "untracked.txt").write_text("synthetic\n")
    elif mismatch == "head":
        _git(repo, "commit", "--quiet", "--allow-empty", "-m", "Changed source identity")
        expected = "commit changed"
    else:
        tree = "f" * 64
        expected = "source tree changed"
    result = _probe(repo, commit, tree)
    assert result.returncode == 17, result.stdout + result.stderr
    assert expected in result.stdout


@pytest.mark.parametrize("origin", ("wheel-layout", "other-checkout", "mixed", "untracked-module"))
def test_inspected_git_cannot_launder_a_different_executing_package(source_repo, tmp_path, origin):
    repo, commit, tree = source_repo
    import_path, extra = None, None
    if origin == "wheel-layout":
        import_path = tmp_path / "site-packages"
        shutil.copytree(repo / "src/empirical_lawhood", import_path / "empirical_lawhood")
    elif origin == "other-checkout":
        worktree = tmp_path / "other-checkout"
        _git(repo, "worktree", "add", "--quiet", "--detach", str(worktree), commit)
        import_path = worktree / "src"
    else:
        extra = (repo / "src/empirical_lawhood" if origin == "untracked-module" else tmp_path) / "origin_probe.py"
        extra.write_text("VALUE = 1\n", encoding="utf-8")
    result = _probe(repo, commit, tree, import_path=import_path, extra=extra)
    assert result.returncode == 17, result.stdout + result.stderr
    assert any(message in result.stdout for message in (
        "not the inspected target source tree", "outside the inspected target source tree", "not tracked",
        "executing source origin cannot be verified",
    )), result.stdout
