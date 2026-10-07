"""Publication arguments and closed generator paths retain their boundaries.

SPDX-License-Identifier: MPL-2.0
"""

import argparse
import importlib.util
from pathlib import Path

import pytest

from scripts._descriptor_discovery import (
    discover_descriptors,
    module_name,
    require_safe_output,
)


ROOT = Path(__file__).resolve().parents[1]


def _storage():
    spec = importlib.util.spec_from_file_location(
        "paper_storage", ROOT / "paper/source/storage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("supplied", ((), ("--storage-root", "/tmp/root"), ("--storage-mount", "/tmp")))
def test_publication_requires_both_explicit_storage_arguments(supplied):
    parser = argparse.ArgumentParser()
    _storage().add_storage_arguments(parser)
    with pytest.raises(SystemExit) as failure:
        parser.parse_args(supplied)
    assert failure.value.code == 2


def test_publication_preserves_mounted_containment_and_checkout_refusal(tmp_path, monkeypatch):
    storage = _storage()
    parser = argparse.ArgumentParser()
    storage.add_storage_arguments(parser)
    mount = tmp_path / "mount"
    root = mount / "selected-root"
    args = parser.parse_args(["--storage-root", str(root), "--storage-mount", str(mount)])
    monkeypatch.setattr(Path, "is_mount", lambda path: path == mount)
    output = root / "new-output"
    assert storage.external_directory(output, args, parser) == output
    for refused in (root, mount / "another-root/output", ROOT / "paper/outputs"):
        with pytest.raises(SystemExit) as failure:
            storage.external_directory(refused, args, parser)
        assert failure.value.code == 2
    monkeypatch.setattr(Path, "is_mount", lambda path: False)
    with pytest.raises(SystemExit):
        storage.external_directory(output, args, parser)


def _package(tmp_path):
    package = tmp_path / "src/package"
    root = package / "adapters/methods"
    root.mkdir(parents=True)
    return package, root


def test_descriptor_discovery_is_closed_sorted_and_module_names_unchanged(tmp_path):
    package, root = _package(tmp_path)
    first = root / "alpha/extension_bundle.py"
    last = root / "zeta/extension_bundle.py"
    for path in (last, first):
        path.parent.mkdir()
        path.write_text("contribution = None\n")
    other = package / "outside/extension_bundle.py"
    other.parent.mkdir()
    other.write_text("ignored\n")
    assert discover_descriptors(package_root=package, roots=(root,), name="extension_bundle.py", kind="extension") == (first, last)
    assert module_name(first, package_root=package) == "package.adapters.methods.alpha.extension_bundle"
    with pytest.raises(ValueError, match="absent or unsafe"):
        discover_descriptors(package_root=package, roots=(root / "absent",), name="extension_bundle.py", kind="extension")


@pytest.mark.parametrize("linked", ("root", "ancestor", "descriptor"))
def test_both_descriptor_kinds_refuse_links(tmp_path, linked):
    package, root = _package(tmp_path)
    real = package / "real"
    real.mkdir()
    for name in ("extension_bundle.py", "executable_binding.py"):
        (real / name).write_text("descriptor\n")
    if linked == "root":
        root.rmdir()
        root.symlink_to(real, target_is_directory=True)
    elif linked == "ancestor":
        (root / "linked").symlink_to(real, target_is_directory=True)
    else:
        for name in ("extension_bundle.py", "executable_binding.py"):
            (root / name).symlink_to(real / name)
    for kind, name in (("extension", "extension_bundle.py"), ("executable", "executable_binding.py")):
        with pytest.raises(ValueError, match="symlink"):
            discover_descriptors(package_root=package, roots=(root,), name=name, kind=kind)


def test_generated_output_refuses_linked_member_and_parent(tmp_path):
    package, root = _package(tmp_path)
    output = root / "generated.py"
    require_safe_output(output, package_root=package)
    real = package / "real"
    real.mkdir()
    real_file = real / "generated.py"
    real_file.write_text("preserved\n")
    output.symlink_to(real_file)
    with pytest.raises(ValueError, match="not a regular file"):
        require_safe_output(output, package_root=package)
    linked = package / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        require_safe_output(linked / "generated.py", package_root=package)
    assert real_file.read_text() == "preserved\n"
