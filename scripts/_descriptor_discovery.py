"""Closed descriptor discovery and safe generated-output paths.

SPDX-License-Identifier: MPL-2.0
"""

from pathlib import Path


def module_name(path: Path, *, package_root: Path) -> str:
    return ".".join(path.relative_to(package_root.parent).with_suffix("").parts)


def require_safe_path(path: Path, *, root: Path, description: str) -> None:
    """Reject links and escapes before resolving hides their ancestry."""

    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} has no allowed ancestor: {path}") from error
    cursor = path
    while True:
        if cursor.is_symlink():
            raise ValueError(f"{description} traverses a symlink: {path}")
        if cursor == root:
            break
        cursor = cursor.parent
    resolved_root = root.resolve(strict=True)
    try:
        path.resolve(strict=True).relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{description} escapes its allowed root: {path}") from error


def discover_descriptors(
    *, package_root: Path, roots: tuple[Path, ...], name: str, kind: str
) -> tuple[Path, ...]:
    values: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"allowlisted {kind} root is absent or unsafe: {root}")
        require_safe_path(root, root=package_root, description=f"{kind} root")
        for path in root.rglob("*"):
            if path.is_symlink() and (path.is_dir() or path.name == name):
                raise ValueError(f"{kind} discovery traverses a symlink: {path}")
            if path.name != name:
                continue
            require_safe_path(path, root=root, description=f"{kind} descriptor")
            if not path.is_file():
                raise ValueError(f"{kind} descriptor is not a regular file: {path}")
            values.append(path)
    if len(set(values)) != len(values):
        raise ValueError(f"{kind} descriptor discovery contains duplicate paths")
    return tuple(sorted(values))


def require_safe_output(path: Path, *, package_root: Path) -> None:
    require_safe_path(
        path.parent, root=package_root, description="generated output parent"
    )
    if path.is_symlink() or path.exists() and not path.is_file():
        raise ValueError(f"generated output is not a regular file: {path}")
