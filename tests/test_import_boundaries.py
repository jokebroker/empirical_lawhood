"""Keep production dependencies inside the documented source layers.

SPDX-License-Identifier: MPL-2.0
"""

import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src" / "empirical_lawhood"
LAYERS = {
    "kernel": 0,
    "planning": 1,
    "runtime": 2,
    "adapters": 3,
    "infrastructure": 3,
    "api": 4,
    "examples": 5,
    "cli": 6,
}


def test_source_imports_follow_documented_layers():
    violations = []
    for path in sorted(SOURCE.rglob("*.py")):
        relative = path.relative_to(SOURCE)
        owner = relative.parts[0]
        if owner not in LAYERS:
            continue
        package = ("empirical_lawhood", *relative.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(package[: len(package) - node.level + 1]) + (
                        "." + base if base else ""
                    )
                modules = [base, *(base + "." + alias.name for alias in node.names)]
            for module in modules:
                parts = module.split(".")
                if len(parts) < 2 or parts[0] != "empirical_lawhood":
                    continue
                target = parts[1]
                if target not in LAYERS:
                    continue
                # The installed CLI's demo entry is the sole production consumer
                # of examples. Scientific and application code use native owners.
                allowed_demo = (
                    target == "examples" and relative.as_posix() == "cli/app.py"
                )
                imports_example = target == "examples" and owner != "examples"
                if not allowed_demo and (
                    LAYERS[target] > LAYERS[owner] or imports_example
                ):
                    violations.append(f"{relative}:{node.lineno} -> {module}")
    assert not violations, "Outward source imports:\n" + "\n".join(
        sorted(set(violations))
    )
