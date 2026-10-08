# SPDX-License-Identifier: MPL-2.0
"""Check an exact clean commit in a fresh checkout and retain a reviewable packet."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "portable": ((), ["tests", "-m", "not native and not held"]),
    "native-open": (("open-simulators", 'reaction-response-simulators'),
                    ["tests", "-m", "native", "--ignore=tests/test_brian2_native_science.py"]),
    "native-brian2": ((), ["tests/test_brian2_native_science.py", "-m", "native"]),
    "held-reactor": ((), ["tests/test_reactor_batch_input.py", "tests/test_reactor_prepared_input.py",
                          "tests/test_reactor_held_native_science.py", "tests/test_reactor_upstream_binding.py", "-m", "held"]),
    "held-response": ((), ['tests/test_prepared_response_held_source.py', 'tests/test_information_prediction_held_source.py',
                      'tests/test_causal_contrasts_held_source.py', 'tests/test_finite_response_assignment.py',
                      'tests/test_finite_response_stage_input.py', "-m", "held"]),
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def seal_log_snapshot(live: Path, snapshot: Path) -> tuple[int, str]:
    """Bind a new inode to at most the bytes observed when capture stopped.

    A descendant can retain the live inode after subprocess.run returns or is interrupted.
    Renaming that file would not freeze the evidence recorded in the manifest.
    """
    checksum = hashlib.sha256()
    captured = 0
    with live.open("rb") as source, snapshot.open("xb") as destination:
        remaining = os.fstat(source.fileno()).st_size
        while remaining:
            chunk = source.read(min(remaining, 1024 * 1024))
            if not chunk:
                break
            destination.write(chunk)
            checksum.update(chunk)
            captured += len(chunk)
            remaining -= len(chunk)
    return captured, checksum.hexdigest()


def output(command: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def source_identity(root: Path) -> dict[str, object]:
    if output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root):
        raise SystemExit("Release gate requires a clean committed source tree; commit reviewed changes first.")
    commit = output(["git", "rev-parse", "HEAD"], cwd=root)
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", "--full-tree", "HEAD"], cwd=root)
    return {
        "commit": commit, "tree_inventory_sha256": digest(tree), "clean": True,
        "tags_at_commit": output(["git", "tag", "--points-at", commit], cwd=root).splitlines(),
        "uv_lock_sha256": digest((root / "uv.lock").read_bytes()),
    }


def compare_wheel_source(checkout: Path, wheel: Path) -> list[dict[str, object]]:
    """Check every source package byte, including vendor pins, protocols and migrations."""
    names = output(["git", "ls-files", "--", "src/empirical_lawhood"], cwd=checkout).splitlines()
    expected = {name.removeprefix("src/"): (checkout / name).read_bytes() for name in names}
    with zipfile.ZipFile(wheel) as archive:
        observed = {name: archive.read(name) for name in archive.namelist()
                    if name.startswith("empirical_lawhood/") and not name.endswith("/")}
    if observed != expected:
        added = sorted(observed.keys() - expected.keys())
        missing = sorted(expected.keys() - observed.keys())
        changed = sorted(name for name in observed.keys() & expected.keys() if observed[name] != expected[name])
        raise RuntimeError(f"wheel/source mismatch: added={added}, missing={missing}, changed={changed}")
    return [{"path": name, "size_bytes": len(payload), "sha256": digest(payload)}
            for name, payload in sorted(observed.items())]


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Use the selected Python 3.11 environment: uv run --no-sync python scripts/release_check.py ...")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New packet directory outside the checkout.")
    parser.add_argument("--environment-root", type=Path,
                        help="New directory for fresh environments and temporary files on a POSIX filesystem.")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="portable")
    parser.add_argument("--offline", action="store_true", help="Require already cached Python/build/dependency downloads.")
    parser.add_argument("--portable-evidence", type=Path,
                        help="Reuse one authenticated precommit portable suite with the exact final tree; never rerun it.")
    args = parser.parse_args()
    prior_portable = None
    if args.portable_evidence is not None:
        if args.profile != "portable":
            raise SystemExit("Precommit portable evidence is supported only for the portable artifact gate")
        if __package__:
            from .release_portable_evidence import authenticate_portable_evidence
        else:
            from release_portable_evidence import authenticate_portable_evidence
        prior_portable = authenticate_portable_evidence(args.portable_evidence.absolute(), ROOT)
    source = source_identity(ROOT)
    distribution_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    packet = args.output_dir.expanduser().absolute()
    if (".." in packet.parts or packet.is_relative_to(ROOT) or packet.exists()
            or any(part.is_symlink() for part in (packet, *packet.parents))):
        raise SystemExit("Select a new output directory outside the source checkout.")
    environment_root = args.environment_root.expanduser().absolute() if args.environment_root else None
    if environment_root is not None:
        if (environment_root.is_relative_to(ROOT) or environment_root.is_relative_to(packet)
                or packet.is_relative_to(environment_root)
                or ".." in environment_root.parts or environment_root.exists()
                or any(part.is_symlink() for part in (environment_root, *environment_root.parents))):
            raise SystemExit("Select a new environment root outside the source checkout and packet.")
    # Observe prerequisites before creating a packet that needs terminal records.
    uv_version = output(["uv", "--version"], cwd=ROOT)
    checkout = packet / "checkout"
    project_environment = environment_root / "project" if environment_root else checkout / ".venv"
    wheel_environment = environment_root / "wheel" if environment_root else packet / "wheel-env"
    brian2_environment = (environment_root / "brian2" if environment_root else
                          checkout / "experiments/neuron-current-response/native-env/.venv")
    logs = packet / "logs"
    manifest = {
        "schema": 'empirical-lawhood/release/check-manifest', "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(), "source": source,
        "profile": args.profile, "uv": uv_version,
        "distribution_version": distribution_version,
        "host": {"os": platform.system(), "architecture": platform.machine()},
        "checks": [], "artifacts": [],
        "environment_paths": {"project": str(project_environment), "wheel": str(wheel_environment),
                              "brian2": str(brian2_environment)},
        "scientific_ceiling": "Software conformance and exposed development checks; no new qualification or admission.",
        "log_evidence": "Bound snapshots contain observed bytes; descendants may continue writing unbound live logs.",
    }
    offline = ["--offline"] if args.offline else []
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for inherited in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTEST_ADDOPTS"):

        environment.pop(inherited, None)

    environment["COVERAGE_RCFILE"] = str(checkout / ".coveragerc")
    environment.update({name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})

    def run(name: str, command: list[str], *, cwd: Path = checkout,
            env_overrides: dict[str, str] | None = None) -> None:
        print(f"release gate: {name}", flush=True)
        began = time.monotonic()
        log = logs / f"{name}.log"
        check = {"name": name, "command": command, "exit_code": None,
                 "disposition": "RUNNING", "log": None, "sha256": None}
        manifest["checks"].append(check)
        try:
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=cwd, env=environment | (env_overrides or {}),
                                        stdout=stream, stderr=subprocess.STDOUT, text=True)
        except BaseException as error:
            check.update(disposition="INTERRUPTED", exception_type=type(error).__name__,
                         diagnostic=str(error), seconds=round(time.monotonic() - began, 3))
            if log.exists():
                check["unbound_live_log"] = log.relative_to(packet).as_posix()
                snapshot = logs / f"{name}.interrupted.log"
                try:
                    captured, checksum = seal_log_snapshot(log, snapshot)
                except OSError as snapshot_error:
                    check["snapshot_failure"] = str(snapshot_error)
                else:
                    check.update(log=snapshot.relative_to(packet).as_posix(),
                                 sha256=checksum, size_bytes=captured)
            raise
        check.update(exit_code=result.returncode,
                     disposition="FAILED" if result.returncode else "PASSED",
                     seconds=round(time.monotonic() - began, 3),
                     unbound_live_log=log.relative_to(packet).as_posix())
        snapshot = logs / f"{name}.sealed.log"
        try:
            captured, checksum = seal_log_snapshot(log, snapshot)
        except BaseException as error:
            check.update(disposition="EVIDENCE_FAILURE", exception_type=type(error).__name__,
                         diagnostic=str(error))
            raise
        check.update(log=snapshot.relative_to(packet).as_posix(), sha256=checksum, size_bytes=captured)
        if result.returncode:
            raise RuntimeError(f"{name} failed (exit {result.returncode}); see {log}")

    try:
        packet.mkdir(parents=True)
        logs.mkdir()
        if environment_root is not None:
            environment_root.mkdir(parents=True)
            temporary_files = environment_root / "temporary-files"
            temporary_files.mkdir()
            environment["TMPDIR"] = str(temporary_files)
            environment["UV_PROJECT_ENVIRONMENT"] = str(project_environment)
            manifest["temporary_files"] = str(temporary_files)
        run("clean-checkout", ["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(checkout)], cwd=packet)
        if source_identity(checkout) != source:
            raise RuntimeError("fresh checkout identity differs from selected source")
        groups, tests = PROFILES[args.profile]
        group_flags = [flag for group in ("reactor-example", "build", *groups) for flag in ("--group", group)]
        run("locked-install", ["uv", "sync", "--locked", "--python", "3.11.14", *offline, *group_flags])
        if args.profile == "native-brian2":
            run("brian2-install", ["uv", "sync", "--locked", "--python", "3.11.14", *offline],
                cwd=checkout / "experiments/neuron-current-response/native-env",
                env_overrides={"UV_PROJECT_ENVIRONMENT": str(brian2_environment)})
            environment["EMPIRICAL_LAWHOOD_BRIAN2_PYTHON"] = str(brian2_environment / "bin/python")
        uv_python = ["uv", "run", "--no-sync", "python"]
        run("environment", [*uv_python, "-c",
            "import importlib.metadata as m,json,platform,sys; from pathlib import Path; "
            "import empirical_lawhood; "
            "from empirical_lawhood.infrastructure.source_origin import require_executing_target_source; "
            "root=Path.cwd().resolve(); require_executing_target_source(root); "
            f"selected={distribution_version!r}; "
            "assert empirical_lawhood.__version__ == m.version('empirical-lawhood') == selected, "
            "'source/module/distribution version mismatch'; "
            "print(json.dumps({'python':platform.python_version(), 'executable':sys.executable, "
            "'package_origin':empirical_lawhood.__file__, 'selected_version':selected, "
            "'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())},indent=2))"])
        if args.profile == "portable":
            run("frozen-rc-numerical-inputs", [*uv_python, "scripts/check_frozen_rc_inputs.py"])
        # Refuse inexpensive source drift before the complete portable suite.
        run("static-errors", ["uv", "run", "--no-sync", "ruff", "check", "src", "tests", "scripts"])
        for generator in ("extension_bundle_aggregate", "executable_binding_aggregate", "cli_reference", "test_fixture", "operator_examples", "workflow_index", "config_schemas", "integration_examples"):
            run(f"generated-{generator}", [*uv_python, f"scripts/generate_{generator}.py", "--check"])
        run("documentation-links", [*uv_python, "scripts/check_documentation.py"])
        if args.profile == "portable" and prior_portable is None:
            environment["COVERAGE_FILE"] = str(packet / ".coverage")
            run("coverage-configuration", [*uv_python, "scripts/check_coverage_inventory.py",
                                           "--checkout", str(checkout)])
            run("tests", [*uv_python, "-m", "coverage", "run", "-m", "pytest",
                          "-q", "-ra", "--fail-on-skip", "-o",
                          f"cache_dir={packet / 'pytest-cache'}", "-o",
                          "tmp_path_retention_policy=failed", "-o",
                          "tmp_path_retention_count=1", *tests])
            run("critical-coverage", [*uv_python, "-m", "coverage", "report"])
            run("critical-coverage-json", [*uv_python, "-m", "coverage", "json",
                                            "-o", str(packet / "coverage.json")])
            run("critical-coverage-inventory", [*uv_python, "scripts/check_coverage_inventory.py",
                                                "--checkout", str(checkout), "--coverage-json",
                                                str(packet / "coverage.json")])
            manifest["coverage"] = {"path": "coverage.json",
                                    "sha256": digest((packet / "coverage.json").read_bytes()),
                                    "scope": ".coveragerc; parent test process only; no repository-wide coverage claim"}
        elif args.profile != "portable":
            run("tests", ["uv", "run", "--no-sync", "pytest", "-q", "-ra", "--fail-on-skip", *tests])
        if prior_portable is not None:
            # Preserve precommit attribution and its original immutable files;
            # this gate only adds a verified exact-tree join to the clean commit.
            import shutil
            retained = packet / "precommit-portable"
            retained.mkdir()
            selected_path = args.portable_evidence.absolute()
            shutil.copyfile(selected_path, retained / selected_path.name)
            bindings = [{"path": item["log"]} for item in prior_portable["checks"]]
            bindings += [prior_portable["coverage"], prior_portable["candidate_inventory"]]
            for binding in bindings:
                destination = retained / binding["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(selected_path.parent / binding["path"], destination)
            manifest["precommit_portable"] = {
                "path": "precommit-portable/" + selected_path.name,
                "sha256": digest(selected_path.read_bytes()),
                "candidate_tree": prior_portable["candidate_tree"],
                "final_commit": source["commit"], "suite_invoked_in_this_gate": False,
                "attribution": "Tests ran before this commit on its exact unchanged staged candidate.",
            }
            run("prior-portable-tree-binding", [*uv_python, "-c",
                "from pathlib import Path; from scripts.release_portable_evidence import authenticate_portable_evidence; "
                f"authenticate_portable_evidence(Path({str(retained / selected_path.name)!r}), Path.cwd()); "
                "print('one passing precommit suite authenticated to this exact tree; no tests executed')"])
        if args.profile == "portable":
            artifacts = packet / "artifacts"
            # With neither --wheel nor --sdist, uv builds the wheel THROUGH the
            # newly built sdist. The backend and all its dependencies are locked.
            run("build", ["uv", "build", "--python", str(project_environment / "bin/python"),
                          "--no-build-isolation", *offline, "--out-dir", str(artifacts)])
            wheels = tuple(artifacts.glob("*.whl"))
            sdists = tuple(artifacts.glob("*.tar.gz"))
            if len(wheels) != 1 or len(sdists) != 1:
                raise RuntimeError("expected one new wheel and one new sdist")
            inventory = compare_wheel_source(checkout, wheels[0])
            (packet / "package-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
            constraints = packet / "wheel-requirements.txt"
            run("wheel-pins", ["uv", "export", "--frozen", "--no-dev", "--group", "reactor-example",
                               "--no-emit-project", "--format", "requirements-txt", "--output-file", str(constraints)])
            wheel_env = wheel_environment
            run("wheel-environment", ["uv", "venv", "--python", "3.11.14", *offline, str(wheel_env)])
            python = wheel_env / "bin/python"
            run("wheel-dependencies", ["uv", "pip", "install", "--python", str(python), *offline,
                                       "--require-hashes", "-r", str(constraints)])
            run("wheel-install", ["uv", "pip", "install", "--python", str(python), *offline, "--no-deps", str(wheels[0])])
            outside = packet / "installed-smoke"
            outside.mkdir()
            run("installed-smoke", [str(python), str(checkout / "scripts/installed_wheel_smoke.py"),
                                     "--checkout", str(checkout), "--output-dir", str(outside)], cwd=outside)
            manifest["artifacts"] = [{"path": path.relative_to(packet).as_posix(),
                                      "size_bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}
                                     for path in (*wheels, *sdists)]
            manifest["packaged_members"] = len(inventory)
            manifest["package_inventory_sha256"] = digest((packet / "package-inventory.json").read_bytes())
        if source_identity(checkout) != source or source_identity(ROOT) != source:
            raise RuntimeError("source identity changed during the gate")
        manifest["status"] = "PASSED"
    except BaseException as error:
        manifest["status"] = "FAILED"
        manifest["failure"] = str(error)
        manifest["failure_type"] = type(error).__name__
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        status = {
            "schema": 'empirical-lawhood/release/status',
            "distribution_version": distribution_version,
            "publication_status": "LOCAL_CANDIDATE_OWNER_REVIEW_PENDING",
            "selected_source": source, "profile": args.profile,
            "software_status": manifest["status"], "host": manifest["host"],
            "finished_utc": manifest["finished_utc"],
            "manifest_sha256": digest(manifest_bytes),
            "artifacts": manifest["artifacts"], "checks": manifest["checks"],
            "scientific_qualification": {
                "status": "NOT_PERFORMED", "identity": None, "verdict": None,
                "ceiling": manifest["scientific_ceiling"],
            },
            "historical_evidence": "Separate original source identities; receipts not relabeled.",
            "paper_primary_evidence": (
                "Separate selected paper preservation/build evidence; original experimental record "
                "availability is documented by paper provenance. This software gate performs no "
                "paper build or experimental reproduction."
            ),
            "packet_access": "Local maintainer review packet; request permitted access through repository support.",
        }
        # mkdir can be interrupted after creating the directory. Finalize any
        # created packet, including a failure before logs or environments exist.
        if packet.is_dir():
            (packet / "release-manifest.json").write_bytes(manifest_bytes)
            (packet / "release-status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(f"release gate passed: {packet / 'release-manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
