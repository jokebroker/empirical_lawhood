# SPDX-License-Identifier: MPL-2.0
"""Run the supported portable suite once on an unchanged staged candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

# These scripts are also importable by focused tests from the repository root.
if __package__:
    from .release_check import seal_log_snapshot
    from .release_portable_evidence import SCHEMA, candidate_inventory, inventory_digest
else:
    from release_check import seal_log_snapshot
    from release_portable_evidence import SCHEMA, candidate_inventory, inventory_digest

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    packet = args.output_dir.absolute()
    if (".." in packet.parts or packet.exists() or packet.is_relative_to(ROOT)
            or any(part.is_symlink() for part in (packet, *packet.parents))):
        raise SystemExit("Select one new ordinary evidence directory outside the checkout")
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode or subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT):
        raise SystemExit("Stage all intended files and settle the loaded candidate before its suite")
    inventory = candidate_inventory(ROOT)
    if any(Path(row["path"]).name.startswith("FINAL") and row["path"].endswith(".md")
           or Path(row["path"]).name == "LOWFRICTION.md" for row in inventory):
        raise SystemExit("Working plans must remain outside the release candidate")
    tree = subprocess.check_output(["git", "write-tree"], cwd=ROOT, text=True).strip()
    packet.mkdir(parents=True)
    (packet / "logs").mkdir()
    inventory_path = packet / "candidate-inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    record = {"schema": SCHEMA, "version": "1.0.0", "status": "RUNNING", "candidate_tree": tree,
              "attribution": "Precommit staged candidate; this is not a clean-commit test execution.",
              "parent_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "inventory_before_sha256": inventory_digest(inventory), "inventory_after_sha256": None,
              "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
              "full_suite_invocations": 0, "checks": [], "candidate_inventory": {
                  "path": inventory_path.name, "size_bytes": inventory_path.stat().st_size,
                  "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest()},
              "coverage": None, "scientific_qualification": "NOT_PERFORMED",
              "test_environment_policy": {"pytest_addopts": None,
                  "coverage_rcfile_sha256": hashlib.sha256((ROOT / ".coveragerc").read_bytes()).hexdigest()}}
    environment = dict(os.environ, COVERAGE_FILE=str(packet / ".coverage"),
                       COVERAGE_RCFILE=str(ROOT / ".coveragerc"), PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTEST_ADDOPTS", None)
    # Plugins can import numerical libraries before pytest_sessionstart runs.
    # Set the supported profile before launching any selected interpreter.
    environment.update({name: "1" for name in THREAD_ENVIRONMENT})
    python = ["uv", "run", "--no-sync", "python"]

    def persist():
        temporary = packet / "precommit-portable-evidence.pending.json"
        with temporary.open("w") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(packet / "precommit-portable-evidence.json")

    persist()

    def run(name, command):
        print(f"candidate check: {name}", flush=True)
        began = time.monotonic()
        live = packet / "logs" / (name + ".live.log")
        sealed = packet / "logs" / (name + ".log")
        check = {"name": name, "command": command, "exit_code": None, "disposition": "RUNNING"}
        record["checks"].append(check)
        persist()
        try:
            with live.open("xb") as stream:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            check.update(exit_code=result.returncode, disposition="PASSED" if result.returncode == 0 else "FAILED")
        except BaseException:
            check["disposition"] = "INTERRUPTED"
            raise
        finally:
            if live.is_file():
                size, checksum = seal_log_snapshot(live, sealed)
                check.update(log=sealed.relative_to(packet).as_posix(), size_bytes=size, sha256=checksum,
                             seconds=round(time.monotonic() - began, 3))
            persist()
        if check["exit_code"] != 0:
            raise RuntimeError(f"{name} failed; preserve this packet and use focused repair checks")

    try:
        run("environment", [*python, "-c", "import importlib.metadata as m,json,os,platform,sys; "
            "from pathlib import Path; import empirical_lawhood; "
            "from empirical_lawhood.infrastructure.source_origin import require_executing_target_source; "
            "require_executing_target_source(Path.cwd()); "
            "assert empirical_lawhood.__version__ == m.version('empirical-lawhood'); "
            "assert platform.python_version() == '3.11.14'; "
            "print(json.dumps({'python':platform.python_version(),'executable':sys.executable,"
            "'package_origin':empirical_lawhood.__file__,'thread_environment':{k:os.environ[k] for k in "
            f"{THREAD_ENVIRONMENT!r}"
            "},'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())},indent=2))"])
        run("coverage-configuration", [*python, "scripts/check_coverage_inventory.py", "--checkout", str(ROOT)])
        record["full_suite_invocations"] = 1
        run("tests", [*python, "-m", "coverage", "run", "-m", "pytest", "-q", "-ra", "--fail-on-skip",
            "-o", f"cache_dir={packet / 'pytest-cache'}", "-o", "tmp_path_retention_policy=failed",
            "-o", "tmp_path_retention_count=1", "tests", "-m", "not native and not held"])
        run("critical-coverage", [*python, "-m", "coverage", "report"])
        run("critical-coverage-json", [*python, "-m", "coverage", "json", "-o", str(packet / "coverage.json")])
        run("critical-coverage-inventory", [*python, "scripts/check_coverage_inventory.py", "--checkout", str(ROOT),
            "--coverage-json", str(packet / "coverage.json")])
        coverage = packet / "coverage.json"
        record["coverage"] = {"path": coverage.name, "size_bytes": coverage.stat().st_size,
                              "sha256": hashlib.sha256(coverage.read_bytes()).hexdigest()}
        after = inventory_digest(candidate_inventory(ROOT))
        record["inventory_after_sha256"] = after
        if after != record["inventory_before_sha256"] or subprocess.check_output(["git", "write-tree"], cwd=ROOT, text=True).strip() != tree:
            raise RuntimeError("Staged or loaded source changed during its suite")
        record["status"] = "PASSED"
    except BaseException as error:
        record.update(status="INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED",
                      failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
    print(f"candidate suite passed; preserve exact tree before committing: {tree}", flush=True)


if __name__ == "__main__":
    main()
