'Injected excluded solver control material-solver executor and typed QE output extraction.'

from __future__ import annotations

from empirical_lawhood._required_inputs import required_external_path

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
import os
import re
import subprocess
import time
from typing import Final, Protocol

from .contracts import ExcludedSolverControlConfig, FIXED_EXECUTOR_IMPLEMENTATION_ID, SolverRunObservation
from .source import DOCKER_EXECUTABLE


PROFILE_ID: Final = 'ambient-pressure-superconductor-qe76-epw61-local-ext4'


@dataclass(frozen=True, slots=True)
class SolverExecutionRequest:
    request_id: str
    profile_id: str
    config_sha256: str
    environment_image_sha256: str
    qe_binary_sha256: str
    input_sha256: str
    pseudopotential_sha256: str
    wall_time_seconds: int
    cpu_cores: int


@dataclass(frozen=True, slots=True)
class SolverExecutionResult:
    executor_implementation_id: str
    exit_code: int
    stdout: bytes
    stderr: bytes
    wall_time_seconds: Decimal
    peak_scratch_bytes: int
    timed_out: bool = False


class MaterialSolverExecutor(Protocol):
    """One narrow fixed-profile solver port; implementations receive no argv."""

    profile_id: str

    def execute(self, request: SolverExecutionRequest) -> SolverExecutionResult: ...


class SyntheticMaterialSolverExecutor:
    'Deterministic synthetic solver extractor fixture fixture that exercises the real typed extractor.'

    profile_id = PROFILE_ID
    executor_implementation_id = 'executor.ambient-pressure-superconductor-synthetic-fixture'

    def __init__(self, *, malformed: bool = False, exit_code: int = 0) -> None:
        self.malformed = malformed
        self.exit_code = exit_code
        self.execution_count = 0

    def execute(self, request: SolverExecutionRequest) -> SolverExecutionResult:
        if request.profile_id != self.profile_id:
            raise ValueError("synthetic executor profile binding differs")
        self.execution_count += 1
        stdout = (
            b"not a Quantum ESPRESSO output\n"
            if self.malformed
            else b"""Program PWSCF v.7.6 starts
     number of atoms/cell      =  1
     number of electrons       = 14.00
     iteration #  1
     iteration #  2
!    total energy              =   -95.1298432100 Ry
     estimated scf accuracy    <       1.0E-13 Ry
     convergence has been achieved in   2 iterations
     PWSCF        :   0h 0m 1.25s CPU    0h 0m 1.30s WALL
     JOB DONE.
"""
        )
        return SolverExecutionResult(
            executor_implementation_id=self.executor_implementation_id,
            exit_code=self.exit_code,
            stdout=stdout,
            stderr=b"",
            wall_time_seconds=Decimal("1.30"),
            peak_scratch_bytes=len(stdout),
        )


def _bounded_file_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    if not path.is_file() or path.is_symlink():
        return b""
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError("QE diagnostic output exceeds its fixed extraction bound")
    return path.read_bytes()


def _workspace_bytes(workspace: Path) -> int:
    total = 0
    for root, directories, filenames in os.walk(workspace, followlinks=False):
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for filename in filenames:
            path = Path(root) / filename
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total


class FixedQEExecutor:
    """Qualified local QE profile using a fixed digest-bound image and no network.

    The privileged helper mounts one prebuilt ext4 environment because the
    authoritative external volume is VFAT and cannot represent the symlinks
    required by a QE build.  The command template is implementation-owned;
    only a registered profile key is selectable from scientific config.
    """

    profile_id = PROFILE_ID
    executor_implementation_id = FIXED_EXECUTOR_IMPLEMENTATION_ID

    _SCRIPT: Final = r"""
set -euo pipefail
run_identity="$1"
expected_binary_sha="$2"
expected_input_sha="$3"
expected_pseudo_sha="$4"
cpu_cores="$5"

mkdir -p /environment
mount -o loop,ro /environment.ext4 /environment
cleanup() { cd /; umount /environment; }
trap cleanup EXIT INT TERM

binary="/environment/qe/bin/pw.x"
control_root="/environment/qe/control/epw-pb-wosoc-scf"
observed_binary_sha="$(sha256sum "$binary" | cut -d ' ' -f 1)"
test "$observed_binary_sha" = "$expected_binary_sha"
observed_input_sha="$(sha256sum "$control_root/scf.in" | cut -d ' ' -f 1)"
test "$observed_input_sha" = "$expected_input_sha"
observed_pseudo_sha="$(sha256sum "$control_root/pb_s.UPF" | cut -d ' ' -f 1)"
test "$observed_pseudo_sha" = "$expected_pseudo_sha"

run_root="/workspace/run-$run_identity"
test ! -e "$run_root"
mkdir -p "$run_root/pseudo" "$run_root/out"
cp "$control_root/scf.in" "$run_root/scf.in"
cp "$control_root/pb_s.UPF" "$run_root/pseudo/pb_s.UPF"
sed -i "s#pseudo_dir = '../../pp/'#pseudo_dir = './pseudo/'#" "$run_root/scf.in"
sed -i "s#outdir='./'#outdir='./out/'#" "$run_root/scf.in"
cd "$run_root"
set +e
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  mpirun -np "$cpu_cores" "$binary" -in scf.in > pw.stdout 2> pw.stderr
solver_status="$?"
set -e
cp pw.stdout /workspace/pw.stdout
cp pw.stderr /workspace/pw.stderr
printf '%s\n' "$solver_status" > /workspace/pw.exit-code
exit "$solver_status"
""".strip()

    def __init__(self, *, container_image_digest: str) -> None:
        self.container_image_digest = container_image_digest

    @staticmethod
    def _workspace() -> Path:
        workspace = Path.cwd().resolve()
        root = required_external_path("EMPIRICAL_LAWHOOD_AP_SCRATCH_ROOT").resolve()
        if workspace != root and root not in workspace.parents:
            raise ValueError("QE execution workspace is outside task-scoped external scratch")
        if workspace.is_symlink():
            raise ValueError("QE execution workspace cannot be a symlink")
        return workspace

    def execute(self, request: SolverExecutionRequest) -> SolverExecutionResult:
        if request.profile_id != self.profile_id:
            raise ValueError("QE executor profile binding differs")
        if not request.qe_binary_sha256:
            raise ValueError("QE executor requires a frozen binary digest")
        if not request.input_sha256 or not request.pseudopotential_sha256:
            raise ValueError("QE executor requires frozen control-input digests")
        if request.cpu_cores != 2:
            raise ValueError('excluded solver control fixed QE profile requires exactly two MPI ranks')
        environment_image = required_external_path("EMPIRICAL_LAWHOOD_AP_ENVIRONMENT_IMAGE")
        if not environment_image.is_file() or environment_image.is_symlink():
            raise ValueError('qualified excluded solver control solver environment is unavailable')
        started = time.monotonic()
        environment_digest = sha256()
        with environment_image.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                environment_digest.update(block)
        if environment_digest.hexdigest() != request.environment_image_sha256:
            raise ValueError('excluded solver control solver environment image digest differs')
        workspace = self._workspace()
        run_identity = sha256(request.request_id.encode("utf-8")).hexdigest()[:24]
        container_name = f'empirical-lawhood-ambient-pressure-superconductor-{run_identity}'
        command = (
            DOCKER_EXECUTABLE,
            "run",
            "--pull",
            "never",
            "--rm",
            "--name",
            container_name,
            "--privileged",
            "--network",
            "none",
            "--cpus",
            str(request.cpu_cores),
            "--memory",
            "8g",
            "-v",
            f'{environment_image}:/environment.ext4:ro',
            "-v",
            f'{workspace}:/workspace:rw',
            self.container_image_digest,
            "/bin/bash",
            "-c",
            self._SCRIPT,
            'ambient-pressure-superconductor-qe-executor',
            run_identity,
            request.qe_binary_sha256,
            request.input_sha256,
            request.pseudopotential_sha256,
            str(request.cpu_cores),
        )
        remaining_seconds = request.wall_time_seconds - (time.monotonic() - started)
        peak_scratch_bytes = _workspace_bytes(workspace)
        if remaining_seconds <= 0:
            timed_out = True
            exit_code = 124
            launcher_stdout = b""
            launcher_stderr = b"environment verification exhausted the task wall-time budget"
        else:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
            deadline = started + request.wall_time_seconds
            timed_out = False
            while process.poll() is None:
                peak_scratch_bytes = max(
                    peak_scratch_bytes,
                    _workspace_bytes(workspace),
                )
                if time.monotonic() >= deadline:
                    timed_out = True
                    process.kill()
                    break
                time.sleep(0.25)
            launcher_stdout, launcher_stderr = process.communicate()
            if timed_out:
                exit_code = 124
            else:
                exit_code = process.returncode
            peak_scratch_bytes = max(
                peak_scratch_bytes,
                _workspace_bytes(workspace),
            )
        if timed_out:
            subprocess.run(
                (DOCKER_EXECUTABLE, "rm", "--force", container_name),
                check=False,
                capture_output=True,
                timeout=30,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
        elapsed = Decimal(str(time.monotonic() - started))
        solver_stdout = _bounded_file_bytes(workspace / "pw.stdout", maximum_bytes=4 * 1024**2)
        solver_stderr = _bounded_file_bytes(workspace / "pw.stderr", maximum_bytes=4 * 1024**2)
        stdout = solver_stdout or launcher_stdout
        stderr = (
            solver_stderr + (b"\n" if solver_stderr and launcher_stderr else b"") + launcher_stderr
        )
        return SolverExecutionResult(
            executor_implementation_id=self.executor_implementation_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            wall_time_seconds=elapsed,
            peak_scratch_bytes=peak_scratch_bytes,
            timed_out=timed_out,
        )


_ENERGY = re.compile(rb"!\s+total energy\s+=\s+([-+0-9.EeDd]+)\s+Ry")
_ACCURACY = re.compile(rb"estimated scf accuracy\s+<\s+([-+0-9.EeDd]+)\s+Ry")
_ELECTRONS = re.compile(rb"number of electrons\s+=\s+([-+0-9.EeDd]+)")
_ATOMS = re.compile(rb"number of atoms/cell\s+=\s+([0-9]+)")
_CONVERGED = re.compile(rb"convergence has been achieved in\s+([0-9]+)\s+iterations")


def _number(match: re.Match[bytes] | None) -> Decimal | None:
    if match is None:
        return None
    try:
        return Decimal(match.group(1).decode("ascii").replace("D", "E").replace("d", "e"))
    except (UnicodeDecodeError, InvalidOperation) as error:
        raise ValueError("QE emitted a malformed decimal operand") from error


def _last_number(pattern: re.Pattern[bytes], payload: bytes) -> Decimal | None:
    """Return the final iterative QE operand, not its first provisional value."""

    matches = tuple(pattern.finditer(payload))
    return _number(None if not matches else matches[-1])


def extract_qe_observation(
    *, config: ExcludedSolverControlConfig, result: SolverExecutionResult, observation_id: str
) -> SolverRunObservation:
    """Extract only bounded, declared QE operands with their native units."""

    if len(result.stdout) > 4 * 1024**2 or len(result.stderr) > 4 * 1024**2:
        raise ValueError('QE output exceeds the excluded solver control typed extractor bound')
    convergence = _CONVERGED.search(result.stdout)
    atoms = _ATOMS.search(result.stdout)
    reasons: list[str] = []
    if result.timed_out:
        reasons.append("solver-wall-time-exceeded")
    if result.exit_code != 0:
        reasons.append("solver-exit-nonzero")
    if b"JOB DONE." not in result.stdout:
        reasons.append("qe-job-done-marker-missing")
    if convergence is None:
        reasons.append("scf-convergence-marker-missing")
    energy = _number(_ENERGY.search(result.stdout))
    accuracy = _last_number(_ACCURACY, result.stdout)
    electrons = _number(_ELECTRONS.search(result.stdout))
    if energy is None:
        reasons.append("total-energy-operand-missing")
    if accuracy is None:
        reasons.append("scf-accuracy-operand-missing")
    if electrons is None:
        reasons.append("electron-count-operand-missing")
    if atoms is None:
        reasons.append("atom-count-operand-missing")
    return SolverRunObservation(
        observation_id=observation_id,
        config_sha256=config.payload_sha256,
        solver_profile_id=config.solver_profile_id,
        executor_implementation_id=result.executor_implementation_id,
        qe_binary_sha256=config.qe_binary_sha256,
        control_case_id=config.control_case_id,
        completed=result.exit_code == 0 and b"JOB DONE." in result.stdout,
        converged=convergence is not None,
        exit_code=result.exit_code,
        total_energy_Ry=energy,
        estimated_accuracy_Ry=accuracy,
        electron_count=electrons,
        atom_count=(int(atoms.group(1)) if atoms is not None else None),
        scf_iterations=(int(convergence.group(1)) if convergence is not None else None),
        wall_time_seconds=result.wall_time_seconds,
        peak_scratch_bytes=result.peak_scratch_bytes,
        stdout_sha256=sha256(result.stdout).hexdigest(),
        stderr_sha256=sha256(result.stderr).hexdigest(),
        reason_codes=tuple(sorted(set(reasons))),
    )


__all__ = [
    "FixedQEExecutor",
    "MaterialSolverExecutor",
    "PROFILE_ID",
    "SolverExecutionRequest",
    "SolverExecutionResult",
    "SyntheticMaterialSolverExecutor",
    "extract_qe_observation",
]
