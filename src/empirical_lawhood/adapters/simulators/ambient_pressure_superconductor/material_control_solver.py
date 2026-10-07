'Fixed-profile QE/PH/EPW execution and bounded material control output extraction.\n\nThe scientific configuration selects one of a closed set of workflow profile\nidentities.  It never supplies argv, shell, import, path, or callable values.\nEach profile consumes exact source bytes already qualified by ``ap4_source``,\nruns without network access, and emits a deterministic raw-output archive plus\ntyped native-unit operands.  The executor does not adjudicate a material\nclass or authorize a science freeze.\n'

from __future__ import annotations

import os
import re
import subprocess
import tarfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from gzip import GzipFile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Final

from empirical_lawhood._required_inputs import (
    required_external_path,
    required_external_sha256,
)

from .material_control_contracts import ControlClass
from .material_control_raw import encode_raw_archive_hdf5
from .material_control_source import CONTROL_INPUT_ARCHIVE, DOCKER_EXECUTABLE, ENVIRONMENT_IMAGE, TUTORIAL_ARCHIVE, TUTORIAL_ARCHIVE_SHA256, _binary_sha256s, _external_path

MAXIMUM_LOG_BYTES: Final = 32 * 1024**2
MATERIAL_CONTROL_CONTAINER_RUN_LABEL: Final = 'empirical-lawhood.material-control-run-scope'
MATERIAL_CONTROL_CONTAINER_TASK_LABEL: Final = 'empirical-lawhood.material-control-task-scope'


class WorkflowKind(StrEnum):
    POSITIVE = "POSITIVE"
    PHONON_STABILITY = "PHONON_STABILITY"
    SCF_CLASS = "SCF_CLASS"


class WorkflowSource(StrEnum):
    EPW_SUPERCONDUCTING_REFERENCE = "EPW_SUPERCONDUCTING_REFERENCE"
    SSSP_CONTROL = "SSSP_CONTROL"


@dataclass(frozen=True, slots=True)
class MaterialControlWorkflowProfile:
    profile_id: str
    source: WorkflowSource
    structure_id: str
    view_id: str
    control_class: ControlClass
    kind: WorkflowKind
    source_root: str
    prefix: str
    cpu_cores: int
    memory_bytes: int


def _profile(
    profile_id: str,
    *,
    source: WorkflowSource,
    structure_id: str,
    view_id: str,
    control_class: ControlClass,
    kind: WorkflowKind,
    source_root: str,
    prefix: str,
) -> MaterialControlWorkflowProfile:
    return MaterialControlWorkflowProfile(
        profile_id=profile_id,
        source=source,
        structure_id=structure_id,
        view_id=view_id,
        control_class=control_class,
        kind=kind,
        source_root=source_root,
        prefix=prefix,
        cpu_cores=8,
        memory_bytes=24 * 1024**3,
    )


_TUTORIAL_PROFILES: Final = (
    _profile(
        'workflow.material-control-tutorial04-mgb2',
        source=WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE,
        structure_id='structure.calibration-mgb2-alb2',
        view_id="view.epw-tutorial04-pbe-pseudodojo-workflow-reference",
        control_class=ControlClass.POSITIVE_ANISOTROPIC,
        kind=WorkflowKind.POSITIVE,
        source_root="tutorial04",
        prefix="mgb2",
    ),
    _profile(
        'workflow.material-control-tutorial04-pb',
        source=WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE,
        structure_id='structure.calibration-pb-fcc',
        view_id="view.epw-tutorial04-pbe-pseudodojo-workflow-reference",
        control_class=ControlClass.POSITIVE_ISOTROPIC,
        kind=WorkflowKind.POSITIVE,
        source_root="tutorial04",
        prefix="pb",
    ),
)


def _science_profiles() -> tuple[MaterialControlWorkflowProfile, ...]:
    rows = (
        (
            'calibration-c-diamond',
            'structure.calibration-c-diamond',
            ControlClass.INSULATOR,
            WorkflowKind.SCF_CLASS,
            "diamond",
        ),
        (
            'calibration-cu-fcc',
            'structure.calibration-cu-fcc',
            ControlClass.NORMAL_METAL,
            WorkflowKind.SCF_CLASS,
            "cu",
        ),
        (
            'calibration-mgb2-alb2',
            'structure.calibration-mgb2-alb2',
            ControlClass.POSITIVE_ANISOTROPIC,
            WorkflowKind.POSITIVE,
            "mgb2",
        ),
        (
            'calibration-pb-fcc',
            'structure.calibration-pb-fcc',
            ControlClass.POSITIVE_ISOTROPIC,
            WorkflowKind.POSITIVE,
            "pb",
        ),
        (
            'calibration-pb-sc-compressed',
            'structure.calibration-pb-sc-compressed',
            ControlClass.DYNAMICALLY_UNSTABLE,
            WorkflowKind.PHONON_STABILITY,
            "pbsc",
        ),
    )
    result: list[MaterialControlWorkflowProfile] = []
    for view_id in ("view.pbe-efficiency-base", "view.pbe-precision-refined"):
        view_name = view_id.removeprefix("view.")
        for case, structure_id, control_class, kind, prefix in rows:
            result.append(
                _profile(
                    f'workflow.material-control-{view_name}-{case}',
                    source=WorkflowSource.SSSP_CONTROL,
                    structure_id=structure_id,
                    view_id=view_id,
                    control_class=control_class,
                    kind=kind,
                    source_root=f'material-control-control-inputs/{view_name}/{case}',
                    prefix=prefix,
                )
            )
    return tuple(result)


MATERIAL_CONTROL_WORKFLOW_PROFILES: Final = tuple(
    sorted(_TUTORIAL_PROFILES + _science_profiles(), key=lambda value: value.profile_id)
)
_PROFILE_BY_ID: Final = {value.profile_id: value for value in MATERIAL_CONTROL_WORKFLOW_PROFILES}


def resolve_material_control_workflow_profile(profile_id: str) -> MaterialControlWorkflowProfile:
    try:
        return _PROFILE_BY_ID[profile_id]
    except KeyError as error:
        raise ValueError('unknown material control workflow profile') from error


@dataclass(frozen=True, slots=True)
class MaterialControlWorkflowRequest:
    request_id: str
    run_id: str
    task_id: str
    profile_id: str
    wall_time_seconds: int
    output_bytes: int


@dataclass(frozen=True, slots=True)
class MaterialControlWorkflowOperands:
    exit_code: int
    timed_out: bool
    terminal_step_id: str
    wall_time_seconds: Decimal
    peak_scratch_bytes: int
    scf_completed: bool
    scf_converged: bool
    total_energy_Ry: Decimal
    scf_accuracy_Ry: Decimal
    scf_repeat_evaluable: bool
    scf_repeat_residual_Ry: Decimal
    fermi_evaluable: bool
    fermi_energy_eV: Decimal
    band_gap_eV: Decimal
    phonon_evaluable: bool
    minimum_phonon_frequency_cm1: Decimal
    epw_evaluable: bool
    electron_phonon_lambda: Decimal
    tc_estimate_K: Decimal
    gap_min_meV: Decimal
    gap_max_meV: Decimal
    raw_archive_path: Path
    raw_archive_sha256: str
    raw_archive_bytes: int
    raw_hdf5_path: Path
    raw_hdf5_sha256: str
    raw_hdf5_bytes: int
    reason_codes: tuple[str, ...]


class FileTaskOutputSource:
    """Single-use bounded source for the generic streamed-artifact writer."""

    def __init__(self, path: Path) -> None:
        if not path.is_file() or path.is_symlink():
            raise ValueError('material control streamed output is not a regular file')
        self.path = path
        self._closed = False

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        if self._closed or maximum_chunk_bytes <= 0:
            raise ValueError('material control streamed output source is closed or unbounded')
        with self.path.open("rb") as stream:
            while block := stream.read(maximum_chunk_bytes):
                yield block

    def close(self) -> None:
        self._closed = True


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _container_scope_id(value: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,199}", value) is None:
        raise ValueError('material control container scope identity is invalid')
    return sha256(value.encode("utf-8")).hexdigest()


def _assert_no_preexisting_run_container(run_id: str) -> str:
    """Fail closed before a new solver if an earlier same-run container survived."""

    scope_id = _container_scope_id(run_id)
    observed = subprocess.run(
        (
            DOCKER_EXECUTABLE,
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f'label={MATERIAL_CONTROL_CONTAINER_RUN_LABEL}={scope_id}',
        ),
        check=False,
        capture_output=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    if observed.returncode != 0 or observed.stderr:
        raise ValueError('material control same-run container preflight could not be verified')
    container_ids = tuple(value for value in observed.stdout.decode("ascii").splitlines() if value)
    if any(re.fullmatch(r"[0-9a-f]{12,64}", value) is None for value in container_ids):
        raise ValueError('material control same-run container preflight returned an invalid identity')
    if container_ids:
        raise ValueError('material control same-run solver container survived a prior task')
    return scope_id


def _force_remove_container(container_name: str) -> bool:
    removed = subprocess.run(
        (DOCKER_EXECUTABLE, "rm", "--force", container_name),
        check=False,
        capture_output=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    return removed.returncode == 0 and not removed.stderr


def _safe_extract(archive_path: Path, destination: Path, *, required_root: str) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError('material control source archive contains an unsafe member')
            if path.parts[0] != required_root:
                continue
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError('material control source archive contains a non-regular selected member')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('material control selected archive member cannot be read')
            payload = stream.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError('material control selected archive member size differs')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(payload)


def _workspace() -> Path:
    workspace = Path.cwd().resolve()
    root = required_external_path("EMPIRICAL_LAWHOOD_AP_SCRATCH_ROOT").resolve()
    if workspace != root and root not in workspace.parents:
        raise ValueError('material control execution workspace is outside task-scoped external scratch')
    if workspace.is_symlink():
        raise ValueError('material control execution workspace cannot be a symlink')
    return workspace


def cleanup_material_control_workflow_scratch(operands: MaterialControlWorkflowOperands) -> None:
    """Remove one completed generated workflow tree after its HDF stream is copied."""

    workspace = _workspace()
    run_root = operands.raw_archive_path.parent.resolve()
    if (
        run_root.parent != workspace
        or re.fullmatch(r"workflow-[0-9a-f]{24}", run_root.name) is None
        or run_root.is_symlink()
        or operands.raw_hdf5_path.parent.resolve() != run_root
    ):
        raise ValueError('material control scratch cleanup target is outside its exact generated workflow')
    for current, directories, filenames in os.walk(run_root, topdown=False, followlinks=False):
        directory = Path(current)
        for filename in filenames:
            path = directory / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError('material control scratch cleanup encountered a non-regular file')
            path.unlink()
        for name in directories:
            path = directory / name
            if path.is_symlink() or not path.is_dir():
                raise ValueError('material control scratch cleanup encountered a non-directory')
            path.rmdir()
    run_root.rmdir()


def _prune_workflow_to_archive(run_root: Path, archive_path: Path) -> None:
    """Discard non-published intermediates after the selected raw tar is durable."""

    if archive_path.parent != run_root or archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError('material control packaging prune lacks its exact raw archive')
    for current, directories, filenames in os.walk(run_root, topdown=False, followlinks=False):
        directory = Path(current)
        for filename in filenames:
            path = directory / filename
            if path == archive_path:
                continue
            if path.is_symlink():
                path.unlink()
                continue
            if not path.is_file():
                raise ValueError('material control packaging prune encountered a non-regular file')
            path.unlink()
        for name in directories:
            path = directory / name
            if path.is_symlink():
                path.unlink()
                continue
            if not path.is_dir():
                raise ValueError('material control packaging prune encountered a non-directory')
            path.rmdir()


def _verify_science_repeat_inputs(source_root: Path, profile: MaterialControlWorkflowProfile) -> None:
    """Prove that the two SCF executions share exact prepared input bytes."""

    if profile.source is not WorkflowSource.SSSP_CONTROL:
        return
    profile_root = source_root / profile.source_root
    if profile.kind is WorkflowKind.SCF_CLASS:
        paths = (profile_root / "scf/scf.in", profile_root / "scf/scf.in")
    elif profile.kind is WorkflowKind.PHONON_STABILITY:
        paths = (profile_root / "scf/scf.in", profile_root / "phonon/scf.in")
    else:
        paths = (profile_root / "phonon/scf.in", profile_root / "epw/scf.in")
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError('material control exact-repeat SCF input is unavailable')
    if _hash_file(paths[0]) != _hash_file(paths[1]):
        raise ValueError('material control exact-repeat SCF inputs differ')


def _tree_bytes(root: Path) -> int:
    total = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(current) / name).is_symlink()]
        for filename in filenames:
            path = Path(current) / filename
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total


def _tutorial_script(profile: MaterialControlWorkflowProfile) -> str:
    if profile.prefix == "pb":
        return r"""
cd /workspace/source/tutorial04/exercise1/phonon
run pw phonon-scf scf.in
run ph phonon-ph ph.in
printf 'pb\n' | "$python" "$pp" > pp.stdout 2> pp.stderr
run_aux q2r phonon-q2r q2r.in
run_aux matdyn phonon-matdyn matdyn.in
cd /workspace/source/tutorial04/exercise1/epw
run pw epw-scf ../phonon/scf.in
run pw epw-nscf nscf.in
run epw epw-coarse epw1.in
run epw epw-eliashberg epw2.in
""".strip()
    if profile.prefix == "mgb2":
        return r"""
cd /workspace/source/tutorial04/exercise2/phonon
run pw phonon-scf scf.in
run ph phonon-ph ph.in
printf 'mgb2\n' | "$python" "$pp" > pp.stdout 2> pp.stderr
run_aux q2r phonon-q2r q2r.in
run_aux matdyn phonon-matdyn matdyn.in
cd /workspace/source/tutorial04/exercise2/epw1
run pw epw-scf scf.in
run pw epw-nscf nscf.in
run epw epw-fsr epw1.in
""".strip()
    raise ValueError("unsupported tutorial workflow profile")


def _science_script(profile: MaterialControlWorkflowProfile) -> str:
    root = f'/workspace/source/{profile.source_root}'
    if profile.kind is WorkflowKind.SCF_CLASS:
        return f'\ncd {root}/scf\nmkdir -p out\nrun pw scf scf.in\nmv out out-first\nmkdir out\nrun pw scf-repeat scf.in\n'.strip()
    standalone_scf = f'\ncd {root}/scf\nmkdir -p out\nrun pw class-scf scf.in\n'.strip()
    phonon = f"""\ncd {root}/phonon\nmkdir -p out\nrun pw phonon-scf scf.in\nrun ph phonon-ph ph.in\nstage_phonon_tree {profile.prefix}\nprintf '{profile.prefix}\\n' | "$python" "$pp" > pp.stdout 2> pp.stderr\nunstage_phonon_tree {profile.prefix}\nrun_aux q2r phonon-q2r q2r.in\nrun_aux matdyn phonon-matdyn matdyn.in\n""".strip()
    if profile.kind is WorkflowKind.PHONON_STABILITY:
        return standalone_scf + "\n" + phonon
    if profile.kind is WorkflowKind.POSITIVE:
        return (
            phonon
            + f'\n\ncd {root}/epw\nmkdir -p out\nrun pw epw-scf scf.in\nrun pw epw-nscf nscf.in\nrun epw epw-coarse epw1.in\nrun epw epw-eliashberg epw2.in\n'
        ).strip()
    raise ValueError("unsupported SSSP workflow kind")


def _container_script(profile: MaterialControlWorkflowProfile) -> str:
    workflow = (
        _tutorial_script(profile)
        if profile.source is WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE
        else _science_script(profile)
    )
    script = (
        r"""
set -euo pipefail
mkdir -p /environment
mount -o loop,ro /environment.ext4 /environment
cleanup() { cd /; umount /environment; }
trap cleanup EXIT INT TERM

bin=/environment/qe/bin
aux=/environment/qe/source/bin
python=/usr/local/bin/python3
pp=/environment/qe/source/EPW/bin/pp.py
ranks="$1"

test -x "$python"

test "$(sha256sum "$bin/pw.x" | cut -d ' ' -f 1)" = "__PW_SHA256__"
test "$(sha256sum "$bin/ph.x" | cut -d ' ' -f 1)" = "__PH_SHA256__"
test "$(sha256sum "$bin/epw.x" | cut -d ' ' -f 1)" = "__EPW_SHA256__"
test "$(sha256sum "$aux/q2r.x" | cut -d ' ' -f 1)" = "__Q2R_SHA256__"
test "$(sha256sum "$aux/matdyn.x" | cut -d ' ' -f 1)" = "__MATDYN_SHA256__"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMP_NUM_THREADS=1
printf 'stage.container-start\n' > /workspace/terminal-stage.txt
run() {
  executable="$1"; label="$2"; input="$3"
  printf 'stage.%s\n' "$label" > /workspace/terminal-stage.txt
  mpirun -np "$ranks" "$bin/$executable.x" -nk "$ranks" -in "$input" \
    > "$label.stdout" 2> "$label.stderr"
}
run_aux() {
  executable="$1"; label="$2"; input="$3"
  printf 'stage.%s\n' "$label" > /workspace/terminal-stage.txt
  mpirun -np "$ranks" "$aux/$executable.x" -in "$input" \
    > "$label.stdout" 2> "$label.stderr"
}
stage_phonon_tree() {
  prefix="$1"
  test -d "out/$prefix.save"
  test -d "out/_ph0"
  ln -s "out/$prefix.save" "$prefix.save"
  ln -s "out/_ph0" _ph0
  printf 'stage.phonon-postprocess\n' > /workspace/terminal-stage.txt
}
unstage_phonon_tree() {
  prefix="$1"
  test -L "$prefix.save"
  test -L _ph0
  unlink "$prefix.save"
  unlink _ph0
}
""".strip()
        + "\n"
        + workflow
        + "\nprintf 'stage.workflow-complete\\n' > /workspace/terminal-stage.txt\n"
        + "\n"
    )
    expected = dict(_binary_sha256s())
    for executable in ("pw.x", "ph.x", "epw.x", "q2r.x", "matdyn.x"):
        marker = f"__{executable.removesuffix('.x').upper()}_SHA256__"
        script = script.replace(marker, expected[executable])
    return script


def _number(value: bytes) -> Decimal:
    try:
        return Decimal(value.decode("ascii").replace("D", "E").replace("d", "e"))
    except (UnicodeDecodeError, InvalidOperation) as error:
        raise ValueError("solver emitted a malformed decimal operand") from error


_ENERGY = re.compile(rb"!\s+total energy\s+=\s+([-+0-9.EeDd]+)\s+Ry")
_ACCURACY = re.compile(rb"estimated scf accuracy\s+<\s+([-+0-9.EeDd]+)\s+Ry")
_CONVERGED = re.compile(rb"convergence has been achieved")
_FERMI = re.compile(rb"the Fermi energy is\s+([-+0-9.EeDd]+)\s+ev", re.IGNORECASE)
_GAP = re.compile(
    rb"highest occupied, lowest unoccupied level \(ev\):\s+"
    rb"([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)",
    re.IGNORECASE,
)
_FREQUENCY = re.compile(
    rb"freq\s*\([^)]*\)\s*=\s*[-+0-9.EeDd]+\s*\[THz\]\s*=\s*"
    rb"([-+0-9.EeDd]+)\s*\[cm-1\]",
    re.IGNORECASE,
)


def _bounded_logs(root: Path) -> bytes:
    result = bytearray()
    for path in sorted(root.rglob("*.stdout")):
        if path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAXIMUM_LOG_BYTES or len(result) + size > 8 * MAXIMUM_LOG_BYTES:
            raise ValueError('material control solver logs exceed the fixed extraction bound')
        result.extend(path.read_bytes())
        result.extend(b"\n")
    return bytes(result)


def _scf_repeat_residual(root: Path) -> tuple[bool, Decimal]:
    """Return the largest final-energy disagreement between repeated SCFs."""

    final_energies: list[Decimal] = []
    for path in sorted(root.rglob("*scf*.stdout")):
        if path.is_symlink() or not path.is_file():
            continue
        if "nscf" in path.name:
            continue
        if path.stat().st_size > MAXIMUM_LOG_BYTES:
            raise ValueError('material control SCF repeat log exceeds extraction bound')
        matches = _ENERGY.findall(path.read_bytes())
        if matches:
            final_energies.append(_number(matches[-1]))
    if len(final_energies) < 2:
        return False, Decimal(0)
    return True, max(final_energies) - min(final_energies)


def _numeric_rows(path: Path) -> tuple[tuple[Decimal, ...], ...]:
    rows: list[tuple[Decimal, ...]] = []
    if path.stat().st_size > MAXIMUM_LOG_BYTES:
        raise ValueError('material control scientific text output exceeds extraction bound')
    for line in path.read_bytes().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith((b"#", b"!")):
            continue
        try:
            row = tuple(_number(item) for item in stripped.split())
        except ValueError:
            continue
        if row:
            rows.append(row)
    return tuple(rows)


def _temperature_from_name(path: Path) -> Decimal | None:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", path.name)
    if match is None:
        return None
    return Decimal(match.group(1))


def parse_epw_a2f_integrated_lambda(payload: bytes) -> Decimal:
    """Read the first EPW smearing's integrated coupling, never a spectral peak.

    EPW writes one frequency column, N spectral columns, then N cumulative
    2*alpha2F/omega columns.  The footer repeats the N terminal integrals.
    An ambiguous or truncated table cannot supply a material control operand.
    """

    if not payload or len(payload) > MAXIMUM_LOG_BYTES:
        raise ValueError("EPW alpha2F table is empty or exceeds the extraction bound")
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("EPW alpha2F table is not ASCII") from error
    if not lines or "w[meV] a2f and integrated 2*a2f/w" not in lines[0]:
        raise ValueError("EPW alpha2F table lacks its frequency/integral header")

    rows: list[tuple[Decimal, ...]] = []
    terminal: tuple[Decimal, ...] | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "Integrated el-ph coupling":
            if terminal is not None or index + 1 >= len(lines):
                raise ValueError("EPW alpha2F has ambiguous integrated-coupling footer")
            footer = lines[index + 1].strip()
            if not footer.startswith("#"):
                raise ValueError("EPW alpha2F integrated-coupling footer is malformed")
            try:
                terminal = tuple(Decimal(value) for value in footer[1:].split())
            except (InvalidOperation, ValueError) as error:
                raise ValueError("EPW alpha2F footer has a nonnumeric coupling") from error
            continue
        try:
            row = tuple(Decimal(value) for value in stripped.split())
        except (InvalidOperation, ValueError):
            if stripped:
                try:
                    Decimal(stripped.split()[0])
                except InvalidOperation:
                    pass
                else:
                    raise ValueError("EPW alpha2F has a malformed spectral row") from None
            continue
        if row:
            rows.append(row)

    if not rows or terminal is None:
        raise ValueError("EPW alpha2F lacks spectral rows or integrated coupling")
    width = len(rows[0])
    if width < 3 or width % 2 != 1 or any(len(row) != width for row in rows):
        raise ValueError("EPW alpha2F spectral/integral column layout differs")
    smear_count = (width - 1) // 2
    if len(terminal) != smear_count:
        raise ValueError("EPW alpha2F footer and smearing count differ")
    previous_frequency = Decimal(0)
    previous_integrals = (Decimal(0),) * smear_count
    rounding = Decimal("0.0000002")
    for row in rows:
        frequency = row[0]
        spectral = row[1 : 1 + smear_count]
        integrals = row[1 + smear_count :]
        if (
            any(not value.is_finite() or value < 0 for value in row)
            or frequency <= previous_frequency
            or any(value + rounding < old for value, old in zip(integrals, previous_integrals))
            or any(value < 0 for value in spectral)
        ):
            raise ValueError("EPW alpha2F frequency or cumulative integral is invalid")
        previous_frequency = frequency
        previous_integrals = integrals
    if any(
        not value.is_finite() or value < 0 or abs(value - observed) > rounding
        for value, observed in zip(terminal, previous_integrals)
    ):
        raise ValueError("EPW alpha2F footer disagrees with integrated table")
    return terminal[0]


def _epw_operands(root: Path, prefix: str) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    a2f_paths = tuple(root.rglob(f'{prefix}.a2f'))
    if len(a2f_paths) > 1:
        raise ValueError("EPW alpha2F has multiple ambiguous producer files")
    if a2f_paths:
        path = a2f_paths[0]
        if path.is_symlink() or not path.is_file():
            raise ValueError("EPW alpha2F is not a regular producer file")
        if path.stat().st_size > MAXIMUM_LOG_BYTES:
            raise ValueError("EPW alpha2F exceeds the extraction bound")
        lambda_ = parse_epw_a2f_integrated_lambda(path.read_bytes())
    else:
        lambda_ = Decimal(0)

    gaps_by_temperature: list[tuple[Decimal, Decimal, Decimal]] = []
    for pattern in (f'{prefix}.imag_iso_*', f'{prefix}.imag_aniso_gap0_*'):
        for path in root.rglob(pattern):
            if "gap0" in path.name and "aniso_gap0" not in path.name:
                continue
            temperature = _temperature_from_name(path)
            if temperature is None:
                continue
            rows = _numeric_rows(path)
            values: list[Decimal] = []
            for row in rows:
                if "imag_iso" in path.name and len(row) >= 3:
                    values.append(abs(row[2]) * 1000)
                elif "imag_aniso_gap0" in path.name and len(row) >= 2:
                    # EPW's gap0 convenience table is already in meV:
                    # distance, Delta_nk[meV], T[K], scaled, unscaled.
                    values.append(abs(row[1]))
            if values:
                gaps_by_temperature.append((temperature, min(values), max(values)))

    if not gaps_by_temperature:
        # EPW may omit the gap0 convenience file; use the smallest Matsubara
        # slice of the full anisotropic table (omega, xi, Z, Delta).
        for path in root.rglob(f'{prefix}.imag_aniso_*'):
            if "gap0" in path.name:
                continue
            temperature = _temperature_from_name(path)
            if temperature is None:
                continue
            rows = _numeric_rows(path)
            values = [abs(row[3]) * 1000 for row in rows if len(row) >= 4]
            if values:
                gaps_by_temperature.append((temperature, min(values), max(values)))

    positive = sorted(
        (row for row in gaps_by_temperature if row[2] >= Decimal("0.001")),
        key=lambda value: value[0],
    )
    if not positive:
        return (
            lambda_,
            Decimal(0),
            Decimal(0),
            Decimal(0),
        )
    highest = positive[-1]
    higher = sorted(
        row[0] for row in gaps_by_temperature if row[0] > highest[0] and row[2] < Decimal("0.001")
    )
    tc = highest[0] if not higher else (highest[0] + higher[0]) / 2
    lowest_temperature = min(positive, key=lambda value: value[0])
    return (
        lambda_,
        tc,
        lowest_temperature[1],
        lowest_temperature[2],
    )


def _archive_selected(root: Path, target: Path, *, output_limit: int) -> tuple[str, int]:
    files: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or path == target:
            continue
        relative = path.relative_to(root)
        if any(part.startswith("._") for part in relative.parts):
            continue
        if "pseudo" in relative.parts or path.suffix == ".in":
            continue
        if "_ph0" in relative.parts:
            continue
        if path.name.startswith("wfc") and path.suffix == ".dat":
            continue
        files.append((relative.as_posix(), path))

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        with (
            GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
        ):
            for name, path in files:
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        output.flush()
        os.fsync(output.fileno())
    size = target.stat().st_size
    if size <= 0 or size > output_limit // 2:
        raise ValueError('material control raw output archive exceeds its frozen output budget')
    return _hash_file(target), size


def _terminal_step_id(root: Path) -> str:
    marker = root / "terminal-stage.txt"
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_size > 256:
        return "stage.unavailable"
    try:
        value = marker.read_text(encoding="ascii").strip()
    except UnicodeDecodeError:
        return "stage.unavailable"
    return value if re.fullmatch(r"stage\.[a-z0-9.-]+", value) else "stage.unavailable"


class FixedMaterialControlWorkflowExecutor:
    'Local no-network executor for the closed material control control profile set.'

    executor_implementation_id = 'executor.ambient-pressure-superconductor-material-control-qe76-epw61'

    def execute(self, request: MaterialControlWorkflowRequest) -> MaterialControlWorkflowOperands:
        profile = resolve_material_control_workflow_profile(request.profile_id)
        if request.wall_time_seconds <= 0 or request.output_bytes <= 0:
            raise ValueError('material control workflow request requires positive bounded resources')
        if profile.cpu_cores != 8:
            raise ValueError('material control workflow profile CPU identity differs')
        run_scope_id = _assert_no_preexisting_run_container(request.run_id)
        task_scope_id = _container_scope_id(request.task_id)
        workspace = _workspace()
        run_id = sha256(request.request_id.encode()).hexdigest()[:24]
        run_root = workspace / f'workflow-{run_id}'
        if run_root.exists() or run_root.is_symlink():
            raise ValueError('material control workflow run root is not fresh')
        run_root.mkdir()
        source_root = run_root / "source"
        source_archive = (
            _external_path(TUTORIAL_ARCHIVE)
            if profile.source is WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE
            else _external_path(CONTROL_INPUT_ARCHIVE)
        )
        expected_source_sha = (
            TUTORIAL_ARCHIVE_SHA256
            if profile.source is WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE
            else required_external_sha256('EMPIRICAL_LAWHOOD_MATERIAL_CONTROL_CONTROL_INPUT_ARCHIVE_SHA256')
        )
        if not source_archive.is_file() or source_archive.is_symlink():
            raise ValueError('material control workflow source archive is unavailable')
        if _hash_file(source_archive) != expected_source_sha:
            raise ValueError('material control workflow source archive digest differs')
        environment_image = _external_path(ENVIRONMENT_IMAGE)
        if not environment_image.is_file() or environment_image.is_symlink():
            raise ValueError('material control workflow solver environment is unavailable')
        _safe_extract(
            source_archive,
            source_root,
            required_root=profile.source_root.split("/", maxsplit=1)[0],
        )
        _verify_science_repeat_inputs(source_root, profile)

        container_name = f'empirical-lawhood-material-control-{run_id}'
        command = (
            DOCKER_EXECUTABLE,
            "run",
            "--pull",
            "never",
            "--detach",
            "--name",
            container_name,
            "--label",
            f'{MATERIAL_CONTROL_CONTAINER_RUN_LABEL}={run_scope_id}',
            "--label",
            f'{MATERIAL_CONTROL_CONTAINER_TASK_LABEL}={task_scope_id}',
            "--privileged",
            "--network",
            "none",
            "--cpus",
            str(profile.cpu_cores),
            "--memory",
            str(profile.memory_bytes),
            "-v",
            f'{environment_image}:/environment.ext4:ro',
            "-v",
            f'{run_root}:/workspace:rw',
            "sha256:" + required_external_sha256("EMPIRICAL_LAWHOOD_AP_CONTAINER_IMAGE_SHA256"),
            "/bin/bash",
            "-c",
            _container_script(profile),
            'ambient-pressure-superconductor-material-control-workflow',
            str(profile.cpu_cores),
        )
        started = time.monotonic()
        launched = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=120,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        if (
            launched.returncode != 0
            or len(launched.stdout) > 256
            or len(launched.stderr) > 64 * 1024
        ):
            _force_remove_container(container_name)
            raise ValueError('material control detached solver container could not be started')
        container_id = launched.stdout.decode("ascii").strip()
        if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
            _force_remove_container(container_name)
            raise ValueError('material control detached solver returned an invalid container identity')
        timed_out = False
        peak = _tree_bytes(run_root)
        deadline = started + request.wall_time_seconds
        exit_code: int | None = None
        try:
            while exit_code is None:
                peak = max(peak, _tree_bytes(run_root))
                if time.monotonic() >= deadline:
                    timed_out = True
                    subprocess.run(
                        (DOCKER_EXECUTABLE, "kill", container_name),
                        check=False,
                        capture_output=True,
                        timeout=30,
                        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                    )
                    exit_code = 124
                    break
                state = subprocess.run(
                    (
                        DOCKER_EXECUTABLE,
                        "inspect",
                        "--format",
                        "{{.State.Running}}\t{{.State.ExitCode}}\t{{.State.OOMKilled}}",
                        container_name,
                    ),
                    check=False,
                    capture_output=True,
                    timeout=30,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                )
                if state.returncode != 0 or state.stderr or len(state.stdout) > 128:
                    raise ValueError('material control detached solver state could not be inspected')
                fields = state.stdout.decode("ascii").strip().split("\t")
                if len(fields) != 3 or fields[0] not in {"true", "false"}:
                    raise ValueError('material control detached solver state is malformed')
                if fields[0] == "false":
                    observed_exit = int(fields[1])
                    exit_code = 137 if fields[2] == "true" else observed_exit
                    break
                time.sleep(min(30.0, max(1.0, deadline - time.monotonic())))
            container_logs = subprocess.run(
                (DOCKER_EXECUTABLE, "logs", container_name),
                check=False,
                capture_output=True,
                timeout=30,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
            if (
                container_logs.returncode != 0
                or len(container_logs.stdout) > 1024**2
                or len(container_logs.stderr) > 1024**2
            ):
                raise ValueError('material control detached container logs could not be collected within bounds')
            launcher_stdout = launched.stdout + container_logs.stdout
            launcher_stderr = launched.stderr + container_logs.stderr
        except BaseException:
            _force_remove_container(container_name)
            raise
        if not _force_remove_container(container_name):
            raise ValueError('material control detached solver container could not be removed')
        if exit_code is None:
            raise RuntimeError('material control detached solver exited without a typed status')
        (run_root / "launcher.stdout").write_bytes(launcher_stdout)
        (run_root / "launcher.stderr").write_bytes(launcher_stderr)
        elapsed = Decimal(str(time.monotonic() - started))
        peak = max(peak, _tree_bytes(run_root))
        terminal_step_id = _terminal_step_id(run_root)
        logs = _bounded_logs(source_root)
        energies = tuple(_number(value) for value in _ENERGY.findall(logs))
        accuracies = tuple(_number(value) for value in _ACCURACY.findall(logs))
        fermi = tuple(_number(value) for value in _FERMI.findall(logs))
        gap_matches = tuple(_GAP.finditer(logs))
        frequencies = tuple(_number(value) for value in _FREQUENCY.findall(logs))
        lambda_, tc, gap_min, gap_max = _epw_operands(source_root, profile.prefix)
        scf_repeat_evaluable, scf_repeat_residual = _scf_repeat_residual(source_root)
        reasons: list[str] = []
        if timed_out:
            reasons.append("reason.solver-wall-time-exceeded")
        if exit_code != 0:
            reasons.append("reason.solver-exit-nonzero")
        if not energies:
            reasons.append("reason.scf-total-energy-missing")
        if not accuracies:
            reasons.append("reason.scf-accuracy-missing")
        if not scf_repeat_evaluable:
            reasons.append("reason.scf-repeat-operands-missing")
        if profile.kind is not WorkflowKind.SCF_CLASS and not frequencies:
            reasons.append("reason.phonon-frequency-operands-missing")
        if profile.kind is WorkflowKind.POSITIVE and gap_max <= 0:
            reasons.append("reason.epw-pairing-operands-missing")

        band_gap = Decimal(0)
        if gap_matches:
            last = gap_matches[-1]
            band_gap = max(Decimal(0), _number(last.group(2)) - _number(last.group(1)))
        raw_path = run_root / 'material-control-raw-output.tar.gz'
        raw_sha, raw_size = _archive_selected(
            run_root,
            raw_path,
            output_limit=request.output_bytes,
        )
        peak = max(peak, _tree_bytes(run_root))
        _prune_workflow_to_archive(run_root, raw_path)
        raw_hdf5_path = run_root / 'material-control-raw-output.h5'
        raw_hdf5_sha, raw_hdf5_size = encode_raw_archive_hdf5(
            archive_path=raw_path,
            output_path=raw_hdf5_path,
            profile_id=profile.profile_id,
            archive_sha256=raw_sha,
        )
        peak = max(peak, _tree_bytes(run_root))
        scf_completed = b"JOB DONE." in logs and bool(energies)
        scf_converged = _CONVERGED.search(logs) is not None
        return MaterialControlWorkflowOperands(
            exit_code=exit_code,
            timed_out=timed_out,
            terminal_step_id=terminal_step_id,
            wall_time_seconds=elapsed,
            peak_scratch_bytes=peak,
            scf_completed=scf_completed,
            scf_converged=scf_converged,
            total_energy_Ry=energies[-1] if energies else Decimal(0),
            scf_accuracy_Ry=accuracies[-1] if accuracies else Decimal(0),
            scf_repeat_evaluable=scf_repeat_evaluable,
            scf_repeat_residual_Ry=scf_repeat_residual,
            fermi_evaluable=bool(fermi),
            fermi_energy_eV=fermi[-1] if fermi else Decimal(0),
            band_gap_eV=band_gap,
            phonon_evaluable=bool(frequencies),
            minimum_phonon_frequency_cm1=(min(frequencies) if frequencies else Decimal(0)),
            epw_evaluable=gap_max > 0 and lambda_ > 0,
            electron_phonon_lambda=lambda_,
            tc_estimate_K=tc,
            gap_min_meV=gap_min,
            gap_max_meV=gap_max,
            raw_archive_path=raw_path,
            raw_archive_sha256=raw_sha,
            raw_archive_bytes=raw_size,
            raw_hdf5_path=raw_hdf5_path,
            raw_hdf5_sha256=raw_hdf5_sha,
            raw_hdf5_bytes=raw_hdf5_size,
            reason_codes=tuple(sorted(set(reasons))),
        )


__all__ = [
    "MATERIAL_CONTROL_WORKFLOW_PROFILES",
    'MaterialControlWorkflowOperands',
    'MaterialControlWorkflowProfile',
    'MaterialControlWorkflowRequest',
    "FileTaskOutputSource",
    "FixedMaterialControlWorkflowExecutor",
    "WorkflowKind",
    "WorkflowSource",
    'cleanup_material_control_workflow_scratch',
    "parse_epw_a2f_integrated_lambda",
    'resolve_material_control_workflow_profile',
]
