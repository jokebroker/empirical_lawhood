'Exact held-source closure and read-only excluded solver control asset qualification.'

from __future__ import annotations

from empirical_lawhood._required_inputs import required_external_path, required_external_sha256

from hashlib import sha256
import json
from pathlib import Path
import stat
import subprocess
import tarfile
from typing import Final

from .contracts import ExcludedSolverControlConfig, ExcludedSolverControlSourceManifest, SOURCE_MANIFEST_SOURCE_ID


DOCKER_EXECUTABLE: Final = "/usr/bin/docker"
UPSTREAM_SCF_MEMBER: Final = "q-e-EPW-6.1/EPW/examples/pb/woSOC/epw/scf.in"
UPSTREAM_SCF_SHA256: Final = "b858551313da526937b4d9f837931ecb124440be0753c3ee27519fb2f4ca6c68"
ENVIRONMENT_IMAGE_BYTES: Final = 4_089_446_400
CONTAINER_IMAGE_BYTES: Final = 280_106_726
def _internal_assets() -> tuple[tuple[str, str, str | None, int], ...]:
    return (
        ("build-manifest", "/environment/qe/build-package-versions.tsv", required_external_sha256("EMPIRICAL_LAWHOOD_AP_BUILD_MANIFEST_SHA256"), 6_286),
        ("derived-scf-input", "/environment/qe/control/epw-pb-wosoc-scf/scf.in", None, 541),
        ("epw.x", "/environment/qe/bin/epw.x", required_external_sha256("EMPIRICAL_LAWHOOD_AP_EPW_BINARY_SHA256"), 30_330_976),
        ("ph.x", "/environment/qe/bin/ph.x", required_external_sha256("EMPIRICAL_LAWHOOD_AP_PH_BINARY_SHA256"), 25_841_152),
        ("pseudopotential", "/environment/qe/control/epw-pb-wosoc-scf/pb_s.UPF", None, 410_130),
        ("pw.x", "/environment/qe/bin/pw.x", None, 23_896_616),
    )


def expected_excluded_solver_control_source_manifest(config: ExcludedSolverControlConfig) -> ExcludedSolverControlSourceManifest:
    """Construct the frozen manifest without touching external state."""

    return ExcludedSolverControlSourceManifest(
        source_id=SOURCE_MANIFEST_SOURCE_ID,
        release_id="epw-6.1-qe-7.6",
        upstream_archive_sha256=config.source_archive_sha256,
        upstream_archive_bytes=config.source_archive_bytes,
        upstream_member_count=config.source_member_count,
        upstream_expanded_bytes=config.source_expanded_bytes,
        upstream_scf_input_sha256=UPSTREAM_SCF_SHA256,
        derived_scf_input_sha256=config.input_sha256,
        derivation_id='derivation.pb-wosoc-scf-k14-degauss-0p025',
        pseudopotential_sha256=config.pseudopotential_sha256,
        environment_profile_id=config.solver_profile_id,
        environment_image_sha256=config.environment_image_sha256,
        environment_image_bytes=ENVIRONMENT_IMAGE_BYTES,
        container_image_digest=config.container_image_digest,
        container_image_bytes=CONTAINER_IMAGE_BYTES,
        binary_sha256s=tuple(
            sorted(
                (
                    ("epw.x", required_external_sha256("EMPIRICAL_LAWHOOD_AP_EPW_BINARY_SHA256")),
                    ("ph.x", required_external_sha256("EMPIRICAL_LAWHOOD_AP_PH_BINARY_SHA256")),
                    ("pw.x", config.qe_binary_sha256),
                )
            )
        ),
        build_manifest_sha256=required_external_sha256("EMPIRICAL_LAWHOOD_AP_BUILD_MANIFEST_SHA256"),
        control_component_scope='scope.pb-wosoc-derived-scf-only',
        scientific_scope_limitations=tuple(
            sorted(
                (
                    "no-alpha2f-coupling-tc-or-gap-operand",
                    "no-electron-phonon-execution",
                    "no-phonon-execution",
                    "no-superconductivity-or-target-material-claim",
                    "scf-component-is-a-declared-upstream-input-derivative",
                )
            )
        ),
        network_required=False,
    )


def _hash_regular_file(path: Path, *, expected_size: int) -> str:
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f'excluded solver control held asset is not a regular non-symlink file: {path}')
    if observed.st_size != expected_size:
        raise ValueError(f'excluded solver control held asset size differs: {path}')
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _verify_upstream_member() -> None:
    with tarfile.open(required_external_path("EMPIRICAL_LAWHOOD_AP_SOURCE_ARCHIVE"), mode="r:gz") as archive:
        member = archive.getmember(UPSTREAM_SCF_MEMBER)
        if not member.isfile() or member.size <= 0 or member.size > 1024**2:
            raise ValueError('excluded solver control upstream SCF member is absent or outside its bound')
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError('excluded solver control upstream SCF member cannot be read')
        payload = stream.read(member.size + 1)
    if len(payload) != member.size or sha256(payload).hexdigest() != UPSTREAM_SCF_SHA256:
        raise ValueError('excluded solver control upstream SCF member identity differs')


def _local_container_metadata(image_digest: str) -> tuple[str, int]:
    completed = subprocess.run(
        (DOCKER_EXECUTABLE, "image", "inspect", image_digest),
        check=False,
        capture_output=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    if completed.returncode != 0 or len(completed.stdout) > 4 * 1024**2:
        raise ValueError('excluded solver control local runtime image is unavailable')
    try:
        document = json.loads(completed.stdout)
        image = document[0]
        observed_id = image["Id"]
        observed_size = image["Size"]
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError('excluded solver control local runtime image metadata is malformed') from error
    if not isinstance(observed_id, str) or not isinstance(observed_size, int):
        raise ValueError('excluded solver control local runtime image metadata types differ')
    return observed_id, observed_size


def _verify_environment_members(config: ExcludedSolverControlConfig) -> None:
    script = r"""
set -euo pipefail
mkdir -p /environment
mount -o loop,ro /environment.ext4 /environment
cleanup() { cd /; umount /environment; }
trap cleanup EXIT INT TERM
for specification in \
  'build-manifest:/environment/qe/build-package-versions.tsv' \
  'derived-scf-input:/environment/qe/control/epw-pb-wosoc-scf/scf.in' \
  'epw.x:/environment/qe/bin/epw.x' \
  'ph.x:/environment/qe/bin/ph.x' \
  'pseudopotential:/environment/qe/control/epw-pb-wosoc-scf/pb_s.UPF' \
  'pw.x:/environment/qe/bin/pw.x'
do
  label="${specification%%:*}"
  path="${specification#*:}"
  digest="$(sha256sum "$path" | cut -d ' ' -f 1)"
  size="$(stat -c '%s' "$path")"
  printf '%s\t%s\t%s\n' "$label" "$digest" "$size"
done
""".strip()
    completed = subprocess.run(
        (
            DOCKER_EXECUTABLE,
            "run",
            "--pull",
            "never",
            "--rm",
            "--privileged",
            "--network",
            "none",
            "-v",
            f"{required_external_path('EMPIRICAL_LAWHOOD_AP_ENVIRONMENT_IMAGE')}:/environment.ext4:ro",
            config.container_image_digest,
            "/bin/bash",
            "-c",
            script,
        ),
        check=False,
        capture_output=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    if completed.returncode != 0 or len(completed.stdout) > 64 * 1024:
        raise ValueError('excluded solver control environment member inspection failed')
    expected = {
        label: (
            digest
            if digest is not None
            else (
                config.input_sha256
                if label == "derived-scf-input"
                else config.pseudopotential_sha256
                if label == "pseudopotential"
                else config.qe_binary_sha256
            ),
            size,
        )
        for label, _path, digest, size in _internal_assets()
    }
    observed: dict[str, tuple[str, int]] = {}
    try:
        for line in completed.stdout.decode("ascii").splitlines():
            label, digest, size = line.split("\t")
            observed[label] = (digest, int(size))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError('excluded solver control environment member inspection output is malformed') from error
    if observed != expected:
        raise ValueError('excluded solver control environment member identities differ')


def inspect_excluded_solver_control_external_assets(config: ExcludedSolverControlConfig) -> ExcludedSolverControlSourceManifest:
    'Reopen and verify every held byte identity needed by the excluded solver control SCF act.'

    if (
        _hash_regular_file(
            required_external_path("EMPIRICAL_LAWHOOD_AP_SOURCE_ARCHIVE"),
            expected_size=config.source_archive_bytes,
        )
        != config.source_archive_sha256
    ):
        raise ValueError('excluded solver control source archive digest differs')
    _verify_upstream_member()
    if (
        _hash_regular_file(
            required_external_path("EMPIRICAL_LAWHOOD_AP_ENVIRONMENT_IMAGE"),
            expected_size=ENVIRONMENT_IMAGE_BYTES,
        )
        != config.environment_image_sha256
    ):
        raise ValueError('excluded solver control environment image digest differs')
    observed_image_id, observed_image_size = _local_container_metadata(
        config.container_image_digest
    )
    if (
        observed_image_id != config.container_image_digest
        or observed_image_size != CONTAINER_IMAGE_BYTES
    ):
        raise ValueError('excluded solver control local runtime image identity differs')
    _verify_environment_members(config)
    return expected_excluded_solver_control_source_manifest(config)


__all__ = [
    "DOCKER_EXECUTABLE",
    'expected_excluded_solver_control_source_manifest',
    'inspect_excluded_solver_control_external_assets',
]
