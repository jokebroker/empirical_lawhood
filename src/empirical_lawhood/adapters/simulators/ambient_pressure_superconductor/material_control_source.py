'Exact public Pb/MgB2 control-bundle custody for the material control follow-up.'

from __future__ import annotations

from empirical_lawhood._required_inputs import (
    required_external_path,
    required_external_sha256,
)

from hashlib import sha256
import json
from pathlib import Path
import stat
import subprocess
import tarfile
from typing import Final

from .material_source_design_contracts import SourceAssetLock, SourceMemberLock, SourceOutcomeRole
from .material_source_design_source import _tar_inventory
from .material_control_contracts import MaterialControlControlBundleQualification


TUTORIAL_ARCHIVE: Final = Path(
    'sources/ambient-pressure-superconductor/epw-tutorial04-20260530/tutorial04.tar.gz'
)
TUTORIAL_PAGE: Final = Path(
    'sources/ambient-pressure-superconductor/epw-tutorial04-20260530/epw-superconducting-properties-20260530.html'
)
CONTROL_INPUT_ARCHIVE: Final = Path(
    'sources/ambient-pressure-superconductor/material-control-control-inputs/material-control-control-inputs.tar.gz'
)
ENVIRONMENT_IMAGE: Final = Path(
    'solver-environments/ambient-pressure-superconductor-qe-7.6-epw-6.1-local-ext4.ext4'
)
DOCKER_EXECUTABLE: Final = "/usr/bin/docker"
TUTORIAL_ARCHIVE_SHA256: Final = (
    "dc06d7703cb1d8d1bc1717addbde951a90e67060ba5012162a2eff6143778973"
)
TUTORIAL_PAGE_SHA256: Final = (
    "89aa317def08f97a0018bf36579740650f8e44b6d5264346827e75b3ba37f528"
)

_MEMBER_ROWS: Final = (
    (
        "member.tutorial04-b-pseudo",
        "tutorial04/pseudo/B.upf",
        207_734,
        "0684ccd9b6b3fded340d7b690c90745f075018366eb4e6c876736eb56111f799",
    ),
    (
        "member.tutorial04-ir-basis",
        "tutorial04/sparse-ir/ir_nlambda6_ndigit8.dat",
        1_319_453,
        "fe9a9d25c5e4575e648c66cfa53d2601460051c4c951fd82af869f49b3d01133",
    ),
    (
        "member.tutorial04-mg-pseudo",
        "tutorial04/pseudo/Mg.upf",
        207_480,
        "15cdd9380687a47a1f4bc4ab5fe1f0c6d9c50891350fd8534c8f292b3ce0be14",
    ),
    (
        "member.tutorial04-mgb2-epw-fbw",
        "tutorial04/exercise2/epw2/epw2.in",
        954,
        "c2f8508f22350aaa462063e87e660c3bf39884b8ab7d14b167c505b729e2d027",
    ),
    (
        "member.tutorial04-mgb2-epw-fsr",
        "tutorial04/exercise2/epw1/epw1.in",
        1_169,
        "c90068429538bfda0b84473ffe63b3715b4149ae9c7dba3eb418bf86ae4ce9fb",
    ),
    (
        "member.tutorial04-mgb2-epw-sparse-ir",
        "tutorial04/exercise2/epw3/epw3.in",
        1_053,
        "097d1e08bc64520475dcaf6e217a0bf288454e01c7d1149cc42b23c62c97efca",
    ),
    (
        "member.tutorial04-mgb2-nscf",
        "tutorial04/exercise2/epw1/nscf.in",
        11_777,
        "398dcba9133e240bc769bb7e65b9caccb1e5c7fb655a28c0604b20ad64cf40ff",
    ),
    (
        "member.tutorial04-mgb2-ph",
        "tutorial04/exercise2/phonon/ph.in",
        159,
        "2aca2996c177fc1870612a1eed77d61278b5ea6342f1e4e97ad542a715843af1",
    ),
    (
        "member.tutorial04-mgb2-scf",
        "tutorial04/exercise2/phonon/scf.in",
        780,
        "0e9e1b7398057af88e556bdbc2174eb0c2d0b952009c36596112148f72d1bab0",
    ),
    (
        "member.tutorial04-pb-epw-coarse",
        "tutorial04/exercise1/epw/epw1.in",
        1_722,
        "f782a3aacde155d41b897b535270fd27837ece11dacdb2a043e48583ae2c11b8",
    ),
    (
        "member.tutorial04-pb-epw-eliashberg",
        "tutorial04/exercise1/epw/epw2.in",
        940,
        "ca0727c9e9ea386a4900a5f1190f0d47c15b971bf81238493ae85a595e7fb2b2",
    ),
    (
        "member.tutorial04-pb-nscf",
        "tutorial04/exercise1/epw/nscf.in",
        11_676,
        "bf5272bf13649b5cd1037c2bb4738a6e885a7427eb71c5aed7cb31d0b0a95dba",
    ),
    (
        "member.tutorial04-pb-ph",
        "tutorial04/exercise1/phonon/ph.in",
        266,
        "abebc377cb380e9c5c25f21c02c2bbef4a4e7c3253c8426cbbb5d34c2214d602",
    ),
    (
        "member.tutorial04-pb-pseudo",
        "tutorial04/pseudo/Pb.upf",
        308_691,
        "1c58a8a738fff0b37a3ccbe1c9d45dc5dca11435930a81e628e1d1db363558b1",
    ),
    (
        "member.tutorial04-pb-scf",
        "tutorial04/exercise1/phonon/scf.in",
        678,
        "a676d0873f32054a8c320de562294ef95089f0da9c22712d6cb3191e28d14bbe",
    ),
)


def _external_path(relative: Path) -> Path:
    return required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT") / relative


def _binary_sha256s() -> tuple[tuple[str, str], ...]:
    return (
        ("epw.x", required_external_sha256("EMPIRICAL_LAWHOOD_AP_EPW_BINARY_SHA256")),
        (
            "matdyn.x",
            required_external_sha256("EMPIRICAL_LAWHOOD_AP_MATDYN_BINARY_SHA256"),
        ),
        ("ph.x", required_external_sha256("EMPIRICAL_LAWHOOD_AP_PH_BINARY_SHA256")),
        ("pw.x", required_external_sha256("EMPIRICAL_LAWHOOD_AP_PW_BINARY_SHA256")),
        ("q2r.x", required_external_sha256("EMPIRICAL_LAWHOOD_AP_Q2R_BINARY_SHA256")),
    )


def _members() -> tuple[SourceMemberLock, ...]:
    return tuple(
        SourceMemberLock(
            member_id=member_id,
            member_locator=locator,
            size_bytes=size,
            sha256=digest,
        )
        for member_id, locator, size, digest in sorted(_MEMBER_ROWS)
    )


def expected_material_control_control_bundle(
    *, base_source_sha256: str
) -> MaterialControlControlBundleQualification:
    control_input_sha256 = required_external_sha256(
        'EMPIRICAL_LAWHOOD_MATERIAL_CONTROL_CONTROL_INPUT_ARCHIVE_SHA256'
    )
    tutorial = SourceAssetLock(
        source_id='source.ambient-pressure-superconductor.epw-tutorial04-20260530',
        role_id="role.pb-mgb2-positive-workflow-control-bundle",
        release_id="release.epw-tutorial04-20260530",
        relative_locator=(
            'sources/ambient-pressure-superconductor/epw-tutorial04-20260530/tutorial04.tar.gz'
        ),
        public_locator="https://docs.epw-code.org/tutorials/tutorial_04/index.html",
        content_sha256=TUTORIAL_ARCHIVE_SHA256,
        size_bytes=547_271,
        published_checksum=f'qualified-download-sha256:{TUTORIAL_ARCHIVE_SHA256}',
        license_id="license.official-tutorial-read-execute",
        license_scope=(
            "The official EPW page explicitly offers this bundle for download and execution; "
            "the bundled PseudoDojo UPF headers state GNU GPL terms. No redistribution claim."
        ),
        outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
        archive_member_count=120,
        archive_regular_file_count=109,
        archive_directory_count=11,
        archive_symlink_count=0,
        archive_expanded_regular_bytes=2_168_550,
        archive_maximum_member_bytes=1_319_453,
        member_locks=_members(),
        qualification_checks=tuple(
            sorted(
                (
                    "archive-bounds-and-paths-verified",
                    "exact-control-input-members-hashed",
                    "official-download-route-verified",
                    "pseudo-gpl-headers-verified",
                    "target-outcomes-absent",
                )
            )
        ),
    )
    page = SourceAssetLock(
        source_id='source.ambient-pressure-superconductor.epw-tutorial04-page-20260530',
        role_id="role.control-procedure-and-public-reference-values",
        release_id="release.epw-tutorial04-page-20260530",
        relative_locator=(
            'sources/ambient-pressure-superconductor/epw-tutorial04-20260530/'
            "epw-superconducting-properties-20260530.html"
        ),
        public_locator="https://docs.epw-code.org/tutorials/tutorial_04/index.html",
        content_sha256=TUTORIAL_PAGE_SHA256,
        size_bytes=130_615,
        published_checksum=f'qualified-download-sha256:{TUTORIAL_PAGE_SHA256}',
        license_id="license.website-read-and-cite",
        license_scope="Read-only official procedure/reference capture; no page redistribution.",
        outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
        archive_member_count=0,
        archive_regular_file_count=0,
        archive_directory_count=0,
        archive_symlink_count=0,
        archive_expanded_regular_bytes=0,
        archive_maximum_member_bytes=0,
        member_locks=(),
        qualification_checks=tuple(
            sorted(("exact-page-bytes-hashed", "official-procedure-date-verified"))
        ),
    )
    control_inputs = SourceAssetLock(
        source_id='source.ambient-pressure-superconductor.material-control-control-inputs',
        role_id='role.calibration-exact-sssp-control-input-bundle',
        release_id='release.ambient-pressure-superconductor-material-control-control-inputs',
        relative_locator=(
            'sources/ambient-pressure-superconductor/material-control-control-inputs/material-control-control-inputs.tar.gz'
        ),
        public_locator="derived:material-source-design-roster-science-plus-qualified-sssp-1.3.0",
        content_sha256=control_input_sha256,
        size_bytes=5_929_276,
        published_checksum=f'locally-issued-sha256:{control_input_sha256}',
        license_id="license.base-qualified-sssp-terms-plus-project-inputs",
        license_scope=(
            "Project-authored QE inputs plus the exact SSSP pseudopotential members already "
            "qualified by the imported base-source record; no new licence claim."
        ),
        outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
        archive_member_count=106,
        archive_regular_file_count=63,
        archive_directory_count=43,
        archive_symlink_count=0,
        archive_expanded_regular_bytes=18_128_248,
        archive_maximum_member_bytes=2_296_427,
        member_locks=(
            SourceMemberLock(
                member_id='member.material-control-control-input-manifest',
                member_locator="material-control-control-inputs/manifest.json",
                size_bytes=12_615,
                sha256=required_external_sha256(
                    'EMPIRICAL_LAWHOOD_MATERIAL_CONTROL_CONTROL_INPUT_MANIFEST_SHA256'
                ),
            ),
        ),
        qualification_checks=tuple(
            sorted(
                (
                    'material-source-design-roster-and-science-fingerprints-bound',
                    "archive-bounds-and-paths-verified",
                    "exact-input-manifest-member-hashed",
                    "target-contact-count-zero",
                )
            )
        ),
    )
    return MaterialControlControlBundleQualification(
        qualification_id='qualification.ambient-pressure-superconductor-material-control-pb-mgb2-control-bundle',
        base_source_qualification_sha256=base_source_sha256,
        gauge_covariant_source_extension_sha256=required_external_sha256(
            'EMPIRICAL_LAWHOOD_GAUGE_COVARIANT_SOURCE_EXTENSION_SHA256'
        ),
        tutorial_asset=tutorial,
        tutorial_page_asset=page,
        control_input_asset=control_inputs,
        environment_profile_id='environment.ambient-pressure-superconductor-qe76-epw61-local-ext4',
        environment_image_sha256=required_external_sha256(
            "EMPIRICAL_LAWHOOD_AP_ENVIRONMENT_IMAGE_SHA256"
        ),
        container_image_digest="sha256:"
        + required_external_sha256("EMPIRICAL_LAWHOOD_AP_CONTAINER_IMAGE_SHA256"),
        binary_sha256s=_binary_sha256s(),
        workflow_control_ids=('structure.calibration-mgb2-alb2', 'structure.calibration-pb-fcc'),
        science_view_ids=(
            "view.pbe-efficiency-base",
            "view.pbe-precision-refined",
        ),
        tutorial_view_id="view.epw-tutorial04-pbe-pseudodojo-workflow-reference",
        credentials_required=False,
        clickthrough_required=False,
        paid_resource_required=False,
        target_outcomes_projected=False,
        limitations=tuple(
            sorted(
                (
                    "limitation.tutorial-control-is-pseudodojo-not-sssp",
                    "limitation.tutorial-reference-values-are-not-local-recurrence",
                    "limitation.workflow-pass-cannot-calibrate-sssp-view-tolerances",
                    "limitation.mgb2-full-bandwidth-temperature-support-stops-at-45k",
                )
            )
        ),
    )


def _hash_regular(path: Path, *, expected_size: int) -> str:
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f'material control held source is not a regular non-symlink file: {path}')
    if observed.st_size != expected_size:
        raise ValueError(f'material control held source size differs: {path}')
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _verify_members(asset: SourceAssetLock) -> None:
    expected = {value.member_locator: value for value in asset.member_locks}
    with tarfile.open(_external_path(TUTORIAL_ARCHIVE), mode="r:gz") as archive:
        for locator, lock in expected.items():
            member = archive.getmember(locator)
            if not member.isfile() or member.size != lock.size_bytes:
                raise ValueError(f'material control tutorial member shape differs: {locator}')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'material control tutorial member cannot be read: {locator}')
            payload = stream.read(lock.size_bytes + 1)
            if (
                len(payload) != lock.size_bytes
                or sha256(payload).hexdigest() != lock.sha256
            ):
                raise ValueError(f'material control tutorial member digest differs: {locator}')


def _verify_environment(bundle: MaterialControlControlBundleQualification) -> None:
    environment_image = _external_path(ENVIRONMENT_IMAGE)
    if (
        _hash_regular(environment_image, expected_size=4_089_446_400)
        != bundle.environment_image_sha256
    ):
        raise ValueError('material control solver environment image digest differs')
    binary_lines = " ".join(name for name, _digest in bundle.binary_sha256s)
    script = f"""\nset -euo pipefail\nmkdir -p /environment\nmount -o loop,ro /environment.ext4 /environment\ncleanup() {{ cd /; umount /environment; }}\ntrap cleanup EXIT INT TERM\nfor name in {binary_lines}; do\n  path="/environment/qe/bin/$name"\n  if test ! -e "$path"; then\n    path="/environment/qe/source/bin/$name"\n  fi\n  printf '%s\\t%s\\n' "$name" "$(sha256sum "$path" | cut -d ' ' -f 1)"\ndone\n""".strip()
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
            f'{environment_image}:/environment.ext4:ro',
            bundle.container_image_digest,
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
        raise ValueError('material control environment inspection failed')
    try:
        observed = tuple(
            sorted(
                tuple(line.split("\t"))
                for line in completed.stdout.decode("ascii").splitlines()
            )
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError('material control environment inspection output is malformed') from error
    if observed != bundle.binary_sha256s:
        raise ValueError('material control executable identities differ')
    inspected = subprocess.run(
        (DOCKER_EXECUTABLE, "image", "inspect", bundle.container_image_digest),
        check=False,
        capture_output=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        image = json.loads(inspected.stdout)[0]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('material control container image metadata is unavailable') from error
    if inspected.returncode != 0 or image.get("Id") != bundle.container_image_digest:
        raise ValueError('material control container image identity differs')


def inspect_material_control_control_bundle(
    *, base_source_sha256: str
) -> MaterialControlControlBundleQualification:
    bundle = expected_material_control_control_bundle(base_source_sha256=base_source_sha256)
    root = required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT").resolve()
    for asset, relative in (
        (bundle.tutorial_asset, TUTORIAL_ARCHIVE),
        (bundle.tutorial_page_asset, TUTORIAL_PAGE),
        (bundle.control_input_asset, CONTROL_INPUT_ARCHIVE),
    ):
        path = root / relative
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError('material control source resolves outside external scientific storage')
        if _hash_regular(path, expected_size=asset.size_bytes) != asset.content_sha256:
            raise ValueError(f'material control source digest differs: {asset.source_id}')
    if _tar_inventory(root / TUTORIAL_ARCHIVE) != (
        120,
        109,
        11,
        0,
        2_168_550,
        1_319_453,
    ):
        raise ValueError('material control tutorial archive inventory differs')
    _verify_members(bundle.tutorial_asset)
    if _tar_inventory(root / CONTROL_INPUT_ARCHIVE) != (
        106,
        63,
        43,
        0,
        18_128_248,
        2_296_427,
    ):
        raise ValueError('material control control-input archive inventory differs')
    with tarfile.open(root / CONTROL_INPUT_ARCHIVE, mode="r:gz") as archive:
        lock = bundle.control_input_asset.member_locks[0]
        member = archive.getmember(lock.member_locator)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError('material control control-input manifest cannot be read')
        payload = stream.read(lock.size_bytes + 1)
        if (
            len(payload) != lock.size_bytes
            or sha256(payload).hexdigest() != lock.sha256
        ):
            raise ValueError('material control control-input manifest identity differs')
    _verify_environment(bundle)
    return bundle


__all__ = [
    'expected_material_control_control_bundle',
    'inspect_material_control_control_bundle',
]
