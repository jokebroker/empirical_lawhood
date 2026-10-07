'Read-only qualification of the exact ambient pressure superconductor material source design public roster.'

from __future__ import annotations

from empirical_lawhood._required_inputs import required_external_path, required_external_sha256

import csv
from decimal import Decimal
from hashlib import md5, sha256
import io
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
from typing import Final
import zipfile

from .material_source_design_contracts import MATERIAL_SOURCE_DESIGN_SOURCE_PLAN_ID, MaterialSourceDesignSourceQualification, PseudopotentialLock, SourceAssetLock, SourceMemberLock, SourceOutcomeRole


EPW_ARCHIVE: Final = Path('sources/ambient-pressure-superconductor/epw-6.1/q-e-EPW-6.1.tar.gz')
THREE_DSC_ARCHIVE: Final = Path(
    'sources/ambient-pressure-superconductor/'
    "3dsc-publication-2471dd51a298a854cb4f365ebd39e72c7cbf3634/"
    "3DSC-2471dd51a298a854cb4f365ebd39e72c7cbf3634.tar.gz"
)
SSSP_EFFICIENCY_TAR: Final = Path(
    'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-efficiency/SSSP_1.3.0_PBE_efficiency.tar.gz'
)
SSSP_EFFICIENCY_JSON: Final = Path(
    'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-efficiency/SSSP_1.3.0_PBE_efficiency.json'
)
SSSP_LICENSE: Final = Path(
    'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-efficiency/LICENSE.txt'
)
SSSP_README: Final = Path(
    'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-efficiency/README.md'
)
SSSP_PRECISION_TAR: Final = Path(
    'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-precision/SSSP_1.3.0_PBE_precision.tar.gz'
)
SSSP_PRECISION_JSON: Final = Path(
    'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-precision/SSSP_1.3.0_PBE_precision.json'
)
SSSP_PBESOL_PRECISION_TAR: Final = Path(
    'sources/ambient-pressure-superconductor/sssp-1.3.0-pbesol-precision/SSSP_1.3.0_PBEsol_precision.tar.gz'
)
SSSP_PBESOL_PRECISION_JSON: Final = Path(
    'sources/ambient-pressure-superconductor/sssp-1.3.0-pbesol-precision/SSSP_1.3.0_PBEsol_precision.json'
)
GAO_PAPER: Final = Path(
    'sources/ambient-pressure-superconductor/method-literature/gao-2502.18281v1.pdf'
)
SEMENOK_PAPER: Final = Path(
    'sources/ambient-pressure-superconductor/method-literature/semenok-2407.12922v2.pdf'
)
XI_CODE: Final = Path(
    'sources/ambient-pressure-superconductor/stability-parameter-xi-v1.1/eyuzbash-stability-parameter-xi-v1.1.zip'
)
EXCLUDED_SOLVER_CONTROL_ENVIRONMENT_IMAGE: Final = Path(
    'solver-environments/ambient-pressure-superconductor-qe-7.6-epw-6.1-local-ext4.ext4'
)

THREE_DSC_CSV_MEMBER: Final = (
    "3DSC-2471dd51a298a854cb4f365ebd39e72c7cbf3634/superconductors_3D/data/final/MP/3DSC_MP.csv"
)


def _member(member_id: str, member_locator: str, size_bytes: int, digest: str) -> SourceMemberLock:
    return SourceMemberLock(
        member_id=member_id,
        member_locator=member_locator,
        size_bytes=size_bytes,
        sha256=digest,
    )


_EPW_MEMBERS: Final = tuple(
    sorted(
        (
            _member(
                "member.epw61.readme",
                "q-e-EPW-6.1/README.md",
                4_367,
                "87299e2ae62a98b738a9b9fa05aa436d67cb31d5ef8dffeb7f393f89344b8932",
            ),
            _member(
                "member.epw61.pb-wosoc-scf",
                "q-e-EPW-6.1/EPW/examples/pb/woSOC/epw/scf.in",
                536,
                "b858551313da526937b4d9f837931ecb124440be0753c3ee27519fb2f4ca6c68",
            ),
            _member(
                "member.epw61.pb-wosoc-epw",
                "q-e-EPW-6.1/EPW/examples/pb/woSOC/epw/epw.in",
                1_286,
                "fc5fcbc01db6d6791f3e3f6d57c74156da98563ec9afd2c729a44316311d8eb6",
            ),
            _member(
                "member.epw61.pb-wsoc-epw",
                "q-e-EPW-6.1/EPW/examples/pb/wSOC/epw/epw.in",
                1_311,
                "26d1617c5af22f17fb7a035d20edd5c66e92923c996276c2c073e0fb0d6d539c",
            ),
            _member(
                "member.epw61.pb-pseudo",
                "q-e-EPW-6.1/EPW/examples/pb/pp/pb_s.UPF",
                410_130,
                "24fe8cc951a89521da24dc6bd936b462078927a6970132848a355857a82d27af",
            ),
            _member(
                "member.epw61.mgb2-scf",
                "q-e-EPW-6.1/EPW/examples/mgb2/epw/scf.in",
                801,
                "9bf505c8abf8cc5aea574c3b519417ef126772d5072e3bcb3979ac08008ee68c",
            ),
            _member(
                "member.epw61.mgb2-epw",
                "q-e-EPW-6.1/EPW/examples/mgb2/epw/epw.in",
                1_411,
                "41953df87e2a61b8e6427845905048794f6b13d9f8bf791e4ba3c19b9c30e301",
            ),
            _member(
                "member.epw61.mgb2-b-pseudo",
                "q-e-EPW-6.1/EPW/examples/mgb2/pp/B.pz-vbc.UPF",
                22_160,
                "eb7aa71704ded328f1c78b2e16b630d14fc97d688b0b23b51a56d541c97e1f31",
            ),
            _member(
                "member.epw61.mgb2-mg-pseudo",
                "q-e-EPW-6.1/EPW/examples/mgb2/pp/Mg.pz-n-vbc.UPF",
                30_214,
                "5357982ec1f8139c2cde411073103f9b33d5482a45e2401bcf1b8aea228db076",
            ),
            _member(
                "member.epw61.diamond-scf",
                "q-e-EPW-6.1/EPW/examples/diamond/epw/scf.in",
                748,
                "95c0cee1f48891f730ab235cb78094de92668ffde16b76cd4cde8a78eae2c276",
            ),
            _member(
                "member.epw61.diamond-c-pseudo",
                "q-e-EPW-6.1/EPW/examples/diamond/pp/C_3.98148.UPF",
                78_562,
                "cc147806e663aa2c487cff67fc287a27d846edd438c7f4fc15d8851409244f2a",
            ),
        ),
        key=lambda value: value.member_id,
    )
)

_THREE_DSC_MEMBERS: Final = tuple(
    sorted(
        (
            _member(
                "member.3dsc.license",
                "3DSC-2471dd51a298a854cb4f365ebd39e72c7cbf3634/LICENSE.md",
                1_423,
                "bcf9a13dc5bf11da712a495ba9944b9667a3ae43a1b74a3fe1154ea35b3b8842",
            ),
            _member(
                "member.3dsc.readme",
                "3DSC-2471dd51a298a854cb4f365ebd39e72c7cbf3634/README.md",
                11_383,
                "da79076f31004833ab71a826744dfc3886a316b1240a3caec1594dbf8f4a9802",
            ),
            _member(
                "member.3dsc.mp-csv",
                THREE_DSC_CSV_MEMBER,
                9_457_406,
                "353b86f9c60505d11a3e25df8312da02ff88fd8e27af05b632dec4739d52eb24",
            ),
        ),
        key=lambda value: value.member_id,
    )
)

_XI_MEMBERS: Final = tuple(
    sorted(
        (
            _member(
                "member.xi-v1-1.license",
                "eyuzbash-stability-parameter-xi-f05ec2c/LICENSE",
                1_112,
                "eb7a937b70d254da8e33a62fe9a3cea62824857a845389a76c7aa52284cbd247",
            ),
            _member(
                "member.xi-v1-1.readme",
                "eyuzbash-stability-parameter-xi-f05ec2c/README.md",
                2_583,
                "afc7ced1dda33a38ab3c93598ae20b191300906587ef9d302dd3799ceff015f1",
            ),
            _member(
                "member.xi-v1-1.python",
                "eyuzbash-stability-parameter-xi-f05ec2c/python/compute_xi.py",
                2_091,
                "3ab3d98e2926b7277d7a66996c0582aa6af53c1f3abf466944f644e44de82057",
            ),
            _member(
                "member.xi-v1-1.hg-reference",
                "eyuzbash-stability-parameter-xi-f05ec2c/data/a2F_Hg_0GPa.txt",
                3_465,
                "3a858d6baeda0e81d5b842611b75e851635ee4f0cb9edc28d7585b322b1a2440",
            ),
        ),
        key=lambda value: value.member_id,
    )
)


def _asset(
    *,
    source_id: str,
    role_id: str,
    release_id: str,
    relative_locator: str,
    public_locator: str,
    content_sha256: str,
    size_bytes: int,
    published_checksum: str,
    license_id: str,
    license_scope: str,
    outcome_role: SourceOutcomeRole,
    inventory: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    member_locks: tuple[SourceMemberLock, ...] = (),
    checks: tuple[str, ...],
) -> SourceAssetLock:
    return SourceAssetLock(
        source_id=source_id,
        role_id=role_id,
        release_id=release_id,
        relative_locator=relative_locator,
        public_locator=public_locator,
        content_sha256=content_sha256,
        size_bytes=size_bytes,
        published_checksum=published_checksum,
        license_id=license_id,
        license_scope=license_scope,
        outcome_role=outcome_role,
        archive_member_count=inventory[0],
        archive_regular_file_count=inventory[1],
        archive_directory_count=inventory[2],
        archive_symlink_count=inventory[3],
        archive_expanded_regular_bytes=inventory[4],
        archive_maximum_member_bytes=inventory[5],
        member_locks=member_locks,
        qualification_checks=tuple(sorted(checks)),
    )


def _expected_assets() -> tuple[SourceAssetLock, ...]:
    sssp_url = "https://archive.materialscloud.org/records/gvag2-y0636"
    values = (
        _asset(
            source_id='source.ambient-pressure-superconductor.3dsc-publication-2471dd51',
            role_id="role.historical-structure-dedup-reference",
            release_id="release.3dsc-publication-2471dd51",
            relative_locator=(
                'sources/ambient-pressure-superconductor/'
                "3dsc-publication-2471dd51a298a854cb4f365ebd39e72c7cbf3634/"
                "3DSC-2471dd51a298a854cb4f365ebd39e72c7cbf3634.tar.gz"
            ),
            public_locator=(
                "https://github.com/aimat-lab/3DSC/tree/2471dd51a298a854cb4f365ebd39e72c7cbf3634"
            ),
            content_sha256="84cf2eef4db3e60fd1fe5a0f321b05d68653a7eb03293fbdb0012ee31a0b314d",
            size_bytes=8_965_223,
            published_checksum="commit:2471dd51a298a854cb4f365ebd39e72c7cbf3634",
            license_id="license.cc-by-4-0-and-mit",
            license_scope="3DSC_MP data are CC BY 4.0; software is MIT; ICSD data are excluded.",
            outcome_role=SourceOutcomeRole.HISTORICAL_OUTCOMES_NOT_PROJECTED,
            inventory=(15_835, 15_784, 51, 0, 47_846_064, 9_457_406),
            member_locks=_THREE_DSC_MEMBERS,
            checks=(
                "archive-bounds-verified",
                "commit-identity-verified",
                "license-scope-verified",
                "structural-projection-excludes-tc-values",
            ),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.excluded-solver-control-environment-image',
            role_id='role.excluded-solver-control-qualified-environment-reference',
            release_id='release.qe76-epw61-local-ext4',
            relative_locator='solver-environments/ambient-pressure-superconductor-qe-7.6-epw-6.1-local-ext4.ext4',
            public_locator="derived-locally-from-qe-7.6-and-epw-6.1",
            content_sha256=required_external_sha256(
                "EMPIRICAL_LAWHOOD_AP_ENVIRONMENT_IMAGE_SHA256"
            ),
            size_bytes=4_089_446_400,
            published_checksum='derived-excluded-solver-control-corrective-solver-control-qualified-environment',
            license_id="license.gpl-2-or-later-derived-environment",
            license_scope='Exact excluded solver control-qualified local environment; material source design-transport nomination material closure remains open.',
            outcome_role=SourceOutcomeRole.NO_OUTCOMES,
            checks=('excluded-solver-control-corrective-solver-control-route-qualified', "physical-sha256-verified"),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.epw61-archive',
            role_id="role.solver-control-and-method-source",
            release_id="release.epw-6-1-qe-source",
            relative_locator='sources/ambient-pressure-superconductor/epw-6.1/q-e-EPW-6.1.tar.gz',
            public_locator="https://github.com/QEF/q-e/tree/EPW-6.1",
            content_sha256="3e2246135aa66477d6c81aabbbb742fe57ff0e7fc51a8e14dd6b51d79f26bb68",
            size_bytes=87_890_918,
            published_checksum="release:EPW-6.1",
            license_id="license.gpl-2-or-later",
            license_scope="Quantum ESPRESSO distribution, including EPW examples and bundled controls.",
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            inventory=(8_173, 7_360, 810, 3, 276_451_958, 7_056_961),
            member_locks=_EPW_MEMBERS,
            checks=(
                "archive-bounds-verified",
                "control-inputs-hashed",
                "gpl-terms-verified",
                "relative-symlinks-contained",
            ),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.gao-2502-18281v1',
            role_id="role.empirical-tradeoff-method-paper",
            release_id="release.arxiv-2502-18281v1",
            relative_locator='sources/ambient-pressure-superconductor/method-literature/gao-2502.18281v1.pdf',
            public_locator="https://arxiv.org/abs/2502.18281v1",
            content_sha256="b3e1755d84ce782cd2607cf279f31cc9b4219dc532133d913f3fbabc1e817c2f",
            size_bytes=4_670_692,
            published_checksum="arxiv:2502.18281v1",
            license_id="license.arxiv-read-and-cite",
            license_scope="Read-only cited method evidence; no paper content is redistributed.",
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            checks=("exact-arxiv-version-verified", "pdf-physically-hashed"),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.semenok-2407-12922v2',
            role_id="role.spectrum-stability-method-paper",
            release_id="release.arxiv-2407-12922v2",
            relative_locator='sources/ambient-pressure-superconductor/method-literature/semenok-2407.12922v2.pdf',
            public_locator="https://arxiv.org/abs/2407.12922v2",
            content_sha256="4a678d5a0430a96be4edc85ab3b30b3aa9ef4e484bddebde98771548833318ef",
            size_bytes=2_012_457,
            published_checksum="arxiv:2407.12922v2",
            license_id="license.arxiv-read-and-cite",
            license_scope="Read-only cited method evidence; no paper content is redistributed.",
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            checks=("exact-arxiv-version-verified", "pdf-physically-hashed"),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.sssp130-license',
            role_id="role.pseudopotential-license-manifest",
            release_id="release.sssp-1-3-0",
            relative_locator='sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-efficiency/LICENSE.txt',
            public_locator=sssp_url,
            content_sha256="8028d9d7fcad7932bb5551dc0a60cff6c6b7f7aa2bd480f516a2faaa09d943a1",
            size_bytes=3_673,
            published_checksum="md5:c0f9df05fb4cd4921f4caa25606815ef",
            license_id="license.sssp-mixed-per-file",
            license_scope="Bundle metadata CC BY 4.0; selected pseudo license is bound per family.",
            outcome_role=SourceOutcomeRole.NO_OUTCOMES,
            checks=("physical-sha256-verified", "published-md5-verified"),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.sssp130-pbe-efficiency-json',
            role_id="role.primary-pseudopotential-metadata",
            release_id="release.sssp-1-3-0-pbe-efficiency",
            relative_locator=(
                'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-efficiency/SSSP_1.3.0_PBE_efficiency.json'
            ),
            public_locator=sssp_url,
            content_sha256="2c8642a70308ecd072a209a30c160bf51423560bb202116cb31fd5275eff814a",
            size_bytes=22_376,
            published_checksum="md5:3153c4b20fc90a44fba0236627525644",
            license_id="license.cc-by-4-0-metadata",
            license_scope="SSSP metadata and recommended cutoff table.",
            outcome_role=SourceOutcomeRole.NO_OUTCOMES,
            checks=("published-md5-verified", "selected-elements-closed"),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.sssp130-pbe-efficiency-tar',
            role_id="role.primary-pseudopotential-bytes",
            release_id="release.sssp-1-3-0-pbe-efficiency",
            relative_locator=(
                'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-efficiency/SSSP_1.3.0_PBE_efficiency.tar.gz'
            ),
            public_locator=sssp_url,
            content_sha256="7a85b71fa3d68df1b5ed33c55c7057681fbf10377ae3b3eac9392924d9189f12",
            size_bytes=59_513_522,
            published_checksum="md5:a58f1b3373f330179fd0832c48bb9a52",
            license_id="license.sssp-mixed-per-file",
            license_scope="Selected pseudopotentials use their family-specific GPL terms.",
            outcome_role=SourceOutcomeRole.NO_OUTCOMES,
            inventory=(104, 103, 1, 0, 166_266_051, 3_615_077),
            checks=(
                "archive-bounds-verified",
                "published-md5-verified",
                "selected-pseudopotential-sha256-verified",
            ),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.sssp130-pbe-precision-json',
            role_id="role.refined-pseudopotential-metadata",
            release_id="release.sssp-1-3-0-pbe-precision",
            relative_locator=(
                'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-precision/SSSP_1.3.0_PBE_precision.json'
            ),
            public_locator=sssp_url,
            content_sha256="7b7b648915e54c8397cd8143a7385c17eeed3104ad1236cecd905126d338a4e3",
            size_bytes=22_485,
            published_checksum="md5:1692c5c9ce89e1c7c783f8f0eee0cbfa",
            license_id="license.cc-by-4-0-metadata",
            license_scope="SSSP metadata and recommended cutoff table.",
            outcome_role=SourceOutcomeRole.NO_OUTCOMES,
            checks=("published-md5-verified", "selected-elements-closed"),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.sssp130-pbe-precision-tar',
            role_id="role.refined-pseudopotential-bytes",
            release_id="release.sssp-1-3-0-pbe-precision",
            relative_locator=(
                'sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-precision/SSSP_1.3.0_PBE_precision.tar.gz'
            ),
            public_locator=sssp_url,
            content_sha256="d91db6b4b3788501d535a5b84ebabf3859ea3e3ac6ea154c4be3718da50f0c85",
            size_bytes=62_963_841,
            published_checksum="md5:fde94756886f32ada7bf597547557eb5",
            license_id="license.sssp-mixed-per-file",
            license_scope="Selected pseudopotentials use their family-specific GPL terms.",
            outcome_role=SourceOutcomeRole.NO_OUTCOMES,
            inventory=(104, 103, 1, 0, 176_388_121, 3_615_077),
            checks=(
                "archive-bounds-verified",
                "published-md5-verified",
                "selected-pseudopotential-sha256-verified",
            ),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.sssp130-pbesol-precision-json',
            role_id="role.independent-pseudopotential-metadata",
            release_id="release.sssp-1-3-0-pbesol-precision",
            relative_locator=(
                'sources/ambient-pressure-superconductor/sssp-1.3.0-pbesol-precision/SSSP_1.3.0_PBEsol_precision.json'
            ),
            public_locator=sssp_url,
            content_sha256="00afc6c3a277f70dab8c9b905ef29af7d3baab73163a0c5b2c3cc04ad8eed98f",
            size_bytes=22_556,
            published_checksum="md5:c355de4dc3a2067b34f257d01368322a",
            license_id="license.cc-by-4-0-metadata",
            license_scope="SSSP metadata and recommended cutoff table.",
            outcome_role=SourceOutcomeRole.NO_OUTCOMES,
            checks=("published-md5-verified", "selected-elements-closed"),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.sssp130-pbesol-precision-tar',
            role_id="role.independent-pseudopotential-bytes",
            release_id="release.sssp-1-3-0-pbesol-precision",
            relative_locator=(
                'sources/ambient-pressure-superconductor/sssp-1.3.0-pbesol-precision/'
                "SSSP_1.3.0_PBEsol_precision.tar.gz"
            ),
            public_locator=sssp_url,
            content_sha256="795e95e990dcf612181ec7e82e8c82bde4519256b9e81f055c335cad0fdf574c",
            size_bytes=42_714_898,
            published_checksum="md5:443422c2ac5fd367a8ddb7aa6433077a",
            license_id="license.sssp-mixed-per-file",
            license_scope="Selected pseudopotentials use their family-specific GPL terms.",
            outcome_role=SourceOutcomeRole.NO_OUTCOMES,
            inventory=(86, 85, 1, 0, 118_577_088, 3_431_357),
            checks=(
                "archive-bounds-verified",
                "published-md5-verified",
                "selected-pseudopotential-sha256-verified",
            ),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.sssp130-readme',
            role_id="role.pseudopotential-format-manifest",
            release_id="release.sssp-1-3-0",
            relative_locator='sources/ambient-pressure-superconductor/sssp-1.3.0-pbe-efficiency/README.md',
            public_locator=sssp_url,
            content_sha256="8fa20d9daaec17545a8fbb4ec415dbdd43156f229df3b97e2f53ee3fba0cdc8b",
            size_bytes=2_359,
            published_checksum="md5:23264819e0fa3ac4726a25f8c48b8a29",
            license_id="license.cc-by-4-0-metadata",
            license_scope="SSSP format and cutoff documentation.",
            outcome_role=SourceOutcomeRole.NO_OUTCOMES,
            checks=("physical-sha256-verified", "published-md5-verified"),
        ),
        _asset(
            source_id='source.ambient-pressure-superconductor.xi-code-v1-1',
            role_id="role.spectrum-stability-reference-implementation",
            release_id="release.zenodo-15557831-v1-1",
            relative_locator=(
                'sources/ambient-pressure-superconductor/stability-parameter-xi-v1.1/'
                "eyuzbash-stability-parameter-xi-v1.1.zip"
            ),
            public_locator="https://doi.org/10.5281/zenodo.15557831",
            content_sha256="2a863df08b09a19898c49a7f372a57953c3983b1894fcfdd82234a10daddfbb0",
            size_bytes=10_140,
            published_checksum="md5:a1afbaed706ce0ef69d2d0c081fae26f",
            license_id="license.mit",
            license_scope="Published Mathematica, MATLAB and Python reference implementations.",
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            inventory=(12, 7, 5, 0, 20_886, 9_881),
            member_locks=_XI_MEMBERS,
            checks=(
                "archive-bounds-verified",
                "reference-output-independently-reproduced",
                "mit-license-verified",
                "published-md5-verified",
                "reference-implementation-hashed",
            ),
        ),
    )
    return tuple(sorted(values, key=lambda value: value.source_id))


# element, filename, bytes, md5, sha256, wfc, rho, family
_PSEUDO_ROWS: Final = {
    "set.sssp130-pbe-efficiency": (
        (
            "Al",
            "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
            1762723,
            "cfc449ca30b5f3223ec38ddd88ac046d",
            "fd7b78921e6d0939095b681c328732668eaefd1dee6882fdbc03be463550cc97",
            "30",
            "240",
            "100PAW",
        ),
        (
            "B",
            "b_pbe_v1.4.uspp.F.UPF",
            306486,
            "cc6de2960df11db49a60e589f9ebb39b",
            "01a3696c14b1b245b9741165ffea38120b879286cddbeb956142f8fa922941ee",
            "35",
            "280",
            "GBRV-1.4",
        ),
        (
            "C",
            "C.pbe-n-kjpaw_psl.1.0.0.UPF",
            911843,
            "5d2aebdfa2cae82b50a7e79e9516da0f",
            "9900d1efd50b9848e31849f39094b33348486b400ee51e0f3922f716137cf3d7",
            "45",
            "360",
            "100PAW",
        ),
        (
            "Cu",
            "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf",
            1855080,
            "619f40885d92a09a85a8b37550532d0c",
            "b31028b2bae60cd9903260715a49b4c6d2b6dc654558c87023fa5206e427a16d",
            "90",
            "720",
            "100PAW-low",
        ),
        (
            "Ge",
            "ge_pbe_v1.4.uspp.F.UPF",
            579995,
            "9c9eaa91e581c3f09632fb3098b2c6b2",
            "1b95046ab0f3228e0404719ca66d0309af5a62313be0b22171b963566c90430b",
            "40",
            "320",
            "GBRV-1.4",
        ),
        (
            "Mg",
            "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
            1013584,
            "24ecedc7f3e3cbe212e682f4413594e4",
            "f057a22e5d8484686a1b3fe3724aa01af4236fe8a845e5c2f772c472f107639f",
            "30",
            "240",
            "031PAW",
        ),
        (
            "Nb",
            "Nb.pbe-spn-kjpaw_psl.0.3.0.UPF",
            2052531,
            "411d72ad547312f8017e2943ceca08cc",
            "ccf07218ea34daaa63ff57d9f44ca8b4b529728cccbc8f74dd5fc804ac01e48e",
            "40",
            "320",
            "031PAW",
        ),
        (
            "Pb",
            "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
            2296427,
            "9d431e6316058b74ade52399a6cf67da",
            "b67ab5c757bc89933ccd96deecdf3b629c72783a38b6c23d78457e4abb61219d",
            "40",
            "320",
            "031PAW",
        ),
        (
            "Si",
            "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
            1299382,
            "0b0bb1205258b0d07b9f9672cf965d36",
            "ae3aefd0811f9499dbc4a72f1f9ae02ef4fc7f3568bf6f559b68668719c69e2b",
            "30",
            "240",
            "100US",
        ),
        (
            "Sn",
            "Sn_pbe_v1.uspp.F.UPF",
            597547,
            "4cf58ce39ec5d5d420df3dd08604eb00",
            "2985a215143f6458f0a627315a843123d534bff482dc09b40415af4605359431",
            "60",
            "480",
            "GBRV-1.2",
        ),
        (
            "Ta",
            "Ta_pbe_v1.uspp.F.UPF",
            792110,
            "f8bbe9446314a3b8ea5d9f3e3836c939",
            "e0850b94d7007d47691d131db1fa5ffdaaa6f6a573c1ebf1fb425e46ed891ee1",
            "45",
            "360",
            "GBRV-1.2",
        ),
        (
            "Ti",
            "ti_pbe_v1.4.uspp.F.UPF",
            577420,
            "88a00a6731bd790ddea75d31a80cb452",
            "747afa52fa17dc061e9eff72bcea534e81ffa5a4a5d33af3eb8fc9f1b3aee580",
            "35",
            "280",
            "GBRV-1.4",
        ),
        (
            "V",
            "v_pbe_v1.4.uspp.F.UPF",
            582160,
            "22b79981416ebb76fdaf5b1b8640f6fb",
            "4fc8afef98a911c82908fb9142edf71b2f598dd98762b355a6d291d834fb6d0c",
            "35",
            "280",
            "GBRV-1.4",
        ),
    ),
    "set.sssp130-pbe-precision": (
        (
            "Al",
            "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
            1762723,
            "cfc449ca30b5f3223ec38ddd88ac046d",
            "fd7b78921e6d0939095b681c328732668eaefd1dee6882fdbc03be463550cc97",
            "30",
            "240",
            "100PAW",
        ),
        (
            "B",
            "B_pbe_v1.01.uspp.F.UPF",
            306486,
            "d081ebb89d0c768e112975f650467a00",
            "7a1249cabd9c3afd7708ab1d8a10faa8f9b432831d9d2ec91ec23859e2ee0c17",
            "55",
            "440",
            "GBRV-1.2",
        ),
        (
            "C",
            "C.pbe-n-kjpaw_psl.1.0.0.UPF",
            911843,
            "5d2aebdfa2cae82b50a7e79e9516da0f",
            "9900d1efd50b9848e31849f39094b33348486b400ee51e0f3922f716137cf3d7",
            "45",
            "360",
            "100PAW",
        ),
        (
            "Cu",
            "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf",
            1855080,
            "619f40885d92a09a85a8b37550532d0c",
            "b31028b2bae60cd9903260715a49b4c6d2b6dc654558c87023fa5206e427a16d",
            "90",
            "720",
            "100PAW-low",
        ),
        (
            "Ge",
            "ge_pbe_v1.4.uspp.F.UPF",
            579995,
            "9c9eaa91e581c3f09632fb3098b2c6b2",
            "1b95046ab0f3228e0404719ca66d0309af5a62313be0b22171b963566c90430b",
            "45",
            "360",
            "GBRV-1.4",
        ),
        (
            "Mg",
            "mg_pbe_v1.4.uspp.F.UPF",
            669349,
            "8ffbd8f729fef71095aac5bd8316fb1f",
            "84a2ce998da82ed0d4e82253bbad1ee281cae4c3001fceba688e54f450c096bf",
            "45",
            "360",
            "GBRV-1.4",
        ),
        (
            "Nb",
            "Nb.pbe-spn-kjpaw_psl.0.3.0.UPF",
            2052531,
            "411d72ad547312f8017e2943ceca08cc",
            "ccf07218ea34daaa63ff57d9f44ca8b4b529728cccbc8f74dd5fc804ac01e48e",
            "40",
            "320",
            "031PAW",
        ),
        (
            "Pb",
            "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
            2296427,
            "9d431e6316058b74ade52399a6cf67da",
            "b67ab5c757bc89933ccd96deecdf3b629c72783a38b6c23d78457e4abb61219d",
            "45",
            "360",
            "031PAW",
        ),
        (
            "Si",
            "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
            1299382,
            "0b0bb1205258b0d07b9f9672cf965d36",
            "ae3aefd0811f9499dbc4a72f1f9ae02ef4fc7f3568bf6f559b68668719c69e2b",
            "30",
            "240",
            "100US",
        ),
        (
            "Sn",
            "Sn_pbe_v1.uspp.F.UPF",
            597547,
            "4cf58ce39ec5d5d420df3dd08604eb00",
            "2985a215143f6458f0a627315a843123d534bff482dc09b40415af4605359431",
            "70",
            "560",
            "GBRV-1.2",
        ),
        (
            "Ta",
            "Ta_pbe_v1.uspp.F.UPF",
            792110,
            "f8bbe9446314a3b8ea5d9f3e3836c939",
            "e0850b94d7007d47691d131db1fa5ffdaaa6f6a573c1ebf1fb425e46ed891ee1",
            "50",
            "400",
            "GBRV-1.2",
        ),
        (
            "Ti",
            "ti_pbe_v1.4.uspp.F.UPF",
            577420,
            "88a00a6731bd790ddea75d31a80cb452",
            "747afa52fa17dc061e9eff72bcea534e81ffa5a4a5d33af3eb8fc9f1b3aee580",
            "40",
            "320",
            "GBRV-1.4",
        ),
        (
            "V",
            "v_pbe_v1.4.uspp.F.UPF",
            582160,
            "22b79981416ebb76fdaf5b1b8640f6fb",
            "4fc8afef98a911c82908fb9142edf71b2f598dd98762b355a6d291d834fb6d0c",
            "40",
            "320",
            "GBRV-1.4",
        ),
    ),
    "set.sssp130-pbesol-precision": (
        (
            "Al",
            "Al.pbesol-n-kjpaw_psl.1.0.0.UPF",
            1780933,
            "3401bcfaaf5f9b08e46a00870a1ef39d",
            "801a04431015c4326c8891f7f7b31e06ab51e5c0ad32493ee7436d28e82bd7a1",
            "30",
            "240",
            "100PAW",
        ),
        (
            "B",
            "b_pbesol_v1.4.uspp.F.UPF",
            306489,
            "369b69bef2c3aeed4453bd8b804fd6a8",
            "87ce207ef55d64e2f16efa9c368a1914c1dbaf1414bd204e2d72410f4bbd2641",
            "55",
            "440",
            "GBRV-1.2",
        ),
        (
            "C",
            "C.pbesol-n-kjpaw_psl.1.0.0.UPF",
            921277,
            "5f3a56537b2bcef3348431d8311bb923",
            "d379b4cd2f440c357b310c3772554838c13124d7967e59e045429002efe7055b",
            "45",
            "360",
            "100PAW",
        ),
        (
            "Cu",
            "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf",
            1855072,
            "e985f55843d4aeb8486834e0a057895d",
            "4aaa1d2269f0696ea232b3d0b8a901d66ebf58cc9ccf9cb81c383676b8dbee6e",
            "90",
            "720",
            "100PAW-low",
        ),
        (
            "Ge",
            "ge_pbesol_v1.4.uspp.F.UPF",
            579998,
            "7bed0412183f5e553a4d27ec7023354d",
            "659ce92c49ab47a83ace5623561c7d4e557d34e1954565cbd53ba31406f288f8",
            "45",
            "360",
            "GBRV-1.4",
        ),
        (
            "Mg",
            "mg_pbesol_v1.4.uspp.F.UPF",
            669352,
            "4059f37963cf5867752dbf5ab2400678",
            "09bed4073b6c51cc09a8c19b1b2a5c32b900310386dd3f9173e1ec754b46325d",
            "45",
            "360",
            "GBRV-1.4",
        ),
        (
            "Nb",
            "Nb.pbesol-spn-kjpaw_psl.0.3.0.UPF",
            2073648,
            "3501bca1474143575298708b2d32ca28",
            "6910088685586744219051b12125da12ee78e9c70694295c3c59195ec75f4842",
            "40",
            "320",
            "031PAW",
        ),
        (
            "Pb",
            "Pb.pbesol-dn-kjpaw_psl.0.2.2.UPF",
            2320110,
            "df0e0ac60519e244e20a4bf230ae894d",
            "29bbae4cb95b640f7d62971ca84c96abe308886b9a8bdc910b27c2907a5bb808",
            "45",
            "360",
            "031PAW",
        ),
        (
            "Si",
            "Si.pbesol-n-rrkjus_psl.1.0.0.UPF",
            1591447,
            "c4212819de858c94c3a1644338846ac9",
            "82e6d35f71dfa816fb8fcc7530464af3e724703d176d885ce5b35cc341ab3e4a",
            "30",
            "240",
            "100US",
        ),
        (
            "Sn",
            "sn_pbesol_v1.4.uspp.F.UPF",
            601582,
            "463fec1d5456f1e96ec1bfcad4b7b038",
            "5fe98cd4c11ae4ae5dfd2827cb744148e3582660e9f69730b5eeb9c964c23cb1",
            "70",
            "560",
            "GBRV-1.2",
        ),
        (
            "Ta",
            "ta_pbesol_v1.uspp.F.UPF",
            792109,
            "5dd23d6859724cbb2c0d9b6294d5f84c",
            "34fb4d2ca944250bcdbec370146045115d5a495752b22b899777d8ba3ccd6b8d",
            "50",
            "400",
            "GBRV-1.2",
        ),
        (
            "Ti",
            "ti_pbesol_v1.4.uspp.F.UPF",
            577423,
            "193cf2c0c3c6a4da613d69453fa7e3e9",
            "9f50adf8d3f6a1b98981d5fb94205e53a523f405883dcb060c06b5eb93805f69",
            "40",
            "320",
            "GBRV-1.4",
        ),
        (
            "V",
            "v_pbesol_v1.4.uspp.F.UPF",
            582163,
            "72fa7d0034c41d8adc50bbc8c632b9f9",
            "b8ad92b3ca6dd6a2cf5fbc53eba1d30bd52bf311c9a9037491bb49893857689c",
            "40",
            "320",
            "GBRV-1.4",
        ),
    ),
}


def _pseudo_license(family: str) -> str:
    if family.startswith("GBRV"):
        return "license.gpl-3"
    return "license.gpl-2-or-later"


def _expected_pseudopotentials() -> tuple[PseudopotentialLock, ...]:
    values: list[PseudopotentialLock] = []
    for set_id, rows in _PSEUDO_ROWS.items():
        functional = "PBEsol" if "pbesol" in set_id else "PBE"
        for element, filename, size, checksum, digest, wfc, rho, family in rows:
            values.append(
                PseudopotentialLock(
                    pseudopotential_id=(f"pseudo.{set_id.removeprefix('set.')}.{element.lower()}"),
                    set_id=set_id,
                    element=element,
                    filename=filename,
                    size_bytes=size,
                    md5=checksum,
                    sha256=digest,
                    functional=functional,
                    family=family,
                    wavefunction_cutoff_Ry=Decimal(wfc),
                    charge_density_cutoff_Ry=Decimal(rho),
                    license_id=_pseudo_license(family),
                )
            )
    return tuple(sorted(values, key=lambda value: value.pseudopotential_id))


def expected_material_source_design_source_qualification() -> MaterialSourceDesignSourceQualification:
    return MaterialSourceDesignSourceQualification(
        qualification_id='qualification.ambient-pressure-superconductor-material-source-design-public-roster',
        source_plan_id=MATERIAL_SOURCE_DESIGN_SOURCE_PLAN_ID,
        assets=_expected_assets(),
        pseudopotentials=_expected_pseudopotentials(),
        selected_element_symbols=tuple(
            sorted(("Al", "B", "C", "Cu", "Ge", "Mg", "Nb", "Pb", "Si", "Sn", "Ta", "Ti", "V"))
        ),
        three_dsc_row_count=5_773,
        three_dsc_column_count=92,
        three_dsc_target_column_present=True,
        three_dsc_target_values_projected=False,
        xi_reference_value=Decimal("0.140605"),
        xi_reference_temperature=Decimal("3.227492"),
        xi_reference_tolerance=Decimal("0.000001"),
        credentials_required=False,
        clickthrough_required=False,
        paid_resource_required=False,
        exact_public_terms_accepted=True,
        source_values_used_for_candidate_ranking=False,
        limitations=tuple(
            sorted(
                (
                    "3dsc-corpus-is-historical-dedup-reference-not-candidate-ranker",
                    'excluded-solver-control-environment-does-not-close-material-source-design-transport-nomination-material-capabilities',
                    "arxiv-papers-are-read-and-cite-sources-not-redistributed-assets",
                    "original-http-response-headers-not-preserved-for-preexisting-downloads",
                    "sssp-pseudopotential-licenses-remain-family-specific",
                    "xi-reference-quadrature-reproduces-with-scipy-roundoff-warning",
                )
            )
        ),
    )


_ASSET_PATHS: Final = {
    'source.ambient-pressure-superconductor.3dsc-publication-2471dd51': THREE_DSC_ARCHIVE,
    'source.ambient-pressure-superconductor.excluded-solver-control-environment-image': EXCLUDED_SOLVER_CONTROL_ENVIRONMENT_IMAGE,
    'source.ambient-pressure-superconductor.epw61-archive': EPW_ARCHIVE,
    'source.ambient-pressure-superconductor.gao-2502-18281v1': GAO_PAPER,
    'source.ambient-pressure-superconductor.semenok-2407-12922v2': SEMENOK_PAPER,
    'source.ambient-pressure-superconductor.sssp130-license': SSSP_LICENSE,
    'source.ambient-pressure-superconductor.sssp130-pbe-efficiency-json': SSSP_EFFICIENCY_JSON,
    'source.ambient-pressure-superconductor.sssp130-pbe-efficiency-tar': SSSP_EFFICIENCY_TAR,
    'source.ambient-pressure-superconductor.sssp130-pbe-precision-json': SSSP_PRECISION_JSON,
    'source.ambient-pressure-superconductor.sssp130-pbe-precision-tar': SSSP_PRECISION_TAR,
    'source.ambient-pressure-superconductor.sssp130-pbesol-precision-json': SSSP_PBESOL_PRECISION_JSON,
    'source.ambient-pressure-superconductor.sssp130-pbesol-precision-tar': SSSP_PBESOL_PRECISION_TAR,
    'source.ambient-pressure-superconductor.sssp130-readme': SSSP_README,
    'source.ambient-pressure-superconductor.xi-code-v1-1': XI_CODE,
}


def _hash_regular(path: Path, *, expected_size: int) -> tuple[str, str]:
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f'material source design source is not a regular non-symlink file: {path}')
    if observed.st_size != expected_size:
        raise ValueError(f'material source design source size differs: {path}')
    sha = sha256()
    md5_digest = md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            sha.update(block)
            md5_digest.update(block)
    return sha.hexdigest(), md5_digest.hexdigest()


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _tar_inventory(path: Path) -> tuple[int, int, int, int, int, int]:
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        regular = tuple(member for member in members if member.isfile())
        directories = tuple(member for member in members if member.isdir())
        symlinks = tuple(member for member in members if member.issym())
        if any(not _safe_member_name(member.name) for member in members):
            raise ValueError('material source design tar archive contains an unsafe member name')
        if any(member.islnk() for member in members):
            raise ValueError('material source design tar archive contains a hard link')
        if any(not (member.isfile() or member.isdir() or member.issym()) for member in members):
            raise ValueError('material source design tar archive contains an unsupported member type')
        # SSSP archives legitimately begin with a POSIX ``.`` directory whose
        # PurePosixPath has no parts.  A root is needed only for symlink
        # containment, so derive it from the first non-dot member.
        root = next(
            (parts[0] for member in members if (parts := PurePosixPath(member.name).parts)),
            "",
        )
        for member in symlinks:
            parent = PurePosixPath(member.name).parent
            target = parent.joinpath(member.linkname)
            parts: list[str] = []
            for part in target.parts:
                if part in {"", "."}:
                    continue
                if part == "..":
                    if not parts:
                        raise ValueError('material source design tar symlink escapes its archive root')
                    parts.pop()
                else:
                    parts.append(part)
            if not parts or parts[0] != root:
                raise ValueError('material source design tar symlink escapes its archive root')
        return (
            len(members),
            len(regular),
            len(directories),
            len(symlinks),
            sum(member.size for member in regular),
            max((member.size for member in regular), default=0),
        )


def _zip_inventory(path: Path) -> tuple[int, int, int, int, int, int]:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if any(not _safe_member_name(member.filename) for member in members):
            raise ValueError('material source design zip archive contains an unsafe member name')
        directories = tuple(member for member in members if member.is_dir())
        regular = tuple(member for member in members if not member.is_dir())
        return (
            len(members),
            len(regular),
            len(directories),
            0,
            sum(member.file_size for member in regular),
            max((member.file_size for member in regular), default=0),
        )


def _verify_tar_members(path: Path, locks: tuple[SourceMemberLock, ...]) -> None:
    if not locks:
        return
    with tarfile.open(path, mode="r:gz") as archive:
        for lock in locks:
            member = archive.getmember(lock.member_locator)
            if not member.isfile() or member.size != lock.size_bytes:
                raise ValueError(f'material source design tar member differs: {lock.member_id}')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'material source design tar member cannot be opened: {lock.member_id}')
            payload = stream.read(lock.size_bytes + 1)
            if len(payload) != lock.size_bytes or sha256(payload).hexdigest() != lock.sha256:
                raise ValueError(f'material source design tar member digest differs: {lock.member_id}')


def _verify_zip_members(path: Path, locks: tuple[SourceMemberLock, ...]) -> None:
    with zipfile.ZipFile(path) as archive:
        for lock in locks:
            info = archive.getinfo(lock.member_locator)
            if info.is_dir() or info.file_size != lock.size_bytes:
                raise ValueError(f'material source design zip member differs: {lock.member_id}')
            payload = archive.read(info)
            if len(payload) != lock.size_bytes or sha256(payload).hexdigest() != lock.sha256:
                raise ValueError(f'material source design zip member digest differs: {lock.member_id}')


def _three_dsc_shape() -> tuple[int, int, bool]:
    archive_path = required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT") / THREE_DSC_ARCHIVE
    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = archive.getmember(THREE_DSC_CSV_MEMBER)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("3DSC structural projection member cannot be opened")
        text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
        comment = text.readline()
        if not comment.startswith("# "):
            raise ValueError("3DSC CSV comment preamble differs")
        reader = csv.reader(text)
        header = next(reader)
        target_present = "tc" in header
        row_count = 0
        for row in reader:
            # Count and shape-check rows without copying, storing or exposing values.
            if len(row) != len(header):
                raise ValueError("3DSC CSV row width differs")
            row_count += 1
        return row_count, len(header), target_present


_SSSP_BINDINGS: Final = {
    "set.sssp130-pbe-efficiency": (SSSP_EFFICIENCY_JSON, SSSP_EFFICIENCY_TAR),
    "set.sssp130-pbe-precision": (SSSP_PRECISION_JSON, SSSP_PRECISION_TAR),
    "set.sssp130-pbesol-precision": (
        SSSP_PBESOL_PRECISION_JSON,
        SSSP_PBESOL_PRECISION_TAR,
    ),
}


def _verify_pseudopotentials(expected: tuple[PseudopotentialLock, ...]) -> None:
    root = required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT")
    by_set: dict[str, list[PseudopotentialLock]] = {}
    for lock in expected:
        by_set.setdefault(lock.set_id, []).append(lock)
    for set_id, locks in by_set.items():
        json_relative, tar_relative = _SSSP_BINDINGS[set_id]
        json_path, tar_path = root / json_relative, root / tar_relative
        document = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("SSSP JSON root differs")
        with tarfile.open(tar_path, mode="r:gz") as archive:
            for lock in locks:
                metadata = document.get(lock.element)
                if not isinstance(metadata, dict):
                    raise ValueError(f'SSSP element metadata missing: {set_id}/{lock.element}')
                expected_metadata = {
                    "filename": lock.filename,
                    "md5": lock.md5,
                    # JSON numbers arrive as binary floats.  Convert through their
                    # decimal spelling so an exact published cutoff such as 30 Ry
                    # remains Decimal("30") rather than its binary approximation.
                    "cutoff_wfc": Decimal(str(metadata["cutoff_wfc"])),
                    "cutoff_rho": Decimal(str(metadata["cutoff_rho"])),
                }
                if (
                    metadata.get("filename") != expected_metadata["filename"]
                    or metadata.get("md5") != expected_metadata["md5"]
                    or expected_metadata["cutoff_wfc"] != lock.wavefunction_cutoff_Ry
                    or expected_metadata["cutoff_rho"] != lock.charge_density_cutoff_Ry
                    or metadata.get("pseudopotential") != lock.family
                ):
                    raise ValueError(f'SSSP metadata differs: {set_id}/{lock.element}')
                member = archive.getmember(f'./{lock.filename}')
                if not member.isfile() or member.size != lock.size_bytes:
                    raise ValueError(f'SSSP pseudo size differs: {set_id}/{lock.element}')
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f'SSSP pseudo cannot be opened: {set_id}/{lock.element}')
                payload = stream.read(lock.size_bytes + 1)
                if (
                    len(payload) != lock.size_bytes
                    or md5(payload, usedforsecurity=False).hexdigest() != lock.md5
                    or sha256(payload).hexdigest() != lock.sha256
                ):
                    raise ValueError(f'SSSP pseudo digest differs: {set_id}/{lock.element}')


def _verify_xi_reference(expected: MaterialSourceDesignSourceQualification) -> None:
    """Reproduce the locked Hg fixture without executing downloaded code."""

    import mpmath
    import numpy as np
    from scipy import integrate, interpolate, optimize
    from scipy.integrate import IntegrationWarning
    import warnings

    data_member = "eyuzbash-stability-parameter-xi-f05ec2c/data/a2F_Hg_0GPa.txt"
    xi_path = required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT") / XI_CODE
    with zipfile.ZipFile(xi_path) as archive:
        data = np.loadtxt(io.BytesIO(archive.read(data_member)))
    omega_values = data[:, 0]
    alpha2f_values = data[:, 1]
    alpha2f = interpolate.interp1d(
        omega_values,
        alpha2f_values,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    mpmath.mp.dps = 25
    x_table = np.linspace(1e-6, 20, 20_000)
    g_table = np.array(
        [
            float(
                6 * x**2
                + 12 * x**3 * mpmath.im(mpmath.polygamma(1, 1j * x))
                + 6 * x**4 * mpmath.re(mpmath.polygamma(2, 1j * x))
            )
            for x in x_table
        ]
    )
    g = interpolate.interp1d(
        x_table,
        g_table,
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
    )
    omega_min = float(omega_values.min())
    omega_max = float(omega_values.max())
    split_points = np.unique(
        np.concatenate(
            [
                np.logspace(np.log10(omega_min), np.log10(0.003), 5),
                np.linspace(0.003, omega_max, 4),
            ]
        )
    )

    def xi_at(temperature: float) -> float:
        total = 0.0
        for lower, upper in zip(split_points[:-1], split_points[1:], strict=True):
            # The reference implementation emits a roundoff warning on the
            # locked Hg fixture.  It is recorded in the qualification limits;
            # exact six-place output reproduction remains the adjudicated check.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IntegrationWarning)
                part, _error = integrate.quad(
                    lambda omega: (
                        g(omega / (2 * np.pi * temperature)) * (2 * alpha2f(omega) / omega)
                    ),
                    lower,
                    upper,
                    limit=200,
                )
            total += float(part)
        return total

    optimum = optimize.minimize_scalar(
        lambda temperature: -xi_at(temperature),
        bounds=(1e-5, omega_max / 2),
        method="bounded",
    )
    observed_temperature = Decimal(f'{optimum.x:.6f}')
    observed_xi = Decimal(f'{-optimum.fun:.6f}')
    if (
        abs(observed_temperature - expected.xi_reference_temperature)
        > expected.xi_reference_tolerance
        or abs(observed_xi - expected.xi_reference_value) > expected.xi_reference_tolerance
    ):
        raise ValueError("locked xi reference fixture does not reproduce")


def inspect_material_source_design_external_sources() -> MaterialSourceDesignSourceQualification:
    'Reopen and verify every exact source needed at the material source design freeze boundary.'

    expected = expected_material_source_design_source_qualification()
    root = required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT")
    for asset in expected.assets:
        path = root / _ASSET_PATHS[asset.source_id]
        observed_sha, observed_md5 = _hash_regular(path, expected_size=asset.size_bytes)
        if observed_sha != asset.content_sha256:
            raise ValueError(f'material source design source SHA-256 differs: {asset.source_id}')
        if asset.published_checksum.startswith("md5:") and (
            observed_md5 != asset.published_checksum.removeprefix("md5:")
        ):
            raise ValueError(f'material source design published MD5 differs: {asset.source_id}')
        expected_inventory = (
            asset.archive_member_count,
            asset.archive_regular_file_count,
            asset.archive_directory_count,
            asset.archive_symlink_count,
            asset.archive_expanded_regular_bytes,
            asset.archive_maximum_member_bytes,
        )
        if asset.relative_locator.endswith(".tar.gz"):
            if _tar_inventory(path) != expected_inventory:
                raise ValueError(f'material source design tar inventory differs: {asset.source_id}')
            _verify_tar_members(path, asset.member_locks)
        elif asset.relative_locator.endswith(".zip"):
            if _zip_inventory(path) != expected_inventory:
                raise ValueError(f'material source design zip inventory differs: {asset.source_id}')
            _verify_zip_members(path, asset.member_locks)
        elif any(expected_inventory):
            raise ValueError(f'nonarchive material source design asset has archive inventory: {asset.source_id}')
    _verify_pseudopotentials(expected.pseudopotentials)
    _verify_xi_reference(expected)
    rows, columns, target_present = _three_dsc_shape()
    if (
        rows != expected.three_dsc_row_count
        or columns != expected.three_dsc_column_count
        or target_present != expected.three_dsc_target_column_present
    ):
        raise ValueError("3DSC structural-only projection shape differs")
    return expected


__all__ = [
    'EXCLUDED_SOLVER_CONTROL_ENVIRONMENT_IMAGE',
    "EPW_ARCHIVE",
    "GAO_PAPER",
    "SEMENOK_PAPER",
    "SSSP_EFFICIENCY_JSON",
    "SSSP_EFFICIENCY_TAR",
    "SSSP_PBESOL_PRECISION_JSON",
    "SSSP_PBESOL_PRECISION_TAR",
    "SSSP_PRECISION_JSON",
    "SSSP_PRECISION_TAR",
    "THREE_DSC_ARCHIVE",
    "XI_CODE",
    'expected_material_source_design_source_qualification',
    'inspect_material_source_design_external_sources',
]
