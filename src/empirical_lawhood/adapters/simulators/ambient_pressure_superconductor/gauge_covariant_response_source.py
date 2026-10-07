'Exact public-source extension for ambient pressure superconductor amendment gauge covariant response.'

from __future__ import annotations

from empirical_lawhood._required_inputs import required_external_path

from hashlib import sha256
from pathlib import Path
import stat
from typing import Final

from .gauge_covariant_response_contracts import GaugeCovariantResponseSourceQualification
from .material_source_design_contracts import SourceAssetLock, SourceOutcomeRole
from .material_source_design_source import _tar_inventory


SOURCE_RELATIVE_ROOT: Final = 'sources/ambient-pressure-superconductor/gauge-covariant-response-staged'


def _lock(
    *,
    source_id: str,
    role_id: str,
    release_id: str,
    filename: str,
    public_locator: str,
    content_sha256: str,
    size_bytes: int,
    published_checksum: str,
    license_id: str,
    license_scope: str,
    outcome_role: SourceOutcomeRole,
    inventory: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    checks: tuple[str, ...],
) -> SourceAssetLock:
    return SourceAssetLock(
        source_id=source_id,
        role_id=role_id,
        release_id=release_id,
        relative_locator=f'{SOURCE_RELATIVE_ROOT}/{filename}',
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
        member_locks=(),
        qualification_checks=tuple(sorted(checks)),
    )


def _assets() -> tuple[SourceAssetLock, ...]:
    values = (
        _lock(
            source_id='source.ambient-pressure-superconductor.gauge-covariant-response-hiorth-2603-10955v1',
            role_id='role.uniform-pairing-current-ab-initio-stiffness-method',
            release_id="release.arxiv-2603-10955v1",
            filename="hiorth-2603.10955v1.pdf",
            public_locator="https://arxiv.org/abs/2603.10955v1",
            content_sha256="91ff97d8d733340bc044128462ff856e8549ca92e130c7017d670fc153f91a7d",
            size_bytes=3_409_375,
            published_checksum="arxiv:2603.10955v1",
            license_id="license.arxiv-read-and-cite",
            license_scope="Read-only cited method evidence; no paper content is redistributed.",
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            checks=("exact-arxiv-version-verified", "pdf-physically-hashed"),
        ),
        _lock(
            source_id='source.ambient-pressure-superconductor.gauge-covariant-response-materials-project-phase-diagrams',
            role_id="role.zero-kelvin-phase-diagram-method-documentation",
            release_id="release.materials-project-docs-2026-08-02",
            filename="materials-project-phase-diagrams.html",
            public_locator=(
                "https://docs.materialsproject.org/methodology/materials-methodology/"
                "thermodynamic-stability/phase-diagrams-pds"
            ),
            content_sha256="ddbee1419ea1cb4b0ae4a1f5caed47a565a5dcb43c010d5704aaee159f6b4e3e",
            size_bytes=727_419,
            published_checksum="qualified-local-html:2026-08-02",
            license_id="license.website-read-and-cite",
            license_scope=(
                "Read-only cited method documentation; no API access or target values are used."
            ),
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            checks=("html-physically-hashed", "method-only-role-verified"),
        ),
        _lock(
            source_id='source.ambient-pressure-superconductor.gauge-covariant-response-sctk-1-2-1',
            role_id="role.independent-finite-temperature-scdft-method",
            release_id="release.sctk-1-2-1-qe6-7",
            filename="sctk-1.2.1-qe6.7.tar.gz",
            public_locator="https://github.com/mitsuaki1987/sctk/tree/sctk1.2.1-qe6.7",
            content_sha256="68039d62b21294228cdc20a498f7086d5f0359e4870711d3dd31895490b7ba61",
            size_bytes=1_755_853,
            published_checksum="git:14b4958eb6c247e03f2704045221dd8508f9fdef",
            license_id="license.gpl-3-or-later",
            license_scope="SCTK source and documentation under GPL-3-or-later.",
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            inventory=(143, 130, 13, 0, 6_620_866, 1_024_028),
            checks=(
                "archive-bounds-verified",
                "git-tag-commit-verified",
                "license-verified",
            ),
        ),
        _lock(
            source_id='source.ambient-pressure-superconductor.gauge-covariant-response-superconga-1-1-0',
            role_id="role.secondary-finite-receiver-method",
            release_id="release.superconga-1-1-0",
            filename="superconga-v1.1.0.tar.gz",
            public_locator="https://gitlab.com/superconga/superconga/-/tags/v1.1.0",
            content_sha256="caa606e71128bd77b159496e3bfb9ee9d252f128fe7fc626bb75e36d4b424dbc",
            size_bytes=2_109_149,
            published_checksum="git:59dc5d2bc51cc50910a484e6c20bc6869a07b4f0",
            license_id="license.lgpl-3-0",
            license_scope="SuperConga source under LGPL-3.0; secondary receiver use only.",
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            inventory=(310, 250, 60, 0, 5_143_955, 1_021_073),
            checks=(
                "archive-bounds-verified",
                "git-tag-commit-verified",
                "license-verified",
            ),
        ),
        _lock(
            source_id='source.ambient-pressure-superconductor.gauge-covariant-response-text-mined-synthesis-fca0a994',
            role_id="role.nonpromoting-synthesis-method-reference",
            release_id="release.text-mined-synthesis-fca0a994",
            filename="text-mined-synthesis-fca0a994.tar.gz",
            public_locator="https://github.com/CederGroupHub/text-mined-synthesis_public",
            content_sha256="269122a1b133a861391cd3f7c4809a741de631e2d2ea5c66e6ebe1b1514e078e",
            size_bytes=63_796_820,
            published_checksum="git:fca0a994ddb36b37d39b803e8a21880aaecb1d71",
            license_id="license.unspecified-read-and-cite-only",
            license_scope=(
                "No explicit repository licence was found; retained only for read/cite method "
                "inspection and prohibited from promoting synthesis reachability."
            ),
            outcome_role=SourceOutcomeRole.HISTORICAL_OUTCOMES_NOT_PROJECTED,
            inventory=(87, 65, 22, 0, 79_019_035, 24_246_985),
            checks=(
                "archive-bounds-verified",
                "git-commit-verified",
                "license-absence-recorded",
                "promoting-route-use-prohibited",
            ),
        ),
        _lock(
            source_id='source.ambient-pressure-superconductor.gauge-covariant-response-wannier90-3-1-0',
            role_id="role.dft-to-tight-binding-compatibility-method",
            release_id="release.wannier90-3-1-0",
            filename="wannier90-v3.1.0.tar.gz",
            public_locator="https://github.com/wannier-developers/wannier90/releases/tag/v3.1.0",
            content_sha256="40651a9832eb93dec20a8360dd535262c261c34e13c41b6755fa6915c936b254",
            size_bytes=101_211_573,
            published_checksum="git:1d6b187374a2d50b509e5e79e2cab01a79ff7ce1",
            license_id="license.gpl-2-0",
            license_scope="Wannier90 source under GPL-2.0.",
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            inventory=(1_434, 1_116, 218, 100, 171_168_540, 15_170_909),
            checks=(
                "archive-bounds-verified",
                "contained-relative-symlinks-verified",
                "git-tag-commit-verified",
                "license-verified",
            ),
        ),
        _lock(
            source_id='source.ambient-pressure-superconductor.gauge-covariant-response-watanabe-2501-13722v2',
            role_id="role.generalized-cfop-ward-method",
            release_id="release.arxiv-2501-13722v2",
            filename="watanabe-2501.13722v2.pdf",
            public_locator="https://arxiv.org/abs/2501.13722v2",
            content_sha256="ffa6c31503faf622a32eef461f14ab7b2f446e3b0ec8da4b55a84de093013f53",
            size_bytes=2_062_765,
            published_checksum="arxiv:2501.13722v2",
            license_id="license.arxiv-read-and-cite",
            license_scope="Read-only cited method evidence; no paper content is redistributed.",
            outcome_role=SourceOutcomeRole.METHOD_EVIDENCE,
            checks=("exact-arxiv-version-verified", "pdf-physically-hashed"),
        ),
    )
    return tuple(sorted(values, key=lambda value: value.source_id))


def expected_gauge_covariant_response_source_qualification(
    *, base_source_sha256: str
) -> GaugeCovariantResponseSourceQualification:
    return GaugeCovariantResponseSourceQualification(
        qualification_id='qualification.ambient-pressure-superconductor-gauge-covariant-response-gauge-covariant-response-extension',
        base_source_qualification_sha256=base_source_sha256,
        extension_assets=_assets(),
        credentials_required=False,
        clickthrough_required=False,
        paid_resource_required=False,
        exact_public_use_terms_accepted=True,
        material_target_values_projected=False,
        synthesis_corpus_promoting_route_evidence=False,
        limitations=tuple(
            sorted(
                (
                    'limitation.gauge-covariant-response-materials-project-document-is-zero-kelvin-method-only',
                    'limitation.gauge-covariant-response-uniform-pairing-current-zero-temperature-uniform-pairing-is-nonpromoting',
                    'limitation.gauge-covariant-response-sctk-independent-method-is-not-installed-in-qualified-environment',
                    'limitation.gauge-covariant-response-superconga-requires-gpu-and-is-secondary-receiver-only',
                    'limitation.gauge-covariant-response-synthesis-corpus-has-no-explicit-license-and-cannot-promote-route',
                    'limitation.gauge-covariant-response-source-acquisition-http-headers-not-preserved',
                    'limitation.gauge-covariant-response-wannier-sources-do-not-supply-material-specific-orbitals',
                )
            )
        ),
    )


def _hash_regular(path: Path, expected_size: int) -> str:
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f'gauge covariant response source is not a regular non-symlink file: {path}')
    if observed.st_size != expected_size:
        raise ValueError(f'gauge covariant response source size differs: {path}')
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def inspect_gauge_covariant_response_external_sources(*, base_source_sha256: str) -> GaugeCovariantResponseSourceQualification:
    expected = expected_gauge_covariant_response_source_qualification(base_source_sha256=base_source_sha256)
    external_root = required_external_path("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT")
    root = external_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError('gauge covariant response external scientific root is unavailable')
    for asset in expected.extension_assets:
        path = external_root / asset.relative_locator
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError('gauge covariant response source resolves outside the external scientific root')
        if _hash_regular(path, asset.size_bytes) != asset.content_sha256:
            raise ValueError(f'gauge covariant response source digest differs: {asset.source_id}')
        if asset.archive_member_count:
            observed_inventory = _tar_inventory(path)
            expected_inventory = (
                asset.archive_member_count,
                asset.archive_regular_file_count,
                asset.archive_directory_count,
                asset.archive_symlink_count,
                asset.archive_expanded_regular_bytes,
                asset.archive_maximum_member_bytes,
            )
            if observed_inventory != expected_inventory:
                raise ValueError(f'gauge covariant response archive inventory differs: {asset.source_id}')
    return expected


__all__ = [
    "SOURCE_RELATIVE_ROOT",
    'expected_gauge_covariant_response_source_qualification',
    'inspect_gauge_covariant_response_external_sources',
]
