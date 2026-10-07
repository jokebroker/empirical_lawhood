"""Deterministic, outcome-blind VCC core-input materialization."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
import math
from typing import BinaryIO, ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .codecs import encode_target_features
from .config import (
    ENSEMBL_113_GTF_SHA256,
    ENSEMBL_113_PEPTIDE_SHA256,
)
from .contracts import (
    VirtualCellContractError,
    VirtualCellSourceManifest,
    VirtualCellSourceObject,
    VirtualCellSplit,
)
from .features import TargetFeatureBuildReceipt, build_ensembl_target_features
from .ports import VirtualCellSourcePort


_MAXIMUM_ROSTER_BYTES = 1024**2
_MAXIMUM_GTF_BYTES = 128 * 1024**2
_MAXIMUM_PEPTIDE_BYTES = 64 * 1024**2
_ROSTER_HEADER = ("target_gene", "n_cells", "median_umi_per_cell")
_EXPECTED_TARGET_COUNTS = {
    VirtualCellSplit.TRAIN: 150,
    VirtualCellSplit.VALIDATION: 50,
    VirtualCellSplit.TEST: 100,
}


@dataclass(frozen=True, slots=True)
class TargetFeatureMaterializationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/target-feature-materialization-receipt'
    )

    receipt_id: str
    source_manifest_sha256: str
    roster_source_object_ids: tuple[str, ...]
    roster_source_sha256s: tuple[str, ...]
    build_receipt_sha256: str
    arrow_sha256: str
    arrow_size_bytes: int
    target_count: int
    feature_count: int
    feature_provenance_sha256: str
    outcome_access: OutcomeAccess
    response_derived: bool
    post_cutoff: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        for digest_name, digest_value in (
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("build_receipt_sha256", self.build_receipt_sha256),
            ("arrow_sha256", self.arrow_sha256),
            ("feature_provenance_sha256", self.feature_provenance_sha256),
        ):
            validate_sha256(digest_value, field_name=digest_name)
        require_sorted_unique_strings(
            self.roster_source_object_ids,
            field_name="roster_source_object_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.roster_source_sha256s,
            field_name="roster_source_sha256s",
            allow_empty=False,
        )
        for value in self.roster_source_sha256s:
            validate_sha256(value, field_name="roster_source_sha256s")
        if (
            len(self.roster_source_object_ids) != 3
            or len(self.roster_source_sha256s) != 3
        ):
            raise ValueError(
                "target-feature materialization requires all three split rosters"
            )
        for count_name, count_value in (
            ("arrow_size_bytes", self.arrow_size_bytes),
            ("target_count", self.target_count),
            ("feature_count", self.feature_count),
        ):
            if isinstance(count_value, bool) or count_value <= 0:
                raise ValueError(f"{count_name} must be positive")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("target features must remain outcome blind")
        if self.response_derived or self.post_cutoff:
            raise ValueError("target features must be response-blind and pre-cutoff")


@dataclass(frozen=True, slots=True)
class TargetFeatureMaterialization:
    arrow_payload: bytes
    build_receipt: TargetFeatureBuildReceipt
    materialization_receipt: TargetFeatureMaterializationReceipt


def _bounded_payload(stream: BinaryIO, *, maximum_bytes: int, label: str) -> bytes:
    payload = stream.read(maximum_bytes + 1)
    if not isinstance(payload, bytes) or not payload:
        raise VirtualCellContractError(f"{label} is empty or not binary")
    if len(payload) > maximum_bytes:
        raise VirtualCellContractError(f"{label} exceeds its frozen byte bound")
    if stream.read(1) != b"":
        raise VirtualCellContractError(f"{label} exceeds its frozen byte bound")
    return payload


def decode_perturbation_roster(
    stream: BinaryIO,
    *,
    split: VirtualCellSplit,
) -> tuple[str, ...]:
    """Decode only the three legal prefix fields from one bounded split roster."""

    if split not in _EXPECTED_TARGET_COUNTS:
        raise ValueError("perturbation roster requires TRAIN, VALIDATION or TEST")
    payload = _bounded_payload(
        stream,
        maximum_bytes=_MAXIMUM_ROSTER_BYTES,
        label=f"{split.value} perturbation roster",
    )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise VirtualCellContractError("perturbation roster is not UTF-8") from error
    reader = csv.DictReader(StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != _ROSTER_HEADER:
        raise VirtualCellContractError("perturbation roster header differs")
    targets: list[str] = []
    for row in reader:
        if None in row or set(row) != set(_ROSTER_HEADER):
            raise VirtualCellContractError("perturbation roster row is malformed")
        target = row["target_gene"]
        try:
            cells = int(row["n_cells"])
            median = float(row["median_umi_per_cell"])
        except (TypeError, ValueError) as error:
            raise VirtualCellContractError(
                "perturbation roster values are malformed"
            ) from error
        if not target or cells <= 0 or not math.isfinite(median) or median <= 0:
            raise VirtualCellContractError("perturbation roster values are invalid")
        targets.append(target)
    expected = _EXPECTED_TARGET_COUNTS[split]
    if len(targets) != expected or len(set(targets)) != expected:
        raise VirtualCellContractError("perturbation roster cardinality differs")
    return tuple(sorted(targets))


def _roster_source(
    manifest: VirtualCellSourceManifest,
    split: VirtualCellSplit,
) -> VirtualCellSourceObject:
    values = tuple(
        value
        for value in manifest.objects
        if value.split is split and value.role == "prefix/target-roster"
    )
    if len(values) != 1 or values[0].sealed:
        raise VirtualCellContractError(
            "source manifest lacks one outcome-blind split roster"
        )
    return values[0]


def build_target_feature_materialization(
    *,
    expected: TargetFeatureMaterializationReceipt,
    source_manifest: VirtualCellSourceManifest,
    source_port: VirtualCellSourcePort,
    gtf_stream: BinaryIO,
    peptide_fasta_stream: BinaryIO,
) -> TargetFeatureMaterialization:
    """Build and requalify the exact 300-target Ensembl-113 Arrow artifact."""

    if expected.source_manifest_sha256 != source_manifest.fingerprint():
        raise VirtualCellContractError("explicit feature freeze binds another source")
    roster_sources = tuple(
        _roster_source(source_manifest, split)
        for split in (
            VirtualCellSplit.TRAIN,
            VirtualCellSplit.VALIDATION,
            VirtualCellSplit.TEST,
        )
    )
    split_targets: list[tuple[str, ...]] = []
    for source in roster_sources:
        with source_port.open_object(
            source,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ) as stream:
            split_targets.append(decode_perturbation_roster(stream, split=source.split))
    target_ids = tuple(sorted(value for values in split_targets for value in values))
    if len(target_ids) != 300 or len(set(target_ids)) != 300:
        raise VirtualCellContractError(
            "combined target roster is not the exact disjoint 300"
        )
    gtf_payload = _bounded_payload(
        gtf_stream,
        maximum_bytes=_MAXIMUM_GTF_BYTES,
        label="Ensembl 113 GTF",
    )
    peptide_payload = _bounded_payload(
        peptide_fasta_stream,
        maximum_bytes=_MAXIMUM_PEPTIDE_BYTES,
        label="Ensembl 113 peptide FASTA",
    )
    if sha256(gtf_payload).hexdigest() != ENSEMBL_113_GTF_SHA256:
        raise VirtualCellContractError("Ensembl 113 GTF SHA-256 differs")
    if sha256(peptide_payload).hexdigest() != ENSEMBL_113_PEPTIDE_SHA256:
        raise VirtualCellContractError("Ensembl 113 peptide SHA-256 differs")
    matrix, build_receipt = build_ensembl_target_features(
        target_ids=target_ids,
        gtf_stream=BytesIO(gtf_payload),
        peptide_fasta_stream=BytesIO(peptide_payload),
        gtf_sha256=ENSEMBL_113_GTF_SHA256,
        peptide_fasta_sha256=ENSEMBL_113_PEPTIDE_SHA256,
    )
    arrow_payload = encode_target_features(matrix)
    if matrix.provenance_sha256 != expected.feature_provenance_sha256:
        raise VirtualCellContractError(
            "target-feature provenance differs from the freeze"
        )
    if build_receipt.fingerprint() != expected.build_receipt_sha256:
        raise VirtualCellContractError(
            "target-feature build receipt differs from the freeze"
        )
    if (
        len(arrow_payload) != expected.arrow_size_bytes
        or sha256(arrow_payload).hexdigest() != expected.arrow_sha256
    ):
        raise VirtualCellContractError(
            "target-feature Arrow bytes differ from the freeze"
        )
    receipt = TargetFeatureMaterializationReceipt(
        receipt_id="materialization.virtual-cell-2025-ensembl-113-features",
        source_manifest_sha256=source_manifest.fingerprint(),
        roster_source_object_ids=tuple(
            sorted(value.object_id for value in roster_sources)
        ),
        roster_source_sha256s=tuple(sorted(value.sha256 for value in roster_sources)),
        build_receipt_sha256=build_receipt.fingerprint(),
        arrow_sha256=sha256(arrow_payload).hexdigest(),
        arrow_size_bytes=len(arrow_payload),
        target_count=len(matrix.target_ids),
        feature_count=len(matrix.feature_ids),
        feature_provenance_sha256=matrix.provenance_sha256,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        response_derived=False,
        post_cutoff=False,
    )
    return TargetFeatureMaterialization(
        arrow_payload=arrow_payload,
        build_receipt=build_receipt,
        materialization_receipt=receipt,
    )


__all__ = [
    "TargetFeatureMaterialization",
    "TargetFeatureMaterializationReceipt",
    "build_target_feature_materialization",
    "decode_perturbation_roster",
]
