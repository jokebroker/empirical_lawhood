"""Historically eligible, outcome-blind Ensembl target features.

The transform is deliberately small and deterministic.  It derives gene-span,
chromosome/strand and protein-composition features from Ensembl release 113;
no perturbation response, leaderboard value or fitted foundation model enters
the feature matrix.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import gzip
import hashlib
import math
import re
from typing import BinaryIO, ClassVar

import numpy as np

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .analysis import TargetFeatureMatrix
from .contracts import VirtualCellContractError


_TRANSFORM_ID = "ensembl-113-gene-protein-composition"
_MAX_GTF_DECOMPRESSED_BYTES = 4 * 1024**3
_MAX_FASTA_DECOMPRESSED_BYTES = 1024**3
_MAX_LINE_BYTES = 4 * 1024**2
_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
# No synonym is silently substituted.  In particular, the 2025 roster's TAZ
# is intentionally left unmapped: mapping it to the WWTR1 protein also called
# "TAZ" would conflate distinct genes.
_TARGET_ALIASES: dict[str, str] = {}
_ATTRIBUTE_PATTERN = re.compile(r'(\S+) "([^"]*)"')


@dataclass(frozen=True, slots=True)
class _GeneAnnotation:
    gene_id: str
    gene_name: str
    chromosome: str
    strand: str
    start: int
    end: int
    gene_biotype: str


@dataclass(frozen=True, slots=True)
class TargetFeatureMapping(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/target-feature-mapping'

    target_id: str
    mapped_gene_name: str
    ensembl_gene_id: str | None
    protein_sequence_available: bool
    alias_applied: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_nonempty(self.target_id, field_name="target_id")
        validate_nonempty(self.mapped_gene_name, field_name="mapped_gene_name")
        if self.ensembl_gene_id is not None:
            validate_nonempty(self.ensembl_gene_id, field_name="ensembl_gene_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.ensembl_gene_id is None and not self.reason_codes:
            raise ValueError("unmapped target feature requires a reason")
        if self.protein_sequence_available and self.ensembl_gene_id is None:
            raise ValueError("protein sequence cannot exist without a gene mapping")


@dataclass(frozen=True, slots=True)
class TargetFeatureBuildReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/target-feature-build-receipt'

    receipt_id: str
    transform_id: str
    ensembl_release: str
    gtf_sha256: str
    peptide_fasta_sha256: str
    target_roster_sha256: str
    feature_ids_sha256: str
    matrix_sha256: str
    mappings: tuple[TargetFeatureMapping, ...]
    mapped_gene_count: int
    protein_covered_count: int
    response_derived: bool
    post_cutoff: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.transform_id, field_name="transform_id")
        validate_stable_id(self.ensembl_release, field_name="ensembl_release")
        for name, value in (
            ("gtf_sha256", self.gtf_sha256),
            ("peptide_fasta_sha256", self.peptide_fasta_sha256),
            ("target_roster_sha256", self.target_roster_sha256),
            ("feature_ids_sha256", self.feature_ids_sha256),
            ("matrix_sha256", self.matrix_sha256),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_ids(self.mappings, attribute="target_id", field_name="mappings")
        if self.mapped_gene_count != sum(
            value.ensembl_gene_id is not None for value in self.mappings
        ):
            raise ValueError("mapped gene count is not derived from mappings")
        if self.protein_covered_count != sum(
            value.protein_sequence_available for value in self.mappings
        ):
            raise ValueError("protein coverage count is not derived from mappings")
        if self.response_derived or self.post_cutoff:
            raise ValueError("Lane H target features must be response-blind and pre-cutoff")


def _registry_digest(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _bounded_text_lines(
    stream: BinaryIO,
    *,
    maximum_decompressed_bytes: int,
) -> Iterator[str]:
    observed = 0
    try:
        with gzip.GzipFile(fileobj=stream, mode="rb") as decompressed:
            for raw in decompressed:
                if not isinstance(raw, bytes) or len(raw) > _MAX_LINE_BYTES:
                    raise VirtualCellContractError("feature-source line exceeds its bound")
                observed += len(raw)
                if observed > maximum_decompressed_bytes:
                    raise VirtualCellContractError("feature source exceeds decompressed byte bound")
                try:
                    yield raw.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as error:
                    raise VirtualCellContractError("feature source is not UTF-8") from error
    except (EOFError, OSError) as error:
        raise VirtualCellContractError(
            "feature source is not a valid complete gzip stream"
        ) from error


def _attributes(raw: str) -> dict[str, str]:
    return {key: value for key, value in _ATTRIBUTE_PATTERN.findall(raw)}


def _read_gene_annotations(
    stream: BinaryIO,
    *,
    requested_names: frozenset[str],
) -> dict[str, tuple[_GeneAnnotation, ...]]:
    annotations: dict[str, list[_GeneAnnotation]] = {}
    for line in _bounded_text_lines(
        stream,
        maximum_decompressed_bytes=_MAX_GTF_DECOMPRESSED_BYTES,
    ):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 9 or fields[2] != "gene":
            continue
        attributes = _attributes(fields[8])
        gene_name = attributes.get("gene_name")
        gene_id = attributes.get("gene_id")
        if gene_name not in requested_names or gene_id is None:
            continue
        try:
            start = int(fields[3])
            end = int(fields[4])
        except ValueError as error:
            raise VirtualCellContractError("GTF gene interval is not integral") from error
        if start <= 0 or end < start:
            raise VirtualCellContractError("GTF gene interval is invalid")
        annotation = _GeneAnnotation(
            gene_id=gene_id.split(".", 1)[0],
            gene_name=gene_name,
            chromosome=fields[0],
            strand=fields[6],
            start=start,
            end=end,
            gene_biotype=attributes.get("gene_biotype", attributes.get("gene_type", "unknown")),
        )
        candidates = annotations.setdefault(gene_name, [])
        if annotation not in candidates:
            candidates.append(annotation)
    return {
        gene_name: tuple(sorted(values, key=lambda value: value.gene_id))
        for gene_name, values in annotations.items()
    }


def _fasta_gene_id(header: str) -> str | None:
    match = re.search(r"(?:^|\s)gene:([^\s]+)", header)
    return None if match is None else match.group(1).split(".", 1)[0]


def _read_longest_proteins(
    stream: BinaryIO,
    *,
    requested_gene_ids: frozenset[str],
) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_gene: str | None = None
    chunks: list[str] = []

    def finish() -> None:
        if current_gene is None:
            return
        sequence = "".join(chunks).upper()
        if not sequence or any(value not in set(_AMINO_ACIDS) | {"X", "*"} for value in sequence):
            raise VirtualCellContractError("Ensembl peptide FASTA contains invalid residues")
        sequence = sequence.rstrip("*")
        if len(sequence) > len(sequences.get(current_gene, "")):
            sequences[current_gene] = sequence

    for line in _bounded_text_lines(
        stream,
        maximum_decompressed_bytes=_MAX_FASTA_DECOMPRESSED_BYTES,
    ):
        if line.startswith(">"):
            finish()
            candidate = _fasta_gene_id(line[1:])
            current_gene = candidate if candidate in requested_gene_ids else None
            chunks = []
        elif current_gene is not None:
            chunks.append(line.strip())
    finish()
    return sequences


def _sequence_features(sequence: str | None) -> list[float]:
    if sequence is None or not sequence:
        return [0.0] * (len(_AMINO_ACIDS) + 9) + [1.0]
    length = len(sequence)
    counts = {acid: sequence.count(acid) / length for acid in _AMINO_ACIDS}
    entropy = -sum(value * math.log(value) for value in counts.values() if value > 0)
    grouped = (
        sum(counts[value] for value in "DE"),
        sum(counts[value] for value in "KRH"),
        sum(counts[value] for value in "AVILMFWY"),
        sum(counts[value] for value in "FWY"),
        sum(counts[value] for value in "STNQ"),
        counts["P"],
        counts["G"],
        entropy,
    )
    return [math.log1p(length), *(counts[value] for value in _AMINO_ACIDS), *grouped, 0.0]


def virtual_cell_feature_ids() -> tuple[str, ...]:
    values = (
        "gene_log1p_span_bp",
        "gene_is_protein_coding",
        "gene_strand_positive",
        "gene_chr_autosome",
        "gene_chr_x",
        "gene_chr_y",
        "gene_chr_mt",
        "protein_log1p_length",
        *(f"protein_fraction_{acid.lower()}" for acid in _AMINO_ACIDS),
        "protein_fraction_acidic",
        "protein_fraction_basic",
        "protein_fraction_hydrophobic",
        "protein_fraction_aromatic",
        "protein_fraction_polar",
        "protein_fraction_proline",
        "protein_fraction_glycine",
        "protein_composition_entropy",
        "protein_missing",
        "gene_annotation_missing",
    )
    return tuple(values)


def build_ensembl_target_features(
    *,
    target_ids: tuple[str, ...],
    gtf_stream: BinaryIO,
    peptide_fasta_stream: BinaryIO,
    gtf_sha256: str,
    peptide_fasta_sha256: str,
) -> tuple[TargetFeatureMatrix, TargetFeatureBuildReceipt]:
    "Build the frozen Ensembl-113 gene/protein target-feature transform."

    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise ValueError("target roster must be nonempty and unique")
    validate_sha256(gtf_sha256, field_name="gtf_sha256")
    validate_sha256(peptide_fasta_sha256, field_name="peptide_fasta_sha256")
    mapped_names = {target: _TARGET_ALIASES.get(target, target) for target in target_ids}
    annotation_candidates = _read_gene_annotations(
        gtf_stream,
        requested_names=frozenset(mapped_names.values()),
    )
    proteins = _read_longest_proteins(
        peptide_fasta_stream,
        requested_gene_ids=frozenset(
            value.gene_id for values in annotation_candidates.values() for value in values
        ),
    )
    annotations = {
        gene_name: max(
            values,
            key=lambda value: (
                len(proteins.get(value.gene_id, "")),
                value.gene_biotype == "protein_coding",
                value.end - value.start,
                value.gene_id,
            ),
        )
        for gene_name, values in annotation_candidates.items()
    }
    feature_ids = virtual_cell_feature_ids()
    rows: list[list[float]] = []
    mappings: list[TargetFeatureMapping] = []
    for target in target_ids:
        mapped_name = mapped_names[target]
        annotation = annotations.get(mapped_name)
        sequence = None if annotation is None else proteins.get(annotation.gene_id)
        reasons: tuple[str, ...]
        if annotation is None:
            gene_features = [0.0] * 7
            reasons = ("ENSEMBL_GENE_NAME_UNMAPPED",)
        else:
            chromosome = annotation.chromosome.removeprefix("chr").upper()
            gene_features = [
                math.log1p(annotation.end - annotation.start + 1),
                float(annotation.gene_biotype == "protein_coding"),
                float(annotation.strand == "+"),
                float(chromosome.isdigit() and 1 <= int(chromosome) <= 22),
                float(chromosome == "X"),
                float(chromosome == "Y"),
                float(chromosome in {"M", "MT"}),
            ]
            reasons = () if sequence is not None else ("ENSEMBL_PROTEIN_SEQUENCE_UNAVAILABLE",)
        rows.append([*gene_features, *_sequence_features(sequence), float(annotation is None)])
        mappings.append(
            TargetFeatureMapping(
                target_id=target,
                mapped_gene_name=mapped_name,
                ensembl_gene_id=None if annotation is None else annotation.gene_id,
                protein_sequence_available=sequence is not None,
                alias_applied=target != mapped_name,
                reason_codes=reasons,
            )
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(target_ids), len(feature_ids)) or not np.all(np.isfinite(matrix)):
        raise VirtualCellContractError("target feature transform produced an invalid matrix")
    provenance = hashlib.sha256()
    for value in (
        _TRANSFORM_ID,
        gtf_sha256,
        peptide_fasta_sha256,
        _registry_digest(target_ids),
        _registry_digest(feature_ids),
    ):
        provenance.update(value.encode("utf-8"))
    feature_matrix = TargetFeatureMatrix(
        target_ids=target_ids,
        feature_ids=feature_ids,
        values=matrix,
        provenance_sha256=provenance.hexdigest(),
    )
    matrix_digest = hashlib.sha256(np.ascontiguousarray(matrix, dtype="<f8").tobytes()).hexdigest()
    receipt = TargetFeatureBuildReceipt(
        receipt_id="target-features.virtual-cell-2025-ensembl-113",
        transform_id=_TRANSFORM_ID,
        ensembl_release="ensembl-113",
        gtf_sha256=gtf_sha256,
        peptide_fasta_sha256=peptide_fasta_sha256,
        target_roster_sha256=_registry_digest(target_ids),
        feature_ids_sha256=_registry_digest(feature_ids),
        matrix_sha256=matrix_digest,
        mappings=tuple(sorted(mappings, key=lambda value: value.target_id)),
        mapped_gene_count=sum(value.ensembl_gene_id is not None for value in mappings),
        protein_covered_count=sum(value.protein_sequence_available for value in mappings),
        response_derived=False,
        post_cutoff=False,
    )
    return feature_matrix, receipt


__all__ = [
    "TargetFeatureBuildReceipt",
    "TargetFeatureMapping",
    "build_ensembl_target_features",
    "virtual_cell_feature_ids",
]
