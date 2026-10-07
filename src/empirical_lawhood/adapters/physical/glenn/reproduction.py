"""Read-only in-memory preparation of the pinned Glenn reproduction inputs.

This service deliberately stops before persistence, catalog mutation,
authorization, lineage construction, or scientific adjudication.  It opens the
three exact already-held inputs through the runtime source guard, performs the
fixed archive transform and deterministic serialization, and records only a
strict operational comparison with the frozen historical CSV.
"""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.dataset_authority import DatasetStorageScope
from empirical_lawhood.runtime.dataset_io import (
    DatasetSourceGuardReceipt,
    DatasetSourceOpener,
    GuardedDatasetSource,
)

from .comparison import (
    GlennHistoricalCSVInput,
    GlennLogicalEquivalence,
    compare_historical_observations_csv,
)
from .contracts import GLENN_2026_PROFILE, GlennAdapterError, GlennArchiveInput
from .manifests import GlennRetainedComparisonInputs, GLENN_ARCHIVE_RELATIVE_LOCATOR, GLENN_ARCHIVE_SHA256, GLENN_ARCHIVE_SIZE_BYTES, GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR, GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES, GLENN_READ_SET_RELATIVE_LOCATOR, GLENN_READ_SET_SIZE_BYTES
from .transform import (
    GlennParquetArtifact,
    GlennTransformResult,
    serialize_glenn_parquet,
    transform_glenn_archive,
)


_SOURCE_SCOPE_ID = "scope.glenn-transform-source"
_SELECTOR_SCOPE_ID = "scope.glenn-source-read-set"
_HISTORICAL_SCOPE_ID = "scope.glenn-historical-comparison"


def _require_fixed_profile_consistency() -> None:
    if (
        GLENN_2026_PROFILE.expected_archive_size_bytes != GLENN_ARCHIVE_SIZE_BYTES
        or GLENN_2026_PROFILE.expected_archive_sha256 != GLENN_ARCHIVE_SHA256
    ):
        raise RuntimeError(
            "fixed Glenn transform and source identity profiles disagree"
        )


@dataclass(frozen=True, slots=True)
class GlennReproductionPreparationRequest:
    """Exact read scopes and aggregate distinct-input budget for preparation."""

    historical_inputs: GlennRetainedComparisonInputs

    source_scope: DatasetStorageScope
    selector_scope: DatasetStorageScope
    historical_scope: DatasetStorageScope
    maximum_distinct_input_bytes: int
    trusted_at_utc: str

    def __post_init__(self) -> None:
        _require_fixed_profile_consistency()
        if not isinstance(self.source_scope, DatasetStorageScope):
            raise ValueError("source_scope must be a DatasetStorageScope")
        if not isinstance(self.selector_scope, DatasetStorageScope):
            raise ValueError("selector_scope must be a DatasetStorageScope")
        if not isinstance(self.historical_scope, DatasetStorageScope):
            raise ValueError("historical_scope must be a DatasetStorageScope")
        if (
            self.source_scope.scope_id != _SOURCE_SCOPE_ID
            or self.source_scope.relative_prefix != GLENN_ARCHIVE_RELATIVE_LOCATOR
        ):
            raise ValueError("source_scope differs from the fixed Glenn archive scope")
        if (
            self.selector_scope.scope_id != _SELECTOR_SCOPE_ID
            or self.selector_scope.relative_prefix != GLENN_READ_SET_RELATIVE_LOCATOR
        ):
            raise ValueError(
                "selector_scope differs from the fixed Glenn read-set scope"
            )
        if (
            self.historical_scope.scope_id != _HISTORICAL_SCOPE_ID
            or self.historical_scope.relative_prefix
            != GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR
        ):
            raise ValueError(
                "historical_scope differs from the fixed Glenn comparison scope"
            )
        input_keys = {
            (
                scope.storage_root.object_fingerprint,
                scope.relative_prefix,
            )
            for scope in (
                self.source_scope,
                self.selector_scope,
                self.historical_scope,
            )
        }
        if len(input_keys) != 3:
            raise ValueError(
                "archive, read-set, and historical inputs must be distinct"
            )
        exact_distinct_bytes = (
            GLENN_ARCHIVE_SIZE_BYTES
            + GLENN_READ_SET_SIZE_BYTES
            + GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES
        )
        if (
            isinstance(self.maximum_distinct_input_bytes, bool)
            or not isinstance(self.maximum_distinct_input_bytes, int)
            or self.maximum_distinct_input_bytes != exact_distinct_bytes
        ):
            raise ValueError(
                "maximum_distinct_input_bytes must equal the fixed three-input envelope"
            )
        parse_utc_timestamp(self.trusted_at_utc, field_name="trusted_at_utc")


@dataclass(frozen=True, slots=True)
class GlennReproductionPreparationResult:
    """In-memory operational preparation without persistence or claim promotion."""

    historical_inputs: GlennRetainedComparisonInputs

    artifact: GlennParquetArtifact
    transform_audit: GlennTransformResult
    equivalence: GlennLogicalEquivalence
    source_guard_receipt: DatasetSourceGuardReceipt
    selector_guard_receipt: DatasetSourceGuardReceipt
    historical_guard_receipt: DatasetSourceGuardReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, GlennParquetArtifact):
            raise ValueError("artifact must be a GlennParquetArtifact")
        if not isinstance(self.transform_audit, GlennTransformResult):
            raise ValueError("transform_audit must be a GlennTransformResult")
        if not isinstance(self.equivalence, GlennLogicalEquivalence):
            raise ValueError("equivalence must be a GlennLogicalEquivalence")
        if not isinstance(self.source_guard_receipt, DatasetSourceGuardReceipt):
            raise ValueError("source_guard_receipt must be completed guard evidence")
        if not isinstance(self.selector_guard_receipt, DatasetSourceGuardReceipt):
            raise ValueError("selector_guard_receipt must be completed guard evidence")
        if not isinstance(self.historical_guard_receipt, DatasetSourceGuardReceipt):
            raise ValueError(
                "historical_guard_receipt must be completed guard evidence"
            )
        fixed_receipt_facts = (
            (
                self.source_guard_receipt,
                GLENN_ARCHIVE_RELATIVE_LOCATOR,
                GLENN_ARCHIVE_SIZE_BYTES,
                GLENN_ARCHIVE_SHA256,
            ),
            (
                self.selector_guard_receipt,
                GLENN_READ_SET_RELATIVE_LOCATOR,
                GLENN_READ_SET_SIZE_BYTES,
                self.historical_inputs.read_set_sha256,
            ),
            (
                self.historical_guard_receipt,
                GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR,
                GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES,
                self.historical_inputs.observations_sha256,
            ),
        )
        if any(
            receipt.relative_locator != locator
            or receipt.observed_size_bytes != size
            or receipt.observed_sha256 != digest
            or receipt.network_bytes != 0
            for receipt, locator, size, digest in fixed_receipt_facts
        ):
            raise ValueError(
                "reproduction guard receipts differ from fixed input identities"
            )
        if len({receipt.verified_at_utc for receipt, *_ in fixed_receipt_facts}) != 1:
            raise ValueError("reproduction guard receipts use different trusted times")
        if (
            self.artifact.logical_sha256 != self.transform_audit.logical_sha256
            or self.equivalence.transformed_logical_sha256
            != self.transform_audit.logical_sha256
        ):
            raise ValueError(
                "preparation components disagree on follow-up logical identity"
            )
        if (
            self.source_guard_receipt.guard_receipt_id
            != self.transform_audit.archive_audit.guard_evidence_id
        ):
            raise ValueError("source guard receipt differs from the transform audit")
        if (
            self.historical_guard_receipt.guard_receipt_id
            != self.equivalence.historical_guard_evidence_id
        ):
            raise ValueError("historical guard receipt differs from the comparison")
        if (
            self.equivalence.scientific_verdict_issued
            or self.equivalence.lineage_parent_created
        ):
            raise ValueError("reproduction preparation cannot issue verdict or lineage")


def _completed_receipt(
    guarded: GuardedDatasetSource,
    *,
    expected_scope: DatasetStorageScope,
    expected_size_bytes: int,
    expected_sha256: str,
    trusted_at_utc: str,
) -> DatasetSourceGuardReceipt:
    receipt = guarded.guard_receipt
    if (
        receipt.source_scope != expected_scope
        or receipt.relative_locator != expected_scope.relative_prefix
        or receipt.observed_size_bytes != expected_size_bytes
        or receipt.observed_sha256 != expected_sha256
        or receipt.verified_at_utc != trusted_at_utc
        or receipt.guard_receipt_id != guarded.guard_receipt_id
        or receipt.network_bytes != 0
    ):
        raise GlennAdapterError(
            "completed source guard receipt differs from its exact request"
        )
    return receipt


def prepare_glenn_reproduction(
    request: GlennReproductionPreparationRequest,
    *,
    source_opener: DatasetSourceOpener,
) -> GlennReproductionPreparationResult:
    """Prepare exact follow-up bytes and historical comparison without writing.

    ``GLENN_2026_PROFILE`` and all three physical input identities are module-owned
    constants, not caller-selected fields.  Guard receipts are deliberately
    accessed only after their corresponding context has exited cleanly.
    """

    if not isinstance(request, GlennReproductionPreparationRequest):
        raise TypeError("request must be a GlennReproductionPreparationRequest")

    with source_opener.open_exact(
        source_scope=request.source_scope,
        expected_size_bytes=GLENN_ARCHIVE_SIZE_BYTES,
        expected_sha256=GLENN_ARCHIVE_SHA256,
        maximum_bytes=GLENN_ARCHIVE_SIZE_BYTES,
        trusted_at_utc=request.trusted_at_utc,
    ) as guarded_source:
        if not isinstance(guarded_source, GuardedDatasetSource):
            raise GlennAdapterError("source opener returned an invalid guarded source")
        transform_audit = transform_glenn_archive(
            GlennArchiveInput(
                stream=guarded_source.stream,
                locator=request.source_scope.relative_prefix,
                guard_evidence_id=guarded_source.guard_receipt_id,
            ),
            profile=GLENN_2026_PROFILE,
        )
        artifact = serialize_glenn_parquet(transform_audit.table)

    source_receipt = _completed_receipt(
        guarded_source,
        expected_scope=request.source_scope,
        expected_size_bytes=GLENN_ARCHIVE_SIZE_BYTES,
        expected_sha256=GLENN_ARCHIVE_SHA256,
        trusted_at_utc=request.trusted_at_utc,
    )

    with source_opener.open_exact(
        source_scope=request.selector_scope,
        expected_size_bytes=GLENN_READ_SET_SIZE_BYTES,
        expected_sha256=request.historical_inputs.read_set_sha256,
        maximum_bytes=GLENN_READ_SET_SIZE_BYTES,
        trusted_at_utc=request.trusted_at_utc,
    ) as guarded_selector:
        if not isinstance(guarded_selector, GuardedDatasetSource):
            raise GlennAdapterError(
                "source opener returned an invalid guarded selector"
            )
        selector_payload = guarded_selector.stream.read(GLENN_READ_SET_SIZE_BYTES + 1)
        if (
            not isinstance(selector_payload, bytes)
            or len(selector_payload) != GLENN_READ_SET_SIZE_BYTES
            or b"\x00" in selector_payload
        ):
            raise GlennAdapterError(
                "Glenn read-set bytes violate their exact bounded contract"
            )

    selector_receipt = _completed_receipt(
        guarded_selector,
        expected_scope=request.selector_scope,
        expected_size_bytes=GLENN_READ_SET_SIZE_BYTES,
        expected_sha256=request.historical_inputs.read_set_sha256,
        trusted_at_utc=request.trusted_at_utc,
    )

    with source_opener.open_exact(
        source_scope=request.historical_scope,
        expected_size_bytes=GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES,
        expected_sha256=request.historical_inputs.observations_sha256,
        maximum_bytes=GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES,
        trusted_at_utc=request.trusted_at_utc,
    ) as guarded_historical:
        if not isinstance(guarded_historical, GuardedDatasetSource):
            raise GlennAdapterError("source opener returned an invalid guarded source")
        equivalence = compare_historical_observations_csv(
            transform_audit.table,
            GlennHistoricalCSVInput(
                stream=guarded_historical.stream,
                locator=request.historical_scope.relative_prefix,
                guard_evidence_id=guarded_historical.guard_receipt_id,
                expected_size_bytes=GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES,
                expected_sha256=request.historical_inputs.observations_sha256,
            ),
        )

    historical_receipt = _completed_receipt(
        guarded_historical,
        expected_scope=request.historical_scope,
        expected_size_bytes=GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES,
        expected_sha256=request.historical_inputs.observations_sha256,
        trusted_at_utc=request.trusted_at_utc,
    )

    return GlennReproductionPreparationResult(
        historical_inputs=request.historical_inputs,
        artifact=artifact,
        transform_audit=transform_audit,
        equivalence=equivalence,
        source_guard_receipt=source_receipt,
        selector_guard_receipt=selector_receipt,
        historical_guard_receipt=historical_receipt,
    )


__all__ = [
    "GlennReproductionPreparationRequest",
    "GlennReproductionPreparationResult",
    "prepare_glenn_reproduction",
]
