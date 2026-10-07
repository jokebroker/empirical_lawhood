"""Read-only application service over the bounded dataset repository port."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from heapq import merge
from typing import Any

from empirical_lawhood.planning.datasets import (
    AcquisitionState,
    CustodyState,
    DatasetBindingRole,
    DatasetFamily,
    DatasetMaterialization,
    DatasetRelease,
    ExternalIdentifier,
)

from .datasets import (
    DatasetCatalogCursor,
    DatasetCatalogPage,
    DatasetCatalogProjectionAnchor,
    DatasetCatalogProjectionState,
    DatasetCatalogQuery,
    DatasetCatalogQueryBoundary,
    DatasetCatalogRecord,
    DatasetCatalogWorkLimit,
    DatasetRepository,
    DatasetCatalogSnapshot,
    dataset_catalog_record_key,
)


_TrustedKey = tuple[str, str]


class DatasetCatalogReadAuthenticationError(RuntimeError):
    """A local projection result differed from its injected external authority."""


class DatasetCatalogTrustedQueryWorkLimitExceeded(RuntimeError):
    """Trusted-snapshot page verification exceeded its declared pure-work limit."""

    def __init__(self, *, records_examined: int, record_limit: int) -> None:
        self.records_examined = records_examined
        self.record_limit = record_limit
        super().__init__("trusted dataset snapshot query exceeded its record work limit")


@dataclass(frozen=True, slots=True)
class _TrustedPage:
    records: tuple[DatasetCatalogRecord, ...]
    records_examined: int
    has_more: bool
    next_key: _TrustedKey | None


class _TrustedDatasetSnapshotIndex:
    """Bounded immutable filter index derived only from the injected snapshot."""

    _CATEGORIES = (
        "family_ids",
        "release_ids",
        "provider_ids",
        "external_identifiers",
        "custody_states",
        "acquisition_states",
        "experiment_spec_ids",
        "roles",
    )

    def __init__(self, snapshot: DatasetCatalogSnapshot) -> None:
        records: dict[_TrustedKey, DatasetCatalogRecord] = {}
        mutable_indexes: dict[str, dict[object, list[_TrustedKey]]] = {
            category: {} for category in self._CATEGORIES
        }

        def include(
            record: DatasetCatalogRecord,
            *memberships: tuple[str, object | None],
        ) -> None:
            kind, record_id = dataset_catalog_record_key(record)
            key = (kind.value, record_id)
            if key in records:
                raise ValueError("trusted dataset snapshot contains a duplicate record key")
            records[key] = record
            for category, value in memberships:
                if value is None:
                    continue
                mutable_indexes[category].setdefault(value, []).append(key)

        releases = {record.release_id: record for record in snapshot.releases}
        for family in snapshot.families:
            include(
                family,
                ("family_ids", family.family_id),
                ("provider_ids", family.provider_id),
                *(
                    ("external_identifiers", identifier)
                    for identifier in family.external_identifiers
                ),
            )
        for release_record in snapshot.releases:
            include(
                release_record,
                ("family_ids", release_record.family_id),
                ("release_ids", release_record.release_id),
                ("provider_ids", release_record.provider_id),
                *(
                    ("external_identifiers", identifier)
                    for identifier in release_record.external_identifiers
                ),
            )
        for observation in snapshot.observations:
            include(
                observation,
                ("provider_ids", observation.provider_id),
                *(
                    ("external_identifiers", identifier)
                    for identifier in observation.provider_object_ids
                ),
            )
        for materialization in snapshot.materializations:
            release = releases[materialization.release_id]
            include(
                materialization,
                ("family_ids", release.family_id),
                ("release_ids", release.release_id),
                ("provider_ids", release.provider_id),
                ("custody_states", materialization.custody_state),
            )
        for attempt in snapshot.acquisition_attempts:
            release = releases[attempt.release_id]
            include(
                attempt,
                ("family_ids", release.family_id),
                ("release_ids", release.release_id),
                ("provider_ids", release.provider_id),
                ("acquisition_states", attempt.acquisition_state),
            )
        for binding in snapshot.bindings:
            release = releases[binding.release_id]
            include(
                binding,
                ("family_ids", release.family_id),
                ("release_ids", release.release_id),
                ("provider_ids", release.provider_id),
                ("experiment_spec_ids", binding.experiment_spec_id),
                ("roles", binding.role),
            )

        self.records = records
        self.sorted_keys = tuple(sorted(records))
        self.indexes = {
            category: {value: tuple(sorted(keys)) for value, keys in values.items()}
            for category, values in mutable_indexes.items()
        }
        self.memberships = {
            category: {value: frozenset(keys) for value, keys in values.items()}
            for category, values in self.indexes.items()
        }
        self.families = {record.family_id: record for record in snapshot.families}
        self.releases = {record.release_id: record for record in snapshot.releases}
        self.materializations = {
            record.materialization_id: record for record in snapshot.materializations
        }

    @staticmethod
    def _requested(
        boundary: DatasetCatalogQueryBoundary,
    ) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        return tuple(
            (category, tuple(values))
            for category, values in (
                ("family_ids", boundary.family_ids),
                ("release_ids", boundary.release_ids),
                ("provider_ids", boundary.provider_ids),
                ("external_identifiers", boundary.external_identifiers),
                ("custody_states", boundary.custody_states),
                ("acquisition_states", boundary.acquisition_states),
                ("experiment_spec_ids", boundary.experiment_spec_ids),
                ("roles", boundary.roles),
            )
            if values
        )

    def page(self, query: DatasetCatalogQuery) -> _TrustedPage:
        requested = self._requested(query.boundary)
        candidates: tuple[_TrustedKey, ...] | None = None
        base_category: str | None = None
        base_values: tuple[Any, ...] = ()
        if requested:
            estimates = tuple(
                (
                    sum(len(self.indexes[category].get(value, ())) for value in values),
                    category,
                    values,
                )
                for category, values in requested
            )
            estimate, base_category, base_values = min(
                estimates,
                key=lambda value: (value[0], value[1]),
            )
            if estimate == 0:
                return _TrustedPage((), 0, False, None)
        else:
            candidates = self.sorted_keys

        after_key = (
            None
            if query.cursor is None
            else (
                query.cursor.after_record_kind.value,
                query.cursor.after_record_id,
            )
        )
        if candidates is not None:
            offset = 0 if after_key is None else bisect_right(candidates, after_key)
            candidate_iterator = iter(candidates[offset:])
        else:
            assert base_category is not None
            base_lists = tuple(self.indexes[base_category].get(value, ()) for value in base_values)
            sliced = tuple(
                values[0 if after_key is None else bisect_right(values, after_key) :]
                for values in base_lists
            )
            candidate_iterator = merge(*sliced)

        selected: list[DatasetCatalogRecord] = []
        records_examined = 0
        previous: _TrustedKey | None = None
        for key in candidate_iterator:
            if key == previous:
                continue
            previous = key
            records_examined += 1
            if records_examined > query.boundary.work_limit.max_records_examined:
                raise DatasetCatalogTrustedQueryWorkLimitExceeded(
                    records_examined=records_examined,
                    record_limit=query.boundary.work_limit.max_records_examined,
                )
            if any(
                not any(key in self.memberships[category].get(value, ()) for value in values)
                for category, values in requested
            ):
                continue
            selected.append(self.records[key])
            if len(selected) > query.limit:
                break

        has_more = len(selected) > query.limit
        page_records = tuple(selected[: query.limit])
        next_key = None
        if has_more and page_records:
            kind, record_id = dataset_catalog_record_key(page_records[-1])
            next_key = (kind.value, record_id)
        return _TrustedPage(
            records=page_records,
            records_examined=records_examined,
            has_more=has_more,
            next_key=next_key,
        )


@dataclass(frozen=True, slots=True)
class DatasetCatalogLookup:
    """One public lookup request before its current snapshot is authenticated."""

    work_limit: DatasetCatalogWorkLimit
    limit: int = 100
    cursor: DatasetCatalogCursor | None = None
    family_ids: tuple[str, ...] = ()
    release_ids: tuple[str, ...] = ()
    provider_ids: tuple[str, ...] = ()
    external_identifiers: tuple[ExternalIdentifier, ...] = ()
    custody_states: tuple[CustodyState, ...] = ()
    acquisition_states: tuple[AcquisitionState, ...] = ()
    experiment_spec_ids: tuple[str, ...] = ()
    roles: tuple[DatasetBindingRole, ...] = ()


class DatasetCatalogReadService:
    """Pin every page to the repository's authenticated projection state."""

    def __init__(
        self,
        repository: DatasetRepository,
        *,
        trusted_snapshot: DatasetCatalogSnapshot,
        trusted_anchor: DatasetCatalogProjectionAnchor,
    ) -> None:
        if not isinstance(trusted_snapshot, DatasetCatalogSnapshot):
            raise TypeError("trusted_snapshot must be a DatasetCatalogSnapshot")
        if not isinstance(trusted_anchor, DatasetCatalogProjectionAnchor):
            raise TypeError("trusted_anchor must be a DatasetCatalogProjectionAnchor")
        if not trusted_anchor.validates_snapshot(trusted_snapshot):
            raise ValueError("trusted projection anchor differs from its exact snapshot")
        self.repository = repository
        self.trusted_snapshot = trusted_snapshot
        self.trusted_anchor = trusted_anchor
        self._trusted_index = _TrustedDatasetSnapshotIndex(trusted_snapshot)

    def _authenticate_state(self, state: object) -> None:
        if state != self.trusted_anchor.projection_state:
            raise DatasetCatalogReadAuthenticationError(
                "local dataset projection state differs from its trusted anchor"
            )

    def authenticate_projection(
        self,
        work_limit: DatasetCatalogWorkLimit,
    ) -> DatasetCatalogProjectionState:
        """Authenticate current local state without disclosing catalog records."""

        if not isinstance(work_limit, DatasetCatalogWorkLimit):
            raise TypeError("work_limit must be a DatasetCatalogWorkLimit")
        state = self.repository.projection_state(work_limit)
        self._authenticate_state(state)
        return state

    def page(self, lookup: DatasetCatalogLookup) -> DatasetCatalogPage:
        if not isinstance(lookup, DatasetCatalogLookup):
            raise TypeError("lookup must be a DatasetCatalogLookup")
        state = self.repository.projection_state(lookup.work_limit)
        self._authenticate_state(state)
        boundary = DatasetCatalogQueryBoundary(
            snapshot_fingerprint=state.snapshot_fingerprint,
            projection_state_fingerprint=state.fingerprint(),
            projection_anchor_fingerprint=self.trusted_anchor.fingerprint(),
            work_limit=lookup.work_limit,
            family_ids=lookup.family_ids,
            release_ids=lookup.release_ids,
            provider_ids=lookup.provider_ids,
            external_identifiers=lookup.external_identifiers,
            custody_states=lookup.custody_states,
            acquisition_states=lookup.acquisition_states,
            experiment_spec_ids=lookup.experiment_spec_ids,
            roles=lookup.roles,
        )
        query = DatasetCatalogQuery(
            boundary=boundary,
            limit=lookup.limit,
            cursor=lookup.cursor,
        )
        page = self.repository.query(query)
        expected = self._trusted_index.page(query)
        observed_next_key = (
            None
            if page.next_cursor is None
            else (
                page.next_cursor.after_record_kind.value,
                page.next_cursor.after_record_id,
            )
        )
        if (
            page.boundary != boundary
            or page.limit != lookup.limit
            or page.records != expected.records
            or page.has_more != expected.has_more
            or observed_next_key != expected.next_key
        ):
            raise DatasetCatalogReadAuthenticationError(
                "local dataset page differs from the trusted snapshot"
            )
        return page

    def family(self, family_id: str) -> DatasetFamily | None:
        observed = self.repository.family(family_id)
        expected = self._trusted_index.families.get(family_id)
        if observed != expected:
            raise DatasetCatalogReadAuthenticationError(
                "local dataset family differs from the trusted snapshot"
            )
        return expected

    def release(self, release_id: str) -> DatasetRelease | None:
        observed = self.repository.release(release_id)
        expected = self._trusted_index.releases.get(release_id)
        if observed != expected:
            raise DatasetCatalogReadAuthenticationError(
                "local dataset release differs from the trusted snapshot"
            )
        return expected

    def materialization(self, materialization_id: str) -> DatasetMaterialization | None:
        observed = self.repository.materialization(materialization_id)
        expected = self._trusted_index.materializations.get(materialization_id)
        if observed != expected:
            raise DatasetCatalogReadAuthenticationError(
                "local dataset materialization differs from the trusted snapshot"
            )
        return expected


__all__ = [
    "DatasetCatalogLookup",
    "DatasetCatalogReadAuthenticationError",
    "DatasetCatalogReadService",
    "DatasetCatalogTrustedQueryWorkLimitExceeded",
]
