'Dataset identity, custody, observation and binding tables.\n\nSix entity projections have a normalized external-identifier association and\na singleton projection-state control row. Indexed associations support exact\nbounded queries; the state row gives a constant-work snapshot boundary.\nMulti-valued metadata remains bounded canonical JSON. Released definitions\nrequire a migration when changed; runtime SQLAlchemy objects remain separate.'

from __future__ import annotations

from typing import Final

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

from empirical_lawhood.planning.dataset_limits import (
    MAX_DATASET_INTEGER as _MAX_DATASET_INTEGER,
    MAX_DATASET_RECORD_CANONICAL_BYTES,
)

from .catalog_tables import CATALOG_METADATA, NAMING_CONVENTION

MAX_DATASET_ID_BYTES: Final = 128
MAX_DATASET_INTEGER: Final = _MAX_DATASET_INTEGER
MAX_DATASET_SCHEMA_BYTES: Final = 300
MAX_DATASET_VERSION_BYTES: Final = 40
MAX_DATASET_NAME_BYTES: Final = 256
MAX_DATASET_TEXT_BYTES: Final = 4_096
MAX_DATASET_LOCATOR_BYTES: Final = 2_048
MAX_EXTERNAL_IDENTIFIER_BYTES: Final = 1_024
MAX_DATASET_COLLECTION_ITEMS: Final = 256
MAX_DATASET_PROVIDER_OBJECTS: Final = 4_096
MAX_DATASET_REASON_CODES: Final = 128
MAX_CANONICAL_JSON_ESCAPE_EXPANSION: Final = 6
MAX_KEYWORDS_JSON_BYTES: Final = (
    MAX_DATASET_COLLECTION_ITEMS
    * (MAX_DATASET_NAME_BYTES * MAX_CANONICAL_JSON_ESCAPE_EXPANSION + 3)
    + 2
)
MAX_IDS_JSON_BYTES: Final = (
    MAX_DATASET_COLLECTION_ITEMS * (MAX_DATASET_ID_BYTES + 3) + 2
)
MAX_REASON_CODES_JSON_BYTES: Final = (
    MAX_DATASET_REASON_CODES * (MAX_DATASET_ID_BYTES + 3) + 2
)
# The largest contract shape is one 4,096-object provider observation.  This
# conservative ceiling covers worst-case JSON escaping of every provider value,
# all other bounded collections and envelope overhead while remaining compact
# metadata rather than scientific rows or provider response bytes.
MAX_DATASET_RECORD_JSON_BYTES: Final = MAX_DATASET_RECORD_CANONICAL_BYTES
DATASET_METADATA: Final[MetaData] = MetaData(naming_convention=NAMING_CONVENTION)
for _catalog_table in CATALOG_METADATA.sorted_tables:
    _catalog_table.to_metadata(DATASET_METADATA)


def _id(name: str, *, nullable: bool = False) -> Column[str]:
    return Column(name, String(MAX_DATASET_ID_BYTES), nullable=nullable)


def _sha256(name: str, *, nullable: bool = False) -> Column[bytes]:
    return Column(name, LargeBinary(32), nullable=nullable)


def _bounded_utf8(
    column: str,
    maximum_bytes: int,
    *,
    nullable: bool = False,
    allow_empty: bool = False,
) -> CheckConstraint:
    lower_bound = "0" if allow_empty else "1"
    bounded = (
        f"length(CAST({column} AS BLOB)) >= {lower_bound} "
        f"AND length(CAST({column} AS BLOB)) <= {maximum_bytes}"
    )
    expression = f"{column} IS NULL OR ({bounded})" if nullable else bounded
    return CheckConstraint(expression, name=f"{column}_bytes")


def _sha256_length(column: str, *, nullable: bool = False) -> CheckConstraint:
    expression = (
        f"{column} IS NULL OR length({column}) = 32"
        if nullable
        else f"length({column}) = 32"
    )
    return CheckConstraint(expression, name=f"{column}_length")


def _canonical_json_array(
    column: str,
    maximum_bytes: int,
    *,
    maximum_items: int = MAX_DATASET_COLLECTION_ITEMS,
) -> CheckConstraint:
    return CheckConstraint(
        f"json_valid({column}) = 1 "
        f"AND json_type({column}) = 'array' "
        f"AND json({column}) || char(10) = {column} "
        f"AND json_array_length({column}) <= {maximum_items} "
        f"AND length(CAST({column} AS BLOB)) <= {maximum_bytes}",
        name=f"{column}_canonical_bounded_array",
    )


def _canonical_json_object(column: str, maximum_bytes: int) -> CheckConstraint:
    return CheckConstraint(
        f"json_valid({column}) = 1 "
        f"AND json_type({column}) = 'object' "
        f"AND json({column}) || char(10) = {column} "
        f"AND length(CAST({column} AS BLOB)) <= {maximum_bytes}",
        name=f"{column}_canonical_bounded_object",
    )


def _enum(column: str, values: tuple[str, ...]) -> CheckConstraint:
    choices = ", ".join(f"'{value}'" for value in values)
    return CheckConstraint(f"{column} IN ({choices})", name=f"{column}_value")


def _utc_timestamp_predicate(column: str) -> str:
    """SQLite predicate matching the kernel's strict 0--6 digit UTC grammar."""

    digits = (
        f"substr({column}, 1, 4) NOT GLOB '*[^0-9]*' "
        f"AND substr({column}, 6, 2) NOT GLOB '*[^0-9]*' "
        f"AND substr({column}, 9, 2) NOT GLOB '*[^0-9]*' "
        f"AND substr({column}, 12, 2) NOT GLOB '*[^0-9]*' "
        f"AND substr({column}, 15, 2) NOT GLOB '*[^0-9]*' "
        f"AND substr({column}, 18, 2) NOT GLOB '*[^0-9]*'"
    )
    fractional = (
        f"(length({column}) = 20 AND substr({column}, 20, 1) = 'Z') "
        f"OR (length({column}) BETWEEN 22 AND 27 "
        f"AND substr({column}, 20, 1) = '.' "
        f"AND substr({column}, -1, 1) = 'Z' "
        f"AND substr({column}, 21, length({column}) - 21) "
        "NOT GLOB '*[^0-9]*')"
    )
    return (
        f"typeof({column}) = 'text' "
        f"AND length(CAST({column} AS BLOB)) BETWEEN 20 AND 27 "
        f"AND substr({column}, 5, 1) = '-' "
        f"AND substr({column}, 8, 1) = '-' "
        f"AND substr({column}, 11, 1) = 'T' "
        f"AND substr({column}, 14, 1) = ':' "
        f"AND substr({column}, 17, 1) = ':' "
        f"AND {digits} "
        f"AND ({fractional}) "
        f"AND date(substr({column}, 1, 10)) = substr({column}, 1, 10) "
        f"AND CAST(substr({column}, 1, 4) AS INTEGER) BETWEEN 1 AND 9999 "
        f"AND CAST(substr({column}, 12, 2) AS INTEGER) BETWEEN 0 AND 23 "
        f"AND CAST(substr({column}, 15, 2) AS INTEGER) BETWEEN 0 AND 59 "
        f"AND CAST(substr({column}, 18, 2) AS INTEGER) BETWEEN 0 AND 59"
    )


def _utc_timestamp(column: str, *, nullable: bool = False) -> CheckConstraint:
    predicate = _utc_timestamp_predicate(column)
    expression = f"{column} IS NULL OR ({predicate})" if nullable else predicate
    return CheckConstraint(expression, name=f"{column}_strict_utc")


def _utc_sort_key(column: str) -> str:
    """Normalize accepted timestamps to fixed six-digit fractional text in SQL."""

    return (
        f"CASE WHEN length({column}) = 20 "
        f"THEN substr({column}, 1, 19) || '.000000Z' "
        f"ELSE substr({column}, 1, length({column}) - 1) "
        f"|| substr('000000', 1, 27 - length({column})) || 'Z' END"
    )


dataset_family = Table(
    "dataset_family",
    DATASET_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("family_id"),
    Column("record_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("record_fingerprint"),
    Column("record_json", Text, nullable=False),
    Column("canonical_name", String(MAX_DATASET_NAME_BYTES), nullable=False),
    _id("provider_id", nullable=True),
    Column("description", Text, nullable=False),
    Column("keywords_json", Text, nullable=False),
    _id("first_observation_id", nullable=True),
    _id("latest_observation_id", nullable=True),
    Column("identity_state", String(32), nullable=False),
    Column("reason_codes_json", Text, nullable=False),
    UniqueConstraint("family_id"),
    UniqueConstraint("record_fingerprint"),
    _bounded_utf8("family_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("record_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("record_fingerprint"),
    _canonical_json_object("record_json", MAX_DATASET_RECORD_JSON_BYTES),
    _bounded_utf8("canonical_name", MAX_DATASET_NAME_BYTES),
    _bounded_utf8("provider_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("description", MAX_DATASET_TEXT_BYTES, allow_empty=True),
    _canonical_json_array("keywords_json", MAX_KEYWORDS_JSON_BYTES),
    _bounded_utf8("first_observation_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("latest_observation_id", MAX_DATASET_ID_BYTES, nullable=True),
    CheckConstraint(
        "(first_observation_id IS NULL) = (latest_observation_id IS NULL)",
        name="observation_ids_indivisible",
    ),
    _enum("identity_state", ("UNRESOLVED", "FAMILY_RESOLVED", "CONFLICT")),
    _canonical_json_array(
        "reason_codes_json",
        MAX_REASON_CODES_JSON_BYTES,
        maximum_items=MAX_DATASET_REASON_CODES,
    ),
    CheckConstraint(
        "identity_state NOT IN ('UNRESOLVED', 'CONFLICT') "
        "OR json_array_length(reason_codes_json) > 0",
        name="unresolved_identity_reasoned",
    ),
)
Index(
    "ix_dataset_family_provider_family",
    dataset_family.c.provider_id,
    dataset_family.c.family_id,
)
Index(
    "ix_dataset_family_identity",
    dataset_family.c.identity_state,
    dataset_family.c.family_id,
)


dataset_release = Table(
    "dataset_release",
    DATASET_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("release_id"),
    Column("family_pk", ForeignKey("dataset_family.pk"), nullable=False),
    Column("record_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("record_fingerprint"),
    Column("record_json", Text, nullable=False),
    Column("identity_state", String(32), nullable=False),
    Column("resolution_class", String(48), nullable=False),
    Column("access_state", String(32), nullable=False),
    _id("provider_id", nullable=True),
    _id("local_selector_id", nullable=True),
    Column("local_selector_kind", String(40)),
    Column("local_selector_schema", String(MAX_DATASET_SCHEMA_BYTES)),
    _sha256("local_selector_sha256", nullable=True),
    Column("local_selector_complete", Integer),
    Column("publication_at_utc", String(32)),
    Column("observed_at_utc", String(32)),
    Column("mutable_snapshot", Integer, nullable=False),
    Column("expected_format_profile_ids_json", Text, nullable=False),
    Column("expected_file_count_minimum", Integer),
    Column("expected_file_count_maximum", Integer),
    Column("expected_byte_count_minimum", BigInteger),
    Column("expected_byte_count_maximum", BigInteger),
    _sha256("manifest_sha256", nullable=True),
    _sha256("expected_physical_sha256", nullable=True),
    _id("licence_evidence_id", nullable=True),
    _id("access_evidence_id", nullable=True),
    Column("evidence_ref_ids_json", Text, nullable=False),
    Column("reason_codes_json", Text, nullable=False),
    UniqueConstraint("release_id"),
    UniqueConstraint("record_fingerprint"),
    _bounded_utf8("release_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("record_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("record_fingerprint"),
    _canonical_json_object("record_json", MAX_DATASET_RECORD_JSON_BYTES),
    _enum(
        "identity_state",
        ("UNRESOLVED", "FAMILY_RESOLVED", "RELEASE_RESOLVED", "CONFLICT"),
    ),
    _enum(
        "resolution_class",
        (
            "PROVIDER_IMMUTABLE_RESOLVED",
            "CAPTURED_SNAPSHOT_RESOLVED",
            "UNRESOLVED_LOCAL_CUSTODY",
            "NOT_APPLICABLE_GENERATED",
        ),
    ),
    _enum(
        "access_state",
        (
            "PUBLIC",
            "REGISTRATION",
            "CREDENTIALLED",
            "AGREEMENT",
            "PAID",
            "UNAVAILABLE",
            "UNKNOWN",
        ),
    ),
    _bounded_utf8("provider_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("local_selector_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("local_selector_schema", MAX_DATASET_SCHEMA_BYTES, nullable=True),
    _sha256_length("local_selector_sha256", nullable=True),
    _enum(
        "local_selector_kind",
        (
            "COMPLETE_RELEASE",
            "MANIFEST",
            "PARTITION",
            "SPECIMEN",
            "SWEEP",
            "TIME_RANGE",
            "SCENARIO",
            "COMPOSITE",
            "OTHER_REGISTERED",
        ),
    ),
    CheckConstraint(
        "(local_selector_id IS NULL "
        "AND local_selector_kind IS NULL "
        "AND local_selector_schema IS NULL "
        "AND local_selector_sha256 IS NULL "
        "AND local_selector_complete IS NULL) "
        "OR (local_selector_id IS NOT NULL "
        "AND local_selector_kind IS NOT NULL "
        "AND local_selector_schema IS NOT NULL "
        "AND local_selector_sha256 IS NOT NULL "
        "AND local_selector_complete IN (0, 1))",
        name="local_selector_indivisible",
    ),
    CheckConstraint(
        "local_selector_kind IS NULL "
        "OR ((local_selector_kind = 'COMPLETE_RELEASE') = (local_selector_complete = 1))",
        name="local_selector_complete_agreement",
    ),
    _utc_timestamp("publication_at_utc", nullable=True),
    _utc_timestamp("observed_at_utc", nullable=True),
    CheckConstraint(
        "publication_at_utc IS NULL OR observed_at_utc IS NULL "
        f"OR {_utc_sort_key('publication_at_utc')} "
        f"<= {_utc_sort_key('observed_at_utc')}",
        name="publication_not_after_observation",
    ),
    CheckConstraint("mutable_snapshot IN (0, 1)", name="mutable_snapshot_boolean"),
    _canonical_json_array("expected_format_profile_ids_json", MAX_IDS_JSON_BYTES),
    CheckConstraint(
        "(expected_file_count_minimum IS NULL) = (expected_file_count_maximum IS NULL) "
        "AND (expected_file_count_minimum IS NULL "
        "OR (expected_file_count_minimum >= 0 "
        f"AND expected_file_count_maximum <= {MAX_DATASET_INTEGER} "
        "AND expected_file_count_minimum <= expected_file_count_maximum))",
        name="expected_file_count_range",
    ),
    CheckConstraint(
        "(expected_byte_count_minimum IS NULL) = (expected_byte_count_maximum IS NULL) "
        "AND (expected_byte_count_minimum IS NULL "
        "OR (expected_byte_count_minimum >= 0 "
        f"AND expected_byte_count_maximum <= {MAX_DATASET_INTEGER} "
        "AND expected_byte_count_minimum <= expected_byte_count_maximum))",
        name="expected_byte_count_range",
    ),
    _sha256_length("manifest_sha256", nullable=True),
    _sha256_length("expected_physical_sha256", nullable=True),
    _bounded_utf8("licence_evidence_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("access_evidence_id", MAX_DATASET_ID_BYTES, nullable=True),
    _canonical_json_array("evidence_ref_ids_json", MAX_IDS_JSON_BYTES),
    _canonical_json_array(
        "reason_codes_json",
        MAX_REASON_CODES_JSON_BYTES,
        maximum_items=MAX_DATASET_REASON_CODES,
    ),
    CheckConstraint(
        "access_state NOT IN ('UNAVAILABLE', 'UNKNOWN') "
        "OR json_array_length(reason_codes_json) > 0",
        name="unknown_access_reasoned",
    ),
)
Index(
    "ix_dataset_release_family_release",
    dataset_release.c.family_pk,
    dataset_release.c.release_id,
)
Index(
    "ix_dataset_release_provider_release",
    dataset_release.c.provider_id,
    dataset_release.c.release_id,
)
Index(
    "ix_dataset_release_access",
    dataset_release.c.access_state,
    dataset_release.c.release_id,
)


dataset_observation = Table(
    "dataset_observation",
    DATASET_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("observation_id"),
    Column("record_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("record_fingerprint"),
    Column("record_json", Text, nullable=False),
    _id("provider_id"),
    _id("adapter_key"),
    Column("adapter_version", String(MAX_DATASET_VERSION_BYTES), nullable=False),
    _sha256("adapter_sha256"),
    _id("query_id"),
    _sha256("canonical_request_sha256"),
    Column("retrieved_at_utc", String(32), nullable=False),
    _sha256("response_sha256"),
    Column("storage_root_pk", ForeignKey("storage_root.pk"), nullable=False),
    Column("response_relative_locator", Text, nullable=False),
    Column("page_index", Integer, nullable=False),
    Column("predecessor_observation_pk", ForeignKey("dataset_observation.pk")),
    _sha256("pagination_token_sha256", nullable=True),
    Column("truncated", Integer, nullable=False),
    Column("complete", Integer, nullable=False),
    Column("rate_limited", Integer, nullable=False),
    Column("schema_drift", Integer, nullable=False),
    Column("discovery_state", String(32), nullable=False),
    Column("candidate_release_ids_json", Text, nullable=False),
    Column("unresolved_lead_ids_json", Text, nullable=False),
    Column("evidence_ref_ids_json", Text, nullable=False),
    Column("reason_codes_json", Text, nullable=False),
    UniqueConstraint("observation_id"),
    UniqueConstraint("record_fingerprint"),
    _bounded_utf8("observation_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("record_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("record_fingerprint"),
    _canonical_json_object("record_json", MAX_DATASET_RECORD_JSON_BYTES),
    _bounded_utf8("provider_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("adapter_key", MAX_DATASET_ID_BYTES),
    _bounded_utf8("adapter_version", MAX_DATASET_VERSION_BYTES),
    _sha256_length("adapter_sha256"),
    _bounded_utf8("query_id", MAX_DATASET_ID_BYTES),
    _sha256_length("canonical_request_sha256"),
    _utc_timestamp("retrieved_at_utc"),
    _sha256_length("response_sha256"),
    _bounded_utf8("response_relative_locator", MAX_DATASET_LOCATOR_BYTES),
    CheckConstraint(
        f"page_index > 0 AND page_index <= {MAX_DATASET_INTEGER}",
        name="positive_bounded_page_index",
    ),
    CheckConstraint(
        "(page_index = 1 AND predecessor_observation_pk IS NULL) "
        "OR (page_index > 1 AND predecessor_observation_pk IS NOT NULL)",
        name="page_predecessor_agreement",
    ),
    _sha256_length("pagination_token_sha256", nullable=True),
    CheckConstraint("truncated IN (0, 1)", name="truncated_boolean"),
    CheckConstraint("complete IN (0, 1)", name="complete_boolean"),
    CheckConstraint("rate_limited IN (0, 1)", name="rate_limited_boolean"),
    CheckConstraint("schema_drift IN (0, 1)", name="schema_drift_boolean"),
    CheckConstraint(
        "complete = 0 OR (truncated = 0 AND rate_limited = 0 AND schema_drift = 0)",
        name="complete_observation_consistent",
    ),
    _enum("discovery_state", ("OBSERVED", "CANDIDATE", "DISMISSED", "WITHDRAWN")),
    _canonical_json_array("candidate_release_ids_json", MAX_IDS_JSON_BYTES),
    _canonical_json_array("unresolved_lead_ids_json", MAX_IDS_JSON_BYTES),
    _canonical_json_array("evidence_ref_ids_json", MAX_IDS_JSON_BYTES),
    _canonical_json_array(
        "reason_codes_json",
        MAX_REASON_CODES_JSON_BYTES,
        maximum_items=MAX_DATASET_REASON_CODES,
    ),
    CheckConstraint(
        "complete = 1 "
        "AND discovery_state NOT IN ('DISMISSED', 'WITHDRAWN') "
        "OR json_array_length(reason_codes_json) > 0",
        name="incomplete_observation_reasoned",
    ),
)
Index(
    "ix_dataset_observation_provider_retrieved",
    dataset_observation.c.provider_id,
    dataset_observation.c.retrieved_at_utc,
    dataset_observation.c.observation_id,
)
Index(
    "ix_dataset_observation_query_page",
    dataset_observation.c.query_id,
    dataset_observation.c.page_index,
)
Index(
    "ix_dataset_observation_discovery",
    dataset_observation.c.discovery_state,
    dataset_observation.c.observation_id,
)


dataset_materialization = Table(
    "dataset_materialization",
    DATASET_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("materialization_id"),
    Column("release_pk", ForeignKey("dataset_release.pk"), nullable=False),
    Column("record_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("record_fingerprint"),
    Column("record_json", Text, nullable=False),
    Column("materialization_class", String(48), nullable=False),
    Column("evidence_class", String(40), nullable=False),
    Column("outcome_access", String(40), nullable=False),
    Column("storage_root_pk", ForeignKey("storage_root.pk"), nullable=False),
    Column("relative_locator", Text, nullable=False),
    _id("selector_id"),
    Column("selector_kind", String(40), nullable=False),
    Column("selector_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("selector_sha256"),
    Column("selector_complete", Integer, nullable=False),
    _sha256("physical_sha256", nullable=True),
    Column("byte_size", BigInteger),
    Column("file_count", Integer),
    Column("media_type", String(256)),
    _id("format_profile_id", nullable=True),
    _id("logical_decoder_key", nullable=True),
    Column("logical_decoder_version", String(MAX_DATASET_VERSION_BYTES)),
    _sha256("logical_decoder_sha256", nullable=True),
    Column("logical_schema", String(MAX_DATASET_SCHEMA_BYTES)),
    _sha256("logical_sha256", nullable=True),
    Column("custody_state", String(32), nullable=False),
    _id("verifier_key", nullable=True),
    Column("verifier_version", String(MAX_DATASET_VERSION_BYTES)),
    _sha256("verifier_sha256", nullable=True),
    _id("verification_policy_id", nullable=True),
    _sha256("verification_policy_sha256", nullable=True),
    Column("verified_at_utc", String(32)),
    Column("verification_evidence_ref_ids_json", Text, nullable=False),
    Column("manifest_relative_locator", Text),
    Column("receipt_relative_locator", Text),
    Column("reason_codes_json", Text, nullable=False),
    UniqueConstraint("materialization_id"),
    UniqueConstraint("record_fingerprint"),
    UniqueConstraint("pk", "release_pk"),
    UniqueConstraint("storage_root_pk", "relative_locator", "selector_sha256"),
    _bounded_utf8("materialization_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("record_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("record_fingerprint"),
    _canonical_json_object("record_json", MAX_DATASET_RECORD_JSON_BYTES),
    _enum(
        "materialization_class",
        (
            "AUTHORITATIVE_SOURCE",
            "AUXILIARY_SOURCE",
            "HISTORICAL_DERIVATIVE",
            "TRANSFORMED_DERIVATIVE",
            "SEALED_EVALUATOR_ONLY",
            "SOFTWARE_RUNTIME_MEDIUM",
            "GENERATED_OBSERVATION",
            "METADATA_OBSERVATION",
            "NON_AUTHORITATIVE_WORKING_COPY",
        ),
    ),
    _enum(
        "evidence_class",
        (
            "EMPIRICAL_SOURCE",
            "ANALYTIC_REFERENCE",
            "SIMULATION",
            "HIL",
            "PHYSICAL_EXPERIMENT",
            "GENERATED_TRUTH",
            "DERIVED",
            "METADATA_ONLY",
            "SOFTWARE",
            "NON_AUTHORITATIVE",
        ),
    ),
    _enum(
        "outcome_access",
        (
            "outcome-blind",
            "development-visible",
            "evaluation-sealed",
            "evaluator-reveal",
            "evaluation-revealed",
            "privileged-truth",
        ),
    ),
    _bounded_utf8("relative_locator", MAX_DATASET_LOCATOR_BYTES),
    _bounded_utf8("selector_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("selector_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("selector_sha256"),
    _enum(
        "selector_kind",
        (
            "COMPLETE_RELEASE",
            "MANIFEST",
            "PARTITION",
            "SPECIMEN",
            "SWEEP",
            "TIME_RANGE",
            "SCENARIO",
            "COMPOSITE",
            "OTHER_REGISTERED",
        ),
    ),
    CheckConstraint("selector_complete IN (0, 1)", name="selector_complete_boolean"),
    CheckConstraint(
        "(selector_kind = 'COMPLETE_RELEASE') = (selector_complete = 1)",
        name="selector_complete_agreement",
    ),
    _sha256_length("physical_sha256", nullable=True),
    CheckConstraint(
        "(physical_sha256 IS NULL AND byte_size IS NULL AND file_count IS NULL) "
        "OR (physical_sha256 IS NOT NULL "
        f"AND byte_size BETWEEN 0 AND {MAX_DATASET_INTEGER} "
        f"AND file_count BETWEEN 1 AND {MAX_DATASET_INTEGER})",
        name="physical_facts_indivisible",
    ),
    _bounded_utf8("media_type", 256, nullable=True),
    _bounded_utf8("format_profile_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("logical_decoder_key", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8(
        "logical_decoder_version",
        MAX_DATASET_VERSION_BYTES,
        nullable=True,
    ),
    _sha256_length("logical_decoder_sha256", nullable=True),
    _bounded_utf8("logical_schema", MAX_DATASET_SCHEMA_BYTES, nullable=True),
    _sha256_length("logical_sha256", nullable=True),
    CheckConstraint(
        "(logical_decoder_key IS NULL "
        "AND logical_decoder_version IS NULL "
        "AND logical_decoder_sha256 IS NULL "
        "AND logical_schema IS NULL "
        "AND logical_sha256 IS NULL) "
        "OR (logical_decoder_key IS NOT NULL "
        "AND logical_decoder_version IS NOT NULL "
        "AND logical_decoder_sha256 IS NOT NULL "
        "AND logical_schema IS NOT NULL "
        "AND logical_sha256 IS NOT NULL)",
        name="logical_identity_indivisible",
    ),
    _enum(
        "custody_state",
        (
            "NOT_PRESENT",
            "UNVERIFIED",
            "PARTIAL",
            "VERIFIED",
            "QUARANTINED",
            "INVALID",
            "MISSING",
        ),
    ),
    _bounded_utf8("verifier_key", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("verifier_version", MAX_DATASET_VERSION_BYTES, nullable=True),
    _sha256_length("verifier_sha256", nullable=True),
    _bounded_utf8("verification_policy_id", MAX_DATASET_ID_BYTES, nullable=True),
    _sha256_length("verification_policy_sha256", nullable=True),
    _utc_timestamp("verified_at_utc", nullable=True),
    CheckConstraint(
        "(verifier_key IS NULL "
        "AND verifier_version IS NULL "
        "AND verifier_sha256 IS NULL "
        "AND verification_policy_id IS NULL "
        "AND verification_policy_sha256 IS NULL "
        "AND verified_at_utc IS NULL "
        "AND json_array_length(verification_evidence_ref_ids_json) = 0) "
        "OR (verifier_key IS NOT NULL "
        "AND verifier_version IS NOT NULL "
        "AND verifier_sha256 IS NOT NULL "
        "AND verification_policy_id IS NOT NULL "
        "AND verification_policy_sha256 IS NOT NULL "
        "AND verified_at_utc IS NOT NULL "
        "AND json_array_length(verification_evidence_ref_ids_json) > 0)",
        name="verification_proof_indivisible",
    ),
    _canonical_json_array("verification_evidence_ref_ids_json", MAX_IDS_JSON_BYTES),
    _bounded_utf8(
        "manifest_relative_locator", MAX_DATASET_LOCATOR_BYTES, nullable=True
    ),
    _bounded_utf8("receipt_relative_locator", MAX_DATASET_LOCATOR_BYTES, nullable=True),
    CheckConstraint(
        "(manifest_relative_locator IS NULL) = (receipt_relative_locator IS NULL)",
        name="external_manifest_receipt_indivisible",
    ),
    _canonical_json_array(
        "reason_codes_json",
        MAX_REASON_CODES_JSON_BYTES,
        maximum_items=MAX_DATASET_REASON_CODES,
    ),
    CheckConstraint(
        "custody_state NOT IN ('PARTIAL', 'QUARANTINED', 'INVALID', 'MISSING') "
        "OR json_array_length(reason_codes_json) > 0",
        name="exceptional_custody_reasoned",
    ),
)
Index(
    "ix_dataset_materialization_release_custody",
    dataset_materialization.c.release_pk,
    dataset_materialization.c.custody_state,
    dataset_materialization.c.materialization_id,
)
Index(
    "ix_dataset_materialization_custody",
    dataset_materialization.c.custody_state,
    dataset_materialization.c.materialization_id,
)


acquisition_attempt = Table(
    "acquisition_attempt",
    DATASET_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("attempt_id"),
    Column("release_pk", ForeignKey("dataset_release.pk"), nullable=False),
    Column("record_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("record_fingerprint"),
    Column("record_json", Text, nullable=False),
    _id("provider_capability_key"),
    _id("source_endpoint_id"),
    _id("preview_evidence_id"),
    Column("preview_evidence_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("preview_sha256"),
    Column("preview_storage_root_pk", ForeignKey("storage_root.pk"), nullable=False),
    Column("preview_relative_locator", Text, nullable=False),
    _sha256("expected_physical_sha256", nullable=True),
    Column("expected_size_bytes", BigInteger, nullable=False),
    Column("expected_format_profile_ids_json", Text, nullable=False),
    _id("authorization_evidence_id", nullable=True),
    Column("authorization_evidence_schema", String(MAX_DATASET_SCHEMA_BYTES)),
    _sha256("authorization_sha256", nullable=True),
    Column("authorization_storage_root_pk", ForeignKey("storage_root.pk")),
    Column("authorization_relative_locator", Text),
    Column("authorization_valid_from_utc", String(32)),
    Column("authorization_valid_until_utc", String(32)),
    Column("requested_bytes", BigInteger, nullable=False),
    Column("observed_bytes", BigInteger, nullable=False),
    Column("storage_root_pk", ForeignKey("storage_root.pk"), nullable=False),
    Column("temporary_relative_locator", Text),
    Column("final_relative_locator", Text),
    Column("retry_count", Integer, nullable=False),
    Column("resumed_from_attempt_pk", ForeignKey("acquisition_attempt.pk")),
    Column("acquisition_state", String(32), nullable=False),
    Column("result_materialization_pk", Integer),
    Column("reason_codes_json", Text, nullable=False),
    Column("evidence_ref_ids_json", Text, nullable=False),
    ForeignKeyConstraint(
        ("result_materialization_pk", "release_pk"),
        ("dataset_materialization.pk", "dataset_materialization.release_pk"),
    ),
    UniqueConstraint("attempt_id"),
    UniqueConstraint("record_fingerprint"),
    _bounded_utf8("attempt_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("record_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("record_fingerprint"),
    _canonical_json_object("record_json", MAX_DATASET_RECORD_JSON_BYTES),
    _bounded_utf8("provider_capability_key", MAX_DATASET_ID_BYTES),
    _bounded_utf8("source_endpoint_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("preview_evidence_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("preview_evidence_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("preview_sha256"),
    _bounded_utf8("preview_relative_locator", MAX_DATASET_LOCATOR_BYTES),
    _sha256_length("expected_physical_sha256", nullable=True),
    CheckConstraint(
        f"expected_size_bytes BETWEEN 0 AND {MAX_DATASET_INTEGER}",
        name="bounded_expected_size",
    ),
    _canonical_json_array("expected_format_profile_ids_json", MAX_IDS_JSON_BYTES),
    _bounded_utf8("authorization_evidence_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8(
        "authorization_evidence_schema", MAX_DATASET_SCHEMA_BYTES, nullable=True
    ),
    _sha256_length("authorization_sha256", nullable=True),
    _bounded_utf8(
        "authorization_relative_locator", MAX_DATASET_LOCATOR_BYTES, nullable=True
    ),
    _utc_timestamp("authorization_valid_from_utc", nullable=True),
    _utc_timestamp("authorization_valid_until_utc", nullable=True),
    CheckConstraint(
        "(authorization_evidence_id IS NULL "
        "AND authorization_evidence_schema IS NULL "
        "AND authorization_sha256 IS NULL "
        "AND authorization_storage_root_pk IS NULL "
        "AND authorization_relative_locator IS NULL "
        "AND authorization_valid_from_utc IS NULL "
        "AND authorization_valid_until_utc IS NULL) "
        "OR (authorization_evidence_id IS NOT NULL "
        "AND authorization_evidence_schema IS NOT NULL "
        "AND authorization_sha256 IS NOT NULL "
        "AND authorization_storage_root_pk IS NOT NULL "
        "AND authorization_relative_locator IS NOT NULL "
        "AND authorization_valid_from_utc IS NOT NULL "
        "AND authorization_valid_until_utc IS NOT NULL "
        f"AND {_utc_sort_key('authorization_valid_from_utc')} "
        f"< {_utc_sort_key('authorization_valid_until_utc')})",
        name="authorization_indivisible",
    ),
    CheckConstraint(
        f"requested_bytes BETWEEN 0 AND {MAX_DATASET_INTEGER} "
        f"AND observed_bytes BETWEEN 0 AND {MAX_DATASET_INTEGER} "
        "AND observed_bytes <= requested_bytes",
        name="bounded_observed_bytes",
    ),
    _bounded_utf8(
        "temporary_relative_locator", MAX_DATASET_LOCATOR_BYTES, nullable=True
    ),
    _bounded_utf8("final_relative_locator", MAX_DATASET_LOCATOR_BYTES, nullable=True),
    CheckConstraint(
        f"retry_count BETWEEN 0 AND {MAX_DATASET_INTEGER}",
        name="bounded_retry_count",
    ),
    CheckConstraint(
        "(retry_count = 0 AND resumed_from_attempt_pk IS NULL) "
        "OR (retry_count > 0 AND resumed_from_attempt_pk IS NOT NULL)",
        name="retry_resume_agreement",
    ),
    _enum(
        "acquisition_state",
        (
            "PREVIEWED",
            "AUTHORITY_REQUIRED",
            "ALREADY_PRESENT",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "BLOCKED",
        ),
    ),
    _canonical_json_array(
        "reason_codes_json",
        MAX_REASON_CODES_JSON_BYTES,
        maximum_items=MAX_DATASET_REASON_CODES,
    ),
    _canonical_json_array("evidence_ref_ids_json", MAX_IDS_JSON_BYTES),
    CheckConstraint(
        "acquisition_state NOT IN ('AUTHORITY_REQUIRED', 'ALREADY_PRESENT', 'FAILED', 'BLOCKED') "
        "OR json_array_length(reason_codes_json) > 0",
        name="terminal_stop_reasoned",
    ),
    CheckConstraint(
        "acquisition_state <> 'ALREADY_PRESENT' "
        "OR (result_materialization_pk IS NOT NULL "
        "AND requested_bytes = 0 AND observed_bytes = 0 "
        "AND temporary_relative_locator IS NULL AND final_relative_locator IS NULL "
        "AND authorization_evidence_id IS NULL)",
        name="already_present_zero_effect",
    ),
    CheckConstraint(
        "acquisition_state NOT IN ('PREVIEWED', 'AUTHORITY_REQUIRED') "
        "OR (observed_bytes = 0 AND authorization_evidence_id IS NULL "
        "AND temporary_relative_locator IS NULL AND final_relative_locator IS NULL "
        "AND result_materialization_pk IS NULL)",
        name="preview_zero_effect",
    ),
    CheckConstraint(
        "acquisition_state <> 'RUNNING' "
        "OR (authorization_evidence_id IS NOT NULL "
        "AND temporary_relative_locator IS NOT NULL "
        "AND final_relative_locator IS NULL AND result_materialization_pk IS NULL)",
        name="running_state_shape",
    ),
    CheckConstraint(
        "acquisition_state <> 'COMPLETED' "
        "OR (authorization_evidence_id IS NOT NULL "
        "AND final_relative_locator IS NOT NULL "
        "AND result_materialization_pk IS NOT NULL AND observed_bytes > 0)",
        name="completed_state_shape",
    ),
    CheckConstraint(
        "acquisition_state NOT IN ('FAILED', 'BLOCKED') "
        "OR (final_relative_locator IS NULL AND result_materialization_pk IS NULL)",
        name="failed_state_no_result",
    ),
)
Index(
    "ix_acquisition_attempt_release_state",
    acquisition_attempt.c.release_pk,
    acquisition_attempt.c.acquisition_state,
    acquisition_attempt.c.attempt_id,
)
Index(
    "ix_acquisition_attempt_state",
    acquisition_attempt.c.acquisition_state,
    acquisition_attempt.c.attempt_id,
)
Index(
    "ix_acquisition_attempt_provider_endpoint",
    acquisition_attempt.c.provider_capability_key,
    acquisition_attempt.c.source_endpoint_id,
)


experiment_dataset_binding = Table(
    "experiment_dataset_binding",
    DATASET_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("binding_id"),
    Column("record_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("record_fingerprint"),
    Column("record_json", Text, nullable=False),
    Column(
        "experiment_spec_id",
        String(MAX_DATASET_ID_BYTES),
        nullable=False,
    ),
    Column("run_id", String(MAX_DATASET_ID_BYTES), ForeignKey("workflow_run.run_id")),
    Column("release_pk", ForeignKey("dataset_release.pk"), nullable=False),
    Column("materialization_pk", Integer, nullable=False),
    _id("selector_id"),
    Column("selector_kind", String(40), nullable=False),
    Column("selector_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("selector_sha256"),
    Column("selector_complete", Integer, nullable=False),
    Column("binding_role", String(40), nullable=False),
    Column("outcome_access", String(40), nullable=False),
    _id("transform_reference_id", nullable=True),
    Column("transform_reference_schema", String(MAX_DATASET_SCHEMA_BYTES)),
    _sha256("transform_reference_sha256", nullable=True),
    Column("transform_storage_root_pk", ForeignKey("storage_root.pk")),
    Column("transform_relative_locator", Text),
    _id("transform_implementation_key", nullable=True),
    Column("transform_implementation_version", String(40)),
    _sha256("transform_implementation_sha256", nullable=True),
    Column("binding_state", String(32), nullable=False),
    Column("reason_codes_json", Text, nullable=False),
    _id("binding_receipt_id", nullable=True),
    Column("binding_receipt_schema", String(MAX_DATASET_SCHEMA_BYTES)),
    _sha256("binding_receipt_sha256", nullable=True),
    Column("binding_receipt_storage_root_pk", ForeignKey("storage_root.pk")),
    Column("binding_receipt_relative_locator", Text),
    Column("evidence_ref_ids_json", Text, nullable=False),
    ForeignKeyConstraint(
        ("materialization_pk", "release_pk"),
        ("dataset_materialization.pk", "dataset_materialization.release_pk"),
    ),
    UniqueConstraint("binding_id"),
    UniqueConstraint("record_fingerprint"),
    _bounded_utf8("binding_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("record_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("record_fingerprint"),
    _canonical_json_object("record_json", MAX_DATASET_RECORD_JSON_BYTES),
    _bounded_utf8("experiment_spec_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("run_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("selector_id", MAX_DATASET_ID_BYTES),
    _bounded_utf8("selector_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("selector_sha256"),
    _enum(
        "selector_kind",
        (
            "COMPLETE_RELEASE",
            "MANIFEST",
            "PARTITION",
            "SPECIMEN",
            "SWEEP",
            "TIME_RANGE",
            "SCENARIO",
            "COMPOSITE",
            "OTHER_REGISTERED",
        ),
    ),
    CheckConstraint("selector_complete IN (0, 1)", name="selector_complete_boolean"),
    CheckConstraint(
        "(selector_kind = 'COMPLETE_RELEASE') = (selector_complete = 1)",
        name="selector_complete_agreement",
    ),
    _enum(
        "binding_role",
        (
            "DEVELOPMENT",
            "CALIBRATION",
            "EVALUATION_SEALED",
            "EVALUATOR_REVEAL",
            "AUXILIARY",
            "COMPARATOR",
            "GENERATED_TRUTH",
            "PARENT",
        ),
    ),
    _enum(
        "outcome_access",
        (
            "outcome-blind",
            "development-visible",
            "evaluation-sealed",
            "evaluator-reveal",
            "evaluation-revealed",
            "privileged-truth",
        ),
    ),
    _bounded_utf8("transform_reference_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8(
        "transform_reference_schema", MAX_DATASET_SCHEMA_BYTES, nullable=True
    ),
    _sha256_length("transform_reference_sha256", nullable=True),
    _bounded_utf8(
        "transform_relative_locator", MAX_DATASET_LOCATOR_BYTES, nullable=True
    ),
    _bounded_utf8("transform_implementation_key", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("transform_implementation_version", 40, nullable=True),
    _sha256_length("transform_implementation_sha256", nullable=True),
    CheckConstraint(
        "(transform_reference_id IS NULL "
        "AND transform_reference_schema IS NULL "
        "AND transform_reference_sha256 IS NULL "
        "AND transform_storage_root_pk IS NULL "
        "AND transform_relative_locator IS NULL "
        "AND transform_implementation_key IS NULL "
        "AND transform_implementation_version IS NULL "
        "AND transform_implementation_sha256 IS NULL) "
        "OR (transform_reference_id IS NOT NULL "
        "AND transform_reference_schema IS NOT NULL "
        "AND transform_reference_sha256 IS NOT NULL "
        "AND transform_storage_root_pk IS NOT NULL "
        "AND transform_relative_locator IS NOT NULL "
        "AND transform_implementation_key IS NOT NULL "
        "AND transform_implementation_version IS NOT NULL "
        "AND transform_implementation_sha256 IS NOT NULL)",
        name="transform_reference_indivisible",
    ),
    _enum("binding_state", ("PROPOSED", "VERIFIED", "STALE", "INVALID")),
    _canonical_json_array(
        "reason_codes_json",
        MAX_REASON_CODES_JSON_BYTES,
        maximum_items=MAX_DATASET_REASON_CODES,
    ),
    _bounded_utf8("binding_receipt_id", MAX_DATASET_ID_BYTES, nullable=True),
    _bounded_utf8("binding_receipt_schema", MAX_DATASET_SCHEMA_BYTES, nullable=True),
    _sha256_length("binding_receipt_sha256", nullable=True),
    _bounded_utf8(
        "binding_receipt_relative_locator", MAX_DATASET_LOCATOR_BYTES, nullable=True
    ),
    CheckConstraint(
        "(binding_receipt_id IS NULL "
        "AND binding_receipt_schema IS NULL "
        "AND binding_receipt_sha256 IS NULL "
        "AND binding_receipt_storage_root_pk IS NULL "
        "AND binding_receipt_relative_locator IS NULL) "
        "OR (binding_receipt_id IS NOT NULL "
        "AND binding_receipt_schema IS NOT NULL "
        "AND binding_receipt_sha256 IS NOT NULL "
        "AND binding_receipt_storage_root_pk IS NOT NULL "
        "AND binding_receipt_relative_locator IS NOT NULL)",
        name="binding_receipt_indivisible",
    ),
    _canonical_json_array("evidence_ref_ids_json", MAX_IDS_JSON_BYTES),
    CheckConstraint(
        "(binding_state = 'VERIFIED' AND binding_receipt_id IS NOT NULL "
        "AND json_array_length(evidence_ref_ids_json) > 0) "
        "OR (binding_state <> 'VERIFIED' AND binding_receipt_id IS NULL)",
        name="verified_binding_receipted",
    ),
    CheckConstraint(
        "binding_state NOT IN ('STALE', 'INVALID') OR json_array_length(reason_codes_json) > 0",
        name="invalid_binding_reasoned",
    ),
    CheckConstraint(
        "binding_role <> 'EVALUATION_SEALED' OR outcome_access = 'evaluation-sealed'",
        name="sealed_role_access",
    ),
    CheckConstraint(
        "binding_role <> 'EVALUATOR_REVEAL' OR outcome_access = 'evaluator-reveal'",
        name="reveal_role_access",
    ),
)
Index(
    "ix_experiment_dataset_binding_experiment_role",
    experiment_dataset_binding.c.experiment_spec_id,
    experiment_dataset_binding.c.binding_role,
    experiment_dataset_binding.c.binding_id,
)
Index(
    "ix_experiment_dataset_binding_role",
    experiment_dataset_binding.c.binding_role,
    experiment_dataset_binding.c.binding_id,
)
Index("ix_experiment_dataset_binding_run", experiment_dataset_binding.c.run_id)


dataset_external_identifier = Table(
    "dataset_external_identifier",
    DATASET_METADATA,
    Column("pk", Integer, primary_key=True),
    Column("identifier_fingerprint", LargeBinary(32), nullable=False),
    Column("owner_kind", String(16), nullable=False),
    Column("family_pk", ForeignKey("dataset_family.pk")),
    Column("release_pk", ForeignKey("dataset_release.pk")),
    Column("observation_pk", ForeignKey("dataset_observation.pk")),
    Column("identifier_kind", String(40), nullable=False),
    _id("namespace"),
    Column("value", Text, nullable=False),
    UniqueConstraint("family_pk", "identifier_kind", "namespace", "value"),
    UniqueConstraint("release_pk", "identifier_kind", "namespace", "value"),
    UniqueConstraint("observation_pk", "identifier_kind", "namespace", "value"),
    _sha256_length("identifier_fingerprint"),
    _enum("owner_kind", ("FAMILY", "RELEASE", "OBSERVATION")),
    CheckConstraint(
        "(owner_kind = 'FAMILY' AND family_pk IS NOT NULL "
        "AND release_pk IS NULL AND observation_pk IS NULL) "
        "OR (owner_kind = 'RELEASE' AND family_pk IS NULL "
        "AND release_pk IS NOT NULL AND observation_pk IS NULL) "
        "OR (owner_kind = 'OBSERVATION' AND family_pk IS NULL "
        "AND release_pk IS NULL AND observation_pk IS NOT NULL)",
        name="exactly_one_owner",
    ),
    _enum(
        "identifier_kind",
        (
            "PROVIDER_RELEASE",
            "ACCESSION",
            "DEPOSIT",
            "REPOSITORY_COMMIT",
            "IMMUTABLE_SNAPSHOT",
            "PROVIDER_COLLECTION",
            "PROVIDER_OBJECT",
            "OTHER_REGISTERED",
        ),
    ),
    _bounded_utf8("namespace", MAX_DATASET_ID_BYTES),
    _bounded_utf8("value", MAX_EXTERNAL_IDENTIFIER_BYTES),
)
Index(
    "ix_dataset_external_identifier_lookup",
    dataset_external_identifier.c.identifier_kind,
    dataset_external_identifier.c.namespace,
    dataset_external_identifier.c.value,
)
Index("ix_dataset_external_identifier_family", dataset_external_identifier.c.family_pk)
Index(
    "ix_dataset_external_identifier_release", dataset_external_identifier.c.release_pk
)
Index(
    "ix_dataset_external_identifier_observation",
    dataset_external_identifier.c.observation_pk,
)


dataset_projection_state = Table(
    "dataset_projection_state",
    DATASET_METADATA,
    Column("state_id", Integer, primary_key=True),
    Column("snapshot_schema", String(MAX_DATASET_SCHEMA_BYTES), nullable=False),
    _sha256("snapshot_fingerprint"),
    Column("family_count", BigInteger, nullable=False),
    Column("release_count", BigInteger, nullable=False),
    Column("observation_count", BigInteger, nullable=False),
    Column("materialization_count", BigInteger, nullable=False),
    Column("acquisition_attempt_count", BigInteger, nullable=False),
    Column("binding_count", BigInteger, nullable=False),
    Column("canonical_byte_count", BigInteger, nullable=False),
    Column("projection_valid", Integer, nullable=False),
    Column("projection_revision", BigInteger, nullable=False),
    CheckConstraint("state_id = 1", name="singleton_state_id"),
    _bounded_utf8("snapshot_schema", MAX_DATASET_SCHEMA_BYTES),
    _sha256_length("snapshot_fingerprint"),
    CheckConstraint(
        "family_count BETWEEN 0 AND {maximum} "
        "AND release_count BETWEEN 0 AND {maximum} "
        "AND observation_count BETWEEN 0 AND {maximum} "
        "AND materialization_count BETWEEN 0 AND {maximum} "
        "AND acquisition_attempt_count BETWEEN 0 AND {maximum} "
        "AND binding_count BETWEEN 0 AND {maximum}".format(maximum=MAX_DATASET_INTEGER),
        name="bounded_entity_counts",
    ),
    CheckConstraint(
        f"canonical_byte_count BETWEEN 0 AND {MAX_DATASET_INTEGER}",
        name="bounded_canonical_byte_count",
    ),
    CheckConstraint("projection_valid IN (0, 1)", name="projection_valid_boolean"),
    CheckConstraint(
        f"projection_revision BETWEEN 0 AND {MAX_DATASET_INTEGER}",
        name="bounded_projection_revision",
    ),
)


DATASET_TABLE_NAMES: Final[tuple[str, ...]] = (
    "dataset_family",
    "dataset_release",
    "dataset_observation",
    "dataset_materialization",
    "acquisition_attempt",
    "experiment_dataset_binding",
    "dataset_external_identifier",
    "dataset_projection_state",
)
DATASET_TABLES: Final[tuple[Table, ...]] = tuple(
    DATASET_METADATA.tables[table_name] for table_name in DATASET_TABLE_NAMES
)
DATASET_VIEW_NAMES: Final[tuple[str, ...]] = ()
DATASET_EXPLICIT_INDEX_NAMES: Final[tuple[str, ...]] = tuple(
    sorted(
        str(index.name)
        for table in DATASET_TABLES
        for index in table.indexes
        if index.name is not None
    )
)
DATASET_PROJECTION_MUTATION_TABLE_NAMES: Final[tuple[str, ...]] = (
    "dataset_family",
    "dataset_release",
    "dataset_observation",
    "dataset_materialization",
    "acquisition_attempt",
    "experiment_dataset_binding",
    "dataset_external_identifier",
)
DATASET_PROJECTION_DIRTY_TRIGGER_NAMES: Final[tuple[str, ...]] = tuple(
    f"trg_{table_name}_projection_dirty_after_{operation}"
    for table_name in DATASET_PROJECTION_MUTATION_TABLE_NAMES
    for operation in ("insert", "update", "delete")
)
DATASET_PROJECTION_DIRTY_TRIGGER_SQL: Final[tuple[str, ...]] = tuple(
    "CREATE TRIGGER "
    f"{trigger_name} AFTER {operation.upper()} ON {table_name} "
    "BEGIN UPDATE dataset_projection_state "
    "SET projection_valid = 0, projection_revision = projection_revision + 1 "
    "WHERE state_id = 1; END"
    for table_name in DATASET_PROJECTION_MUTATION_TABLE_NAMES
    for operation, trigger_name in zip(
        ("insert", "update", "delete"),
        (
            f"trg_{table_name}_projection_dirty_after_insert",
            f"trg_{table_name}_projection_dirty_after_update",
            f"trg_{table_name}_projection_dirty_after_delete",
        ),
        strict=True,
    )
)


def normalize_dataset_projection_trigger_sql(value: str) -> str:
    """Return the exact whitespace-normalized trigger definition used at reads."""

    if not isinstance(value, str):
        raise TypeError("dataset projection trigger SQL must be text")
    normalized = " ".join(value.strip().rstrip(";").split())
    if not normalized or len(normalized.encode("utf-8")) > 4096:
        raise ValueError("dataset projection trigger SQL lies outside its bound")
    return normalized


DATASET_PROJECTION_DIRTY_TRIGGER_DEFINITIONS: Final[tuple[tuple[str, str], ...]] = (
    tuple(
        sorted(
            (
                trigger_name,
                normalize_dataset_projection_trigger_sql(trigger_sql),
            )
            for trigger_name, trigger_sql in zip(
                DATASET_PROJECTION_DIRTY_TRIGGER_NAMES,
                DATASET_PROJECTION_DIRTY_TRIGGER_SQL,
                strict=True,
            )
        )
    )
)
DATASET_SCHEMA_OBJECT_NAMES: Final[tuple[str, ...]] = (
    *DATASET_TABLE_NAMES,
    *DATASET_VIEW_NAMES,
    *DATASET_EXPLICIT_INDEX_NAMES,
    *DATASET_PROJECTION_DIRTY_TRIGGER_NAMES,
)
if len(DATASET_SCHEMA_OBJECT_NAMES) != len(set(DATASET_SCHEMA_OBJECT_NAMES)):
    raise RuntimeError("dataset schema object names must be globally unique")


EMPTY_DATASET_SNAPSHOT_SCHEMA: Final = (
    'empirical-lawhood/runtime/dataset-catalog-snapshot'
)
EMPTY_DATASET_SNAPSHOT_FINGERPRINT: Final = bytes.fromhex(
    "4e4f27b35e3a61d05801e34c90174cdd0c466cd60204a98ab5bf331d19fd3a8c"
)
EMPTY_DATASET_SNAPSHOT_CANONICAL_BYTES: Final = 200
