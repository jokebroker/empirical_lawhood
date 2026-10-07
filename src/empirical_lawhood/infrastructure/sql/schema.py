'Normalized bounded SQLite schema for the rebuildable catalog projection.\n\nThe runtime owns SQLAlchemy objects distinct from the public initialization\nso subsequent migrations cannot alter its released table definitions.'

from __future__ import annotations

from typing import Final

from sqlalchemy import MetaData, Table

from .migrations.catalog_tables import CATALOG_METADATA as _INITIAL_CATALOG_METADATA, NAMING_CONVENTION as _CATALOG_NAMING_CONVENTION, PROJECTION_TABLE_NAMES as _CATALOG_PROJECTION_TABLE_NAMES
from .migrations.dataset_tables import DATASET_METADATA as _DATASET_METADATA, DATASET_TABLE_NAMES as _DATASET_TABLE_NAMES

NAMING_CONVENTION: Final[dict[str, str]] = dict(_CATALOG_NAMING_CONVENTION)
PROJECTION_TABLE_NAMES: Final[tuple[str, ...]] = _CATALOG_PROJECTION_TABLE_NAMES
DATASET_TABLE_NAMES: Final[tuple[str, ...]] = _DATASET_TABLE_NAMES

metadata: Final[MetaData] = MetaData(naming_convention=NAMING_CONVENTION)
for _catalog_table in _INITIAL_CATALOG_METADATA.sorted_tables:
    _catalog_table.to_metadata(metadata)
for _dataset_table_name in DATASET_TABLE_NAMES:
    _DATASET_METADATA.tables[_dataset_table_name].to_metadata(metadata)

storage_root: Final[Table] = metadata.tables["storage_root"]
logical_artifact: Final[Table] = metadata.tables["logical_artifact"]
artifact: Final[Table] = metadata.tables["artifact"]
receipt: Final[Table] = metadata.tables["receipt"]
scientific_object: Final[Table] = metadata.tables["scientific_object"]

projection_tables: Final[dict[str, Table]] = {
    table_name: metadata.tables[table_name] for table_name in PROJECTION_TABLE_NAMES
}

knowledge_edge: Final[Table] = metadata.tables["knowledge_edge"]
metric_definition: Final[Table] = metadata.tables["metric_definition"]
metric_observation: Final[Table] = metadata.tables["metric_observation"]
exploration_attempt: Final[Table] = metadata.tables["exploration_attempt"]
exploration_attempt_reason: Final[Table] = metadata.tables["exploration_attempt_reason"]
exploration_attempt_artifact: Final[Table] = metadata.tables["exploration_attempt_artifact"]
workflow_run: Final[Table] = metadata.tables["workflow_run"]
task_attempt: Final[Table] = metadata.tables["task_attempt"]
lease: Final[Table] = metadata.tables["lease"]
run_event: Final[Table] = metadata.tables["run_event"]

dataset_family: Final[Table] = metadata.tables["dataset_family"]
dataset_release: Final[Table] = metadata.tables["dataset_release"]
dataset_observation: Final[Table] = metadata.tables["dataset_observation"]
dataset_materialization: Final[Table] = metadata.tables["dataset_materialization"]
acquisition_attempt: Final[Table] = metadata.tables["acquisition_attempt"]
experiment_dataset_binding: Final[Table] = metadata.tables["experiment_dataset_binding"]
dataset_external_identifier: Final[Table] = metadata.tables["dataset_external_identifier"]
dataset_projection_state: Final[Table] = metadata.tables["dataset_projection_state"]
dataset_tables: Final[dict[str, Table]] = {
    table_name: metadata.tables[table_name] for table_name in DATASET_TABLE_NAMES
}
