"""Infrastructure composition for authenticated campaign dataset resolution."""

from __future__ import annotations

from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.planning.datasets import ExperimentDatasetBinding
from empirical_lawhood.runtime.dataset_binding import resolve_experiment_dataset_bindings
from empirical_lawhood.runtime.datasets import DatasetCatalogWorkLimit

from .dataset_catalog_access import DatasetCatalogReadSessionProvider


class AuthenticatedDatasetCampaignBindingResolver:
    """Resolve only after the installed projection matches external authority."""

    def __init__(
        self,
        session_provider: DatasetCatalogReadSessionProvider,
        *,
        work_limit: DatasetCatalogWorkLimit,
    ) -> None:
        if not isinstance(session_provider, DatasetCatalogReadSessionProvider):
            raise TypeError("session_provider must be a DatasetCatalogReadSessionProvider")
        if not isinstance(work_limit, DatasetCatalogWorkLimit):
            raise TypeError("work_limit must be a DatasetCatalogWorkLimit")
        self._session_provider = session_provider
        self._work_limit = work_limit

    def resolve(
        self,
        experiment: ExperimentSpec,
        *,
        run_id: str,
    ) -> tuple[ExperimentDatasetBinding, ...]:
        with self._session_provider.open() as service:
            service.authenticate_projection(self._work_limit)
        return resolve_experiment_dataset_bindings(
            experiment=experiment,
            run_id=run_id,
            snapshot=self._session_provider.trusted_snapshot,
            anchor=self._session_provider.trusted_anchor,
        )


__all__ = ["AuthenticatedDatasetCampaignBindingResolver"]
