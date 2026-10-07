"""Dependency-inversion ports used by the stable application facade."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.design import ExperimentProposal
from empirical_lawhood.planning.exploration import ExploratoryFinding, HypothesisSet
from empirical_lawhood.planning.prospective import (
    NominationObligations,
    ProspectiveDesignContext,
    ProspectiveNominationDecision,
)
from empirical_lawhood.runtime.dataset_catalog_service import DatasetCatalogReadService


class DatasetCatalogSessionProvider(Protocol):
    """Open one authenticated read-only dataset catalog session."""

    def open(self) -> AbstractContextManager[DatasetCatalogReadService]: ...


class ProspectiveWorkflow(Protocol):
    """Nominate and design only; authority remains a separate planning gate."""

    def nominate(
        self,
        *,
        hypotheses: HypothesisSet,
        findings: tuple[ExploratoryFinding, ...],
        system: SystemSpec,
        context: ProspectiveDesignContext,
    ) -> tuple[ProspectiveNominationDecision, tuple[NominationObligations, ...]]: ...

    def design(
        self,
        *,
        decision: ProspectiveNominationDecision,
        obligations: tuple[NominationObligations, ...],
        system: SystemSpec,
    ) -> tuple[ExperimentProposal, ...]: ...
