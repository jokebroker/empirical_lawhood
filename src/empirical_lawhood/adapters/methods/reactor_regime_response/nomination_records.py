"""Persisted B nomination choices and complete development census."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from math import isfinite
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .config import ROOTS
from .information_models import FrozenInformationRoster
from .model_records import RegimeFitPackage
from .model_selection import Candidate
from .route_nomination import NominationOpportunity, ROUTES, RouteCandidate


def _number(value: float | None) -> D | None:
    return None if value is None or not isfinite(value) else D(repr(value))


@dataclass(frozen=True, slots=True)
class NominationScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/nomination-score'

    model_id: str
    root_losses_K2: tuple[tuple[str, D], ...]
    mean_loss_K2: D | None
    leaf_contacts: tuple[int, ...]
    reasons: tuple[str, ...]

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> NominationScore:
        return cls(
            candidate.model_id,
            tuple((root, D(repr(value))) for root, value in sorted(candidate.nomination_root_losses_K2.items())),
            _number(candidate.nomination_mean_loss_K2),
            candidate.nomination_leaf_contacts,
            candidate.reasons,
        )


@dataclass(frozen=True, slots=True)
class ReactorRegimeNominationOpportunity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-regime-nomination-opportunity'

    root: str
    choices: tuple[int | None, int | None, int | None, int | None]
    joined_service: bool
    chart_valid: bool
    covered: bool
    support_valid: bool
    preparation_safe: bool
    reasons: tuple[str, ...]

    @classmethod
    def from_opportunity(cls, value: NominationOpportunity) -> ReactorRegimeNominationOpportunity:
        return cls(
            value.root, value.choices, value.joined_service, value.chart_valid,
            value.covered, value.support_valid, value.preparation_safe, value.reasons,
        )


@dataclass(frozen=True, slots=True)
class NominationRoute(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/nomination-route'

    route: str
    coefficient_model_id: str | None
    absolute_model_id: str | None
    direct_model_id: str | None
    provisional_q_temperature: D | None
    provisional_q_cooling: D | None
    mean_contrast_loss_K2: D | None
    joined_service_roots: int
    opportunities: tuple[ReactorRegimeNominationOpportunity, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.route not in ROUTES or not 0 <= self.joined_service_roots <= 16:
            raise ValueError("nomination route or service census differs")
        if self.coefficient_model_id is None and not self.reasons:
            raise ValueError("unavailable route lacks its scientific reason")
        if self.coefficient_model_id is not None and (
            self.absolute_model_id is None
            or len(self.opportunities) != 16
            or tuple(row.root for row in self.opportunities)
            != tuple(root for root, role, _, _ in ROOTS if role == "nomination")
        ):
            raise ValueError("nominated route lacks its all-assigned opportunity census")

    @classmethod
    def from_candidate(cls, route: str, value: RouteCandidate | None, reason: str = "") -> NominationRoute:
        if value is None:
            return cls(route, None, None, None, None, None, None, 0, (), (reason,))
        if value.route != route:
            raise ValueError("nomination candidate route differs")
        return cls(
            route,
            value.coefficient.model_id,
            value.absolute.model_id,
            None if value.direct is None else value.direct.model_id,
            _number(value.provisional_q_temperature),
            _number(value.provisional_q_cooling),
            _number(value.mean_contrast_loss_K2),
            value.joined_service_roots,
            tuple(ReactorRegimeNominationOpportunity.from_opportunity(item) for item in value.opportunities),
            (),
        )


@dataclass(frozen=True, slots=True)
class NominationInformation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/nomination-information'

    pair_id: str
    spec_id: str | None
    arm_losses_K2: tuple[tuple[str, D], ...]
    reasons: tuple[str, ...]

    @classmethod
    def from_selected(
        cls,
        pair_id: str,
        selected: FrozenInformationRoster | None,
        reasons: tuple[str, ...],
    ) -> NominationInformation:
        return cls(
            pair_id,
            None if selected is None else selected.spec.spec_id,
            () if selected is None else tuple((arm, D(repr(loss))) for arm, loss in selected.nomination_losses),
            reasons,
        )


@dataclass(frozen=True, slots=True)
class RegimeNominationPackage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-nomination-package'

    package_id: str
    fit_package: ObjectIdentity
    nomination_causal: tuple[ObjectIdentity, ...]
    nomination_assays: tuple[ObjectIdentity, ...]
    coefficient_scores: tuple[NominationScore, ...]
    absolute_scores: tuple[NominationScore, ...]
    information: tuple[NominationInformation, ...]
    routes: tuple[NominationRoute, NominationRoute, NominationRoute]
    selected_route: str | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        roots = tuple(root for root, role, _, _ in ROOTS if role == "nomination")
        if (
            self.package_id != "reactor-regime-response-nomination-package"
            or self.fit_package.object_schema != RegimeFitPackage.SCHEMA
            or tuple(value.object_id for value in self.nomination_causal)
            != tuple(f"{root}.causal-preparation" for root in roots)
            or tuple(value.object_id for value in self.nomination_assays)
            != tuple(f"{root}.assay-panel" for root in roots)
            or len(self.coefficient_scores) != 32
            or len(self.absolute_scores) != 24
            or tuple(value.pair_id for value in self.information) != ("S_H", "S_I", "S_Q")
            or tuple(value.route for value in self.routes) != ROUTES
            or self.selected_route not in (*ROUTES, None)
            or (
                self.selected_route is not None
                and next(row for row in self.routes if row.route == self.selected_route).coefficient_model_id is None
            )
        ):
            raise ValueError("nomination package changed its fixed development census")
