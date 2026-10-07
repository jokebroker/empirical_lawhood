"""Canonical NIMS material corpus and family-held search-world construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import re
import struct
from statistics import median
from typing import Iterable, Mapping

from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import (
    CandidateView,
    DiscoveryObservation,
    LabelState,
    ObservationOrigin,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import canonical_json_bytes, validate_stable_id

from .contracts import UNPARTITIONED_CANDIDATE_ALGORITHM_ID, MaterialFamilyConfig, PHASE_DISJOINT_CANDIDATE_ALGORITHM_ID, ReproductionDisposition, MaterialFamilyDiscoveryPhase, SourceQualification


# Historical hash domain preserves the frozen family partition and trailing NUL.
_HISTORICAL_PHASE_PARTITION_DOMAIN = b"phase-partition-dev75-v2\x00"


ELEMENTS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
ELEMENT_INDEX = {value: index for index, value in enumerate(ELEMENTS)}
FEATURE_SCHEMA_ID = "features.nims-stoichiometry-presence-f32-236"
FEATURE_WIDTH = len(ELEMENTS) * 2
_FLOAT = re.compile(r"^[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
_FAMILY_SLUG = re.compile(r"[^a-z0-9]+")
_PAIR_PREFIXES = ("ma", "mb", "mc", "md", "me", "mf", "mg", "mh", "mi", "mj", "mo")


@dataclass(frozen=True, slots=True)
class MaterialCandidate:
    candidate_id: str
    formula_sha256: str
    features: tuple[float, ...]
    stratum_id: str
    family_id: str | None
    family_label: str | None
    tc_kelvin: Decimal | None
    label_state: LabelState
    source_row_count: int
    ambiguous_family: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if len(self.features) != FEATURE_WIDTH:
            raise ValueError("material feature width differs")

    def policy_view(self) -> CandidateView:
        return CandidateView(
            candidate_id=self.candidate_id,
            stratum_id=self.stratum_id,
            features=self.features,
        )

    def initial_observation(self) -> DiscoveryObservation:
        if self.tc_kelvin is None:
            raise ValueError("unlabelled material cannot enter the initial labelled set")
        return DiscoveryObservation(
            candidate_id=self.candidate_id,
            origin=ObservationOrigin.INITIAL_LABEL,
            state=self.label_state,
            tc_kelvin=self.tc_kelvin,
            query_cost=Decimal(0),
            round_index=None,
        )


@dataclass(frozen=True, slots=True)
class MaterialCorpus:
    corpus_id: str
    candidates: tuple[MaterialCandidate, ...]
    family_labels: tuple[tuple[str, str], ...]
    qualification: SourceQualification

    def __post_init__(self) -> None:
        validate_stable_id(self.corpus_id, field_name="corpus_id")
        ids = tuple(value.candidate_id for value in self.candidates)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("material corpus candidates must be sorted and unique")
        family_ids = tuple(value[0] for value in self.family_labels)
        if family_ids != tuple(sorted(set(family_ids))):
            raise ValueError("material family labels must be sorted and unique")

    @property
    def by_id(self) -> dict[str, MaterialCandidate]:
        return {value.candidate_id: value for value in self.candidates}


@dataclass(frozen=True, slots=True)
class MaterialSearchWorld:
    world_id: str
    target_family_id: str
    target_family_label: str
    policy_candidates: tuple[CandidateView, ...]
    initial_observations: tuple[DiscoveryObservation, ...]
    candidate_pool_ids: tuple[str, ...]
    target_candidate_ids: frozenset[str]
    candidate_truth: tuple[MaterialCandidate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.world_id, field_name="world_id")
        validate_stable_id(self.target_family_id, field_name="target_family_id")
        policy_ids = {value.candidate_id for value in self.policy_candidates}
        initial_ids = {value.candidate_id for value in self.initial_observations}
        if not initial_ids.issubset(policy_ids):
            raise ValueError("initial labels lack policy-visible features")
        if self.candidate_pool_ids != tuple(sorted(set(self.candidate_pool_ids))):
            raise ValueError("candidate pool IDs must be sorted and unique")
        if initial_ids & set(self.candidate_pool_ids):
            raise ValueError("initial labels leak into the candidate pool")
        if not self.target_candidate_ids.issubset(self.candidate_pool_ids):
            raise ValueError("target candidates must lie in the eligible pool")
        truth_ids = tuple(value.candidate_id for value in self.candidate_truth)
        if truth_ids != self.candidate_pool_ids:
            raise ValueError("candidate truth must align exactly with the sorted pool")


def _strict_decimal(value: str) -> Decimal | None:
    text = value.strip()
    if not _FLOAT.fullmatch(text):
        return None
    try:
        result = Decimal(text)
    except InvalidOperation:
        return None
    if not result.is_finite():
        return None
    return result


def _normal_formula(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]", "", value).casefold()


def _family(value: str) -> tuple[str, str] | None:
    label = " ".join(value.split()).casefold()
    if not label:
        return None
    slug = _FAMILY_SLUG.sub("-", label).strip("-")[:40] or "unmapped"
    suffix = sha256(label.encode("utf-8")).hexdigest()[:10]
    return f"family.{slug}.{suffix}", label


def family_identity(value: str) -> tuple[str, str]:
    """Return the stable exact-``str3`` family identity used by configs."""

    result = _family(value)
    if result is None:
        raise ValueError("blank str3 value has no family identity")
    return result


def _composition(row: Mapping[str, str]) -> tuple[float, ...] | None:
    amounts: dict[str, Decimal] = defaultdict(Decimal)
    present: set[str] = set()
    for prefix in _PAIR_PREFIXES:
        element = row[f"{prefix}1"].strip()
        if not element:
            continue
        if element not in ELEMENT_INDEX:
            return None
        present.add(element)
        amount = _strict_decimal(row[f"{prefix}2"])
        if amount is not None and amount > 0:
            amounts[element] += amount
    if not present:
        return None
    total = sum(amounts.values(), Decimal(0))
    stoichiometry = tuple(
        float(amounts.get(element, Decimal(0)) / total) if total > 0 else 0.0
        for element in ELEMENTS
    )
    presence = tuple(float(element in present) for element in ELEMENTS)
    return (*stoichiometry, *presence)


def _resolve_tc(row: Mapping[str, str]) -> tuple[Decimal | None, bool, bool]:
    tc = _strict_decimal(row["tc"])
    filled_detection_limit = False
    above_ceiling = False
    if tc is None:
        lower_limit = _strict_decimal(row["tcn"])
        if lower_limit is not None and lower_limit < Decimal(5):
            tc = Decimal(0)
            filled_detection_limit = True
    if tc is not None and tc > Decimal(150):
        tc = None
        above_ceiling = True
    return tc, filled_detection_limit, above_ceiling


def build_material_corpus(
    *,
    rows: Iterable[Mapping[str, str]],
    source_manifest_sha256: str,
    qualified_object_ids: tuple[str, ...],
    o_and_m_row_count: int,
    organic_row_count: int,
) -> MaterialCorpus:
    """Aggregate source rows at the canonical material, never nested-row, unit."""

    grouped: dict[str, list[tuple[Mapping[str, str], tuple[float, ...], Decimal | None]]] = (
        defaultdict(list)
    )
    resolved_rows = detection_rows = unlabelled_rows = above_rows = 0
    for row in rows:
        tc, filled, above = _resolve_tc(row)
        detection_rows += int(filled)
        above_rows += int(above)
        if tc is None:
            unlabelled_rows += 1
        else:
            resolved_rows += 1
        formula = _normal_formula(row["element"])
        features = _composition(row)
        if not formula or features is None:
            continue
        grouped[formula].append((row, features, tc))

    candidates: list[MaterialCandidate] = []
    family_labels: dict[str, str] = {}
    ambiguous_count = 0
    for formula, source_rows in grouped.items():
        formula_sha = sha256(formula.encode("ascii")).hexdigest()
        families = {
            parsed for row, _, _ in source_rows if (parsed := _family(row["str3"])) is not None
        }
        family_id: str | None = None
        family_label: str | None = None
        ambiguous = len(families) > 1
        if len(families) == 1:
            family_id, family_label = next(iter(families))
            family_labels[family_id] = family_label
        elif ambiguous:
            ambiguous_count += 1
        tc_values = [value for _, _, value in source_rows if value is not None]
        tc_value = median(tc_values) if tc_values else None
        if tc_value is None:
            state = LabelState.UNLABELLED
        elif tc_value < Decimal(5):
            state = LabelState.EXPLICIT_NEGATIVE
        else:
            state = LabelState.MEASURED
        # Duplicates share a canonical formula. Averaging their explicit
        # numeric-stoichiometry plus presence views is deterministic and never
        # increases replication. Variable stoichiometries contribute presence
        # but never a fabricated numeric coefficient.
        feature_matrix = tuple(value for _, value, _ in source_rows)
        features = tuple(
            sum(row[index] for row in feature_matrix) / len(feature_matrix)
            for index in range(FEATURE_WIDTH)
        )
        element_count = sum(value > 0 for value in features[len(ELEMENTS) :])
        candidates.append(
            MaterialCandidate(
                candidate_id=f"material.{formula_sha[:24]}",
                formula_sha256=formula_sha,
                features=features,
                stratum_id=f"stratum.elements-{min(element_count, 6)}",
                family_id=family_id,
                family_label=family_label,
                tc_kelvin=tc_value,
                label_state=state,
                source_row_count=len(source_rows),
                ambiguous_family=ambiguous,
            )
        )
    ordered = tuple(sorted(candidates, key=lambda value: value.candidate_id))
    qualification = SourceQualification(
        qualification_id='qualification.material-family-discovery-nims-220808',
        source_manifest_sha256=source_manifest_sha256,
        qualified_object_ids=qualified_object_ids,
        o_and_m_row_count=o_and_m_row_count,
        organic_row_count=organic_row_count,
        o_and_m_column_count=191,
        resolved_tc_row_count=resolved_rows,
        detection_limit_negative_row_count=detection_rows,
        unlabelled_row_count=unlabelled_rows,
        above_ceiling_row_count=above_rows,
        canonical_candidate_count=len(ordered),
        ambiguous_family_candidate_count=ambiguous_count,
        exact_reproduction_disposition=(ReproductionDisposition.SOURCE_OPERAND_REQUIRED_SC_EXACT),
        missing_exact_operand_ids=(
            "operand.pekala-exact-code",
            "operand.pekala-exact-materials-project-snapshot",
            "operand.pekala-exact-split",
            "operand.pekala-roost-checkpoint",
            "operand.pekala-str3-family-map",
        ),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    return MaterialCorpus(
        corpus_id='corpus.material-family-discovery-nims-220808',
        candidates=ordered,
        family_labels=tuple(sorted(family_labels.items())),
        qualification=qualification,
    )


def eligible_family_counts(
    corpus: MaterialCorpus,
    config: MaterialFamilyConfig,
) -> tuple[tuple[str, int, int], ...]:
    counts: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    for value in corpus.candidates:
        if value.family_id is None or value.ambiguous_family:
            continue
        counts[value.family_id] += 1
        if value.tc_kelvin is not None and value.tc_kelvin >= config.target_tc_threshold_kelvin:
            positives[value.family_id] += 1
    return tuple(
        sorted(
            (
                family_id,
                counts[family_id],
                positives[family_id],
            )
            for family_id in counts
            if counts[family_id] >= config.minimum_family_candidates
            and positives[family_id] >= config.minimum_family_positive_candidates
        )
    )


def _hash_order(namespace: str, seed: int, *values: str) -> bytes:
    return sha256("\0".join((namespace, str(seed), *values)).encode("ascii")).digest()


def _phase_candidates(
    corpus: MaterialCorpus,
    config: MaterialFamilyConfig,
) -> tuple[MaterialCandidate, ...]:
    """Return the frozen phase partition without consulting candidate outcomes."""

    if (
        config.phase is MaterialFamilyDiscoveryPhase.REPRODUCTION
        or config.canonical_candidate_algorithm_id == UNPARTITIONED_CANDIDATE_ALGORITHM_ID
    ):
        return corpus.candidates
    if (
        config.canonical_candidate_algorithm_id != PHASE_DISJOINT_CANDIDATE_ALGORITHM_ID
    ):  # also guarded by MaterialFamilyConfig
        raise ValueError("SC phase partition algorithm is unknown")

    development = set(config.development_family_ids)
    evaluation = set(config.evaluation_family_ids)

    def assigned_phase(candidate: MaterialCandidate) -> MaterialFamilyDiscoveryPhase:
        if candidate.family_id in development:
            return MaterialFamilyDiscoveryPhase.DEVELOPMENT
        if candidate.family_id in evaluation:
            return MaterialFamilyDiscoveryPhase.EVALUATION
        # Keep every unrostered source family together. Canonical candidates
        # without a unique family are independently assigned by their already
        # deduplicated material ID. Three quarters of the background corpus is
        # reserved for development because the frozen evaluation target roster
        # contains substantially more source material; this split was selected
        # from outcome-blind support counts, not policy results.
        partition_key = candidate.family_id or candidate.candidate_id
        value = int.from_bytes(
            sha256(_HISTORICAL_PHASE_PARTITION_DOMAIN + str(config.world_seed).encode("ascii") + b"\0" + partition_key.encode("ascii")).digest()[:8],
            "big",
        )
        return MaterialFamilyDiscoveryPhase.DEVELOPMENT if value < 3 * 2**62 else MaterialFamilyDiscoveryPhase.EVALUATION

    return tuple(
        candidate for candidate in corpus.candidates if assigned_phase(candidate) is config.phase
    )


def build_world(
    corpus: MaterialCorpus,
    config: MaterialFamilyConfig,
    family_id: str,
) -> MaterialSearchWorld:
    """Build one matched candidate universe with the complete family held out."""

    allowed = (
        config.development_family_ids
        if config.phase.value == "DEVELOPMENT"
        else config.evaluation_family_ids
        if config.phase.value == "EVALUATION"
        else (*config.development_family_ids, *config.evaluation_family_ids)
    )
    if family_id not in allowed:
        raise ValueError("family is not assigned to this SC phase")
    eligible = {value[0] for value in eligible_family_counts(corpus, config)}
    if family_id not in eligible:
        raise ValueError("family lacks the frozen candidate/positive support floor")
    phase_candidates = _phase_candidates(corpus, config)
    by_id = corpus.by_id
    family_label = dict(corpus.family_labels)[family_id]
    target_candidates = tuple(
        sorted(
            (
                value
                for value in phase_candidates
                if value.family_id == family_id
                and value.tc_kelvin is not None
                and value.tc_kelvin >= config.target_tc_threshold_kelvin
            ),
            key=lambda value: _hash_order(
                "target-downsample", config.world_seed, family_id, value.candidate_id
            ),
        )[: config.target_candidates_per_world]
    )
    if len(target_candidates) != config.target_candidates_per_world:
        raise ValueError("target family cannot fill its frozen prevalence quota")
    target_ids = frozenset(value.candidate_id for value in target_candidates)
    distractors = []
    for value in phase_candidates:
        if value.family_id == family_id:
            continue
        if value.tc_kelvin is None:
            distractors.append(value)
            continue
        digest = int.from_bytes(
            _hash_order("measured-pool", config.world_seed, family_id, value.candidate_id)[:8],
            "big",
        )
        if (
            digest % config.measured_candidate_test_modulus
            == config.measured_candidate_test_remainder
        ):
            distractors.append(value)
    distractors.sort(
        key=lambda value: _hash_order(
            "distractor-downsample", config.world_seed, family_id, value.candidate_id
        )
    )
    requested_distractors = config.candidate_pool_size - len(target_candidates)
    if len(distractors) < requested_distractors:
        raise ValueError("source corpus cannot fill the frozen candidate pool")
    pool_ids = tuple(
        sorted(
            (*target_ids, *(value.candidate_id for value in distractors[:requested_distractors]))
        )
    )
    pool_set = set(pool_ids)
    initial = tuple(
        value.initial_observation()
        for value in phase_candidates
        if value.tc_kelvin is not None
        and value.family_id != family_id
        and value.candidate_id not in pool_set
    )
    policy_candidate_ids = tuple(sorted((*pool_ids, *(value.candidate_id for value in initial))))
    world_identity = (
        (
            'material-family-unpartitioned-world',
            config.phase.value,
            config.world_seed,
            family_id,
        )
        if config.canonical_candidate_algorithm_id == UNPARTITIONED_CANDIDATE_ALGORITHM_ID
        else (
            'material-family-phase-disjoint-world',
            config.fingerprint(),
            config.phase.value,
            config.world_seed,
            family_id,
        )
    )
    opaque_world_token = sha256(canonical_json_bytes(world_identity)).hexdigest()[:24]
    return MaterialSearchWorld(
        world_id=f"world.material-family-discovery.{opaque_world_token}",
        target_family_id=family_id,
        target_family_label=family_label,
        policy_candidates=tuple(by_id[value].policy_view() for value in policy_candidate_ids),
        initial_observations=initial,
        candidate_pool_ids=pool_ids,
        target_candidate_ids=target_ids,
        candidate_truth=tuple(by_id[value] for value in pool_ids),
    )


def corpus_identity(corpus: MaterialCorpus) -> str:
    return sha256(
        canonical_json_bytes(
            tuple(
                (
                    value.candidate_id,
                    value.formula_sha256,
                    FEATURE_SCHEMA_ID,
                    sha256(struct.pack(f"<{FEATURE_WIDTH}f", *value.features)).hexdigest(),
                    value.family_id,
                    str(value.tc_kelvin) if value.tc_kelvin is not None else None,
                    value.label_state.value,
                    value.source_row_count,
                )
                for value in corpus.candidates
            )
        )
    ).hexdigest()


__all__ = [
    "ELEMENTS",
    "FEATURE_WIDTH",
    "FEATURE_SCHEMA_ID",
    "MaterialCandidate",
    "MaterialCorpus",
    "MaterialSearchWorld",
    "build_material_corpus",
    "build_world",
    "corpus_identity",
    "eligible_family_counts",
    "family_identity",
]
