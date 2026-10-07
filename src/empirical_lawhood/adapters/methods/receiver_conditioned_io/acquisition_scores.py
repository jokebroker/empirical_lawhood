"""Pure deterministic D-optimal information scores for issued candidate queries."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import CanonicalMatrix, CanonicalVector, decimal_from_float


@dataclass(frozen=True, slots=True)
class DOptimalConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/methods/receiver-conditioned-io/d-optimal-config'

    config_id: str
    regularization: Decimal
    feature_coordinate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_decimal(self.regularization, field_name="regularization", minimum=Decimal(0))
        if self.regularization <= 0:
            raise ValueError("D-optimal regularization must be positive")
        require_sorted_unique_strings(
            self.feature_coordinate_ids,
            field_name="feature_coordinate_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class DOptimalCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/methods/receiver-conditioned-io/d-optimal-candidate'

    candidate_id: str
    action_word: ObjectIdentity
    feature_vector: CanonicalVector
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("D-optimal candidate requires an exact ActionWord")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.eligible and self.reason_codes:
            raise ValueError("eligible D-optimal candidate cannot carry refusal reasons")
        if not self.eligible and not self.reason_codes:
            raise ValueError("ineligible D-optimal candidate requires reasons")


@dataclass(frozen=True, slots=True)
class DOptimalScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/methods/receiver-conditioned-io/d-optimal-score'

    score_id: str
    candidate_id: str
    action_word: ObjectIdentity
    information_gain: NamedDecimal | None
    evidence_matrix: CanonicalMatrix
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("D-optimal score requires an exact ActionWord")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.information_gain is None and not self.reason_codes:
            raise ValueError("refused D-optimal score requires reasons")
        if self.information_gain is not None and self.reason_codes:
            raise ValueError("computed D-optimal score cannot carry reasons")


@dataclass(frozen=True, slots=True)
class DOptimalScoreSet(CanonicalRecord):
    """Scores only; graph-owned acquisition logic consumes them separately."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/methods/receiver-conditioned-io/d-optimal-score-set'

    score_set_id: str
    config: ObjectIdentity
    current_information: ObjectIdentity
    scores: tuple[DOptimalScore, ...]
    ranked_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.score_set_id, field_name="score_set_id")
        if self.config.object_schema != DOptimalConfig.SCHEMA:
            raise ValueError("D-optimal score set requires an exact config")
        if self.current_information.object_schema != CanonicalMatrix.SCHEMA:
            raise ValueError("D-optimal score set requires its information matrix")
        require_sorted_unique_ids(self.scores, attribute="score_id", field_name="scores")
        require_sorted_unique_strings(
            tuple(sorted(self.ranked_candidate_ids)),
            field_name="ranked_candidate_ids",
        )
        eligible = {
            value.candidate_id for value in self.scores if value.information_gain is not None
        }
        if set(self.ranked_candidate_ids) != eligible:
            raise ValueError("D-optimal rank list differs from computable candidate scores")


@dataclass(frozen=True, slots=True)
class DOptimalScorer:
    def score(
        self,
        *,
        score_set_id: str,
        current_information: CanonicalMatrix,
        candidates: tuple[DOptimalCandidate, ...],
        config: DOptimalConfig,
    ) -> DOptimalScoreSet:
        if current_information.row_coordinate_ids != config.feature_coordinate_ids or (
            current_information.column_coordinate_ids != config.feature_coordinate_ids
        ):
            raise ValueError("D-optimal information matrix uses another feature basis")
        base = current_information.as_array() + float(config.regularization) * np.eye(
            len(config.feature_coordinate_ids)
        )
        sign, base_logdet = np.linalg.slogdet(base)
        if sign <= 0:
            raise ValueError("D-optimal regularized information matrix is not positive")
        scores: list[DOptimalScore] = []
        raw: dict[str, float] = {}
        for candidate in sorted(candidates, key=lambda value: value.candidate_id):
            if candidate.feature_vector.coordinate_ids != config.feature_coordinate_ids:
                raise ValueError("D-optimal candidate uses another feature basis")
            if candidate.eligible:
                feature = candidate.feature_vector.as_array()
                updated = base + np.outer(feature, feature)
                updated_sign, updated_logdet = np.linalg.slogdet(updated)
                if updated_sign <= 0:
                    raise ValueError("D-optimal candidate produced invalid information")
                gain = float(updated_logdet - base_logdet)
                raw[candidate.candidate_id] = gain
                result = NamedDecimal(
                    value_id=f"d-optimal-gain.{candidate.candidate_id}",
                    value=decimal_from_float(gain),
                    unit="nat",
                )
                reasons: tuple[str, ...] = ()
            else:
                result = None
                reasons = candidate.reason_codes
            scores.append(
                DOptimalScore(
                    score_id=f"d-optimal-score.{candidate.candidate_id}",
                    candidate_id=candidate.candidate_id,
                    action_word=candidate.action_word,
                    information_gain=result,
                    evidence_matrix=current_information,
                    reason_codes=reasons,
                )
            )
        ranked = tuple(
            candidate_id
            for candidate_id, _ in sorted(raw.items(), key=lambda item: (-item[1], item[0]))
        )
        return DOptimalScoreSet(
            score_set_id=score_set_id,
            config=ObjectIdentity.from_record(config.config_id, config),
            current_information=ObjectIdentity.from_record(
                current_information.matrix_id,
                current_information,
            ),
            scores=tuple(scores),
            ranked_candidate_ids=ranked,
        )
