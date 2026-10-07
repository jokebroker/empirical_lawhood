"""Fixed whole-root cross-validation coordinates, independent of public identities."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreparationCrossfitRootInput:
    context: str
    index: int
    rank: int

    def __post_init__(self) -> None:
        if self.context not in ("assembling", "prepared") or any(
            type(value) is not int or not 0 <= value < 64
            for value in (self.index, self.rank)
        ):
            raise ValueError(
                "preparation cross-validation requires its exact numerical root coordinate"
            )


PREPARATION_CROSSFIT_RANKS = (
    (
        "assembling",
        (
            18,
            61,
            23,
            52,
            28,
            57,
            40,
            17,
            49,
            13,
            39,
            1,
            30,
            42,
            50,
            53,
            9,
            48,
            31,
            14,
            55,
            45,
            22,
            46,
            26,
            62,
            38,
            43,
            56,
            63,
            51,
            19,
            35,
            37,
            59,
            7,
            47,
            29,
            2,
            36,
            32,
            3,
            60,
            0,
            6,
            41,
            25,
            58,
            27,
            24,
            54,
            21,
            16,
            15,
            12,
            20,
            34,
            44,
            33,
            10,
            11,
            8,
            5,
            4,
        ),
    ),
    (
        "prepared",
        (
            54,
            14,
            50,
            30,
            62,
            19,
            42,
            40,
            22,
            39,
            0,
            33,
            49,
            24,
            37,
            63,
            27,
            23,
            46,
            60,
            18,
            10,
            1,
            6,
            47,
            41,
            25,
            57,
            26,
            29,
            52,
            43,
            36,
            53,
            59,
            28,
            45,
            61,
            2,
            38,
            11,
            8,
            13,
            35,
            3,
            20,
            31,
            55,
            32,
            15,
            51,
            12,
            34,
            5,
            44,
            16,
            9,
            17,
            56,
            4,
            21,
            58,
            48,
            7,
        ),
    ),
)

PREPARATION_CROSSFIT_ROOT_INPUTS = tuple(
    PreparationCrossfitRootInput(context, index, rank)
    for context, ranks in PREPARATION_CROSSFIT_RANKS
    for index, rank in enumerate(ranks)
)


def preparation_crossfit_rank(context: str, index: int) -> int:
    if (
        context not in ("assembling", "prepared")
        or type(index) is not int
        or not 0 <= index < 64
    ):
        raise ValueError(
            "preparation cross-validation requires its declared numerical input row"
        )
    return PREPARATION_CROSSFIT_ROOT_INPUTS[
        (0 if context == "assembling" else 64) + index
    ].rank
