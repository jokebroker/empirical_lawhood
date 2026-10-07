"""Explicit stage bootstrap and permutation operands for preparation inference.

Public stage labels identify an inference; these integers select its stream.
The allocation grants no historical qualification, receipt or authority.
"""

from types import MappingProxyType
from typing import Final

from .types import Stage

STAGE_INFERENCE_SEEDS: Final = MappingProxyType({
    'design': MappingProxyType({
        ('joint-grid-bootstrap', None): 5392441178489833185,
        ('joint-memory', 'initial_k'): 10594348367614822144,
        ('joint-memory', 'boundary_occupation'): 9418632000785802752,
        ('joint-memory', 'translation_orbit'): 15870735327548823488,
    }),
    'freeze': MappingProxyType({
        ('joint-grid-bootstrap', None): 10020169148052580752,
        ('joint-memory', 'initial_k'): 14627123609357624901,
        ('joint-memory', 'boundary_occupation'): 6391141125833690991,
        ('joint-memory', 'translation_orbit'): 706471919738331810,
    }),
    'base-compatibility': MappingProxyType({
        ('joint-grid-bootstrap', None): 15991710607159498722,
        ('joint-memory', 'initial_k'): 10066703174419100919,
        ('joint-memory', 'boundary_occupation'): 10587389508885661396,
        ('joint-memory', 'translation_orbit'): 14301122002711505824,
    }),
    'statistical-conformance': MappingProxyType({
        ('joint-grid-bootstrap', None): 7856810575334006840,
        ('joint-memory', 'initial_k'): 11726392031457907108,
        ('joint-memory', 'boundary_occupation'): 11678233287333885412,
        ('joint-memory', 'translation_orbit'): 3672958010354752642,
    }),
    'development': MappingProxyType({
        ('joint-grid-bootstrap', None): 2716936968583560535,
        ('joint-memory', 'initial_k'): 13239027256686162934,
        ('joint-memory', 'boundary_occupation'): 11782238454128810640,
        ('joint-memory', 'translation_orbit'): 13262643506348823180,
    }),
    'evaluation-freeze': MappingProxyType({
        ('joint-grid-bootstrap', None): 7048161908325268921,
        ('joint-memory', 'initial_k'): 971443408083130326,
        ('joint-memory', 'boundary_occupation'): 1192087380353115458,
        ('joint-memory', 'translation_orbit'): 9198283880839667383,
    }),
    'evaluation': MappingProxyType({
        ('joint-grid-bootstrap', None): 2724289547153541376,
        ('joint-memory', 'initial_k'): 11139078291369613239,
        ('joint-memory', 'boundary_occupation'): 9921581655828513640,
        ('joint-memory', 'translation_orbit'): 13741306824578087438,
    }),
    'closeout': MappingProxyType({
        ('joint-grid-bootstrap', None): 8545663707754822233,
        ('joint-memory', 'initial_k'): 7912701713454180869,
        ('joint-memory', 'boundary_occupation'): 13445953133795290877,
        ('joint-memory', 'translation_orbit'): 7335301285577049162,
    }),
})


def stage_inference_seed(stage: Stage, purpose: str, *, factor: str | None = None) -> int:
    """Read one of the declared stage/purpose/factor scientific commitments."""
    try:
        return STAGE_INFERENCE_SEEDS[stage.value][purpose, factor]
    except KeyError as exc:
        raise ValueError("undeclared preparation inference seed purpose or factor") from exc
