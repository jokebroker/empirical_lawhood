"""Unchanged lower scientific reduction with new source and causal-seal identities."""

from typing import Any
from empirical_lawhood.adapters.methods.preparation_applicability.measurement import (
    reduce_panel, validity, lower_choices as lower_choices, service,
)
from .records import ConstructedPreparationMeasuredRoot, numbers
from empirical_lawhood.adapters.methods.preparation_applicability.measurement import requests as requests



def measure_root(prefix: Any, panel: Any, lower: Any, phase: str, scientific_seed: int, sealed: Any) -> Any:
    from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import ConstructedPreparationParents

    parents = ConstructedPreparationParents(
        panel.root_id,
        panel.prefix_sha256,
        panel.selection_sha256,
        tuple(p for p in panel.phases if p.phase == "parent"),
    )
    if (
        sealed.root_id != prefix.root_id
        or sealed.parents_sha256 != parents.fingerprint()
        or panel.lower_seal_sha256 != sealed.fingerprint()
        or sealed.lower_sha256 != lower.fingerprint()
    ):
        raise ValueError("measurement changes the authenticated pre-future operands")
    identity = (prefix.root_id, prefix.fingerprint(), panel.fingerprint(), lower.fingerprint())
    operands = None if prefix.frame_base64 is None else reduce_panel(prefix, panel)
    if operands is None:
        return ConstructedPreparationMeasuredRoot(*identity, False, (), (), (), (), (), (), (), (), ())
    z, y, work = operands
    maxima, conjuncts, margins, mean, width, support = validity(lower, z, y, work)
    if (
        not sealed.complete
        or numbers(mean) != sealed.mean
        or numbers(width) != sealed.width
        or tuple(map(bool, support)) != sealed.support
    ):
        raise ValueError("reduced actual handoff differs from its sealed lower forecast")
    if phase in ("Q", "E"):
        direction, requirement = requests(scientific_seed)
        if (
            tuple(map(int, lower_choices(mean, width, support, direction, requirement).ravel()))
            != sealed.choices
            or tuple(map(int, direction.ravel())) != sealed.directions
            or numbers(requirement) != sealed.requirements
        ):
            raise ValueError("future outcomes altered the sealed lower word choice")
    joint: tuple[int, ...] = ()
    covered: tuple[int, ...] = ()
    attempts: tuple[int, ...] = ()
    false: tuple[int, ...] = ()
    marginals: tuple[int, ...] = ()
    if phase in ("Q", "E"):
        selected, success = service(mean, width, support, y, work, *requests(scientific_seed))
        joint_events = success.all(axis=2)
        joint = tuple(map(int, joint_events.sum(axis=1)))
        covered = tuple(map(int, (joint_events & conjuncts.all(axis=1)[:, None]).sum(axis=1)))
        attempts = tuple(map(int, (selected >= 0).sum(axis=(1, 2))))
        false = tuple(map(int, ((selected >= 0) & ~success).sum(axis=(1, 2))))
        marginals = tuple(map(int, success.sum(axis=1).ravel()))
    return ConstructedPreparationMeasuredRoot(
        *identity,
        True,
        numbers(maxima),
        tuple(map(bool, conjuncts.ravel())),
        numbers(margins),
        numbers(work),
        joint,
        covered,
        attempts,
        false,
        marginals,
        handoff=numbers(z),
    )
