"""Pure development/calibration transformations used by the installed method runner."""

from dataclasses import asdict
from decimal import Decimal as D
import json
from typing import Any
import numpy as np
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode
from empirical_lawhood.adapters.simulators.reactor_causal_response.projection import DevelopmentEpisode, development_rows
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import measured_labels
from .config import EmpiricalRecipe
from .experiment_records import EmpiricalAcquisitionEnvelope, EmpiricalArrayPayload, EmpiricalDiscoveryEnvelope, EmpiricalCalibrationEnvelope, EmpiricalUsefulness
from .transport import CausalReactorRootEnvelope
from .numerical import Observation, digest
from .discovery import discover
from .calibration import calibrate
from .serialization import json_bytes, read_fit
from .payload import DevelopmentBoundReactorResponsePayload
from .development_readout import episode_validity, branch_summary, nomination_shadow
from empirical_lawhood.adapters.simulators.reactor_causal_response.projection import UnmappedBranchWord


def policy_usefulness(
    roots: tuple[CausalReactorRootEnvelope, ...], payload: DevelopmentBoundReactorResponsePayload
) -> EmpiricalUsefulness:
    from .controller import NumericalController, EmpiricalNonattempt
    from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator

    if payload.q is None:
        raise ValueError("unentered calibration has no usefulness trial")
    model = read_fit(json.loads(payload.model_json))
    counts = []
    for envelope in sorted(roots, key=lambda r: r.root):
        raw = envelope.unpack()
        policy = NumericalController(model, float(payload.q), Actuator())
        completed, alternatives = 0, 0
        for time, t, j, dose in raw.observations:
            try:
                policy.step(
                    float(time),
                    dict(t_reactor_k=float(t), t_jacket_k=float(j), dosed_kg=float(dose)),
                    10.0,
                )
            except (ValueError, EmpiricalNonattempt):
                break
            if completed >= len(raw.requests) or not np.array_equal(
                raw.requests[completed, 0], policy.requests[-1]
            ):
                break
            decision = policy.decisions[-1]
            eligible = {c.projection.delivery_key for c in decision.candidates if not c.reasons}
            completed += 1
            alternatives += len(eligible) > 1
            policy.decisions.clear()
        counts.append((envelope.root, completed, alternatives))
    return EmpiricalUsefulness(tuple(counts))


def discovery_from_acquisitions(
    recipe: EmpiricalRecipe, publications: tuple[EmpiricalAcquisitionEnvelope, ...]
) -> EmpiricalDiscoveryEnvelope:
    expected = tuple(
        f"reactor-empirical-{role}-{i:03d}"
        for role, n in (("fit", 16), ("nomination", 8))
        for i in range(n)
    )
    by_root = {p.root: p for p in publications}
    if len(by_root) != len(publications) or set(by_root) != set(expected):
        raise ValueError("development publication census differs")
    episodes = []
    pairs = {}
    failures = []
    for root in expected:
        publication = by_root[root]
        for name in ("exploration-unshifted", "exploration-shifted-one-position", "exploration-shifted-two-positions", "feed-intervention", "jacket-intervention"):
            pair = tuple(
                NativeEpisode.from_envelope(e) for e in publication.episodes if e.episode == name
            )
            if len(pair) != 2 or tuple(e.dt for e in pair) != (1.0, 0.5):
                raise ValueError("development numerical view roster differs")
            pairs[(root, name)] = (pair[0], pair[1])
            if not all(e.complete for e in pair):
                failures.append((root, name, "INCOMPLETE_ASSIGNED_EPISODE"))
                continue
            nominal, refined = pair
            valid = np.full(2880, np.array_equal(nominal.requests, refined.requests), dtype=bool)
            for episode in pair:
                valid &= episode_validity(episode)
            if not all(
                e.observations.shape == (2880, 4)
                and e.stages.shape == (2880, 4)
                and np.isfinite(e.grid).all()
                and np.isfinite(e.observations).all()
                and np.isfinite(e.requests).all()
                and np.array_equal(e.observations[:, 0], np.arange(2880) * 10)
                and e.observations[0, 3] == 0
                and np.all(np.diff(e.observations[:, 3]) >= 0)
                and np.array_equal(e.grid[:, 0], np.arange(int(28800 / e.dt) + 1) * e.dt)
                for e in pair
            ):
                failures.append((root, name, "INVALID_REQUIRED_MEASUREMENT"))
                continue
            labels = np.stack(
                [
                    measured_labels(e.grid[:, 0], e.grid[:, 1], e.grid[:, 4], e.grid[:, 3], e.dt)
                    for e in pair
                ],
                axis=1,
            )
            observations = tuple(Observation(*map(float, row)) for row in nominal.observations)
            episodes.append(
                DevelopmentEpisode(
                    root,
                    publication.role,
                    name,
                    observations,
                    tuple((float(f), float(j)) for f, j in nominal.requests),
                    labels,
                    valid,
                )
            )
    branch = branch_summary(pairs)
    projected = None
    if not failures:
        try:
            projected = development_rows(tuple(episodes))
        except UnmappedBranchWord as error:
            failures.append(("assigned-census", "branch", str(error)))
    result: dict[str, Any]
    if failures:
        result = dict(
            disposition="IDENTIFICATION_NOT_SUPPORTED",
            selected=None,
            failures=failures,
            fit_calls=0,
            assigned_roots=24,
            assigned_episodes=120,
        )
    else:
        assert projected is not None
        result = asdict(discover(projected))
    result["branch_readout"] = branch
    if result["selected"] is not None:
        selected = result["selected"]
        models = tuple(read_fit(result["fits"][i]) for i in (selected, *result["rivals"]))
        result["opportunity"] = nomination_shadow(
            pairs, (models[0], models[1], models[2]), result["nominations"][selected]["q_dev"]
        )
    else:
        result["opportunity"] = {"status": "UNENTERED_IDENTIFICATION_NOT_SUPPORTED"}
    return EmpiricalDiscoveryEnvelope(
        ObjectIdentity.from_record(recipe.config_id, recipe),
        tuple(ObjectIdentity.from_record(root, by_root[root]) for root in expected),
        json_bytes(result).decode(),
        None
        if projected is None
        else EmpiricalArrayPayload.pack(
            {
                "root_index": np.repeat(np.arange(24, dtype=np.uint8), 5 * 2880),
                "episode_index": np.tile(np.repeat(np.arange(5, dtype=np.uint8), 2880), 24),
                "clocks": projected.clocks,
                "actions": projected.actions,
                "features": projected.features,
                "labels": projected.labels,
                "delivery_valid": projected.delivery_valid.astype(np.uint8),
            }
        ),
    )


def calibration_from_roots(
    recipe: EmpiricalRecipe,
    discovery: EmpiricalDiscoveryEnvelope,
    roots: tuple[CausalReactorRootEnvelope, ...],
) -> EmpiricalCalibrationEnvelope:
    if discovery.recipe != ObjectIdentity.from_record(recipe.config_id, recipe):
        raise ValueError("calibration changed frozen discovery recipe")
    data = json.loads(discovery.result_json)
    selected = data["selected"]
    expected = tuple(f"reactor-empirical-calibration-{i:03d}" for i in range(32))
    by_root = {r.root: r for r in roots}
    if tuple(sorted(by_root)) != expected or len(roots) != 32:
        raise ValueError("calibration publication census differs")
    identities = tuple(ObjectIdentity.from_record(r, by_root[r]) for r in expected)
    if selected is None:
        if any(by_root[r].nonattempt_reason != "IDENTIFICATION_NOT_SUPPORTED" for r in expected):
            raise ValueError("an ineligible nominee acquired downstream evidence")
        return EmpiricalCalibrationEnvelope(
            ObjectIdentity.from_record("reactor-empirical-discovery", discovery),
            identities,
            None,
            json_bytes(
                dict(
                    disposition="CALIBRATION_UNUSABLE",
                    q=None,
                    prerequisite="IDENTIFICATION_NOT_SUPPORTED",
                    acquisitions=0,
                )
            ).decode(),
        )
    model = read_fit(data["fits"][selected])
    q_dev = data["nominations"][selected]["q_dev"]
    values = calibrate(tuple(by_root[r].unpack() for r in expected), model, consumer_q=q_dev)
    calibration_digest = digest(
        {
            "traces": [r.object_fingerprint for r in identities],
            "scores": [
                None if s.value is None else format(D(repr(s.value)).normalize(), "f")
                for s in values.scores
            ],
            "q": None if values.q is None else format(D(repr(values.q)).normalize(), "f"),
        }
    )
    payload = DevelopmentBoundReactorResponsePayload(
        json_bytes(asdict(model)).decode(),
        recipe.fingerprint(),
        calibration_digest,
        None if values.q is None or values.q > 1 else D(repr(values.q)),
        development_q=D(repr(q_dev)),
    )
    return EmpiricalCalibrationEnvelope(
        ObjectIdentity.from_record("reactor-empirical-discovery", discovery),
        identities,
        payload,
        json_bytes(asdict(values)).decode(),
    )
