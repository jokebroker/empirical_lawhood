"""Receipt-bound q2 algebra diagnostics, with no optimized-factor claim.

Spectra and product leakage are invariant under simultaneous unitary changes of
frame. Cyclic maxima and the constructive factor can depend on the chosen slow
basis, especially at degeneracy. Identifiability uses the strict original gap.
"""

from __future__ import annotations

import numpy as np

from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import traceless_hermitian_basis

BASIS = traceless_hermitian_basis(4)
IDENTITY = np.eye(4, dtype="<c16")
PAULI = np.asarray(([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]), dtype="<c16")
LEFT = np.asarray([np.kron(p / 2, np.eye(2)) for p in PAULI], dtype="<c16")
RIGHT = np.asarray([np.kron(np.eye(2), p / 2) for p in PAULI], dtype="<c16")


def _triplet(value):
    value = np.asarray(value)
    if value.shape != (3, 4, 4) or value.dtype != np.dtype("complex128") or not np.isfinite(value).all() or np.linalg.norm(value - value.conj().swapaxes(-1, -2)) > 1e-10 * max(np.linalg.norm(value), 1):
        raise ValueError("Algebra operands require a finite Hermitian q2 complex128 triplet")
    return value


def coords(fields):
    return np.einsum("iab,...ab->...i", BASIS.conj(), fields).real


def operator(triplet):
    """Positive double-commutator Gram matrix in the frozen HS basis."""
    triplet = _triplet(triplet)
    comm = triplet[:, None] @ BASIS[None] - BASIS[None] @ triplet[:, None]
    value = np.einsum("aibc,ajbc->ij", comm.conj(), comm).real
    return (value + value.T) / 2


def principal_angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.linalg.svd(a.T @ b, compute_uv=False).min(), 0, 1))))


def algebra(low):
    """Lie closure and complex associative closure are separate diagnostics."""
    low = np.asarray(low, dtype=np.float64)
    if low.shape != (15, 3) or not np.isfinite(low).all() or not np.allclose(low.T @ low, np.eye(3), atol=1e-10, rtol=0):
        raise ValueError("Algebra slow sector needs three orthonormal real HS columns")
    fields = np.einsum("ia,ijk->ajk", low, BASIS)
    leaks, couplings, norms = [], [], []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        comm = -1j * (fields[i] @ fields[j] - fields[j] @ fields[i])
        c = coords(comm)
        norm = float(c @ c)
        leaks.append(float(np.linalg.norm(c - low @ (low.T @ c))**2 / max(norm, 1e-30)))
        couplings.append(float(abs(np.vdot(fields[k], comm).real)))
        norms.append(norm)
    casimir = sum(f @ f for f in fields)
    full = [IDENTITY / 2, *fields]
    residuals, products, jordan = [], [], []
    for i in range(3):
        for j in range(3):
            product = fields[i] @ fields[j]
            projection = sum(np.vdot(f, product) * f for f in full)
            residuals.append(float(np.linalg.norm(product - projection)**2))
            products.append(float(np.linalg.norm(product)**2))
            target = IDENTITY / 2 if i == j else np.zeros_like(IDENTITY)
            jordan.append(float(np.linalg.norm(product + fields[j] @ fields[i] - target)**2))
    return {
        "lie_leakage": max(leaks), "minimum_commutator_coupling": min(couplings),
        "minimum_commutator_norm_squared": min(norms),
        "associative_leakage": sum(residuals) / max(sum(products), 1e-30),
        "casimir_relative": float(np.linalg.norm(casimir - .75 * IDENTITY) / 1.5),
        "jordan": float(np.sqrt(sum(jordan) / 3)),
        "eigenpair": max(float(np.linalg.norm(np.linalg.eigvalsh(f) - [-.5, -.5, .5, .5])) for f in fields),
    }, fields


def exact_factor(fields):
    """Construct a multiplicity-two Pauli factor; do not optimize its fit."""
    _, vectors = np.linalg.eigh(fields[2])
    minus, plus = vectors[:, :2], vectors[:, 2:]
    cross = plus.conj().T @ fields[0] @ minus
    u, _, vh = np.linalg.svd(cross)
    unitary = np.concatenate((plus, minus @ (u @ vh).conj().T), axis=1)
    return unitary @ LEFT @ unitary.conj().T


def sector(triplet):
    triplet = _triplet(triplet)
    op = operator(triplet)
    spectrum, vectors = np.linalg.eigh(op)
    low = vectors[:, :3]
    identifiable = bool(spectrum[3] - spectrum[2] > 1e-10 * max(spectrum[-1], 1e-30) and spectrum[3] > 1e-20)
    result = {"spectrum": spectrum, "projector": low @ low.T,
              "identifiable": identifiable, "reason": None if identifiable else "SLOW_SECTOR_NOT_IDENTIFIABLE",
              "metrics": {"mean_slow_rate": float(spectrum[:3].mean()),
                          "band_ratio": float(spectrum[2] / spectrum[3]) if spectrum[3] > 1e-20 else np.nan,
                          "gap": float(spectrum[3] - spectrum[2])},
              "factor": np.full((3, 4, 4), np.nan, dtype="<c16"),
              "reconstructed": np.full((3, 4, 4), np.nan, dtype="<c16"),
              "operator": op, "low": low}
    if not identifiable:
        return result
    metrics, fields = algebra(low)
    factor = exact_factor(fields)
    fcoords = coords(factor).T
    centered = triplet - np.trace(triplet, axis1=-2, axis2=-1)[:, None, None] * IDENTITY / 4
    reconstructed = (centered + sum(4 * f @ centered @ f for f in factor)) / 4
    tracepart = triplet - centered
    metrics.update({
        "factor_angle_degrees": principal_angle(low, fcoords),
        "commutant_mixing": float(np.linalg.norm(centered - reconstructed)**2 / max(np.linalg.norm(centered)**2, 1e-30)),
        "left_ancestry_angle_degrees": principal_angle(low, coords(LEFT).T),
        "right_ancestry_angle_degrees": principal_angle(low, coords(RIGHT).T),
        "trace_fraction": float(np.linalg.norm(tracepart)**2 / max(np.linalg.norm(triplet)**2, 1e-30)),
    })
    result["metrics"].update(metrics)
    result.update(factor=factor, reconstructed=reconstructed)
    return result


def measure(positions):
    positions = np.asarray(positions)
    if positions.shape != (2, 3, 4, 4):
        raise ValueError("Algebra measurement needs separate X/Y triplets")
    x, y = (sector(positions[index]) for index in (0, 1))
    joint = np.linalg.eigvalsh(x["operator"] + y["operator"])
    return {"x": x, "y": y, "joint_spectrum": joint,
            "cross_metrics": {
                "y_slow_x_rate": float(np.trace(y["low"].T @ x["operator"] @ y["low"]) / 3),
                "x_slow_y_rate": float(np.trace(x["low"].T @ y["operator"] @ x["low"]) / 3),
                "xy_angle_degrees": principal_angle(x["low"], y["low"]),
                "x_norm": float(np.linalg.norm(positions[0])), "y_norm": float(np.linalg.norm(positions[1])),
            }}
