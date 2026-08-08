"""Pure single-qubit mathematics shared by characterization techniques."""

from __future__ import annotations

import numpy as np


PAULIS = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the half-open interval [-pi, pi)."""

    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def same_unitary(left: np.ndarray, right: np.ndarray, atol: float = 1e-8) -> bool:
    """Compare two single-qubit unitaries while ignoring global phase."""

    return abs(abs(np.trace(left.conj().T @ right)) / 2 - 1) < atol


def unitary_bloch_rotation(unitary: np.ndarray) -> np.ndarray:
    """Return the SO(3) Bloch rotation induced by a one-qubit unitary."""

    unitary = np.asarray(unitary, dtype=complex)
    if unitary.shape != (2, 2):
        raise ValueError("A single-qubit unitary must have shape (2, 2).")
    rotation = np.empty((3, 3), dtype=float)
    for row, sigma_out in enumerate(PAULIS):
        for column, sigma_in in enumerate(PAULIS):
            transformed = unitary @ sigma_in @ unitary.conj().T
            rotation[row, column] = float(np.trace(sigma_out @ transformed).real / 2)
    return rotation
