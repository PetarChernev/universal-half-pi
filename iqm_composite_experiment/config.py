"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


QUBIT_SELECTION_METRICS = (
    "prx_fidelity",
    "readout_fidelity",
    "t1",
    "t2",
)


@dataclass(frozen=True)
class Config:
    qubits: tuple[str, ...] | int | None = ("QB19",)
    qubit_selection_metric: str = "prx_fidelity"
    amplitude_errors: tuple[float, ...] = (0.0, 0.15)
    detunings_hz: tuple[float, ...] = (0.0,)
    ramsey_phases: tuple[float, ...] = tuple(np.linspace(0, 2 * np.pi, 5)[:-1])
    tomography_shots: int = 2000
    ramsey_shots: int = 50
    rb_shots: int = 50
    rb_lengths: tuple[int, ...] = (1, 2, 4, 8)
    rb_samples: int = 2
    seed: int = 7
    prx_implementation: str | None = None
    pre_calibration: bool = False
    post_calibration: bool = False

    def __post_init__(self) -> None:
        if self.qubit_selection_metric not in QUBIT_SELECTION_METRICS:
            supported = ", ".join(QUBIT_SELECTION_METRICS)
            raise ValueError(
                f"Unknown qubit-selection metric "
                f"{self.qubit_selection_metric!r}; expected one of: {supported}"
            )
