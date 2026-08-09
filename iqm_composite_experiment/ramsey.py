"""Ramsey transition-phase acquisition and analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from common import Config, Pulla, SequenceSpec, calibrated_prx, composite_gate, correct, measured_parallel_circuit, p1_from_dataset, prx_matrix, run_batch, wrap
from dataset_persistence import persist_dataset


def run(pulla: Pulla, compiler: Any, qubits: Sequence[str], sequences: Sequence[SequenceSpec], config: Config, readout: dict[str, Any], output_directory: Path) -> pd.DataFrame:
    rows = []
    acquisition_index = 0
    for sequence in sequences:
        for amplitude_error in config.amplitude_errors:
            for detuning_hz in config.detunings_hz:
                dataset = acquire(pulla, compiler, qubits, sequence, amplitude_error, detuning_hz, config)
                persist_dataset(dataset, output_directory / f"raw_ramsey_{acquisition_index:04d}.nc")
                acquisition_index += 1
                for q in qubits:
                    row = {"sequence": sequence.name, "pulse_count": len(sequence.phases), "qubit": q, "amplitude_error": amplitude_error, "detuning_hz": detuning_hz}
                    row.update(metrics(correct(p1_from_dataset(dataset, q, "ramsey"), readout[q]), sequence, config.ramsey_phases))
                    rows.append(row)
    return pd.DataFrame(rows)


def acquire(pulla: Pulla, compiler: Any, qubits: Sequence[str], sequence: SequenceSpec, amplitude_error: float, detuning_hz: float, config: Config) -> Any:
    builder = compiler.get_schedule_builder()
    circuits = []
    for phase in config.ramsey_phases:
        operations = {}
        for q in qubits:
            ops = []
            if abs(sequence.target_angle - np.pi) < 1e-6:
                ops.append(calibrated_prx(builder, q, np.pi / 2, 0, config))
            ops.append(composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config))
            ops.append(calibrated_prx(builder, q, np.pi / 2, phase, config))
            operations[q] = ops
        circuits.append(measured_parallel_circuit(builder, qubits, operations, "ramsey"))
    return run_batch(pulla, compiler, circuits, qubits, config.ramsey_shots)


def fit_fringe(phases: Sequence[float], p1: Sequence[float]) -> dict[str, float]:
    phases = np.asarray(phases)
    design = np.column_stack([np.ones(len(phases)), np.cos(phases), np.sin(phases)])
    offset, cosine, sine = np.linalg.lstsq(design, np.asarray(p1), rcond=None)[0]
    return {"offset": float(offset), "contrast": float(math.hypot(cosine, sine)), "phase": float(math.atan2(sine, cosine))}


def ideal_p1(sequence: SequenceSpec, phases: Sequence[float]) -> np.ndarray:
    state = np.array([1.0, 0.0], dtype=complex)
    if abs(sequence.target_angle - np.pi) < 1e-6:
        state = prx_matrix(np.pi / 2, 0) @ state
    state = prx_matrix(sequence.target_angle, 0) @ state
    return np.array([abs((prx_matrix(np.pi / 2, phase) @ state)[1]) ** 2 for phase in phases])


def metrics(p1: np.ndarray, sequence: SequenceSpec, phases: Sequence[float]) -> dict[str, float]:
    measured = fit_fringe(phases, p1)
    ideal = fit_fringe(phases, ideal_p1(sequence, phases))
    shift = wrap(measured["phase"] - ideal["phase"])
    divisor = 2 if abs(sequence.target_angle - np.pi) < 1e-6 else 1
    return {"ramsey_contrast": measured["contrast"], "ramsey_fringe_phase": measured["phase"], "ramsey_fringe_shift": shift, "transition_phase_error": shift / divisor}
