"""Tomography acquisition and analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from common import (
    Config,
    Pulla,
    SequenceSpec,
    calibrated_prx,
    composite_gate,
    correct,
    measured_parallel_circuit,
    p1_from_dataset,
    run_batch,
)
from dataset_persistence import persist_dataset

TOMO_INPUTS = ("0", "1", "+x", "+y")
TOMO_BASES = ("x", "y", "z")



def run(pulla: Pulla, compiler: Any, qubits: Sequence[str], sequences: Sequence[SequenceSpec], config: Config, readout: dict[str, Any], output_directory: Path) -> pd.DataFrame:
    rows = []
    acquisition_index = 0
    for sequence in sequences:
        for amplitude_error in config.amplitude_errors:
            for detuning_hz in config.detunings_hz:
                dataset, metadata = acquire(pulla, compiler, qubits, sequence, amplitude_error, detuning_hz, config)
                persist_dataset(dataset, output_directory / f"raw_tomography_{acquisition_index:04d}.nc")
                acquisition_index += 1
                for q in qubits:
                    p1 = correct(p1_from_dataset(dataset, q, "tomo"), readout[q])
                    ptm, bloch, translation = reconstruct_ptm(p1, metadata)
                    row = {"sequence": sequence.name, "pulse_count": len(sequence.phases), "qubit": q, "amplitude_error": amplitude_error, "detuning_hz": detuning_hz, "translation_x": translation[0], "translation_y": translation[1], "translation_z": translation[2]}
                    row.update(metrics(ptm, bloch, sequence.target_angle))
                    rows.append(row)
    return pd.DataFrame(rows)


def prep_boxes(builder: Any, q: str, state: str, config: Config) -> list[Any]:
    if state == "0":
        return []
    if state == "1":
        return [calibrated_prx(builder, q, np.pi, 0, config)]
    if state == "+x":
        return [calibrated_prx(builder, q, np.pi / 2, np.pi / 2, config)]
    if state == "+y":
        return [calibrated_prx(builder, q, np.pi / 2, np.pi, config)]
    raise ValueError(state)


def analysis_boxes(builder: Any, q: str, basis: str, config: Config) -> list[Any]:
    if basis == "z":
        return []
    if basis == "x":
        return [calibrated_prx(builder, q, np.pi / 2, -np.pi / 2, config)]
    if basis == "y":
        return [calibrated_prx(builder, q, np.pi / 2, 0, config)]
    raise ValueError(basis)


def acquire(
    pulla: Pulla,
    compiler: Any,
    qubits: Sequence[str],
    sequence: SequenceSpec,
    amplitude_error: float,
    detuning_hz: float,
    config: Config,
) -> tuple[Any, list[tuple[str, str]]]:
    builder = compiler.get_schedule_builder()
    circuits, metadata = [], []
    for state in TOMO_INPUTS:
        for basis in TOMO_BASES:
            operations = {
                q: (
                    prep_boxes(builder, q, state, config)
                    + [composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config)]
                    + analysis_boxes(builder, q, basis, config)
                )
                for q in qubits
            }
            circuits.append(measured_parallel_circuit(builder, qubits, operations, "tomo"))
            metadata.append((state, basis))
    return run_batch(pulla, compiler, circuits, qubits, config.tomography_shots), metadata


def reconstruct_ptm(p1: np.ndarray, metadata: Sequence[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = {state: np.zeros(3) for state in TOMO_INPUTS}
    axis = {"x": 0, "y": 1, "z": 2}
    for probability, (state, basis) in zip(p1, metadata, strict=True):
        outputs[state][axis[basis]] = 1 - 2 * probability
    translation = (outputs["0"] + outputs["1"]) / 2
    bloch = np.column_stack([outputs["+x"] - translation, outputs["+y"] - translation, (outputs["0"] - outputs["1"]) / 2])
    ptm = np.zeros((4, 4))
    ptm[0, 0] = 1
    ptm[1:, 0] = translation
    ptm[1:, 1:] = bloch
    return ptm, bloch, translation


def metrics(ptm: np.ndarray, bloch: np.ndarray, target_angle: float) -> dict[str, float]:
    target = np.zeros((4, 4))
    target[0, 0] = 1
    target[1:, 1:] = Rotation.from_rotvec([target_angle, 0, 0]).as_matrix()
    fidelity = float((np.trace(target.T @ ptm) + 2) / 6)
    u, _, vt = np.linalg.svd(bloch)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotvec = Rotation.from_matrix(u @ correction @ vt).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    axis = np.array([1.0, 0.0, 0.0]) if angle < 1e-12 else rotvec / angle
    if abs(target_angle - np.pi) < 1e-6 and axis[0] < 0:
        axis = -axis
    wrap = lambda value: float((value + np.pi) % (2 * np.pi) - np.pi)
    return {
        "process_fidelity": fidelity,
        "process_infidelity": 1 - fidelity,
        "effective_angle": angle,
        "angle_error": wrap(angle - target_angle),
        "axis_x": float(axis[0]), "axis_y": float(axis[1]), "axis_z": float(axis[2]),
        "axis_azimuth": float(math.atan2(axis[1], axis[0])),
        "axis_tilt": float(math.asin(np.clip(axis[2], -1, 1))),
        "axis_distance": float(math.acos(np.clip(axis[0], -1, 1))),
    }
