"""Tomography acquisition and analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from iqm.cpc.compiler.compiler import Compiler
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.timebox import TimeBox
from scipy.spatial.transform import Rotation
from xarray import Dataset

from common import (
    Config,
    Pulla,
    ReadoutMap,
    SequenceSpec,
    calibrated_prx,
    composite_gate,
    correct,
    measured_parallel_circuit,
    paper_phase_to_iqm,
    p1_from_dataset,
    readout_calibration_metadata,
    run_batch,
)
from dataset_persistence import persist_dataset

TomographyInput: TypeAlias = Literal["0", "1", "+x", "+y"]
TomographyBasis: TypeAlias = Literal["x", "y", "z"]
TomographyCoordinate: TypeAlias = tuple[TomographyInput, TomographyBasis]
ResultValue: TypeAlias = str | int | float | bool

TOMO_INPUTS: tuple[TomographyInput, ...] = ("0", "1", "+x", "+y")
TOMO_BASES: tuple[TomographyBasis, ...] = ("x", "y", "z")



def run(pulla: Pulla, compiler: Compiler, qubits: Sequence[str], sequences: Sequence[SequenceSpec], config: Config, readout: ReadoutMap, output_directory: Path) -> pd.DataFrame:
    rows: list[dict[str, ResultValue]] = []
    acquisition_index = 0
    for sequence in sequences:
        for amplitude_error in config.amplitude_errors:
            for detuning_hz in config.detunings_hz:
                dataset, metadata = acquire(pulla, compiler, qubits, sequence, amplitude_error, detuning_hz, config)
                persist_dataset(
                    dataset,
                    output_directory / f"raw_tomography_{acquisition_index:04d}.nc",
                    {
                        "technique": "process_tomography",
                        "acquisition_index": acquisition_index,
                        "sequence": sequence.name,
                        "target_angle_rad": float(sequence.target_angle),
                        "constituent_angle_rad": float(sequence.constituent_angle),
                        "paper_phases_rad": [float(phase) for phase in sequence.phases],
                        "iqm_phases_rad": [paper_phase_to_iqm(phase) for phase in sequence.phases],
                        "amplitude_error": amplitude_error,
                        "detuning_hz": detuning_hz,
                        "qubits": list(qubits),
                        "shots": config.tomography_shots,
                        "readout_calibration_shots": config.tomography_shots,
                        "prx_implementation": config.prx_implementation,
                        "reconstruction": "trace_preserving_linear_inversion",
                        "readout_calibration": readout_calibration_metadata(readout),
                        "circuit_coordinates": [
                            {"input_state": state, "measurement_basis": basis}
                            for state, basis in metadata
                        ],
                    },
                )
                acquisition_index += 1
                for q in qubits:
                    p1 = correct(p1_from_dataset(dataset, q, "tomo"), readout[q])
                    ptm, bloch, translation = reconstruct_ptm(p1, metadata)
                    row: dict[str, ResultValue] = {"sequence": sequence.name, "pulse_count": len(sequence.phases), "qubit": q, "amplitude_error": amplitude_error, "detuning_hz": detuning_hz, "translation_x": float(translation[0]), "translation_y": float(translation[1]), "translation_z": float(translation[2])}
                    row.update({
                        f"ptm_{i}{j}": float(ptm[i, j])
                        for i in range(4)
                        for j in range(4)
                    })
                    row.update(metrics(ptm, bloch, sequence.target_angle))
                    rows.append(row)
    return pd.DataFrame(rows)


def prep_boxes(
    builder: ScheduleBuilder,
    q: str,
    state: TomographyInput,
    config: Config,
) -> list[TimeBox]:
    if state == "0":
        return []
    if state == "1":
        return [calibrated_prx(builder, q, np.pi, 0, config)]
    if state == "+x":
        return [calibrated_prx(builder, q, np.pi / 2, np.pi / 2, config)]
    if state == "+y":
        return [calibrated_prx(builder, q, np.pi / 2, np.pi, config)]
    raise ValueError(state)


def analysis_boxes(
    builder: ScheduleBuilder,
    q: str,
    basis: TomographyBasis,
    config: Config,
) -> list[TimeBox]:
    if basis == "z":
        return []
    if basis == "x":
        return [calibrated_prx(builder, q, np.pi / 2, -np.pi / 2, config)]
    if basis == "y":
        return [calibrated_prx(builder, q, np.pi / 2, 0, config)]
    raise ValueError(basis)


def acquire(
    pulla: Pulla,
    compiler: Compiler,
    qubits: Sequence[str],
    sequence: SequenceSpec,
    amplitude_error: float,
    detuning_hz: float,
    config: Config,
) -> tuple[Dataset, list[TomographyCoordinate]]:
    builder = compiler.get_schedule_builder()
    circuits: list[TimeBox] = []
    metadata: list[TomographyCoordinate] = []
    for state in TOMO_INPUTS:
        for basis in TOMO_BASES:
            operations: dict[str, list[TimeBox]] = {
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


def reconstruct_ptm(
    p1: npt.NDArray[np.float64],
    metadata: Sequence[TomographyCoordinate],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    outputs: dict[TomographyInput, npt.NDArray[np.float64]] = {
        state: np.zeros(3) for state in TOMO_INPUTS
    }
    axis: dict[TomographyBasis, int] = {"x": 0, "y": 1, "z": 2}
    for probability, (state, basis) in zip(p1, metadata, strict=True):
        outputs[state][axis[basis]] = 1 - 2 * probability
    translation = (outputs["0"] + outputs["1"]) / 2
    bloch = np.column_stack([outputs["+x"] - translation, outputs["+y"] - translation, (outputs["0"] - outputs["1"]) / 2])
    ptm = np.zeros((4, 4))
    ptm[0, 0] = 1
    ptm[1:, 0] = translation
    ptm[1:, 1:] = bloch
    return ptm, bloch, translation


def metrics(
    ptm: npt.NDArray[np.float64],
    bloch: npt.NDArray[np.float64],
    target_angle: float,
) -> dict[str, float]:
    target = np.zeros((4, 4))
    target[0, 0] = 1
    target[1:, 1:] = Rotation.from_rotvec([target_angle, 0, 0]).as_matrix()
    fidelity = float((np.trace(target.T @ ptm) + 2) / 6)
    u, singular_values, vt = np.linalg.svd(bloch)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    nearest_rotation = u @ correction @ vt
    rotvec = Rotation.from_matrix(nearest_rotation).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    axis = np.array([1.0, 0.0, 0.0]) if angle < 1e-12 else rotvec / angle
    if abs(target_angle - np.pi) < 1e-6 and axis[0] < 0:
        axis = -axis
        angle = 2 * np.pi - angle
    wrap = lambda value: float((value + np.pi) % (2 * np.pi) - np.pi)
    return {
        "average_gate_fidelity": fidelity,
        "average_gate_infidelity": 1 - fidelity,
        "effective_angle": angle,
        "angle_error": wrap(angle - target_angle),
        "axis_x": float(axis[0]), "axis_y": float(axis[1]), "axis_z": float(axis[2]),
        "axis_azimuth": float(math.atan2(axis[1], axis[0])),
        "axis_tilt": float(math.asin(np.clip(axis[2], -1, 1))),
        "axis_distance": float(math.acos(np.clip(axis[0], -1, 1))),
        "bloch_singular_value_1": float(singular_values[0]),
        "bloch_singular_value_2": float(singular_values[1]),
        "bloch_singular_value_3": float(singular_values[2]),
        "polar_rotation_residual": float(np.linalg.norm(bloch - nearest_rotation)),
    }
