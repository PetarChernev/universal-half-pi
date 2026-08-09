"""Interleaved randomized benchmarking acquisition and analysis."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from iqm.cpc.compiler.compiler import Compiler
from iqm.pulse.builder import ScheduleBuilder
from iqm.pulse.timebox import TimeBox
from scipy.optimize import curve_fit
from xarray import Dataset


from common import (
    Clifford,
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
    prx_matrix,
    readout_calibration_metadata,
    run_batch,
    same_unitary,
)
from dataset_persistence import persist_dataset


class RBPlanItem(TypedDict):
    kind: Literal["reference", "interleaved"]
    length: int
    sample: int
    indices: list[int]
    recovery: int


ResultValue: TypeAlias = str | int | float | bool


def run(pulla: Pulla, compiler: Compiler, qubits: Sequence[str], sequences: Sequence[SequenceSpec], config: Config, readout: ReadoutMap, output_directory: Path) -> pd.DataFrame:
    rows: list[dict[str, ResultValue]] = []
    acquisition_index = 0
    for sequence in sequences:
        for amplitude_error in config.amplitude_errors:
            for detuning_hz in config.detunings_hz:
                dataset, rb_plan = acquire(pulla, compiler, qubits, sequence, amplitude_error, detuning_hz, config)
                persist_dataset(
                    dataset,
                    output_directory / f"raw_rb_{acquisition_index:04d}.nc",
                    {
                        "technique": "interleaved_randomized_benchmarking",
                        "acquisition_index": acquisition_index,
                        "sequence": sequence.name,
                        "target_angle_rad": float(sequence.target_angle),
                        "constituent_angle_rad": float(sequence.constituent_angle),
                        "paper_phases_rad": [float(phase) for phase in sequence.phases],
                        "iqm_phases_rad": [paper_phase_to_iqm(phase) for phase in sequence.phases],
                        "amplitude_error": amplitude_error,
                        "detuning_hz": detuning_hz,
                        "qubits": list(qubits),
                        "shots": config.rb_shots,
                        "readout_calibration_shots": config.tomography_shots,
                        "seed": config.seed,
                        "prx_implementation": config.prx_implementation,
                        "fit_model": "amplitude * decay**length + offset",
                        "readout_calibration": readout_calibration_metadata(readout),
                        "rb_plan": rb_plan,
                    },
                )
                acquisition_index += 1
                for q in qubits:
                    row: dict[str, ResultValue] = {"sequence": sequence.name, "pulse_count": len(sequence.phases), "qubit": q, "amplitude_error": amplitude_error, "detuning_hz": detuning_hz}
                    row.update(metrics(correct(p1_from_dataset(dataset, q, "rb"), readout[q]), rb_plan))
                    rows.append(row)
    return pd.DataFrame(rows)


def clifford_group() -> tuple[Clifford, ...]:
    generators = (
        (prx_matrix(np.pi / 2, 0), (np.pi / 2, 0)),
        (prx_matrix(np.pi / 2, np.pi / 2), (np.pi / 2, np.pi / 2)),
    )
    group = [Clifford(np.eye(2, dtype=complex), tuple())]
    queue = [0]
    while queue and len(group) < 24:
        current = group[queue.pop(0)]
        for unitary, pulse in generators:
            candidate = unitary @ current.unitary
            if any(same_unitary(candidate, element.unitary) for element in group):
                continue
            group.append(Clifford(candidate, current.pulses + (pulse,)))
            queue.append(len(group) - 1)
    if len(group) != 24:
        raise RuntimeError(f"Generated {len(group)} Cliffords instead of 24.")
    return tuple(group)


def clifford_index(
    unitary: npt.NDArray[np.complex128],
    group: Sequence[Clifford],
) -> int:
    for i, element in enumerate(group):
        if same_unitary(unitary, element.unitary, atol=1e-7):
            return i
    raise ValueError("Unitary is not a single-qubit Clifford.")


def clifford_pulses(
    builder: ScheduleBuilder,
    q: str,
    element: Clifford,
    config: Config,
) -> list[TimeBox]:
    return [calibrated_prx(builder, q, angle, phase, config) for angle, phase in element.pulses]


def plan(
    sequence: SequenceSpec,
    config: Config,
) -> tuple[list[RBPlanItem], tuple[Clifford, ...]]:
    group = clifford_group()
    target = prx_matrix(sequence.target_angle, 0)
    clifford_index(target, group)
    rng = np.random.default_rng(config.seed)
    result: list[RBPlanItem] = []
    for length in config.rb_lengths:
        for sample in range(config.rb_samples):
            indices = [int(index) for index in rng.integers(0, 24, size=length)]
            total = np.eye(2, dtype=complex)
            for i in indices:
                total = group[i].unitary @ total
            reference_recovery = clifford_index(total.conj().T, group)
            total = np.eye(2, dtype=complex)
            for i in indices:
                total = group[i].unitary @ total
                total = target @ total
            interleaved_recovery = clifford_index(total.conj().T, group)
            result += [
                {"kind": "reference", "length": length, "sample": sample, "indices": indices, "recovery": reference_recovery},
                {"kind": "interleaved", "length": length, "sample": sample, "indices": indices, "recovery": interleaved_recovery},
            ]
    return result, group


def acquire(pulla: Pulla, compiler: Compiler, qubits: Sequence[str], sequence: SequenceSpec, amplitude_error: float, detuning_hz: float, config: Config) -> tuple[Dataset, list[RBPlanItem]]:
    builder = compiler.get_schedule_builder()
    rb_plan, group = plan(sequence, config)
    circuits: list[TimeBox] = []
    for item in rb_plan:
        operations: dict[str, list[TimeBox]] = {}
        for q in qubits:
            ops: list[TimeBox] = []
            for i in item["indices"]:
                ops += clifford_pulses(builder, q, group[i], config)
                if item["kind"] == "interleaved":
                    ops.append(composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config))
            ops += clifford_pulses(builder, q, group[item["recovery"]], config)
            operations[q] = ops
        circuits.append(measured_parallel_circuit(builder, qubits, operations, "rb"))
    return run_batch(pulla, compiler, circuits, qubits, config.rb_shots), rb_plan


def decay_model(
    length: npt.NDArray[np.float64],
    a: float,
    p: float,
    b: float,
) -> npt.NDArray[np.float64]:
    return a * p ** length + b


def fit_decay(lengths: Sequence[int], survival: Sequence[float]) -> dict[str, float]:
    x = np.asarray(lengths, dtype=float)
    y = np.asarray(survival, dtype=float)
    parameters, covariance = curve_fit(
        decay_model,
        x,
        y,
        p0=(0.45, 0.99, 0.5),
        bounds=((0, 0, 0), (1, 1, 1)),
        maxfev=20_000,
    )
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return {
        "amplitude": float(parameters[0]),
        "decay": float(parameters[1]),
        "offset": float(parameters[2]),
        "amplitude_standard_error": float(standard_errors[0]),
        "decay_standard_error": float(standard_errors[1]),
        "offset_standard_error": float(standard_errors[2]),
        "fit_rmse": float(np.sqrt(np.mean((y - decay_model(x, *parameters)) ** 2))),
    }


def metrics(
    p1: npt.NDArray[np.float64],
    rb_plan: Sequence[RBPlanItem],
) -> dict[str, float | bool]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for probability, item in zip(p1, rb_plan, strict=True):
        grouped.setdefault((item["kind"], item["length"]), []).append(
            float(1 - probability)
        )
    lengths = sorted({item["length"] for item in rb_plan})
    reference = [float(np.mean(grouped[("reference", length)])) for length in lengths]
    interleaved = [float(np.mean(grouped[("interleaved", length)])) for length in lengths]
    reference_fit = fit_decay(lengths, reference)
    interleaved_fit = fit_decay(lengths, interleaved)
    p_reference = reference_fit["decay"]
    p_interleaved = interleaved_fit["decay"]
    if p_reference <= 0:
        p_gate = float("nan")
        p_gate_standard_error = float("nan")
    else:
        p_gate = p_interleaved / p_reference
        p_gate_standard_error = float(np.hypot(
            interleaved_fit["decay_standard_error"] / p_reference,
            p_interleaved * reference_fit["decay_standard_error"] / p_reference**2,
        ))
    result: dict[str, float | bool] = {
        "rb_gate_decay": p_gate,
        "rb_gate_decay_standard_error": p_gate_standard_error,
        "rb_infidelity": 0.5 * (1 - p_gate),
        "rb_estimate_physical": bool(0 <= p_gate <= 1),
    }
    result.update({f"rb_reference_{key}": value for key, value in reference_fit.items()})
    result.update({f"rb_interleaved_{key}": value for key, value in interleaved_fit.items()})
    return result
