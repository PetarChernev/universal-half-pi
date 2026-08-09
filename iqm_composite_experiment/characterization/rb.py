"""Interleaved randomized benchmarking acquisition and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


from common import (
    Clifford,
    Config,
    Pulla,
    SequenceSpec,
    calibrated_prx,
    composite_gate,
    correct,
    measured_parallel_circuit,
    p1_from_dataset,
    prx_matrix,
    run_batch,
    same_unitary,
)
from dataset_persistence import persist_dataset


def run(pulla: Pulla, compiler: Any, qubits: Sequence[str], sequences: Sequence[SequenceSpec], config: Config, readout: dict[str, Any], output_directory: Path) -> pd.DataFrame:
    rows = []
    acquisition_index = 0
    for sequence in sequences:
        for amplitude_error in config.amplitude_errors:
            for detuning_hz in config.detunings_hz:
                dataset, rb_plan = acquire(pulla, compiler, qubits, sequence, amplitude_error, detuning_hz, config)
                persist_dataset(dataset, output_directory / f"raw_rb_{acquisition_index:04d}.nc")
                acquisition_index += 1
                for q in qubits:
                    row = {"sequence": sequence.name, "pulse_count": len(sequence.phases), "qubit": q, "amplitude_error": amplitude_error, "detuning_hz": detuning_hz}
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


def clifford_index(unitary: np.ndarray, group: Sequence[Clifford]) -> int:
    for i, element in enumerate(group):
        if same_unitary(unitary, element.unitary, atol=1e-7):
            return i
    raise ValueError("Unitary is not a single-qubit Clifford.")


def clifford_pulses(builder: Any, q: str, element: Clifford, config: Config) -> list[Any]:
    return [calibrated_prx(builder, q, angle, phase, config) for angle, phase in element.pulses]


def plan(sequence: SequenceSpec, config: Config) -> tuple[list[dict[str, Any]], tuple[Clifford, ...]]:
    group = clifford_group()
    target = prx_matrix(sequence.target_angle, 0)
    clifford_index(target, group)
    rng = np.random.default_rng(config.seed)
    result = []
    for length in config.rb_lengths:
        for sample in range(config.rb_samples):
            indices = rng.integers(0, 24, size=length).tolist()
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


def acquire(pulla: Pulla, compiler: Any, qubits: Sequence[str], sequence: SequenceSpec, amplitude_error: float, detuning_hz: float, config: Config) -> tuple[Any, list[dict[str, Any]]]:
    builder = compiler.get_schedule_builder()
    rb_plan, group = plan(sequence, config)
    circuits = []
    for item in rb_plan:
        operations = {}
        for q in qubits:
            ops = []
            for i in item["indices"]:
                ops += clifford_pulses(builder, q, group[i], config)
                if item["kind"] == "interleaved":
                    ops.append(composite_gate(builder, q, sequence, amplitude_error, detuning_hz, config))
            ops += clifford_pulses(builder, q, group[item["recovery"]], config)
            operations[q] = ops
        circuits.append(measured_parallel_circuit(builder, qubits, operations, "rb"))
    return run_batch(pulla, compiler, circuits, qubits, config.rb_shots), rb_plan


def decay_model(length: np.ndarray, a: float, p: float, b: float) -> np.ndarray:
    return a * p ** length + b


def fit_decay(lengths: Sequence[int], survival: Sequence[float]) -> float:
    parameters, _ = curve_fit(decay_model, np.asarray(lengths, dtype=float), np.asarray(survival, dtype=float), p0=(0.45, 0.99, 0.5), bounds=((0, 0, 0), (1, 1, 1)), maxfev=20_000)
    return float(parameters[1])


def metrics(p1: np.ndarray, rb_plan: Sequence[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for probability, item in zip(p1, rb_plan, strict=True):
        grouped.setdefault((item["kind"], item["length"]), []).append(1 - probability)
    lengths = sorted({item["length"] for item in rb_plan})
    reference = [np.mean(grouped[("reference", length)]) for length in lengths]
    interleaved = [np.mean(grouped[("interleaved", length)]) for length in lengths]
    p_reference = fit_decay(lengths, reference)
    p_interleaved = fit_decay(lengths, interleaved)
    p_gate = p_interleaved / p_reference
    return {"rb_reference_decay": p_reference, "rb_interleaved_decay": p_interleaved, "rb_gate_decay": p_gate, "rb_infidelity": 0.5 * (1 - p_gate)}
