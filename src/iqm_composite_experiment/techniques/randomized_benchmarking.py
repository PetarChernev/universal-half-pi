"""Interleaved randomized benchmarking as a pluggable technique."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit

from ..backend.base import CircuitFactory
from ..models.acquisition import AcquisitionPlan, MeasurementData, PlannedCircuit
from ..models.points import ExperimentPoint
from ..models.results import TechniqueResult
from ..targets.base import OperationUnderTest, require_single_qubit
from ..targets.composite import prx_unitary
from .base import CharacterizationTechnique, TechniqueContext
from .math import same_unitary


@dataclass(frozen=True, eq=False)
class Clifford:
    """One single-qubit Clifford and a calibrated-PRX decomposition."""

    unitary: np.ndarray
    pulses: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RBTrial:
    """Typed coordinate describing one reference or interleaved RB circuit."""

    kind: Literal["reference", "interleaved"]
    length: int
    sample: int
    clifford_indices: tuple[int, ...]
    recovery_index: int


@dataclass(frozen=True)
class RBSettings:
    """Acquisition and randomization settings for interleaved RB."""

    lengths: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    samples: int = 10
    shots: int = 512
    seed: int = 7
    paired_sequences: bool = True

    def __post_init__(self) -> None:
        lengths = tuple(int(length) for length in self.lengths)
        if len(lengths) < 3 or any(length < 1 for length in lengths):
            raise ValueError("RB needs at least three positive sequence lengths.")
        if len(set(lengths)) != len(lengths):
            raise ValueError("RB sequence lengths must be unique.")
        if self.samples < 1 or self.shots < 1:
            raise ValueError("RB samples and shots must be positive.")
        object.__setattr__(self, "lengths", lengths)


@lru_cache(maxsize=1)
def clifford_group() -> tuple[Clifford, ...]:
    """Generate and cache the 24-element single-qubit Clifford group."""

    generators = (
        (prx_unitary(np.pi / 2, 0), (np.pi / 2, 0)),
        (prx_unitary(np.pi / 2, np.pi / 2), (np.pi / 2, np.pi / 2)),
    )
    group = [Clifford(np.eye(2, dtype=complex), ())]
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


def clifford_index(unitary: np.ndarray, group: Sequence[Clifford] | None = None) -> int:
    """Return the group index of a phase-equivalent single-qubit Clifford."""

    elements = clifford_group() if group is None else group
    for index, element in enumerate(elements):
        if same_unitary(unitary, element.unitary, atol=1e-7):
            return index
    raise ValueError("The ideal operation is not a single-qubit Clifford.")


def decay_model(length: np.ndarray, amplitude: float, decay: float, offset: float) -> np.ndarray:
    """Standard RB survival model ``amplitude * decay**length + offset``."""

    return amplitude * decay**length + offset


def fit_decay(lengths: Sequence[int], survival: Sequence[float]) -> float:
    """Fit the RB model and return its constrained decay parameter."""

    parameters, _ = curve_fit(
        decay_model,
        np.asarray(lengths, dtype=float),
        np.asarray(survival, dtype=float),
        p0=(0.45, 0.99, 0.5),
        bounds=((0, 0, 0), (1, 1, 1)),
        maxfev=20_000,
    )
    return float(parameters[1])


class InterleavedRandomizedBenchmarking(CharacterizationTechnique):
    """Reference/interleaved RB for targets with Clifford ideal operations."""

    technique_id = "interleaved_rb"

    def __init__(self, settings: RBSettings | None = None) -> None:
        self.settings = settings or RBSettings()

    def supports(self, target: OperationUnderTest) -> bool:
        if target.ideal_operation.arity != 1:
            return False
        try:
            clifford_index(target.ideal_operation.unitary)
        except ValueError:
            return False
        return True

    def validate(self, point: ExperimentPoint, context: TechniqueContext) -> None:
        require_single_qubit(point.target, "Interleaved randomized benchmarking")
        point.target.validate(point.parameters)
        if not self.supports(point.target):
            raise ValueError("Interleaved RB requires a single-qubit Clifford target.")
        if not context.loci:
            raise ValueError("Interleaved RB needs at least one locus.")

    def _seed_for(self, point: ExperimentPoint) -> int:
        if self.settings.paired_sequences:
            return self.settings.seed
        digest = hashlib.sha256(point.point_id.encode("utf-8")).digest()
        point_seed = int.from_bytes(digest[:8], "little")
        return self.settings.seed ^ point_seed

    def _trials(self, point: ExperimentPoint) -> tuple[RBTrial, ...]:
        group = clifford_group()
        target = point.target.ideal_operation.unitary
        rng = np.random.default_rng(self._seed_for(point))
        trials = []
        for length in self.settings.lengths:
            for sample in range(self.settings.samples):
                indices = tuple(int(index) for index in rng.integers(0, 24, size=length))
                total = np.eye(2, dtype=complex)
                for index in indices:
                    total = group[index].unitary @ total
                reference_recovery = clifford_index(total.conj().T, group)

                total = np.eye(2, dtype=complex)
                for index in indices:
                    total = target @ (group[index].unitary @ total)
                interleaved_recovery = clifford_index(total.conj().T, group)
                trials.extend(
                    [
                        RBTrial("reference", length, sample, indices, reference_recovery),
                        RBTrial("interleaved", length, sample, indices, interleaved_recovery),
                    ]
                )
        return tuple(trials)

    def build_plan(
        self,
        point: ExperimentPoint,
        context: TechniqueContext,
    ) -> AcquisitionPlan:
        group = clifford_group()
        circuits = []
        for circuit_index, trial in enumerate(self._trials(point)):

            def build(factory: CircuitFactory, trial: RBTrial = trial) -> object:
                operations = {}
                for locus in context.loci:
                    sequence = []
                    for index in trial.clifford_indices:
                        sequence.extend(
                            factory.calibrated_prx(locus, angle, phase)
                            for angle, phase in group[index].pulses
                        )
                        if trial.kind == "interleaved":
                            sequence.append(
                                factory.target_operation(point.target, point.parameters, locus)
                            )
                    sequence.extend(
                        factory.calibrated_prx(locus, angle, phase)
                        for angle, phase in group[trial.recovery_index].pulses
                    )
                    operations[locus] = sequence
                return factory.measured_parallel(operations, "rb")

            circuits.append(
                PlannedCircuit(
                    circuit_id=f"rb:{circuit_index:05d}",
                    build=build,
                    coordinates={
                        "kind": trial.kind,
                        "length": trial.length,
                        "sample": trial.sample,
                        "clifford_indices": trial.clifford_indices,
                        "recovery_index": trial.recovery_index,
                    },
                )
            )
        return AcquisitionPlan(
            technique_id=self.technique_id,
            measurement_key="rb",
            shots=self.settings.shots,
            circuits=tuple(circuits),
        )

    def analyze(
        self,
        point: ExperimentPoint,
        plan: AcquisitionPlan,
        measurements: MeasurementData,
    ) -> tuple[TechniqueResult, ...]:
        results = []
        lengths = sorted({int(circuit.coordinates["length"]) for circuit in plan.circuits})
        for locus, probability_data in measurements.probabilities.items():
            grouped: dict[tuple[str, int], list[float]] = {}
            for probability, circuit in zip(probability_data.p1, plan.circuits, strict=True):
                key = (str(circuit.coordinates["kind"]), int(circuit.coordinates["length"]))
                grouped.setdefault(key, []).append(1 - float(probability))
            reference = np.array(
                [np.mean(grouped[("reference", length)]) for length in lengths]
            )
            interleaved = np.array(
                [np.mean(grouped[("interleaved", length)]) for length in lengths]
            )
            reference_decay = fit_decay(lengths, reference)
            interleaved_decay = fit_decay(lengths, interleaved)
            gate_decay = interleaved_decay / reference_decay
            results.append(
                TechniqueResult(
                    point_id=point.point_id,
                    target_id=point.target.operation_id,
                    technique_id=self.technique_id,
                    locus=locus,
                    metrics={
                        "rb_reference_decay": reference_decay,
                        "rb_interleaved_decay": interleaved_decay,
                        "rb_gate_decay": gate_decay,
                        "rb_infidelity": 0.5 * (1 - gate_decay),
                    },
                    artifacts={
                        "lengths": np.asarray(lengths),
                        "reference_survival": reference,
                        "interleaved_survival": interleaved,
                        "raw_p1": probability_data.raw_p1,
                        "analyzed_p1": probability_data.p1,
                    },
                    diagnostics={
                        "seed": self._seed_for(point),
                        "paired_sequences": self.settings.paired_sequences,
                    },
                )
            )
        return tuple(results)
