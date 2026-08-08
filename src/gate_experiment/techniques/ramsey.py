"""Ramsey-based transition-phase characterization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..backend.base import CircuitFactory
from ..models.acquisition import AcquisitionPlan, MeasurementData, PlannedCircuit
from ..models.points import ExperimentPoint
from ..models.results import TechniqueResult
from ..targets.base import OperationUnderTest, require_single_qubit
from ..targets.composite import prx_unitary
from .base import CharacterizationTechnique, TechniqueContext
from .math import unitary_bloch_rotation, wrap_angle


def default_ramsey_phases() -> tuple[float, ...]:
    """Return 16 uniformly spaced phases without duplicating 0 at 2π."""

    return tuple(float(value) for value in np.linspace(0, 2 * np.pi, 17)[:-1])


@dataclass(frozen=True)
class RamseySettings:
    """Settings for the equatorial transition-phase Ramsey protocol."""

    phases: tuple[float, ...] = field(default_factory=default_ramsey_phases)
    shots: int = 1024

    def __post_init__(self) -> None:
        phases = tuple(float(phase) for phase in self.phases)
        if len(phases) < 3 or not all(math.isfinite(phase) for phase in phases):
            raise ValueError("Ramsey fitting needs at least three finite phases.")
        if self.shots < 1:
            raise ValueError("Ramsey shots must be positive.")
        object.__setattr__(self, "phases", phases)


def fit_fringe(phases: Sequence[float], p1: Sequence[float]) -> dict[str, float]:
    """Fit ``offset + cosine*cos(phase) + sine*sin(phase)`` by least squares."""

    phase_values = np.asarray(phases, dtype=float)
    probabilities = np.asarray(p1, dtype=float)
    if phase_values.shape != probabilities.shape:
        raise ValueError("Ramsey phases and probabilities must have equal shape.")
    design = np.column_stack(
        [np.ones(len(phase_values)), np.cos(phase_values), np.sin(phase_values)]
    )
    offset, cosine, sine = np.linalg.lstsq(design, probabilities, rcond=None)[0]
    return {
        "offset": float(offset),
        "contrast": float(math.hypot(cosine, sine)),
        "phase": float(math.atan2(sine, cosine)),
    }


def ideal_ramsey_p1(
    ideal_unitary: np.ndarray,
    phases: Sequence[float],
    *,
    prepare_half_pi: bool,
) -> np.ndarray:
    """Simulate the ideal Ramsey fringe for the configured protocol."""

    state = np.array([1.0, 0.0], dtype=complex)
    if prepare_half_pi:
        state = prx_unitary(np.pi / 2, 0) @ state
    state = np.asarray(ideal_unitary) @ state
    return np.array(
        [abs((prx_unitary(np.pi / 2, phase) @ state)[1]) ** 2 for phase in phases]
    )


def target_rotation(ideal_unitary: np.ndarray) -> tuple[float, np.ndarray]:
    """Return principal Bloch rotation angle and axis for a target unitary."""

    rotvec = Rotation.from_matrix(unitary_bloch_rotation(ideal_unitary)).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    axis = np.array([1.0, 0.0, 0.0]) if angle < 1e-12 else rotvec / angle
    return angle, axis


class TransitionPhaseRamsey(CharacterizationTechnique):
    """Measure transition phase for ideal +X π or π/2 rotations.

    This explicit name reflects the protocol's scientific assumptions.  It is
    not a generic Ramsey experiment: preparation and phase-error normalization
    follow the universal-pulse manuscript protocol.
    """

    technique_id = "transition_phase_ramsey"

    def __init__(self, settings: RamseySettings | None = None) -> None:
        self.settings = settings or RamseySettings()

    def supports(self, target: OperationUnderTest) -> bool:
        if target.ideal_operation.arity != 1:
            return False
        angle, axis = target_rotation(target.ideal_operation.unitary)
        angle_supported = abs(angle - np.pi) < 1e-6 or abs(angle - np.pi / 2) < 1e-6
        return angle_supported and np.dot(axis, np.array([1.0, 0.0, 0.0])) > 1 - 1e-6

    def validate(self, point: ExperimentPoint, context: TechniqueContext) -> None:
        require_single_qubit(point.target, "Transition-phase Ramsey")
        point.target.validate(point.parameters)
        if not self.supports(point.target):
            raise ValueError(
                "TransitionPhaseRamsey requires an ideal +X pi or +X pi/2 rotation."
            )
        if not context.loci:
            raise ValueError("Ramsey characterization needs at least one locus.")

    def build_plan(
        self,
        point: ExperimentPoint,
        context: TechniqueContext,
    ) -> AcquisitionPlan:
        angle, _ = target_rotation(point.target.ideal_operation.unitary)
        prepare_half_pi = abs(angle - np.pi) < 1e-6
        circuits = []
        for index, phase in enumerate(self.settings.phases):

            def build(factory: CircuitFactory, phase: float = phase) -> object:
                operations = {}
                for locus in context.loci:
                    sequence = []
                    if prepare_half_pi:
                        sequence.append(factory.calibrated_prx(locus, np.pi / 2, 0))
                    sequence.extend(
                        [
                            factory.target_operation(point.target, point.parameters, locus),
                            factory.calibrated_prx(locus, np.pi / 2, phase),
                        ]
                    )
                    operations[locus] = sequence
                return factory.measured_parallel(operations, "ramsey")

            circuits.append(
                PlannedCircuit(
                    circuit_id=f"ramsey:{index:03d}",
                    build=build,
                    coordinates={"phase": phase},
                )
            )
        return AcquisitionPlan(
            technique_id=self.technique_id,
            measurement_key="ramsey",
            shots=self.settings.shots,
            circuits=tuple(circuits),
        )

    def analyze(
        self,
        point: ExperimentPoint,
        plan: AcquisitionPlan,
        measurements: MeasurementData,
    ) -> tuple[TechniqueResult, ...]:
        phases = np.array([circuit.coordinates["phase"] for circuit in plan.circuits], dtype=float)
        angle, _ = target_rotation(point.target.ideal_operation.unitary)
        prepare_half_pi = abs(angle - np.pi) < 1e-6
        ideal_p1 = ideal_ramsey_p1(
            point.target.ideal_operation.unitary,
            phases,
            prepare_half_pi=prepare_half_pi,
        )
        ideal_fit = fit_fringe(phases, ideal_p1)
        results = []
        for locus, probability_data in measurements.probabilities.items():
            measured_fit = fit_fringe(phases, probability_data.p1)
            shift = wrap_angle(measured_fit["phase"] - ideal_fit["phase"])
            divisor = 2 if prepare_half_pi else 1
            fitted = (
                measured_fit["offset"]
                + measured_fit["contrast"] * np.cos(phases - measured_fit["phase"])
            )
            results.append(
                TechniqueResult(
                    point_id=point.point_id,
                    target_id=point.target.operation_id,
                    technique_id=self.technique_id,
                    locus=locus,
                    metrics={
                        "ramsey_contrast": measured_fit["contrast"],
                        "ramsey_fringe_phase": measured_fit["phase"],
                        "ramsey_fringe_shift": shift,
                        "transition_phase_error": shift / divisor,
                    },
                    artifacts={
                        "phases": phases,
                        "raw_p1": probability_data.raw_p1,
                        "analyzed_p1": probability_data.p1,
                        "ideal_p1": ideal_p1,
                        "fitted_p1": fitted,
                    },
                    diagnostics={"preparation_half_pi": prepare_half_pi},
                )
            )
        return tuple(results)
