"""Small in-memory backend used by integration tests.

These fakes evaluate ideal 2x2 matrices locally. They contain no Pulla object,
URL, compiler, job submission, or other path to an IQM service.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from gate_experiment.backend.base import BackendSession, ExperimentBackend
from gate_experiment.models.acquisition import (
    AcquisitionPlan,
    Locus,
    MeasurementData,
    ProbabilityData,
)
from gate_experiment.models.parameters import ParameterSet
from gate_experiment.targets.base import OperationUnderTest
from gate_experiment.targets.composite import prx_unitary


class MatrixCircuitFactory:
    """Represent circuit operations by their ideal unitary matrices."""

    def calibrated_prx(self, locus: Locus, angle: float, phase: float) -> np.ndarray:
        return prx_unitary(angle, phase)

    def target_operation(
        self,
        target: OperationUnderTest,
        parameters: ParameterSet,
        locus: Locus,
    ) -> np.ndarray:
        target.validate(parameters)
        return target.ideal_operation.unitary

    def measured_parallel(
        self,
        operations: Mapping[Locus, Sequence[object]],
        measurement_key: str,
    ) -> Mapping[Locus, Sequence[object]]:
        return operations


class MatrixBackendSession(BackendSession):
    """Execute deferred recipes by ideal state-vector propagation."""

    def __init__(self, loci: tuple[Locus, ...]) -> None:
        self._loci = loci
        self.executed_techniques: list[str] = []
        self.closed = False

    @property
    def loci(self) -> tuple[Locus, ...]:
        return self._loci

    def supports_target(self, target: OperationUnderTest) -> bool:
        return target.ideal_operation.arity == 1

    def execute(self, plan: AcquisitionPlan) -> MeasurementData:
        self.executed_techniques.append(plan.technique_id)
        factory = MatrixCircuitFactory()
        values = {locus: [] for locus in self.loci}
        for planned in plan.circuits:
            operations = planned.build(factory)
            for locus in self.loci:
                state = np.array([1.0, 0.0], dtype=complex)
                for operation in operations[locus]:
                    state = np.asarray(operation) @ state
                values[locus].append(float(abs(state[1]) ** 2))
        return MeasurementData(
            plan,
            {locus: ProbabilityData(p1) for locus, p1 in values.items()},
            execution_id=f"memory:{len(self.executed_techniques)}",
        )

    def close(self) -> None:
        self.closed = True


class MatrixBackend(ExperimentBackend):
    """Create and retain one in-memory session for test assertions."""

    def __init__(self) -> None:
        self.session: MatrixBackendSession | None = None

    def open_session(self, requested_qubits: Sequence[str] | None) -> MatrixBackendSession:
        qubits = tuple(requested_qubits or ("QB1",))
        self.session = MatrixBackendSession(tuple((qubit,) for qubit in qubits))
        return self.session
