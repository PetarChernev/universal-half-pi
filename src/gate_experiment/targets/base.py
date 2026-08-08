"""Interfaces for pluggable operations under test."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from ..models.parameters import ParameterSet, ParameterSpec, Scalar


@dataclass(frozen=True, eq=False)
class IdealOperation:
    """The ideal unitary transformation targeted by a physical implementation.

    Global phase is intentionally not normalized: all framework comparisons are
    phase-insensitive.  The unitary is copied and marked read-only so results do
    not change if a notebook later mutates its original NumPy array.
    """

    unitary: np.ndarray
    arity: int
    name: str

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise ValueError("Operation arity must be at least one.")
        matrix = np.array(self.unitary, dtype=complex, copy=True)
        dimension = 2**self.arity
        if matrix.shape != (dimension, dimension):
            raise ValueError(
                f"An arity-{self.arity} unitary must have shape "
                f"{(dimension, dimension)}, not {matrix.shape}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Ideal unitary entries must be finite.")
        if not np.allclose(matrix.conj().T @ matrix, np.eye(dimension), atol=1e-9):
            raise ValueError("Ideal operation is not unitary.")
        matrix.setflags(write=False)
        object.__setattr__(self, "unitary", matrix)


class OperationUnderTest(ABC):
    """Backend-neutral description of a physical operation to characterize.

    Subclasses contain scientific intent and implementation parameters only.
    Hardware realization is delegated to a backend-specific
    :class:`~gate_experiment.backend.base.OperationCompiler`.
    """

    operation_id: str

    @property
    @abstractmethod
    def ideal_operation(self) -> IdealOperation:
        """Return the target transformation used by analysis techniques."""

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        """Return the point parameters understood by this implementation."""

        return ()

    def resolve_parameters(self, parameters: ParameterSet) -> ParameterSet:
        """Validate point parameters and insert target-specific defaults."""

        if not self.operation_id:
            raise ValueError("Operation IDs cannot be empty.")
        return parameters.validate_against(self.parameter_specs)

    def validate(self, parameters: ParameterSet) -> None:
        """Validate that this target can be evaluated at ``parameters``."""

        self.resolve_parameters(parameters)

    def metadata(self) -> dict[str, Scalar]:
        """Return scalar metadata copied into tabular experiment results."""

        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(operation_id={self.operation_id!r})"


def require_single_qubit(target: OperationUnderTest, technique_name: str) -> None:
    """Raise if ``target`` is not a single-qubit operation."""

    if target.ideal_operation.arity != 1:
        raise ValueError(
            f"{technique_name} supports single-qubit operations; "
            f"{target.operation_id!r} has arity {target.ideal_operation.arity}."
        )
