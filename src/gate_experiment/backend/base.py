"""Backend contracts used by the experiment application layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

from ..models.acquisition import AcquisitionPlan, Locus, MeasurementData
from ..models.parameters import ParameterSet
from ..targets.base import OperationUnderTest


class CircuitFactory(Protocol):
    """Minimal circuit-construction API available to deferred recipes.

    Techniques use calibrated PRX rotations for state preparation and analysis,
    but delegate the actual operation under test to the backend compiler registry.
    Returned handles are intentionally opaque and backend-specific.
    """

    def calibrated_prx(self, locus: Locus, angle: float, phase: float) -> object:
        """Build a calibrated equatorial rotation."""

    def target_operation(
        self,
        target: OperationUnderTest,
        parameters: ParameterSet,
        locus: Locus,
    ) -> object:
        """Compile an operation under test for one locus."""

    def measured_parallel(
        self,
        operations: Mapping[Locus, Sequence[object]],
        measurement_key: str,
    ) -> object:
        """Run per-locus operations in parallel and append measurement."""


TargetT = TypeVar("TargetT", bound=OperationUnderTest)


class OperationCompiler(ABC, Generic[TargetT]):
    """Backend-specific realization of one target type."""

    target_type: type[TargetT]

    @abstractmethod
    def compile(
        self,
        target: TargetT,
        parameters: ParameterSet,
        locus: Locus,
        context: Any,
    ) -> object:
        """Compile ``target`` into the backend operation represented by ``context``."""


class OperationCompilerRegistry:
    """Mutable backend-owned registry of operation compiler plugins."""

    def __init__(self, compilers: Sequence[OperationCompiler[Any]] = ()) -> None:
        self._compilers: dict[type[OperationUnderTest], OperationCompiler[Any]] = {}
        for compiler in compilers:
            self.register(compiler)

    def register(self, compiler: OperationCompiler[Any]) -> None:
        """Register exactly one compiler for a target class."""

        target_type = compiler.target_type
        if target_type in self._compilers:
            raise ValueError(f"A compiler for {target_type.__name__} is already registered.")
        self._compilers[target_type] = compiler

    def supports(self, target: OperationUnderTest) -> bool:
        """Return whether a compiler is registered for ``target``."""

        return self._find(type(target)) is not None

    def compile(
        self,
        target: OperationUnderTest,
        parameters: ParameterSet,
        locus: Locus,
        context: Any,
    ) -> object:
        """Resolve and invoke the most specific registered compiler."""

        compiler = self._find(type(target))
        if compiler is None:
            raise TypeError(f"No operation compiler is registered for {type(target).__name__}.")
        return compiler.compile(target, parameters, locus, context)

    def _find(self, target_type: type[OperationUnderTest]) -> OperationCompiler[Any] | None:
        for candidate in target_type.__mro__:
            if candidate in self._compilers:
                return self._compilers[candidate]
        return None


class BackendSession(ABC):
    """An initialized compiler/execution context for one experiment run."""

    @property
    @abstractmethod
    def loci(self) -> tuple[Locus, ...]:
        """Return physical loci selected for this session."""

    @abstractmethod
    def supports_target(self, target: OperationUnderTest) -> bool:
        """Return whether this session can compile ``target``."""

    @abstractmethod
    def execute(self, plan: AcquisitionPlan) -> MeasurementData:
        """Build, compile, execute, and normalize one acquisition plan."""

    def close(self) -> None:
        """Release backend resources; stateless sessions need not override this."""

    def __enter__(self) -> BackendSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class ExperimentBackend(ABC):
    """Factory for backend sessions.

    Opening a production session may retrieve topology and calibration data.
    Merely importing or constructing a backend must not contact a service.
    """

    @abstractmethod
    def open_session(self, requested_qubits: Sequence[str] | None) -> BackendSession:
        """Initialize a session for the requested single-qubit loci."""
