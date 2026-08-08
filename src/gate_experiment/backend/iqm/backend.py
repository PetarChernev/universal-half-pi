"""IQM Pulla 14.x execution adapter.

No service access occurs at import time or when :class:`IQMBackend` is created.
Topology and compiler retrieval happen only in ``open_session``, which allows all
framework tests to use local fake backends.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from iqm.pulla.pulla import Pulla

from ...models.acquisition import AcquisitionPlan, Locus, MeasurementData
from ...targets.base import OperationUnderTest
from ..base import BackendSession, ExperimentBackend, OperationCompilerRegistry
from .circuits import IQMCircuitFactory
from .decoder import IQMDatasetDecoder, ProbabilitySource
from .target_compilers import IQMUniversalCompositePulseCompiler


@dataclass(frozen=True)
class IQMBackendSettings:
    """Hardware-adapter settings independent of scientific technique settings."""

    prx_implementation: str | None = None
    probability_source: ProbabilitySource = ProbabilitySource.EXA_PROBABILITY

    def __post_init__(self) -> None:
        if self.prx_implementation is not None and not isinstance(
            self.prx_implementation, str
        ):
            raise TypeError("PRX implementation must be a string or None.")
        if not isinstance(self.probability_source, ProbabilitySource):
            object.__setattr__(
                self,
                "probability_source",
                ProbabilitySource(self.probability_source),
            )


class IQMBackend(ExperimentBackend):
    """Open IQM execution sessions from an already configured Pulla client."""

    def __init__(
        self,
        pulla: Pulla,
        settings: IQMBackendSettings | None = None,
        registry: OperationCompilerRegistry | None = None,
    ) -> None:
        self._pulla = pulla
        self._settings = settings or IQMBackendSettings()
        self._registry = registry or OperationCompilerRegistry(
            (IQMUniversalCompositePulseCompiler(),)
        )

    def open_session(self, requested_qubits: Sequence[str] | None) -> IQMBackendSession:
        topology = self._pulla.get_chip_topology()
        if hasattr(topology, "qubits_sorted"):
            available = tuple(topology.qubits_sorted)
        else:
            available = tuple(sorted(topology.qubits))
        if requested_qubits is None:
            selected = available
        else:
            missing = sorted(set(requested_qubits) - set(available))
            if missing:
                raise ValueError(f"Unknown qubits: {missing}.")
            selected = tuple(requested_qubits)
        if not selected:
            raise ValueError("At least one qubit must be selected.")
        compiler = self._pulla.get_standard_compiler()
        return IQMBackendSession(
            pulla=self._pulla,
            compiler=compiler,
            qubits=selected,
            settings=self._settings,
            registry=self._registry,
        )


class IQMBackendSession(BackendSession):
    """Compiler and execution state shared by one IQM experiment run."""

    def __init__(
        self,
        *,
        pulla: Pulla,
        compiler: Any,
        qubits: tuple[str, ...],
        settings: IQMBackendSettings,
        registry: OperationCompilerRegistry,
    ) -> None:
        self._pulla = pulla
        self._compiler = compiler
        self._qubits = qubits
        self._settings = settings
        self._registry = registry
        self._decoder = IQMDatasetDecoder(settings.probability_source)

    @property
    def loci(self) -> tuple[Locus, ...]:
        return tuple((qubit,) for qubit in self._qubits)

    def supports_target(self, target: OperationUnderTest) -> bool:
        return self._registry.supports(target)

    def execute(self, plan: AcquisitionPlan) -> MeasurementData:
        """Compile deferred recipes through IQM's builder callback and execute them."""

        def timeboxes(builder: Any) -> list[object]:
            factory = IQMCircuitFactory(
                builder,
                self._registry,
                prx_implementation=self._settings.prx_implementation,
            )
            return [planned.build(factory) for planned in plan.circuits]

        settings = self._compiler.get_settings(
            timeboxes=timeboxes,
            qubits=list(self._qubits),
        )
        settings.set_shots(plan.shots)
        run_definition, context = self._compiler.compile(
            timeboxes=timeboxes,
            components=list(self._qubits),
            settings=settings,
        )
        job = self._pulla.submit_playlist(run_definition, context=context)
        job.wait_for_completion()
        result = job.result(self._compiler)
        if result is None or not hasattr(result, "dataset"):
            raise RuntimeError("IQM job completed without an EXA-style dataset result.")
        execution_id = str(getattr(job, "job_id", "")) or None
        return self._decoder.decode(
            result.dataset,
            plan,
            self.loci,
            execution_id=execution_id,
        )
