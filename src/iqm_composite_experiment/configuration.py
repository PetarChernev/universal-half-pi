"""Validated YAML composition for the plain Python experiment entrypoint.

The framework classes can be constructed directly in notebooks.  This module is
only the repository's composition layer: it translates human-editable YAML into
the same domain objects without embedding hardware or scientific settings in the
entrypoint script.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import yaml

from .backend.iqm import IQMBackendSettings, ProbabilitySource
from .experiment import (
    CharacterizationExperiment,
    ExperimentSettings,
    FailurePolicy,
)
from .models.parameters import Scalar
from .models.points import PointSet, SweepAxis
from .models.results import ExperimentResult
from .readout import (
    IndependentReadoutCorrection,
    IndependentReadoutSettings,
    NoReadoutCorrection,
)
from .targets.base import OperationUnderTest
from .targets.universal import universal_composite_pulses
from .techniques import (
    CharacterizationTechnique,
    InterleavedRandomizedBenchmarking,
    ProcessTomography,
    RBSettings,
    RamseySettings,
    TomographySettings,
    TransitionPhaseRamsey,
)

TechniqueFactory: TypeAlias = Callable[[Mapping[str, Any]], CharacterizationTechnique]


@dataclass(frozen=True)
class ApplicationConfiguration:
    """Fully resolved runtime configuration loaded from YAML."""

    server_url: str | None
    quantum_computer: str | None
    backend_settings: IQMBackendSettings
    experiment: CharacterizationExperiment
    output_csv: Path
    output_view: str

    def result_dataframe(self, result: ExperimentResult) -> pd.DataFrame:
        """Select the configured notebook/CSV view of a structured result."""

        if self.output_view == "metrics":
            return result.metrics_dataframe()
        if self.output_view == "long":
            return result.long_dataframe()
        if self.output_view == "legacy":
            return result.legacy_dataframe()
        raise ValueError(f"Unsupported output view {self.output_view!r}.")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section {path!r} must be a mapping.")
    return dict(value)


def _only_keys(values: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {path!r}: {unknown}.")


def _sequence(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise TypeError(f"Configuration value {path!r} must be a YAML list.")
    return tuple(value)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Configuration value {path!r} must be boolean.")
    return value


def _tomography_factory(values: Mapping[str, Any]) -> CharacterizationTechnique:
    _only_keys(values, {"shots"}, "techniques.tomography")
    return ProcessTomography(TomographySettings(shots=int(values.get("shots", 1024))))


def _ramsey_factory(values: Mapping[str, Any]) -> CharacterizationTechnique:
    _only_keys(values, {"shots", "phases"}, "techniques.transition_phase_ramsey")
    phases = values.get("phases")
    settings = (
        RamseySettings(shots=int(values.get("shots", 1024)))
        if phases is None
        else RamseySettings(
            phases=tuple(
                float(phase)
                for phase in _sequence(
                    phases,
                    "techniques.transition_phase_ramsey.phases",
                )
            ),
            shots=int(values.get("shots", 1024)),
        )
    )
    return TransitionPhaseRamsey(settings)


def _rb_factory(values: Mapping[str, Any]) -> CharacterizationTechnique:
    _only_keys(
        values,
        {"shots", "lengths", "samples", "seed", "paired_sequences"},
        "techniques.interleaved_rb",
    )
    lengths = tuple(
        int(length)
        for length in _sequence(
            values.get("lengths", [1, 2, 4, 8, 16, 32]),
            "techniques.interleaved_rb.lengths",
        )
    )
    return InterleavedRandomizedBenchmarking(
        RBSettings(
            lengths=lengths,
            samples=int(values.get("samples", 10)),
            shots=int(values.get("shots", 512)),
            seed=int(values.get("seed", 7)),
            paired_sequences=_boolean(
                values.get("paired_sequences", True),
                "techniques.interleaved_rb.paired_sequences",
            ),
        )
    )


def default_technique_factories() -> dict[str, TechniqueFactory]:
    """Return factories for the characterization plugins shipped here."""

    return {
        "tomography": _tomography_factory,
        "transition_phase_ramsey": _ramsey_factory,
        "interleaved_rb": _rb_factory,
    }


def load_application_configuration(
    path: str | Path,
    *,
    target_registry: Mapping[str, OperationUnderTest] | None = None,
    technique_factories: Mapping[str, TechniqueFactory] | None = None,
) -> ApplicationConfiguration:
    """Load and validate one YAML experiment definition.

    Custom targets and techniques remain pluggable: notebook or project code can
    pass extended registries without modifying this loader or the experiment
    runner.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    root = _mapping(document, "root")
    _only_keys(root, {"iqm", "experiment", "techniques", "readout", "output"}, "root")

    iqm = _mapping(root.get("iqm", {}), "iqm")
    _only_keys(
        iqm,
        {"server_url", "quantum_computer", "qubits", "prx_implementation", "probability_source"},
        "iqm",
    )
    server_url = iqm.get("server_url")
    if server_url is not None and not isinstance(server_url, str):
        raise TypeError("iqm.server_url must be a string or null.")
    quantum_computer = iqm.get("quantum_computer")
    if quantum_computer is not None and not isinstance(quantum_computer, str):
        raise TypeError("iqm.quantum_computer must be a string or null.")
    qubits_value = iqm.get("qubits")
    qubits = (
        None
        if qubits_value is None
        else tuple(str(qubit) for qubit in _sequence(qubits_value, "iqm.qubits"))
    )
    try:
        probability_source = ProbabilitySource(
            iqm.get("probability_source", ProbabilitySource.EXA_PROBABILITY.value)
        )
    except ValueError as error:
        choices = [source.value for source in ProbabilitySource]
        raise ValueError(f"iqm.probability_source must be one of {choices}.") from error
    backend_settings = IQMBackendSettings(
        prx_implementation=iqm.get("prx_implementation"),
        probability_source=probability_source,
    )

    targets = universal_composite_pulses()
    if target_registry is not None:
        targets.update(target_registry)
    experiment_values = _mapping(root.get("experiment", {}), "experiment")
    _only_keys(experiment_values, {"targets", "failure_policy", "sweep"}, "experiment")
    target_names = _sequence(experiment_values.get("targets", []), "experiment.targets")
    if not target_names:
        raise ValueError("experiment.targets must select at least one target.")
    missing_targets = sorted(set(target_names) - set(targets))
    if missing_targets:
        raise ValueError(f"Unknown experiment targets: {missing_targets}.")
    selected_targets = tuple(targets[str(name)] for name in target_names)

    sweep = _mapping(experiment_values.get("sweep", {}), "experiment.sweep")
    _only_keys(sweep, {"kind", "axes", "fixed"}, "experiment.sweep")
    axes_values = _sequence(sweep.get("axes", []), "experiment.sweep.axes")
    axes = []
    for index, untyped_axis in enumerate(axes_values):
        axis = _mapping(untyped_axis, f"experiment.sweep.axes[{index}]")
        _only_keys(
            axis,
            {"name", "values", "unit", "description"},
            f"experiment.sweep.axes[{index}]",
        )
        axes.append(
            SweepAxis(
                name=str(axis["name"]),
                values=tuple(
                    _sequence(
                        axis["values"],
                        f"experiment.sweep.axes[{index}].values",
                    )
                ),
                unit=axis.get("unit"),
                description=str(axis.get("description", "")),
            )
        )
    fixed_untyped = _mapping(sweep.get("fixed", {}), "experiment.sweep.fixed")
    fixed: dict[str, Scalar] = dict(fixed_untyped)
    sweep_kind = str(sweep.get("kind", "cartesian"))
    if sweep_kind == "cartesian":
        points = PointSet.cartesian(targets=selected_targets, axes=axes, fixed=fixed)
    elif sweep_kind == "zipped":
        points = PointSet.zipped(targets=selected_targets, axes=axes, fixed=fixed)
    else:
        raise ValueError("experiment.sweep.kind must be 'cartesian' or 'zipped'.")

    factories = default_technique_factories()
    if technique_factories is not None:
        factories.update(technique_factories)
    configured_techniques = _mapping(root.get("techniques", {}), "techniques")
    techniques = []
    for name, untyped_settings in configured_techniques.items():
        if name not in factories:
            raise ValueError(f"No technique factory is registered for {name!r}.")
        settings = _mapping(untyped_settings, f"techniques.{name}")
        enabled = _boolean(settings.pop("enabled", True), f"techniques.{name}.enabled")
        if enabled:
            techniques.append(factories[name](settings))

    readout_values = _mapping(root.get("readout", {}), "readout")
    _only_keys(readout_values, {"enabled", "shots"}, "readout")
    readout_enabled = _boolean(readout_values.get("enabled", False), "readout.enabled")
    if readout_enabled and probability_source is not ProbabilitySource.THRESHOLDED_READOUT:
        raise ValueError(
            "Independent readout correction requires iqm.probability_source="
            "'thresholded_readout' to avoid double correction."
        )
    readout = (
        IndependentReadoutCorrection(
            IndependentReadoutSettings(shots=int(readout_values.get("shots", 1024)))
        )
        if readout_enabled
        else NoReadoutCorrection()
    )

    try:
        failure_policy = FailurePolicy(experiment_values.get("failure_policy", "abort"))
    except ValueError as error:
        raise ValueError(
            "experiment.failure_policy must be 'abort' or 'record-and-continue'."
        ) from error
    experiment = CharacterizationExperiment(
        points=points,
        techniques=tuple(techniques),
        settings=ExperimentSettings(qubits=qubits, failure_policy=failure_policy),
        readout=readout,
    )

    output = _mapping(root.get("output", {}), "output")
    _only_keys(output, {"csv", "view"}, "output")
    output_csv = Path(str(output.get("csv", "composite_experiment_results.csv")))
    if not output_csv.is_absolute():
        output_csv = config_path.parent / output_csv
    output_view = str(output.get("view", "legacy"))
    if output_view not in {"metrics", "long", "legacy"}:
        raise ValueError("output.view must be 'metrics', 'long', or 'legacy'.")
    return ApplicationConfiguration(
        server_url=server_url,
        quantum_computer=quantum_computer,
        backend_settings=backend_settings,
        experiment=experiment,
        output_csv=output_csv,
        output_view=output_view,
    )
