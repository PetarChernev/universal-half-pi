# Universal Half-pi Gate Experiments

This repository provides an extensible framework for characterizing physical
quantum operations. The supplied targets are universal composite PRX sequences,
and the supplied techniques are process tomography, transition-phase Ramsey, and
interleaved randomized benchmarking.

The framework separates the object under test from the way it is measured. A new
composite sequence or pulse-shaping strategy implements `OperationUnderTest`; a
new measurement method implements `CharacterizationTechnique`.

## Running The Configured Experiment

1. Activate the existing environment with `mamba activate qti`.
2. Edit `config.yaml`, including `iqm.server_url`.
3. Run `python run_experiment.py`.

There is intentionally no command-line interface. All runtime settings live in
`config.yaml`, and `run_experiment.py` is a plain Python composition script.

Importing the package, loading configuration, or running the tests does not
connect to IQM services. Only executing `run_experiment.py` with a configured URL
constructs `Pulla` and opens a backend session.

## Notebook Use

Framework objects can also be assembled directly in a notebook:

```python
from gate_experiment import (
    CharacterizationExperiment,
    ExperimentSettings,
    PointSet,
    ProcessTomography,
    SweepAxis,
    TomographySettings,
    universal_composite_pulses,
)

targets = universal_composite_pulses()
points = PointSet.cartesian(
    targets=(targets["X1"], targets["X5a"]),
    axes=(
        SweepAxis("amplitude_error", (-0.02, 0.0, 0.02)),
        SweepAxis("detuning_hz", (-2e6, 0.0, 2e6), unit="Hz"),
    ),
)

experiment = CharacterizationExperiment(
    points=points,
    techniques=(ProcessTomography(TomographySettings(shots=1024)),),
    settings=ExperimentSettings(qubits=("QB1",)),
)
```

Execution is deliberately separate from this scientific definition:

```python
from iqm.pulla.pulla import Pulla
from gate_experiment.backend.iqm import IQMBackend
from gate_experiment.experiment import ExperimentRunner

pulla = Pulla("https://your-iqm-server")
result = ExperimentRunner(IQMBackend(pulla)).run(experiment)
result.metrics_dataframe()
```

See `TECHNICAL_SPEC.md` for architecture, extension points, data flow, and
service-free testing guarantees.

## Tests

The test suite uses only pure functions, fake backends, and in-memory datasets:

```bash
mamba run -n qti python -m unittest discover -s tests -v
```
