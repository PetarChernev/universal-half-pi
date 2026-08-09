# Universal Half-pi Gate Experiments

Composite pulse characterization experiments for IQM hardware.

## Setup with pip

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Setup with conda or mamba

```bash
mamba create -n qti python=3.12 pip
mamba activate qti
python -m pip install -e .
```

Replace `mamba` with `conda` if preferred. If the `qti` environment already
exists, activate it and run only the final install command.

## Run

The configured experiment targets the `garnet` quantum computer and writes its
datasets and CSV summaries under `results/`.

```bash
python iqm_composite_experiment/main.py
```
