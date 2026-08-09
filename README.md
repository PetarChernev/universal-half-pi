# Universal Half-pi Gate Experiments

Composite pulse characterization experiments for IQM hardware.

## Setup with pip

Python 3.11 through 3.14 is required by `iqm-pulla`. Python 3.12 is the
recommended version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Setup with conda or mamba

```bash
conda create -n iqm python=3.12 pip
conda activate iqm
python -m pip install -e .
```

Replace `conda` with `mamba` if preferred. If the `iqm` environment already
exists, verify that `python --version` reports a supported version before
running the final install command. An older environment can be updated with:

```bash
conda install -n iqm python=3.12 pip
conda activate iqm
python -m pip install -e .
```

## Run

The configured experiment targets the `garnet` quantum computer and writes its
datasets and CSV summaries under `results/`.

```bash
python iqm_composite_experiment/main.py
```
