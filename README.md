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
datasets, acquisition metadata, and CSV summaries under `results/`.

```bash
python iqm_composite_experiment/main.py
```

All circuits are built and compiled before the first job is submitted. Each
enabled characterization technique is submitted as one job. Depending on
`Config.pre_calibration` and `Config.post_calibration`, that job includes a
ground/excited readout calibration before and/or after its characterization
circuits. If both flags are false, analysis skips readout correction.

The run directory contains `preflight.json` with IQM's estimated QPU runtime
and a `playlists/` directory with one HTML visualization per technique job.
Queue and network time are not included in the estimate.

To generate these preflight artifacts without submitting any jobs:

```bash
python iqm_composite_experiment/main.py --prepare-only
```

The analytic X5 and X9 phases are represented from their exact formulas. Longer
numerical phase vectors use the six-decimal values published in Tables I and II
of `references/Universal_Gates.pdf`; reproducing the paper's machine-precision
cancellation residuals requires the authors' full-precision vectors.
