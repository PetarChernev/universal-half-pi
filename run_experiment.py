"""Run the experiment described by the adjacent ``config.yaml`` file.

This is intentionally a plain Python entrypoint rather than a CLI. Edit the YAML
configuration, then execute ``mamba run -n qti python run_experiment.py``.
"""

from pathlib import Path

from iqm.pulla.pulla import Pulla

from iqm_composite_experiment.backend.iqm import IQMBackend
from iqm_composite_experiment.configuration import load_application_configuration
from iqm_composite_experiment.experiment import ExperimentRunner

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def main() -> None:
    """Load configuration, execute the hardware experiment, and write its table."""

    configuration = load_application_configuration(CONFIG_PATH)
    if not configuration.server_url:
        raise ValueError("Set iqm.server_url in config.yaml before running the experiment.")

    pulla = Pulla(
        configuration.server_url,
        quantum_computer=configuration.quantum_computer,
    )
    backend = IQMBackend(pulla, configuration.backend_settings)
    result = ExperimentRunner(backend).run(configuration.experiment)
    table = configuration.result_dataframe(result)
    table.to_csv(configuration.output_csv, index=False)
    print(table)


if __name__ == "__main__":
    main()
