"""Run the composite-gate experiments.

Comment out any of the three result blocks to disable that experiment.
"""

from datetime import datetime
from pathlib import Path

from iqm.pulla.pulla import Pulla

from common import Config, built_in_sequences, readout_calibration, select_qubits
import rb
import ramsey
import tomography


if __name__ == "__main__":
    registry = built_in_sequences()
    config = Config(qubits=None)
    sequences = [registry["H11a"]]

    pulla = Pulla(quantum_computer="garnet")
    compiler = pulla.get_standard_compiler()
    qubits = select_qubits(pulla, config.qubits)
    readout = readout_calibration(pulla, compiler, qubits, config)
    output_directory = Path("results") / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_directory.mkdir(parents=True)

    # Comment out any block to disable that experiment.
    tomography_results = tomography.run(pulla, compiler, qubits, sequences, config, readout, output_directory)
    tomography_results.to_csv(output_directory / "composite_tomography_results.csv", index=False)

    ramsey_results = ramsey.run(pulla, compiler, qubits, sequences, config, readout, output_directory)
    ramsey_results.to_csv(output_directory / "composite_ramsey_results.csv", index=False)

    rb_results = rb.run(pulla, compiler, qubits, sequences, config, readout, output_directory)
    rb_results.to_csv(output_directory / "composite_rb_results.csv", index=False)
