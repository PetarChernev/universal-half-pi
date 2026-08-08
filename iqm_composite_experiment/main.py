"""Run the composite-gate experiments.

Comment out any of the three result blocks to disable that experiment.
"""

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

    # Comment out any block to disable that experiment.
    tomography_results = tomography.run(pulla, compiler, qubits, sequences, config, readout)
    tomography_results.to_csv("composite_tomography_results.csv", index=False)

    ramsey_results = ramsey.run(pulla, compiler, qubits, sequences, config, readout)
    ramsey_results.to_csv("composite_ramsey_results.csv", index=False)

    rb_results = rb.run(pulla, compiler, qubits, sequences, config, readout)
    rb_results.to_csv("composite_rb_results.csv", index=False)
