# Refactor Plan

1. Add `iqm_composite_experiment/persistence.py`
   - Introduce a single `RunArtifacts` class responsible for every filesystem write.
   - Capture the run start time on construction, but do not create a directory yet.
   - Lazily create a directory such as:
     ```text
     runs/20260809T142233.123456Z/
     ```
   - Serialize content before creating the directory so acquisition or serialization failures do not leave empty run directories.
   - Use atomic temporary-file replacement where practical.
   - Provide centralized methods:
     ```python
     write_acquisition(...)
     write_results(...)
     ```
   - This becomes the only place that needs modification when persistence behavior changes.

2. Add `iqm_composite_experiment/technique.py`
   - Define a small acquisition container:
     ```python
     @dataclass(frozen=True)
     class Acquisition:
         dataset: Any
         metadata: Mapping[str, Any]
     ```
   - Define an abstract `CharacterizationTechnique` base class.
   - Store the shared dependencies on the instance:
     - `pulla`
     - `compiler`
     - `qubits`
     - `config`
     - `readout`
     - `artifacts`
   - Implement the shared `run()` template:
     1. Iterate over sequences, amplitude errors, and detunings.
     2. Call the subclass acquisition method.
     3. Immediately persist the returned dataset.
     4. Only then extract probabilities, correct readout, fit, or reconstruct.
     5. Build the shared result columns.
     6. Delegate technique-specific analysis to the subclass.
     7. Write the non-empty result DataFrame through `RunArtifacts`.
     8. Return the DataFrame for notebook use.

3. Define the subclass interface
   ```python
   class CharacterizationTechnique(ABC):
       technique_id: str

       @abstractmethod
       def acquire(...) -> Acquisition:
           ...

       @abstractmethod
       def analyze_locus(
           self,
           acquisition: Acquisition,
           qubit: str,
           sequence: SequenceSpec,
       ) -> dict[str, Any]:
           ...
   ```
   - The base class owns lifecycle and persistence.
   - Subclasses own circuit construction and scientific analysis.
   - Existing numerical helper functions remain module-level pure functions.

4. Refactor `tomography.py`
   - Add `ProcessTomography(CharacterizationTechnique)`.
   - Move the current `acquire()` behavior into the subclass.
   - Return tomography coordinates as acquisition metadata.
   - Move per-qubit reconstruction into `analyze_locus()`.
   - Keep `prep_boxes`, `analysis_boxes`, `reconstruct_ptm`, and `metrics` as reusable functions.
   - Remove the duplicated top-level sequence/error loops.

5. Refactor `ramsey.py`
   - Add `TransitionPhaseRamsey(CharacterizationTechnique)`.
   - Return the dataset plus configured phases as metadata.
   - Implement per-qubit fringe fitting in `analyze_locus()`.
   - Keep `fit_fringe`, `ideal_p1`, and `metrics` as pure helpers.
   - Remove its duplicated orchestration and persistence responsibilities.

6. Refactor `rb.py`
   - Add `InterleavedRandomizedBenchmarking(CharacterizationTechnique)`.
   - Return the dataset and complete RB plan in `Acquisition.metadata`.
   - Use that persisted plan during analysis so the raw probabilities remain interpretable.
   - Keep Clifford generation, planning, decay fitting, and metric functions independent.
   - Remove its duplicated orchestration loop.

7. Persist raw acquisitions before postprocessing
   - Persist the exact dataset returned by `run_batch()`, before:
     - `p1_from_dataset`
     - readout correction
     - averaging
     - fitting
     - tomography reconstruction
   - Store each dataset as NetCDF.
   - Store an adjacent JSON sidecar containing:
     - technique
     - acquisition index
     - sequence and pulse count
     - amplitude error
     - detuning
     - qubits
     - technique-specific coordinates or RB plan
   - Suggested layout:
     ```text
     runs/20260809T142233.123456Z/
       run.json
       raw/
         tomography/
           0000_H11a.nc
           0000_H11a.json
         ramsey/
           0000_H11a.nc
           0000_H11a.json
         rb/
           0000_H11a.nc
           0000_H11a.json
       results/
         tomography.csv
         ramsey.csv
         rb.csv
     ```
   - Use the acquisition index for uniqueness rather than embedding floating-point values in filenames.

8. Update `common.py`
   - Add an output-root setting to `Config`, defaulting to something like `Path("runs")`.
   - Keep `run_batch()` unchanged: it remains responsible only for execution and returning the IQM dataset.
   - Do not put persistence into `run_batch()`, because it lacks sequence and technique metadata.

9. Update `main.py`
   - Construct one `RunArtifacts` instance for the entire invocation.
   - Pass that same instance to all enabled technique objects.
   - Replace the three direct `to_csv()` calls with class execution:
     ```python
     artifacts = RunArtifacts(config.output_root)

     ProcessTomography(..., artifacts).run(sequences)
     TransitionPhaseRamsey(..., artifacts).run(sequences)
     InterleavedRandomizedBenchmarking(..., artifacts).run(sequences)
     ```
   - Because all objects share the writer, every artifact lands under one timestamped run directory.
   - If readout calibration, setup, acquisition, or interruption happens before the first writable dataset exists, no run directory is created.

10. Verify the critical behavior
   - Constructing `RunArtifacts` creates no directory.
   - Empty sequence collections produce no directory.
   - Acquisition failure before a dataset returns produces no directory.
   - A successful acquisition is written before analysis starts.
   - An analysis failure still leaves its raw acquisition available.
   - NetCDF serialization failure does not create a run directory.
   - All three techniques use the same timestamped directory.
   - RB and tomography metadata round-trip through JSON.
   - Result CSVs are only written when their DataFrames contain rows.

This keeps scientific differences in the three subclasses while centralizing execution order, raw persistence, directory lifecycle, naming, and summary writing.
