### Interactive Event Analysis Workflow

This workflow allows you to analyze and visualize pulse data and event interactions from a ROOT file in an easy and automated way. The `run_interact.py` script simplifies the process by running the necessary steps in sequence.

## Usage Instructions

1. **Prepare the ROOT File Name**
    - Open the `root_file_name.txt` file and enter the full path to the simulation ROOT file you wish to analyze. (optical processes and propogation must be on in the simulation)
    - Example:
      ```
      path/to/mc_FullOP_x45deg_mu+_0_run1_0_Test_10evt_mu+_5.0_5.0.root
      ```

2. **Run `run_interact.py`**
    - Execute the `run_interact.py` script to automate the workflow:
      ```bash
      python run_interact.py
      ```
    - This script will:
      1. Run `makeInteractPulse.py` to process the ROOT file and generate the necessary pulse reconstruction and truth histograms.
      2. If successful, it will automatically run `eventinteract2.py` to allow interactive event analysis.

3. **Interactive Event Analysis**
    - During the execution of `eventinteract2.py`, you will be prompted to select an event number to filter.
    - After selecting an event, you can interactively click on bins with entries to generate and view plots.
    - All generated plots will be saved in the `./plots` directory for later review.

## Notes

- **ROOT File Validation:** The `run_interact.py` script ensures that the ROOT file specified in `root_file_name.txt` exists and is valid before proceeding.
- **Output Files:**
  - The processed histograms are saved in `output.root` by default. If you wish to use a different name, modify the output file name in `makeInteractPulse.py` (line 143).
  - Plots generated during the interactive session are saved in the `./plots` directory.
- **Order of Execution:** The `run_interact.py` script ensures that the steps are executed in the correct order, so you no longer need to manually run `makeInteractPulse.py` and `eventinteract2.py`.
