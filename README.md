# r-aml-uan-hybrid-drl-udds
"R-AML-UAN: Hybrid DRL for acoustic-optical underwater networks with UDDS decoy deception"
# Hybrid DRL UAN for UDDS

This repository contains the source code, output files, and visualization materials used for the Hybrid DRL-based underwater acoustic network (UAN) simulation under the UDDS scenario.

## Repository Contents

- `code/hybrid_drl_v3.cc` — Main C++ source file for the simulation.
- `graph/generate_paper_figures.py` — Script used to generate paper figures.
- `visualization/r-aml-uan-visualizer.html` — HTML visualization of the results.
- `output/hybrid_drl_results.csv` — Main simulation results.
- `output/hybrid_qvalues.csv` — Q-value output data.
- `output/hybrid_decoy_log.csv` — Decoy-related log output.
- `output/simulation_output.txt` — Raw simulation output log.

## Requirements

- C++ compiler.
- Git.
- Python 3.x.
- Any required libraries used in the plotting or visualization scripts.

## How to Run

### 1. Compile the C++ code

Use your preferred compiler or simulation environment to build the source file.

Example:

```bash
g++ code/hybrid_drl_v3.cc -o hybrid_drl_v3
```

If your code depends on a simulator framework, compile it using the appropriate framework instructions.

### 2. Run the simulation

```bash
./hybrid_drl_v3
```

This will generate the output files in the `output/` folder, depending on your implementation.

### 3. Generate figures

Run the plotting script to create publication figures.

```bash
python graph/generate_paper_figures.py
```

### 4. Open the visualization

Open this file in any modern browser:

```text
visualization/r-aml-uan-visualizer.html
```

## Output Files

The repository includes the following generated outputs:

- `output/hybrid_drl_results.csv`
- `output/hybrid_qvalues.csv`
- `output/hybrid_decoy_log.csv`
- `output/simulation_output.txt`

These files support the analysis and figures reported in the paper.

## Data Availability

The simulation code, output files, visualization files, and plotting scripts supporting the findings of this study are publicly available at:

https://github.com/Chandana-art30/r-aml-uan-hybrid-drl-udds

## Citation

If you use this repository, please cite the associated paper.

## Notes

- File and folder names are kept simple for reproducibility.
- The repository is organized so reviewers can quickly locate code, outputs, and visualization materials.
- If your environment requires additional dependencies, list them here.