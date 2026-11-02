# What Does Loading Classical Data Cost a Quantum Kernel?

Code and result files for the paper

> Suyash. *What Does Loading Classical Data Cost a Quantum Kernel? A Budget-Matched Benchmark of Encodings, Accuracy and Circuit Cost.* Submitted to *Quantum Machine Intelligence* (2026).

The scripts compare quantum kernels (angle, IQP, ZZ and amplitude encodings, fidelity and projected constructions) with tuned classical kernels and tree ensembles under one shared tuning budget. They also count the full loading cost of each model: two-qubit gates per circuit, circuits per Gram matrix, shots, and circuits spent on bandwidth tuning. Everything is exact statevector or density-matrix **simulation**; no quantum hardware or IBM account is needed.

Every number, table and figure in the paper is read from the files in `results/` by the scripts in `code/`. Nothing is typed in by hand.

## Contents

```
code/              all Python scripts (run from this folder; they import each other)
results/           the result files used in the paper (CSV and text reports)
results/design/    design-stage power analysis for the equivalence margin
reproduce.bat/.sh  re-run every experiment, then rebuild all outputs (several hours)
make_outputs.bat/.sh  rebuild tables, figures and Online Resources from results/ (a few minutes)
requirements.txt   package versions used for the paper
```

## Setup

Python 3.13 was used for the paper. Other recent versions should work.

```
pip install -r requirements.txt
cd code
python qkcommon.py --selftest     # should print SELFTEST PASSED
```

The OpenML datasets are downloaded on first use and cached in `code/data_cache/`.

## Rebuild the paper's tables and figures (minutes)

Windows: `make_outputs.bat`   Linux/macOS: `bash make_outputs.sh`

This writes the LaTeX tables to `results/tables/`, the figures to `results/figs/` and the Online Resources (supplementary CSV files) to `online_resources/`.

## Re-run all experiments (several hours)

Windows: `reproduce.bat 6`   Linux/macOS: `bash reproduce.sh 6`

The argument is the number of worker processes. The scripts overwrite the files in `results/`. Each step writes a log file `results/log_<step>.txt`. On a laptop with an Intel Core 5 210H CPU and ten workers, the longest steps took about 1.8 h (qubit scaling), 36 min (device replay), 27 min (positive control) and 15 min (main study).

Both runners set the two environment variables the paper uses:

```
QK_TIE=middle                                              # ties broken towards the centre of the grid
QK_S_GRID=0.002,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1,2,4,8  # bandwidth grid
```

If you run a script by hand, set them first. Every study script accepts `--help`, and most accept `--quick` for a small test run.

## Which script produces what

| Script | Paper section | Main outputs |
|---|---|---|
| `multi_dataset_study.py` | Main study: 16 datasets, n = 4, 20 resamples | `mds_*` |
| `qubit_scaling_study.py` | Accuracy from 4 to 12 qubits | `qs_*` |
| `fixed_bandwidth_study.py` | Fixed-bandwidth reference (s = 1), concentration | `fb_*` |
| `gate_counts.py` | Compiled two-qubit gates of the embedding circuits | `gate_counts.csv`, `circuits_per_gram.csv` |
| `loading_counts.py` | Exact amplitude loading: gate counts and cost ratio | `loading_*` |
| `cost_frontier.py`, `lever_table.py` | Loading bill and its levers | `frontier*.csv`, `lever_*` |
| `surrogate_study.py` | Surrogate tuning and Nyström landmarks | `sur_*` |
| `stationarity_study.py` | Kernel class (stationarity index) | `stat_*` |
| `rank_check.py` | Effective dimension of the Gram matrices | `rank_*` |
| `relabel_study.py` | Positive control on engineered labels | `rel_*` |
| `rel_edge_check.py` | Positive control with the grid extended to s = 64 | `rel_edge_*` |
| `rel_surrogate.py` | Surrogate tuning on engineered labels | `relsur_*` |
| `bandwidth_labels.py` | Labelling-bandwidth test | `bwl_*` |
| `dose_response.py` | Detection limit | `dose_*` |
| `reduction_study.py` | Reduction before amplitude loading (H1) | `reduction_*` |
| `shots_device_study.py`, `device_reanalysis.py` | Finite shots, device-model replay | `shots_*`, `device_*` |
| `v2_analysis.py` | Per-dataset equivalence, geometry, pooled reduction analysis | `v2_*` |
| `power_analysis.py` | Design-stage choice of the margin δ = 0.02 | `design/power_analysis.txt` |
| `make_tables.py`, `make_figures.py`, `make_online_resources.py` | Tables, figures, Online Resources | `tables/`, `figs/`, `online_resources/` |

Shared code is in `qkcommon.py` (encodings and kernels), `datasets_cv.py` (datasets and resampling), `gram_spread.py` (shared tuning grid and tie rule) and `extended_study.py` (corrected resampled t test and helpers). `environment_report.py` writes `results/environment.txt`.

The files `results/surrogate_design.txt`, `results/relabel_design.txt` and `results/design_v2.txt` record the designs that were fixed before the corresponding full runs.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Licence

MIT (see `LICENSE`).
