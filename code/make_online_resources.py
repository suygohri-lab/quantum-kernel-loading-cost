"""Write the Online Resources (electronic supplementary material) of the paper from the result files.

Each file is a CSV with a commented header (article title, journal, author, content), as Springer Nature asks for
supplementary files.  Usage:  python make_online_resources.py --results results_v2 --out ../paper/online_resources
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TITLE = ("What Does Loading Classical Data Cost a Quantum Kernel? A Budget-Matched Benchmark of Encodings, "
         "Accuracy and Circuit Cost")
JOURNAL = "Quantum Machine Intelligence"
AUTHOR = "Suyash, Department of Computer Science & Engineering, Punjab Engineering College, Chandigarh, India; e-mail: suyashsuyash.phd26cse@pec.edu.in"

# (number, output name, source file, content description)
RESOURCES = [
    (1, "ESM_1_paired_vs_rbf.csv", "mds_paired_vs_rbf.csv",
     "Main study (n = 4): pooled and per-dataset paired differences of every model to the tuned RBF kernel "
     "(accuracy, balanced accuracy, macro-F1), corrected resampled t, 95% intervals, TOST at +/-0.02"),
    (2, "ESM_2_reduction_table.csv", "reduction_table.csv",
     "H1: all 84 pre-specified reduction configurations and the exploratory latent sets, accuracy change of the "
     "amplitude kernel against the unreduced amplitude kernel, intervals and two-qubit-gate savings"),
    (3, "ESM_3_loading_counts.csv", "loading_counts.csv",
     "Two-qubit gates of exact amplitude loading for five vector families, n = 1 to 12, PennyLane and Qiskit "
     "(optimisation levels 1 and 3, with and without routing)"),
    (4, "ESM_4_qubit_scaling_paired.csv", "qs_paired_vs_rbf.csv",
     "Accuracy from 4 to 12 qubits: pooled paired differences to the tuned RBF kernel for each n"),
    (5, "ESM_5_device_paired.csv", "device_paired.csv",
     "Device-model replay (Online Resource 28, Sect. S6): paired contrasts of each encoding against the angle encoding"),
    (6, "ESM_6_per_dataset_equivalence.csv", "v2_per_dataset_equivalence.csv",
     "Per-dataset equivalence verdicts of the six quantum kernels against the tuned RBF kernel"),
    (7, "ESM_7_reduction_pooled.csv", "v2_reduction_pooled.csv",
     "H1, post-hoc pooled analysis: reduction settings pooled over datasets, all datasets and real datasets only"),
    (8, "ESM_8_geometry_summary.csv", "v2_geometry_summary.csv",
     "Geometric difference g(K_RBF||K_Q) and reverse (three ridge values), centred alignment to the tuned RBF and "
     "kernel-target alignment, median over dataset-resample cells"),
    (9, "ESM_9_finite_shots.csv", "shots_summary.csv",
     "Finite-shot sampling at the ideal hyper-parameters: accuracy loss and Gram-matrix error at 100, 1000 and "
     "10000 shots"),
    (10, "ESM_10_concentration.csv", "fb_concentration.csv",
     "Mean and SD of the off-diagonal fidelity Gram entries at the untuned bandwidth s = 1 against the Haar value, "
     "n = 4 to 12"),
    (11, "ESM_11_ceiling_sensitivity.csv", "v2_ceiling_sensitivity.csv",
     "Sensitivity analysis: pooled differences to the tuned RBF kernel without the datasets on which it reaches "
     ">= 0.99 accuracy"),
    (12, "ESM_12_surrogate_paired.csv", "sur_paired.csv",
     "Surrogate tuning on real labels (RBF map, metric-matched, angle transfer) and Nystrom landmarks: accuracy, balanced "
     "accuracy and macro-F1 differences to full-grid tuning and to the tuned RBF kernel, pooled, outside the pilot, per dataset"),
    (13, "ESM_13_lever_table.csv", "lever_table.csv",
     "Lever table of the loading bill: cost before and after, saving factor and paired accuracy change for every lever"),
    (14, "ESM_14_stationarity.csv", "stat_summary.csv",
     "Stationarity index of every kernel at the tuned bandwidth and at s = 1 (quartiles over the main-study cells)"),
    (15, "ESM_15_stationarity_correlation.csv", "stat_correlation.csv",
     "Spearman correlation over cells between the stationarity index at the tuned bandwidth and the accuracy difference "
     "to the tuned RBF kernel"),
    (16, "ESM_16_rank_uniform.csv", "rank_check.csv",
     "Numerical rank and effective dimension of fidelity Gram matrices on 1000 uniform points, n = 4 to 8, s = 0.1, 0.25, 1"),
    (17, "ESM_17_rank_datasets.csv", "rank_data.csv",
     "Rank and effective dimension of the fidelity Gram matrices on the main-study training blocks (first resample) at "
     "the tuned bandwidth and at s = 1"),
    (18, "ESM_18_positive_control.csv", "rel_paired.csv",
     "Positive control on engineered labels: accuracy differences of every model to the tuned RBF kernel, pooled, real "
     "datasets only and per dataset, n = 4, 6 and 8, IQP and ZZ labels"),
    (19, "ESM_19_positive_control_extended_grid.csv", "rel_edge_paired.csv",
     "Sensitivity check of the positive control: differences to the RBF kernel re-tuned on the grid extended to s = 64"),
    (20, "ESM_20_surrogate_engineered_labels.csv", "relsur_paired.csv",
     "Surrogate tuning on the engineered labels: differences to full-grid tuning and to the tuned RBF kernel"),
    (21, "ESM_21_detection_limit.csv", "dose_paired.csv",
     "Detection limit: advantage of the labelling kernel over the tuned RBF kernel after flipping 0 to 40% of the "
     "engineered labels, pooled and per dataset"),
    (22, "ESM_22_gate_counts.csv", "gate_counts.csv",
     "Two-qubit gates and depth of the state and fidelity circuits of every encoding, n = 2 to 12, all-to-all and routed, "
     "optimisation levels 1 and 3"),
    (23, "ESM_23_evaluation_cost.csv", "frontier.csv",
     "Accuracy and evaluation cost of one Gram matrix (M = 100 training, 50 test points, n = 4): circuits, shots, "
     "circuit and two-qubit gate executions"),
    (25, "ESM_25_software_environment.txt", "environment.txt",
     "Software environment of the runs: Python, NumPy, scikit-learn, Qiskit, PennyLane and other package versions"),
    (26, "ESM_26_labelling_bandwidth.csv", "bwl_paired.csv",
     "Engineered labels at labelling bandwidths 0.15, 0.3, 0.7 and 1 and from IQP circuits with 1 and 3 layers: grid-tuned "
     "kernel, oracle and surrogates against full-grid tuning and against the tuned RBF kernel, n = 4 and 6"),
    (27, "ESM_27_labelling_bandwidth_labels.csv", "bwl_labels.csv",
     "Geometric difference used to build each set of engineered labels of Online Resource 26"),
]


def per_dataset_accuracy(res):
    """Derived resource 24: mean test accuracy per dataset and model over the 20 resamples (main study)."""
    f = res / "mds_cells.csv"
    if not f.exists():
        return None
    c = pd.read_csv(f)
    return c.pivot_table(index="dataset", columns="kernel", values="test_acc").round(4).to_csv()


DERIVED = [(24, "ESM_24_accuracy_per_dataset.csv", per_dataset_accuracy,
            "Main study (n = 4): mean test accuracy of every model on every dataset over the 20 paired resamples")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--out", default="../paper/online_resources")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for num, name, src, content in RESOURCES:
        f = res / src
        if not f.exists():
            print(f"skip Online Resource {num}: {src} not found")
            continue
        head = (f"# Online Resource {num}\n# Article: {TITLE}\n# Journal: {JOURNAL}\n# Author: {AUTHOR}\n"
                f"# Content: {content}\n# Source file: {src}\n")
        (out / name).write_text(head + f.read_text())
        print("wrote", name)
    for num, name, fn, content in DERIVED:
        body = fn(res)
        if body is None:
            print(f"skip Online Resource {num}")
            continue
        head = (f"# Online Resource {num}\n# Article: {TITLE}\n# Journal: {JOURNAL}\n# Author: {AUTHOR}\n"
                f"# Content: {content}\n# Source file: derived from mds_cells.csv by make_online_resources.py\n")
        (out / name).write_text(head + body)
        print("wrote", name)


if __name__ == "__main__":
    main()
