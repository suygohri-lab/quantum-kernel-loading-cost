"""Qubit-scaling study: does the equivalence to a tuned RBF kernel survive more qubits (4, 6, 8, 10, 12)?

For each n the data are reduced by PCA to exactly n components (datasets with fewer than n features are skipped for
that n), so every quantum kernel uses n qubits AND the tuned RBF baseline sees the same n-dimensional data - the
comparison is always like for like.  Same protocol as multi_dataset_study.py (wide bandwidth grid, equal tuning
budget, corrected resampled t, equivalence at +/-0.02).  Statevector simulation only (exact kernel = infinite shots);
12 qubits is the largest size for which the full wide-grid tuning is practical.

Also reported per n: median off-diagonal Gram spread (kernel concentration indicator), grid-edge fractions.

Outputs (--out): qs_cells.csv, qs_paired_vs_rbf.csv, qs_concentration.csv, qs_report.txt
Usage:
  python qubit_scaling_study.py --out results --n-jobs 4
  python qubit_scaling_study.py --out results --quick
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from datasets_cv import DatasetSkip, load_full, resamples
from extended_study import corrected
from multi_dataset_study import evaluate_split

NS = [4, 6, 8, 10, 12]
DATASETS = ["breast_cancer", "wine", "digits38", "digits17", "digits49", "ionosphere", "sonar", "spambase"]


def job(n, name, ri, Xtr, Xte, ytr, yte, extra=False):
    rows = evaluate_split(name, ri, Xtr, Xte, ytr, yte, extra, n_max=n)
    for r in rows:
        r["n_max"] = n
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--ns", nargs="+", type=int, default=NS)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--extra-baselines", action="store_true", help="kNN, gradient boosting, random forest, MLP")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.quick:
        a.datasets, a.ns, a.folds, a.repeats = a.datasets[:2], [4, 8], 2, 1
    jobs, used = [], []
    for name in a.datasets:
        try:
            X, y, inf = load_full(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            continue
        sp = resamples(y, a.folds, a.repeats)
        for n in a.ns:
            if X.shape[1] < n:
                print(f"{name}: {X.shape[1]} features < {n} qubits, skipped for n={n}")
                continue
            used.append((name, n))
            jobs += [(n, name, ri, X[tr], X[te], y[tr], y[te]) for ri, (tr, te) in enumerate(sp)]
    print(f"{len(jobs)} jobs")
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(job)(*j, a.extra_baselines) for j in jobs)
    cells = pd.DataFrame([r for rs in res for r in rs])
    cells.to_csv(out / "qs_cells.csv", index=False)
    ratio = 1.0 / (a.folds - 1)
    rows = []
    for n, g in cells.groupby("n_max"):
        for metric in ("test_acc", "test_bacc"):
            piv = g.pivot_table(index=["dataset", "resample"], columns="kernel", values=metric)
            for k in [c for c in piv.columns if c != "classical/rbf"]:
                d = (piv[k] - piv["classical/rbf"]).unstack("dataset")
                rows.append(dict(n_qubits=n, metric=metric, kernel=k, n_datasets=d.shape[1],
                                 **corrected(d.mean(axis=1).values, 1.0, ratio)))
    st = pd.DataFrame(rows)
    st.to_csv(out / "qs_paired_vs_rbf.csv", index=False)
    conc = cells[cells.kernel.str.contains("fidelity|projected")].groupby(["n_max", "kernel"]).agg(
        offdiag_sd=("offdiag_sd", "median"), offdiag_mean=("offdiag_mean", "median"),
        at_lower_edge=("at_lower_edge", "mean"), at_upper_edge=("at_upper_edge", "mean"),
        acc=("test_acc", "mean")).reset_index()
    conc.to_csv(out / "qs_concentration.csv", index=False)
    from extended_study import S_WIDE
    from gram_spread import tie_rule
    lines = [f"QUBIT SCALING: datasets x n = {sorted(set(used))}",
             f"TUNING: tie rule = {tie_rule()}, bandwidth grid s = {S_WIDE}", f"compute time {time.time() - t0:.0f}s",
             f"design: {a.folds}-fold x {a.repeats}, cap {a.cap}; paired difference to tuned RBF on the same n-dimensional data", "",
             "POOLED PAIRED DIFFERENCE (accuracy)",
             st[st.metric == "test_acc"][["n_qubits", "kernel", "n_datasets", "mean", "lo", "hi", "equivalent_95ci"]].round(4).to_string(index=False),
             "", "KERNEL CONCENTRATION AND GRID EDGES", conc.round(4).to_string(index=False)]
    (out / "qs_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
