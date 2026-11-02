"""Detection limit of the protocol: how small an advantage can it detect?

DESIGN (added after the main analysis, 2026-11-01)
Start from the engineered labels of the positive control (relabel_study.py; same datasets, preprocessing and resamples)
and flip a fraction rho of them to the opposite class, rho in {0, 0.1, 0.2, 0.3, 0.4}.  The flipped points are the first
round(rho N) points of one fixed random permutation per dataset, so the noisy sets are nested.  Flipping shrinks the
advantage of the labelling kernel over the tuned RBF kernel continuously towards zero.  At each rho both kernels are tuned
on the shared grid with the same inner CV, and the paired accuracy difference is tested as in the main study.
Reported, for each labelling kernel (IQP-F, ZZ-F) and n in {4, 6}: pooled difference with 95% interval and whether it is
detected (lower bound > 0), and the share of single datasets on which it is detected, against the size of the
advantage.  The detection limit is the smallest pooled (or per-dataset) advantage that was still detected.

Outputs (--out): dose_cells.csv, dose_paired.csv, dose_report.txt
Usage:  python dose_response.py --out results_v2 --n-jobs 10
        python dose_response.py --out test_out --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVC

from datasets_cv import BASE_SEED, DatasetSkip, preprocess, resamples
from extended_study import S_WIDE, corrected
from gram_spread import tie_rule, tune
from qkcommon import fidelity_gram, states
from relabel_study import REAL, SYNTH, engineered_labels, global_features, raw_dataset

RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]


def one(L, n, rho, name, ri, Xtr_raw, Xte_raw, ytr, yte):
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw, n_max=n)
    ntr = len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + ri
    rows = []

    def rec(kernel, Kd):
        cv, s, C = tune(Kd, ytr, ntr, seed)
        K = Kd[s]
        acc = float((SVC(kernel="precomputed", C=C).fit(K[:ntr, :ntr], ytr).predict(K[ntr:, :ntr]) == yte).mean())
        rows.append(dict(labeller=L, n_qubits=n, rho=rho, dataset=name, resample=ri, kernel=kernel, s_star=s,
                         test_acc=acc))

    Kq = {}
    for s in S_WIDE:
        psi = states(X, s, L, n)
        Kq[s] = fidelity_gram(psi, psi)
    rec(f"{L}/fidelity", Kq)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    rec("classical/rbf", {s: np.exp(-(s ** 2) * d2) for s in S_WIDE})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--datasets", nargs="+", default=REAL + SYNTH)
    ap.add_argument("--n", nargs="+", type=int, default=[4, 6])
    ap.add_argument("--labellers", nargs="+", default=["iqp", "zz"])
    ap.add_argument("--rhos", nargs="+", type=float, default=RHOS)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.quick:
        a.datasets, a.n, a.labellers, a.rhos, a.folds, a.repeats = ["breast_cancer", "synth_r16"], [4], ["iqp"], \
            [0.0, 0.2, 0.4], 3, 1
    jobs = []
    for di, name in enumerate(a.datasets):
        try:
            Xraw = raw_dataset(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            continue
        for n in a.n:
            Z = global_features(Xraw, n)
            for L in a.labellers:
                y0, _ = engineered_labels(Z, L, n)
                perm = np.random.default_rng(BASE_SEED + 1000 + di).permutation(len(y0))
                for rho in a.rhos:
                    y = y0.copy()
                    flip = perm[:int(round(rho * len(y)))]
                    y[flip] = 1 - y[flip]
                    for ri, (tr, te) in enumerate(resamples(y, a.folds, a.repeats)):
                        jobs.append((L, n, rho, name, ri, Z[tr], Z[te], y[tr], y[te]))
    print(f"{len(jobs)} jobs, tie rule {tie_rule()}")
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(one)(*j) for j in jobs)
    cells = pd.DataFrame([r for rs in res for r in rs])
    cells.to_csv(out / "dose_cells.csv", index=False)

    ratio = 1.0 / (a.folds - 1)
    st = []
    for (L, n, rho), g in cells.groupby(["labeller", "n_qubits", "rho"]):
        piv = g.pivot_table(index=["dataset", "resample"], columns="kernel", values="test_acc")
        diff = (piv[f"{L}/fidelity"] - piv["classical/rbf"]).unstack("dataset")
        for scope, cols in (("POOLED", list(diff.columns)), ("POOLED_REAL", [c for c in diff.columns if c in REAL])):
            if cols:
                r = corrected(diff[cols].mean(axis=1).values, 1.0, ratio)
                st.append(dict(labeller=L, n_qubits=n, rho=rho, dataset=scope, **r, detected=bool(r["lo"] > 0)))
        for ds in diff.columns:
            r = corrected(diff[ds].dropna().values, 1.0, ratio)
            st.append(dict(labeller=L, n_qubits=n, rho=rho, dataset=ds, **r, detected=bool(r["lo"] > 0)))
    st = pd.DataFrame(st)
    st.to_csv(out / "dose_paired.csv", index=False)
    pooled = st[st.dataset == "POOLED"]
    per = st[~st.dataset.str.startswith("POOLED")]
    share = per.groupby(["labeller", "n_qubits", "rho"]).agg(mean_advantage=("mean", "mean"),
                                                            datasets_detected=("detected", "sum"),
                                                            datasets=("detected", "size"))
    bins = pd.cut(per["mean"], [-1, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 1])
    curve = per.groupby(bins, observed=True).agg(cases=("detected", "size"), detected=("detected", "mean"))
    det = pooled[pooled.detected]
    lim_pool = det.groupby(["labeller", "n_qubits"])["mean"].min()
    lim_ds = per[per.detected]["mean"].min()
    lines = [f"Label flips rho = {a.rhos}; labelling kernel tuned on the shared grid vs tuned RBF", "",
             "POOLED DIFFERENCE (detected if lo > 0)",
             pooled[["labeller", "n_qubits", "rho", "mean", "lo", "hi", "detected"]].round(4).to_string(index=False), "",
             "SINGLE DATASETS: mean advantage and number detected", share.round(4).to_string(), "",
             "DETECTION RATE OF SINGLE DATASETS BY SIZE OF THE ADVANTAGE", curve.round(3).to_string(), "",
             "SMALLEST POOLED ADVANTAGE DETECTED", lim_pool.round(4).to_string(), "",
             f"SMALLEST SINGLE-DATASET ADVANTAGE DETECTED: {lim_ds:.4f}" if np.isfinite(lim_ds) else "none detected"]
    (out / "dose_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
