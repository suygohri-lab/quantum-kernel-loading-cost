"""Sensitivity check for the positive control (post hoc, added after relabel_study.py showed the tuned RBF kernel at the
upper edge s = 8 of the grid in up to 32% of cells): re-tune the RBF kernel, and the classical closed-form angle
kernels, on the grid extended by s = 16, 32, 64, on exactly the same engineered labels and resamples, and compare the
labelling kernel (tuned on the standard grid, read from rel_cells.csv) with this extended-grid RBF kernel.

Outputs (--out): rel_edge_cells.csv, rel_edge_paired.csv, rel_edge_report.txt
Usage:  python rel_edge_check.py --out results_v2 --n-jobs 10
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
from gram_spread import tune
from qkcommon import bloch, fidelity_gram, projected_gamma, projected_gram, states
from relabel_study import REAL, engineered_labels, global_features, raw_dataset

S_EXT = sorted(set(S_WIDE) | {16.0, 32.0, 64.0})


def one(labeller, n, name, ri, Xtr_raw, Xte_raw, ytr, yte):
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw, n_max=n)
    ntr = len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + ri
    rows = []

    def rec(kernel, Kd):
        cv, s, C = tune(Kd, ytr, ntr, seed)
        K = Kd[s]
        acc = float((SVC(kernel="precomputed", C=C).fit(K[:ntr, :ntr], ytr).predict(K[ntr:, :ntr]) == yte).mean())
        rows.append(dict(labeller=labeller, n_qubits=n, dataset=name, resample=ri, kernel=kernel, s_star=s, C_star=C,
                         test_acc=acc, at_upper_edge=bool(s == S_EXT[-1])))

    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    rec("classical/rbf_ext", {s: np.exp(-(s ** 2) * d2) for s in S_EXT})
    Kf, Kp = {}, {}
    for s in S_EXT:
        psi = states(X, s, "angle", n)
        Kf[s] = fidelity_gram(psi, psi)
        F = bloch(psi, n).reshape(len(X), -1)
        Kp[s] = projected_gram(F, F, projected_gamma(F[:ntr]))
    rec("angle/fidelity_ext", Kf)
    rec("angle/projected_ext", Kp)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--csv-dir", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    rel = pd.read_csv(out / "rel_cells.csv")
    jobs = []
    for name in rel.dataset.unique():
        try:
            Xraw = raw_dataset(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            continue
        for n in sorted(rel.n_qubits.unique()):
            Z = global_features(Xraw, int(n))
            for labeller in sorted(rel.labeller.unique()):
                y, _ = engineered_labels(Z, labeller, int(n))
                for ri, (tr, te) in enumerate(resamples(y, a.folds, a.repeats)):
                    jobs.append((labeller, int(n), name, ri, Z[tr], Z[te], y[tr], y[te]))
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(one)(*j) for j in jobs)
    ext = pd.DataFrame([r for rs in res for r in rs])
    ext.to_csv(out / "rel_edge_cells.csv", index=False)
    allc = pd.concat([rel[["labeller", "n_qubits", "dataset", "resample", "kernel", "s_star", "test_acc"]],
                      ext[["labeller", "n_qubits", "dataset", "resample", "kernel", "s_star", "test_acc"]]])
    ratio = 1.0 / (a.folds - 1)
    st = []
    for (lab, n), g in allc.groupby(["labeller", "n_qubits"]):
        piv = g.pivot_table(index=["dataset", "resample"], columns="kernel", values="test_acc")
        for k in (f"{lab}/fidelity", "angle/fidelity_ext", "angle/projected_ext", "classical/rbf"):
            diff = (piv[k] - piv["classical/rbf_ext"]).unstack("dataset")
            for scope, cols in (("POOLED", list(diff.columns)), ("POOLED_REAL", [c for c in diff.columns if c in REAL])):
                r = corrected(diff[cols].mean(axis=1).values, 1.0, ratio)
                st.append(dict(labeller=lab, n_qubits=n, kernel=k, reference="rbf_ext", dataset=scope, **r,
                               advantage=bool(r["lo"] > 0)))
    st = pd.DataFrame(st)
    st.to_csv(out / "rel_edge_paired.csv", index=False)
    edge = ext.groupby(["labeller", "n_qubits", "kernel"]).at_upper_edge.mean().unstack()
    sm = ext.groupby(["labeller", "n_qubits", "kernel"]).s_star.median().unstack()
    lines = [f"Extended grid: {S_EXT}", "", "SHARE OF CELLS AT THE NEW UPPER EDGE s = 64", edge.round(3).to_string(), "",
             "MEDIAN TUNED s ON THE EXTENDED GRID", sm.round(2).to_string(), "",
             "PAIRED ACCURACY DIFFERENCE TO THE EXTENDED-GRID RBF KERNEL (advantage if lo > 0)",
             st[["labeller", "n_qubits", "dataset", "kernel", "mean", "lo", "hi", "advantage"]].round(4).to_string(index=False)]
    (out / "rel_edge_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
