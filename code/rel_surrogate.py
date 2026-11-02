"""Surrogate tuning where it matters: on the engineered labels of the positive control, where IQP and ZZ beat the RBF
kernel.  Does a surrogate-chosen bandwidth keep the advantage at 1/12 of the tuning circuits?

DESIGN (added after the main analysis, 2026-11-01; the surrogates are those of surrogate_study.py, unchanged)
Same engineered labels, datasets, n, resamples and preprocessing as relabel_study.py.  For each IQP and ZZ kernel
(fidelity and projected), the bandwidth is set by
  metric_map : s = s*_RBF / kappa_k   (kappa from surrogate_study.kappa2)
  rbf_map    : s = 2 s*_RBF
  angle      : s = s* of the angle kernel of the same construction
where s*_RBF and the angle s* are the full-grid choices recorded in rel_cells.csv for the same cell.  One quantum Gram
matrix is built at that s and C is tuned by the same inner CV.  Endpoints (pooled over datasets, corrected resampled t):
  (i)  surrogate - full-grid tuning of the same kernel (non-inferior if lower bound > -0.02)
  (ii) surrogate - tuned RBF (advantage kept if lower bound > 0)
Primary interest: the labelling kernel (IQP-F on IQP labels, ZZ-F on ZZ labels).

Outputs (--out): relsur_cells.csv, relsur_paired.csv, relsur_report.txt
Usage:  python rel_surrogate.py --out results_v2 --n-jobs 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVC

from datasets_cv import BASE_SEED, DatasetSkip, preprocess, resamples
from extended_study import DELTA, corrected
from gram_spread import tune
from relabel_study import REAL, engineered_labels, global_features, raw_dataset
from surrogate_study import RBF_FACTOR, gram, kappa2

KERNELS = ["iqp/fidelity", "iqp/projected", "zz/fidelity", "zz/projected"]


def one(labeller, n, name, ri, Xtr_raw, Xte_raw, ytr, yte, sref):
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw, n_max=n)
    ntr = len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + ri
    rows = []
    for k in KERNELS:
        kind, cons = k.split("/")
        surr = {"metric_map": sref["classical/rbf"] / np.sqrt(kappa2(kind, n)),
                "rbf_map": RBF_FACTOR * sref["classical/rbf"],
                "angle": sref[f"angle/{cons}"]}
        for meth, s in surr.items():
            K = gram(X, s, kind, cons, ntr, n)
            cv, _, C = tune({s: K}, ytr, ntr, seed)
            acc = float((SVC(kernel="precomputed", C=C).fit(K[:ntr, :ntr], ytr).predict(K[ntr:, :ntr]) == yte).mean())
            rows.append(dict(labeller=labeller, n_qubits=n, dataset=name, resample=ri, kernel=k, method=meth,
                             s_used=s, C_star=C, test_acc=acc))
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
    ref = rel.set_index(["labeller", "n_qubits", "dataset", "resample", "kernel"]).s_star
    jobs = []
    for name in rel.dataset.unique():
        try:
            Xraw = raw_dataset(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            continue
        for n in sorted(rel.n_qubits.unique()):
            n = int(n)
            Z = global_features(Xraw, n)
            for L in sorted(rel.labeller.unique()):
                y, _ = engineered_labels(Z, L, n)
                for ri, (tr, te) in enumerate(resamples(y, a.folds, a.repeats)):
                    sref = {k: ref[(L, n, name, ri, k)] for k in ("classical/rbf", "angle/fidelity", "angle/projected")}
                    jobs.append((L, n, name, ri, Z[tr], Z[te], y[tr], y[te], sref))
    print(f"{len(jobs)} jobs")
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(one)(*j) for j in jobs)
    sur = pd.DataFrame([r for rs in res for r in rs])
    sur.to_csv(out / "relsur_cells.csv", index=False)

    base = rel[rel.kernel.isin(KERNELS + ["classical/rbf"])].assign(method="full_grid")
    allc = pd.concat([base[["labeller", "n_qubits", "dataset", "resample", "kernel", "method", "test_acc"]],
                      sur[["labeller", "n_qubits", "dataset", "resample", "kernel", "method", "test_acc"]]])
    ratio = 1.0 / (a.folds - 1)
    st = []
    for (L, n), g in allc.groupby(["labeller", "n_qubits"]):
        piv = g.pivot_table(index=["dataset", "resample"], columns=["kernel", "method"], values="test_acc")
        rbf = piv[("classical/rbf", "full_grid")]
        for k in KERNELS:
            for meth in ("full_grid", "metric_map", "rbf_map", "angle"):
                for refname, refv in (("full_grid", piv[(k, "full_grid")]), ("tuned_rbf", rbf)):
                    if meth == "full_grid" and refname == "full_grid":
                        continue
                    diff = (piv[(k, meth)] - refv).unstack("dataset")
                    for scope, cols in (("POOLED", list(diff.columns)),
                                        ("POOLED_REAL", [c for c in diff.columns if c in REAL])):
                        r = corrected(diff[cols].mean(axis=1).values, 1.0, ratio)
                        st.append(dict(labeller=L, n_qubits=n, kernel=k, method=meth, reference=refname, dataset=scope,
                                       **r, noninferior=bool(r["lo"] > -DELTA), advantage=bool(r["lo"] > 0)))
    st = pd.DataFrame(st)
    st.to_csv(out / "relsur_paired.csv", index=False)
    p = st[st.dataset == "POOLED"]
    lab = p[p.apply(lambda r: r.kernel == f"{r.labeller}/fidelity", axis=1)]
    sm = sur.groupby(["labeller", "n_qubits", "kernel", "method"]).s_used.median().unstack("method")
    sf = rel[rel.kernel.isin(KERNELS)].groupby(["labeller", "n_qubits", "kernel"]).s_star.median().rename("full_grid")
    lines = ["LABELLING KERNEL (IQP-F on IQP labels, ZZ-F on ZZ labels), pooled over datasets",
             "reference full_grid: non-inferior if lo > -0.02;  reference tuned_rbf: advantage kept if lo > 0",
             lab[["labeller", "n_qubits", "method", "reference", "mean", "lo", "hi", "noninferior", "advantage"]]
             .round(4).to_string(index=False), "",
             "ALL IQP / ZZ KERNELS AGAINST FULL-GRID TUNING",
             p[p.reference == "full_grid"][["labeller", "n_qubits", "kernel", "method", "mean", "lo", "hi", "noninferior"]]
             .round(4).to_string(index=False), "",
             "MEDIAN BANDWIDTH (full grid and surrogates)", pd.concat([sf, sm], axis=1).round(3).to_string()]
    (out / "relsur_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
