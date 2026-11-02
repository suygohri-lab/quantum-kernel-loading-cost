"""Fixed-bandwidth reference study: how much of a quantum-kernel result depends on tuning the bandwidth?

Part 1 (accuracy): same 16 datasets, same 20 resamples and the same preprocessing as multi_dataset_study.py (n = 4
qubits).  Every quantum kernel is evaluated at the untuned reference bandwidth s = 1 (projected kernels with the
data-driven width gamma = 1/(3 n Var(F)), unchanged), with only C tuned by the same 3-fold CV.  The tuned results are
read from mds_cells.csv, so the gain (tuned - fixed) is a paired difference on identical splits.  The RBF with the
library default width (rbf_scale in mds_cells.csv) is the classical counterpart of a fixed reference.

Part 2 (concentration): mean and SD of the off-diagonal training Gram entries at s = 1 for n = 4, 6, 8, 10, 12
(PCA to n components; the 8 qubit-scaling datasets; first --conc-resamples resamples), against the Haar value
1/(2^n + 1) for fidelity kernels.

Outputs (--out): fb_cells.csv, fb_gain.csv, fb_concentration.csv, fb_report.txt
Usage:
  python fixed_bandwidth_study.py --out results --n-jobs 6
  python fixed_bandwidth_study.py --out results --quick
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from datasets_cv import ALL_DATASETS, DatasetSkip, load_full, preprocess, resamples
from datasets_cv import BASE_SEED
from extended_study import corrected
from gram_spread import offdiag, tune
from multi_dataset_study import scores
from qkcommon import KINDS, bloch, fidelity_gram, projected_gamma, projected_gram, states
from qubit_scaling_study import DATASETS as QS_DATASETS, NS


def fixed_split(name, ri, Xtr_raw, Xte_raw, ytr, yte, n_max=4):
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw, n_max=n_max)
    n = Xtr.shape[1]
    ntr = len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + ri
    rows = []
    for kind in KINDS:
        psi = states(X, 1.0, kind, n)
        Kf = fidelity_gram(psi, psi)
        F = bloch(psi, n).reshape(len(X), -1)
        Kp = projected_gram(F, F, projected_gamma(F[:ntr]))
        for cons, K in (("fidelity", Kf), ("projected", Kp)):
            cv, s, C = tune({1.0: K}, ytr, ntr, seed)       # only C is searched
            acc, bacc, f1 = scores(K, ytr, yte, ntr, C)
            o = offdiag(K[:ntr, :ntr])
            rows.append(dict(dataset=name, resample=ri, kernel=f"{kind}/{cons}", C_star=C, cv_acc=cv, test_acc=acc,
                             test_bacc=bacc, test_f1=f1, offdiag_mean=float(o.mean()), offdiag_sd=float(o.std()),
                             n_qubits=n))
    return rows


def conc_split(name, ri, Xtr_raw, Xte_raw, ytr, yte, n_max):
    Xtr, _ = preprocess(Xtr_raw, Xte_raw, n_max=n_max)
    n = Xtr.shape[1]
    if n < n_max:
        return []
    rows = []
    for kind in KINDS:
        psi = states(Xtr, 1.0, kind, n)
        o = offdiag(fidelity_gram(psi, psi))
        rows.append(dict(dataset=name, resample=ri, n=n, kind=kind, offdiag_mean=float(o.mean()),
                         offdiag_sd=float(o.std()), haar=1.0 / (2 ** n + 1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--conc-resamples", type=int, default=3)
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = ALL_DATASETS
    ns = NS
    if a.quick:
        names, ns, a.folds, a.repeats, a.conc_resamples = names[:3], [4, 6], 2, 1, 1

    jobs, cjobs = [], []
    for name in names:
        try:
            X, y, inf = load_full(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            continue
        for ri, (tr, te) in enumerate(resamples(y, a.folds, a.repeats)):
            jobs.append((name, ri, X[tr], X[te], y[tr], y[te]))
            if name in QS_DATASETS and ri < a.conc_resamples:
                for n in ns:
                    cjobs.append((name, ri, X[tr], X[te], y[tr], y[te], n))
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(fixed_split)(*j) for j in jobs)
    cells = pd.DataFrame([r for rs in res for r in rs])
    cells.to_csv(out / "fb_cells.csv", index=False)
    cres = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(conc_split)(*j) for j in cjobs)
    conc = pd.DataFrame([r for rs in cres for r in rs])
    conc.to_csv(out / "fb_concentration_cells.csv", index=False)
    print(f"compute time {time.time() - t0:.0f}s")

    # ---- paired gain tuned - fixed (same datasets and resamples) using mds_cells.csv
    mds = pd.read_csv(out / "mds_cells.csv")
    mds = mds[mds.dataset.isin(cells.dataset.unique())]
    ratio = 1.0 / (a.folds - 1)
    gains = []
    for metric in ("test_acc", "test_bacc", "test_f1"):
        tuned = mds.pivot_table(index=["dataset", "resample"], columns="kernel", values=metric)
        fixed = cells.pivot_table(index=["dataset", "resample"], columns="kernel", values=metric)
        for k in fixed.columns:
            diff = (tuned[k] - fixed[k]).unstack("dataset")
            gains.append(dict(metric=metric, kernel=k, dataset="POOLED", fixed_mean=fixed[k].mean(),
                              tuned_mean=tuned[k].loc[fixed.index].mean(), **corrected(diff.mean(axis=1).values, 1.0, ratio)))
            for ds in diff.columns:
                gains.append(dict(metric=metric, kernel=k, dataset=ds, fixed_mean=fixed[k].unstack("dataset")[ds].mean(),
                                  tuned_mean=tuned[k].unstack("dataset")[ds].mean(), **corrected(diff[ds].values, 1.0, ratio)))
        # fixed-bandwidth kernel minus tuned RBF (what an untuned benchmark would have reported)
        for k in fixed.columns:
            diff = (fixed[k] - tuned["classical/rbf"].loc[fixed.index]).unstack("dataset")
            gains.append(dict(metric=metric, kernel=k + " [fixed s=1] - tuned RBF", dataset="POOLED", fixed_mean=np.nan,
                              tuned_mean=np.nan, **corrected(diff.mean(axis=1).values, 1.0, ratio)))
    gdf = pd.DataFrame(gains)
    gdf.to_csv(out / "fb_gain.csv", index=False)

    cc = conc.groupby(["n", "kind"]).agg(offdiag_mean=("offdiag_mean", "mean"), offdiag_sd=("offdiag_sd", "mean"),
                                         haar=("haar", "first"), cells=("dataset", "size")).reset_index()
    cc.to_csv(out / "fb_concentration.csv", index=False)

    acc = gdf[(gdf.metric == "test_acc") & (gdf.dataset == "POOLED") & ~gdf.kernel.str.contains("fixed")]
    vs = gdf[(gdf.metric == "test_acc") & (gdf.dataset == "POOLED") & gdf.kernel.str.contains("fixed")]
    lines = [f"FIXED-BANDWIDTH REFERENCE (s = 1, only C tuned): {len(names)} datasets, {len(jobs)} splits, time {time.time() - t0:.0f}s", "",
             "MEAN TEST ACCURACY, fixed vs tuned (mean over all splits) and POOLED GAIN (tuned - fixed), accuracy",
             acc[["kernel", "fixed_mean", "tuned_mean", "mean", "lo", "hi", "p"]].round(4).to_string(index=False), "",
             "FIXED-BANDWIDTH KERNEL MINUS TUNED RBF (pooled, accuracy)",
             vs[["kernel", "mean", "lo", "hi", "p", "equivalent_95ci"]].round(4).to_string(index=False), "",
             "OFF-DIAGONAL OF THE FIDELITY GRAM AT s = 1 (training block)", cc.round(4).to_string(index=False)]
    (out / "fb_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
