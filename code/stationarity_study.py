"""How far is each kernel from the stationary class (Genton 2001) at its tuned bandwidth and at s = 1?

A kernel is stationary if K(x, z) = K(x + t, z + t) for every shift t.  For each main-study cell (dataset, resample,
kernel) we take the training features (same preprocessing, n = 4), draw P pairs (x, z) of training points and one shift
t ~ U(-T, T)^n per pair (fixed seed), and report
    index = mean |K(x, z) - K(x + t, z + t)| / mean |K(x, z) - mean K(x, z)|
(0 = stationary; about 1 = a shift changes the kernel as much as the kernel varies across pairs).  Angle kernels and the
RBF kernel are stationary by construction (index 0, kept as a check).  Amplitude: (x.z)^2 / (|x|^2 |z|^2).
The bandwidth of each quantum kernel is its tuned s* from mds_cells.csv (main study); every kernel is also evaluated at
s = 1.  Exploratory: Spearman correlation, over cells, between the index at s* and the paired accuracy difference to the
tuned RBF kernel.

Outputs (--out): stat_cells.csv, stat_summary.csv, stat_correlation.csv, stat_report.txt
Usage:  python stationarity_study.py --results results_v2 --out results_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from datasets_cv import BASE_SEED, DatasetSkip, load_full, preprocess, resamples
from qkcommon import bloch, projected_gamma, states

P_PAIRS = 300
T_SHIFT = 0.5


def kernel_pairs(A, B, kind, cons, s, n, gamma=None):
    a, b = states(A, s, kind, n), states(B, s, kind, n)
    if cons == "fidelity":
        return np.abs((a * b.conj()).sum(1)) ** 2
    fa, fb = bloch(a, n).reshape(len(A), -1), bloch(b, n).reshape(len(B), -1)
    return np.exp(-gamma * ((fa - fb) ** 2).sum(1))


def index(k0, k1):
    spread = np.mean(np.abs(k0 - k0.mean()))
    return float(np.mean(np.abs(k0 - k1)) / spread) if spread > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-resamples", type=int, default=None, help="limit resamples per dataset (testing)")
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--cache", default="data_cache")
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res
    out.mkdir(parents=True, exist_ok=True)
    mds = pd.read_csv(res / "mds_cells.csv")
    sstar = mds.set_index(["dataset", "resample", "kernel"]).s_star
    rows = []
    for name in sorted(mds.dataset.unique()):
        try:
            X, y, _ = load_full(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            continue
        for ri, (tr, te) in enumerate(resamples(y, a.folds, a.repeats)):
            if a.max_resamples is not None and ri >= a.max_resamples:
                break
            Xtr, _ = preprocess(X[tr], X[te])
            n = Xtr.shape[1]
            rng = np.random.default_rng(BASE_SEED + ri + 7)
            i, j = rng.integers(0, len(Xtr), (2, P_PAIRS))
            A, B = Xtr[i], Xtr[j]
            T = rng.uniform(-T_SHIFT, T_SHIFT, A.shape)
            for kind in ("angle", "iqp", "zz"):
                for cons in ("fidelity", "projected"):
                    k = f"{kind}/{cons}"
                    for label, s in (("tuned", sstar.get((name, ri, k), np.nan)), ("s=1", 1.0)):
                        if not np.isfinite(s):
                            continue
                        g = None
                        if cons == "projected":
                            g = projected_gamma(bloch(states(Xtr, s, kind, n), n).reshape(len(Xtr), -1))
                        k0 = kernel_pairs(A, B, kind, cons, s, n, g)
                        k1 = kernel_pairs(A + T, B + T, kind, cons, s, n, g)
                        rows.append(dict(dataset=name, resample=ri, kernel=k, bandwidth=label, s=s, index=index(k0, k1)))
            amp = lambda U, V: (U * V).sum(1) ** 2 / np.maximum((U * U).sum(1) * (V * V).sum(1), 1e-24)
            rows.append(dict(dataset=name, resample=ri, kernel="amplitude/fidelity", bandwidth="none", s=np.nan,
                             index=index(amp(A, B), amp(A + T, B + T))))
        print(f"{name}: done")
    cells = pd.DataFrame(rows)
    cells.to_csv(out / "stat_cells.csv", index=False)
    summ = cells.groupby(["kernel", "bandwidth"])["index"].describe(percentiles=[.25, .5, .75])
    summ["median_s"] = cells.groupby(["kernel", "bandwidth"]).s.median()
    summ.to_csv(out / "stat_summary.csv")

    acc = mds.pivot_table(index=["dataset", "resample"], columns="kernel", values="test_acc")
    corr = []
    for k in sorted(cells.kernel.unique()):
        sub = cells[(cells.kernel == k) & (cells.bandwidth.isin(["tuned", "none"]))].set_index(["dataset", "resample"])
        if k not in acc.columns:
            continue
        d = (acc[k] - acc["classical/rbf"]).reindex(sub.index)
        ok = sub["index"].notna() & d.notna()
        if ok.sum() > 5 and sub["index"][ok].std() > 1e-9:
            r, p = stats.spearmanr(sub["index"][ok], d[ok])
            corr.append(dict(kernel=k, cells=int(ok.sum()), spearman_r=r, p=p))
    corr = pd.DataFrame(corr)
    corr.to_csv(out / "stat_correlation.csv", index=False)
    lines = [f"Stationarity index (0 = stationary); {P_PAIRS} pairs per cell, shift U(-{T_SHIFT}, {T_SHIFT}) per feature",
             "", summ[["count", "25%", "50%", "75%", "median_s"]].round(3).to_string(), "",
             "EXPLORATORY: Spearman correlation over cells, index at tuned s vs accuracy difference to tuned RBF",
             corr.round(4).to_string(index=False) if len(corr) else "(none)"]
    (out / "stat_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
