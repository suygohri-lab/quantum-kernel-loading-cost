"""Numerical rank of the fidelity Gram matrices (why Nystrom with a few landmarks can be exact at small n).

For each encoding (angle, IQP, ZZ), n = 4 ... 8 and bandwidth s in {0.1, 0.25, 1}, draw P points uniformly in
[0, pi]^n (fixed seed) and report the number of eigenvalues of the fidelity Gram matrix above tol * largest eigenvalue,
and the effective dimension sum(lambda)^2 / sum(lambda^2).  Upper bounds: 3^n for angle (each qubit contributes the
features 1, cos, sin), 2^n (2^n + 1) / 2 for any real state, 4^n in general.  Also for the main-study features: the
rank on each dataset's training block at the tuned s and at s = 1, read from mds_cells.csv if --results is given (first resample).

Outputs: rank_check.csv, rank_data.csv (optional), rank_report.txt
Usage:  python rank_check.py --out results_v2 --results results_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from datasets_cv import BASE_SEED, DatasetSkip, load_full, preprocess, resamples
from qkcommon import fidelity_gram, states

TOL = 1e-10


def rank_stats(K):
    w = np.clip(np.linalg.eigvalsh((K + K.T) / 2), 0, None)
    return int((w > TOL * w.max()).sum()), float(w.sum() ** 2 / (w ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--results", default=None)
    ap.add_argument("--points", type=int, default=1000)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--csv-dir", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BASE_SEED)
    rows = []
    for n in range(4, 9):
        X = rng.uniform(0, np.pi, (a.points, n))
        for kind in ("angle", "iqp", "zz"):
            for s in (0.1, 0.25, 1.0):
                psi = states(X, s, kind, n)
                r, eff = rank_stats(fidelity_gram(psi, psi))
                rows.append(dict(n=n, kernel=f"{kind}/fidelity", s=s, points=a.points, rank=r, eff_dim=eff,
                                 bound_angle=3 ** n if kind == "angle" else np.nan, bound_real=2 ** n * (2 ** n + 1) // 2,
                                 bound_general=4 ** n))
    t = pd.DataFrame(rows)
    t.to_csv(out / "rank_check.csv", index=False)
    lines = [f"Numerical rank (eigenvalues > {TOL:g} x max) and effective dimension of fidelity Gram matrices, "
             f"{a.points} uniform points", t.pivot_table(index=["kernel", "s"], columns="n", values="rank").to_string(),
             "", "Effective dimension", t.pivot_table(index=["kernel", "s"], columns="n", values="eff_dim").round(1).to_string()]
    if a.results:
        mds = pd.read_csv(Path(a.results) / "mds_cells.csv")
        mds = mds[(mds["resample"] == 0) & mds.kernel.str.endswith("/fidelity") & ~mds.kernel.str.startswith("amplitude")]
        drows = []
        for name in sorted(mds.dataset.unique()):
            try:
                X, y, _ = load_full(name, 500, a.cache, a.csv_dir)
            except DatasetSkip:
                continue
            tr, te = resamples(y)[0]
            Xtr, _ = preprocess(X[tr], X[te])
            for _, r in mds[mds.dataset == name].iterrows():
                kind = r.kernel.split("/")[0]
                row = dict(dataset=name, kernel=r.kernel, s_star=r.s_star, n_train=len(Xtr))
                for tag, s in (("", r.s_star), ("_s1", 1.0)):   # same training block at the tuned s and at s = 1
                    psi = states(Xtr, s, kind, Xtr.shape[1])
                    rk, eff = rank_stats(fidelity_gram(psi, psi))
                    row[f"rank{tag}"], row[f"eff_dim{tag}"] = rk, eff
                drows.append(row)
        d = pd.DataFrame(drows)
        d.to_csv(out / "rank_data.csv", index=False)
        lines += ["", "MAIN-STUDY TRAINING BLOCKS (first resample): rank and effective dimension at the tuned s and at s = 1",
                  d.groupby("kernel")[["n_train", "s_star", "rank", "eff_dim", "rank_s1", "eff_dim_s1"]].median()
                  .round(1).to_string()]
    (out / "rank_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
