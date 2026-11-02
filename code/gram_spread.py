"""Shared tuning routine (bandwidth/C grid search with the tie rule) and the shot rule used by all study scripts.

Run as a script, it performs a small pilot study (n = 4, six datasets, R = 20 paired repetitions) and records, for
every tuned kernel, the spread of its training Gram matrix and the shots needed to resolve it.  The pilot's
cells.csv is the input of the design-stage power analysis (power_analysis.py).

Outputs (in --out):
  cells.csv               one row per (dataset, repetition, kernel): tuned s, C, accuracy, Gram mean/SD ...
  shots.csv               shots per circuit needed for each epsilon rule (long format)

Shot rules (RMS error of the off-diagonal entries of the training Gram matrix <= epsilon):
  fidelity  : each entry ~ Binomial(S, k)/S, so RMS^2 = mean(k(1-k))/S  ->  S = ceil(mean(k(1-k))/eps^2)
              (worst-case  ceil(1/(4 eps^2))  is also reported)
  projected : each Pauli expectation estimated from S shots (3 settings per point); S found by Monte Carlo
  epsilon   : fixed {0.1, 0.05, 0.01}  or  SD(off-diagonal)/c  with c in {1, 2, 4}  ('spread' rule)

Usage:
  python gram_spread.py --out results
  python gram_spread.py --out results --quick        # 2 repetitions, for a fast test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from qkcommon import (BASE_SEED, C_GRID, DATASETS, KINDS, N_QUBITS, S_GRID, bloch, fidelity_gram,
                      load_split, projected_gamma, projected_gram, states)

FIXED_EPS = [0.1, 0.05, 0.01]
SPREAD_C = [1, 2, 4]
SHOT_GRID = [int(25 * 2 ** k) for k in range(0, 17)]  # 25 ... 1,638,400


def tie_rule():
    """Tie-break rule for tune(): 'first' (original design: first candidate in grid order, i.e. the smallest s and C)
    or 'middle' (used in the paper: the tied candidate closest to the middle of the grid).  Set by the environment variable
    QK_TIE so that every script and every joblib worker uses the same rule; default 'first' reproduces the
    original results."""
    import os
    r = os.environ.get("QK_TIE", "first").strip().lower()
    if r not in ("first", "middle"):
        raise ValueError(f"QK_TIE must be 'first' or 'middle', got {r!r}")
    return r


def tune(Kfull, ytr, ntr, seed, s_values=None, tie=None):
    """Select (s, C) by 3-fold CV on the training block.
    Kfull: dict s -> (ntr+nte) x (ntr+nte) Gram matrix (train first).
    Ties (CV accuracy within 1e-12 of the best): tie='first' keeps the first candidate in grid order;
    tie='middle' keeps the tied candidate whose (s, C) grid positions are closest to the grid centre
    (sum of index distances; remaining ties go to the smaller index).  tie=None reads QK_TIE (see tie_rule)."""
    tie = tie or tie_rule()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(ntr), ytr))
    cands = []
    for i_s, (s, K) in enumerate(Kfull.items()):
        Ktr = K[:ntr, :ntr]
        for i_c, C in enumerate(C_GRID):
            acc = []
            for tr, va in folds:
                clf = SVC(kernel="precomputed", C=C).fit(Ktr[np.ix_(tr, tr)], ytr[tr])
                acc.append(clf.score(Ktr[np.ix_(va, tr)], ytr[va]))
            cands.append((float(np.mean(acc)), i_s, i_c, s, C))
    top = max(c[0] for c in cands)
    ties = [c for c in cands if c[0] >= top - 1e-12]
    if tie == "first":
        m, _, _, s, C = ties[0]
    else:
        mid_s, mid_c = (len(Kfull) - 1) / 2.0, (len(C_GRID) - 1) / 2.0
        m, _, _, s, C = min(ties, key=lambda c: (abs(c[1] - mid_s) + abs(c[2] - mid_c), c[1], c[2]))
    return (m, s, C)


def test_acc(K, ytr, yte, ntr, C):
    clf = SVC(kernel="precomputed", C=C).fit(K[:ntr, :ntr], ytr)
    return float(clf.score(K[ntr:, :ntr], yte))


def offdiag(K):
    m = ~np.eye(K.shape[0], dtype=bool)
    return K[m]


def mc_shots_projected(R, gamma, eps, rng):
    """Smallest S in SHOT_GRID with RMS off-diagonal error <= eps (nan if none)."""
    F = R.reshape(R.shape[0], -1)
    K = projected_gram(F, F, gamma)
    mask = ~np.eye(K.shape[0], dtype=bool)
    p = np.clip((1 + F) / 2, 0, 1)
    for S in SHOT_GRID:
        Fh = 2.0 * rng.binomial(S, p) / S - 1.0
        Kh = projected_gram(Fh, Fh, gamma)
        if np.sqrt(np.mean((Kh - K)[mask] ** 2)) <= eps:
            return S
    return np.nan


def run_cell(name, rep, n):
    Xtr, Xte, ytr, yte = load_split(name, rep, n)
    ntr = len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + rep
    rows, shot_rows = [], []
    rng = np.random.default_rng(seed + 17)

    def record(kernel, K, s, C, cvacc, extra=None):
        d = dict(dataset=name, rep=rep, kernel=kernel, s_star=s, C_star=C, cv_acc=cvacc,
                 test_acc=test_acc(K, ytr, yte, ntr, C))
        off = offdiag(K[:ntr, :ntr])
        d.update(offdiag_mean=float(off.mean()), offdiag_sd=float(off.std()))
        if extra:
            d.update(extra)
        rows.append(d)
        return off

    for kind in KINDS:
        Kfid, Kproj, Fs, gam = {}, {}, {}, {}
        for s in S_GRID:
            psi = states(X, s, kind, n)
            Kfid[s] = fidelity_gram(psi, psi)
            R = bloch(psi, n)
            F = R.reshape(len(X), -1)
            g = projected_gamma(F[:ntr])
            Kproj[s] = projected_gram(F, F, g)
            Fs[s], gam[s] = R, g
        # fidelity
        cv, s, C = tune(Kfid, ytr, ntr, seed)
        K = Kfid[s]
        o = offdiag(K[:ntr, :ntr])
        mk = float(np.mean(o * (1 - o)))
        off = record(f"{kind}/fidelity", K, s, C, cv, dict(mean_k1mk=mk))
        sd = off.std()
        for eps in FIXED_EPS:
            shot_rows.append(dict(dataset=name, rep=rep, kernel=f"{kind}/fidelity", rule="fixed", param=eps, eps=eps,
                                  shots=int(np.ceil(max(mk, 1e-12) / eps ** 2)), shots_worstcase=int(np.ceil(1 / (4 * eps ** 2)))))
        for c in SPREAD_C:
            eps = sd / c
            shot_rows.append(dict(dataset=name, rep=rep, kernel=f"{kind}/fidelity", rule="spread", param=c, eps=eps,
                                  shots=int(np.ceil(max(mk, 1e-12) / eps ** 2)), shots_worstcase=int(np.ceil(1 / (4 * eps ** 2)))))
        # projected
        cv, s, C = tune(Kproj, ytr, ntr, seed)
        K = Kproj[s]
        off = record(f"{kind}/projected", K, s, C, cv, dict(gamma=float(gam[s])))
        sd = off.std()
        Rtr = Fs[s][:ntr]
        for eps in FIXED_EPS:
            shot_rows.append(dict(dataset=name, rep=rep, kernel=f"{kind}/projected", rule="fixed", param=eps, eps=eps,
                                  shots=mc_shots_projected(Rtr, gam[s], eps, rng), shots_worstcase=np.nan))
        for c in SPREAD_C:
            eps = sd / c
            shot_rows.append(dict(dataset=name, rep=rep, kernel=f"{kind}/projected", rule="spread", param=c, eps=eps,
                                  shots=mc_shots_projected(Rtr, gam[s], eps, rng), shots_worstcase=np.nan))

    # classical baselines on the same split
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Krbf = {s: np.exp(-(s ** 2) * d2) for s in S_GRID}
    cv, s, C = tune(Krbf, ytr, ntr, seed)
    record("classical/rbf", Krbf[s], s, C, cv)
    g = 1.0 / (X.shape[1] * Xtr.var())
    cv, s, C = tune({"scale": np.exp(-g * d2)}, ytr, ntr, seed)
    record("classical/rbf_scale", np.exp(-g * d2), g, C, cv)
    cv, s, C = tune({"linear": X @ X.T}, ytr, ntr, seed)
    record("classical/linear", X @ X.T, np.nan, C, cv)
    return rows, shot_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n-qubits", type=int, default=N_QUBITS)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps = 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, shots = [], []
    t0 = time.time()
    for name in a.datasets:
        for rep in range(a.reps):
            r, s = run_cell(name, rep, a.n_qubits)
            rows += r
            shots += s
        print(f"{name}: done ({time.time() - t0:.0f}s)")
    cells = pd.DataFrame(rows)
    cells.to_csv(out / "cells.csv", index=False)
    pd.DataFrame(shots).to_csv(out / "shots.csv", index=False)
    print("\nMean test accuracy over datasets and repetitions:")
    print(cells.groupby("kernel").test_acc.mean().round(4).to_string())


if __name__ == "__main__":
    main()
