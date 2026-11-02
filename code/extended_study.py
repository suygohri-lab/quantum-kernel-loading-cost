"""Shared statistics and helpers (corrected resampled t, TOST, geometric difference, bandwidth grid) used by all study
scripts.  Run as a script, it extends the pilot study of gram_spread.py (six datasets, same splits and seeds):
  1. Wide bandwidth grid for EVERY kernel: 10 values 0.002 ... 2 (RBF gamma = s^2) x 4 C values = equal budget.
     Reports how often s lands on either edge of the grid.
  2. Amplitude fidelity kernel |<x|x'>|^2 = (x.x')^2 / (|x|^2 |x'|^2) (closed form; it is a normalised degree-2
     polynomial kernel) and a tuned normalised polynomial baseline (x.x'/(|x||x'|) + c)^2, c in 7 values x 4 C.
     Amplitude loading cost (two-qubit gates of a compute-uncompute circuit) is measured with Qiskit.
  3. Classical-approximability diagnostics between each tuned quantum kernel and the tuned RBF (and tuned polynomial):
     geometric difference g(K1||K2) (Huang et al. 2021 form, ridge-regularised), centred kernel alignment (Cortes et al.
     2012) and relative Frobenius distance, all on trace-normalised training Gram matrices.
  4. Paired statistics vs tuned RBF: corrected resampled t (Nadeau & Bengio), 95% CI, TOST equivalence at +/-0.02,
     per dataset and pooled over datasets; Friedman + Holm-corrected Wilcoxon across datasets.
  5. Optional (--extra-baselines): tuned kNN, gradient boosting, small MLP.  Off by default.

Outputs (--out): ext_cells.csv, ext_paired_vs_rbf.csv, ext_geometry.csv, ext_geometry_summary.csv,
                 ext_gridedge.csv, ext_amplitude_cost.csv, ext_report.txt

Usage:
  python extended_study.py --out results
  python extended_study.py --out results --quick           # 2 repetitions
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from gram_spread import offdiag, test_acc, tune
from qkcommon import (BASE_SEED, DATASETS, KINDS, N_QUBITS, bloch, fidelity_gram, load_split, projected_gamma,
                      projected_gram, states)

S_WIDE_ORIGINAL = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def _s_grid_from_env():
    """Bandwidth grid shared by all studies.  The environment variable QK_S_GRID (comma-separated values, e.g.
    '0.002,0.005,...,2,4,8') overrides the original 10-value grid, so every script and every joblib worker uses the
    same grid.  Unset = original grid (reproduces the original results)."""
    import os
    v = os.environ.get("QK_S_GRID", "").strip()
    if not v:
        return list(S_WIDE_ORIGINAL)
    g = sorted(float(x) for x in v.split(",") if x.strip())
    if len(g) < 2 or g[0] <= 0:
        raise ValueError(f"QK_S_GRID must list at least two positive values, got {v!r}")
    return g


S_WIDE = _s_grid_from_env()
POLY_C0 = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
LAMBDAS = [1e-3, 1e-2, 1e-1]
DELTA = 0.02


# ----------------------------------------------------------------- geometry
def trace_norm(K):
    return K * len(K) / np.trace(K)


def psd_sqrt(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def geo_diff(K1, K2, lam):
    """g(K1||K2) = sqrt( || sqrt(K2) (K1 + lam I)^-1 sqrt(K2) ||_inf ), trace-normalised to Tr = M.
    Large g: K2 has structure K1 cannot reproduce.  ~1: K1 reproduces K2.  (Check against Huang et al. 2021
    before quoting the regularised form; the lam sensitivity is reported.)"""
    A, B = trace_norm(K1), trace_norm(K2)
    S = psd_sqrt(B)
    Mx = S @ np.linalg.solve(A + lam * np.eye(len(A)), S)
    return float(np.sqrt(max(np.linalg.eigvalsh((Mx + Mx.T) / 2).max(), 0.0)))


def cka(K1, K2):
    n = len(K1)
    H = np.eye(n) - np.ones((n, n)) / n
    A, B = H @ K1 @ H, H @ K2 @ H
    return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B)))


def rel_frob(K1, K2):
    A, B = trace_norm(K1), trace_norm(K2)
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


# ----------------------------------------------------------------- statistics
def corrected(d, ntr, nte, delta=DELTA):
    d = np.asarray(d, float)
    R = len(d)
    m = d.mean()
    sd = d.std(ddof=1)
    se = np.sqrt(1.0 / R + nte / ntr) * sd
    t = stats.t.ppf(0.975, R - 1)
    if se == 0:
        return dict(mean=m, lo=m, hi=m, p=0.0 if m != 0 else 1.0, tost_p=0.0 if abs(m) < delta else 1.0,
                    equivalent_95ci=bool(abs(m) < delta))
    p = 2 * stats.t.sf(abs(m) / se, R - 1)
    p_low = stats.t.sf((m + delta) / se, R - 1)   # H0: diff <= -delta
    p_high = stats.t.sf((delta - m) / se, R - 1)  # H0: diff >= +delta
    return dict(mean=m, lo=m - t * se, hi=m + t * se, p=p, tost_p=max(p_low, p_high),
                equivalent_95ci=bool(m - t * se > -delta and m + t * se < delta))


# ----------------------------------------------------------------- amplitude loader cost
def amplitude_loader_cost(d, trials=20, seed=0):
    """Two-qubit gates of a compute-uncompute circuit with two exact amplitude loaders for a dense real vector of
    dimension 2^ceil(log2 d)."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation

    m = int(np.ceil(np.log2(max(d, 2))))
    rng = np.random.default_rng(seed)
    out = {}
    for lv in (1, 3):
        cnt = []
        for _ in range(trials):
            v1, v2 = np.abs(rng.normal(size=2 ** m)), np.abs(rng.normal(size=2 ** m))
            v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
            qc = QuantumCircuit(m)
            qc.append(StatePreparation(v1), range(m))
            qc.append(StatePreparation(v2).inverse(), range(m))
            t = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=lv, seed_transpiler=0)
            cnt.append(t.num_nonlocal_gates())
        out[lv] = float(np.median(cnt))
    return m, out


# ----------------------------------------------------------------- one cell
def run_cell(name, rep, n, extra_baselines):
    Xtr, Xte, ytr, yte = load_split(name, rep, n)
    ntr = len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + rep
    rows, geo_src = [], {}

    def rec(kernel, K, s, C, cv, s_grid=None):
        o = offdiag(K[:ntr, :ntr])
        rows.append(dict(dataset=name, rep=rep, kernel=kernel, s_star=s, C_star=C, cv_acc=cv,
                         test_acc=test_acc(K, ytr, yte, ntr, C),
                         at_lower_edge=bool(s_grid is not None and s == s_grid[0]),
                         at_upper_edge=bool(s_grid is not None and s == s_grid[-1]),
                         offdiag_mean=float(o.mean()), offdiag_sd=float(o.std()),
                         mean_k1mk=float(np.mean(o * (1 - o)))))
        geo_src[kernel] = K[:ntr, :ntr]

    for kind in KINDS:
        Kf, Kp = {}, {}
        for s in S_WIDE:
            psi = states(X, s, kind, n)
            Kf[s] = fidelity_gram(psi, psi)
            F = bloch(psi, n).reshape(len(X), -1)
            Kp[s] = projected_gram(F, F, projected_gamma(F[:ntr]))
        for cons, Kd in (("fidelity", Kf), ("projected", Kp)):
            cv, s, C = tune(Kd, ytr, ntr, seed)
            rec(f"{kind}/{cons}", Kd[s], s, C, cv, S_WIDE)

    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Kr = {s: np.exp(-(s ** 2) * d2) for s in S_WIDE}
    cv, s, C = tune(Kr, ytr, ntr, seed)
    rec("classical/rbf", Kr[s], s, C, cv, S_WIDE)
    g = 1.0 / (X.shape[1] * Xtr.var())
    Ks = np.exp(-g * d2)
    cv, s, C = tune({"scale": Ks}, ytr, ntr, seed)
    rec("classical/rbf_scale", Ks, g, C, cv)
    Kl = X @ X.T
    cv, s, C = tune({"linear": Kl}, ytr, ntr, seed)
    rec("classical/linear", Kl, np.nan, C, cv)

    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    G = Xn @ Xn.T
    Kamp = G ** 2                      # amplitude fidelity kernel (closed form)
    cv, s, C = tune({"amp": Kamp}, ytr, ntr, seed)
    rec("amplitude/fidelity", Kamp, np.nan, C, cv)
    Kpoly = {c: (c + G) ** 2 for c in POLY_C0}
    cv, c0, C = tune(Kpoly, ytr, ntr, seed)
    rec("classical/poly2_normalised", Kpoly[c0], c0, C, cv, POLY_C0)

    if extra_baselines:
        folds = StratifiedKFold(3, shuffle=True, random_state=seed)
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        specs = {
            "classical/knn": (KNeighborsClassifier(), {"n_neighbors": [1, 3, 5, 7, 9, 11, 15]}),
            "classical/gbm": (GradientBoostingClassifier(random_state=seed),
                              {"n_estimators": [50, 100, 200], "max_depth": [1, 2, 3]}),
            "classical/mlp": (MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed),
                              {"alpha": [1e-4, 1e-2, 1.0]}),
        }
        for k, (est, grid) in specs.items():
            gs = GridSearchCV(est, grid, cv=folds).fit(Xtr, ytr)
            rows.append(dict(dataset=name, rep=rep, kernel=k, s_star=np.nan, C_star=np.nan, cv_acc=gs.best_score_,
                             test_acc=float(gs.score(Xte, yte)), at_lower_edge=False, at_upper_edge=False,
                             offdiag_mean=np.nan, offdiag_sd=np.nan, mean_k1mk=np.nan))

    geo = []
    for ref in ("classical/rbf", "classical/poly2_normalised"):
        for k, K in geo_src.items():
            if k == ref:
                continue
            row = dict(dataset=name, rep=rep, kernel=k, reference=ref, cka=cka(K, geo_src[ref]),
                       rel_frobenius=rel_frob(K, geo_src[ref]))
            for lam in LAMBDAS:
                row[f"g_ref_to_k_lam{lam:g}"] = geo_diff(geo_src[ref], K, lam)   # can the reference reproduce k?
                row[f"g_k_to_ref_lam{lam:g}"] = geo_diff(K, geo_src[ref], lam)
            geo.append(row)
    return rows, geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n-qubits", type=int, default=N_QUBITS)
    ap.add_argument("--extra-baselines", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps = 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, geo = [], []
    t0 = time.time()
    for name in a.datasets:
        for rep in range(a.reps):
            r, g = run_cell(name, rep, a.n_qubits, a.extra_baselines)
            rows += r
            geo += g
        print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)
    cells = pd.DataFrame(rows)
    cells.to_csv(out / "ext_cells.csv", index=False)
    gdf = pd.DataFrame(geo)
    gdf.to_csv(out / "ext_geometry.csv", index=False)
    gcols = [c for c in gdf.columns if c not in ("dataset", "rep", "kernel", "reference")]
    gdf.groupby(["reference", "kernel"])[gcols].agg(["mean", "std"]).to_csv(out / "ext_geometry_summary.csv")

    # grid edges
    ge = cells[cells.kernel.str.contains("fidelity|projected|rbf$|poly2")].groupby("kernel").agg(
        at_lower_edge=("at_lower_edge", "mean"), at_upper_edge=("at_upper_edge", "mean"),
        median_s=("s_star", "median"))
    ge.to_csv(out / "ext_gridedge.csv")

    # paired statistics vs tuned RBF (rows: per dataset and pooled)
    ntr, nte = 100, 50
    piv = cells.pivot_table(index=["dataset", "rep"], columns="kernel", values="test_acc")
    stat_rows = []
    for k in [c for c in piv.columns if c != "classical/rbf"]:
        diff = (piv[k] - piv["classical/rbf"]).unstack("dataset")   # rep x dataset
        pooled = corrected(diff.mean(axis=1).values, ntr, nte)
        stat_rows.append(dict(kernel=k, dataset="POOLED", **pooled))
        for ds in diff.columns:
            stat_rows.append(dict(kernel=k, dataset=ds, **corrected(diff[ds].values, ntr, nte)))
    st = pd.DataFrame(stat_rows)
    st.to_csv(out / "ext_paired_vs_rbf.csv", index=False)

    # Friedman + Holm Wilcoxon across datasets (mean over reps)
    means = cells.groupby(["dataset", "kernel"]).test_acc.mean().unstack()
    names = [k for k in means.columns if k not in ("classical/rbf_scale",)]
    fr = stats.friedmanchisquare(*[means[k].values for k in names])
    pv = {k: stats.wilcoxon(means[k].values, means["classical/rbf"].values).pvalue
          for k in names if k != "classical/rbf"}
    order = sorted(pv, key=pv.get)
    holm, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * pv[k]))
        holm[k] = run

    # amplitude loader cost
    d = a.n_qubits
    m, gcnt = amplitude_loader_cost(d)
    amp = cells[cells.kernel == "amplitude/fidelity"]
    pd.DataFrame([dict(qubits=m, twoq_gates_opt1=gcnt[1], twoq_gates_opt3=gcnt[3],
                       offdiag_sd_median=amp.offdiag_sd.median(), mean_k1mk_median=amp.mean_k1mk.median(),
                       note="compute-uncompute circuit with two exact Qiskit StatePreparation loaders, dense real vector")]
                 ).to_csv(out / "ext_amplitude_cost.csv", index=False)

    lines = ["MEAN TEST ACCURACY (all datasets, repetitions)", cells.groupby("kernel").test_acc.mean().round(4).to_string(),
             "", "GRID EDGES (share of cells at the lowest / highest s; wide grid 0.002 ... 2)", ge.round(3).to_string(),
             "", f"FRIEDMAN across {len(means)} datasets, {len(names)} models: chi2={fr.statistic:.2f}, p={fr.pvalue:.4f}",
             "Holm-adjusted Wilcoxon vs tuned RBF (minimum attainable with 6 blocks is 0.031 before adjustment):"]
    lines += [f"  {k}: {holm[k]:.3f}" for k in order]
    lines += ["", "POOLED PAIRED DIFFERENCE VS TUNED RBF (corrected resampled t, 95% CI, TOST delta=0.02)",
              st[st.dataset == "POOLED"][["kernel", "mean", "lo", "hi", "p", "tost_p", "equivalent_95ci"]].round(4).to_string(index=False),
              "", "GEOMETRY vs tuned RBF (mean over cells)",
              gdf[gdf.reference == "classical/rbf"].groupby("kernel")[["cka", "rel_frobenius", "g_ref_to_k_lam0.001",
                                                                          "g_k_to_ref_lam0.001"]].mean().round(3).to_string(),
              "", f"AMPLITUDE LOADER: {m} qubits, two-qubit gates of compute-uncompute circuit: opt1={gcnt[1]:.0f}, opt3={gcnt[3]:.0f}"]
    (out / "ext_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
