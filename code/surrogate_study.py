"""Surrogate tuning (method) and Nystrom landmarks (lever): two ways to cut the number of loading-circuit executions.

PRE-SPECIFIED DESIGN (fixed on 2026-10-31, before any result of this script was seen; written to surrogate_design.txt)
Same datasets, cap, resamples, preprocessing (n = 4), bandwidth grid, C grid, inner 3-fold CV and tie rule as the main
study (multi_dataset_study.py).  Set QK_TIE and QK_S_GRID exactly as for the main run.

Part A - surrogate tuning.  Full-grid tuning builds one quantum Gram matrix per bandwidth (|S| matrices); a surrogate
chooses the bandwidth from a classical kernel and builds ONE quantum Gram matrix.  C is then tuned by inner CV on that
single matrix (C costs no circuits).
  A1 'rbf_map'  : s = 2 * s*_RBF, where s*_RBF is the bandwidth of the tuned RBF kernel exp(-s^2 ||x-x'||^2).  The
                  factor 2 comes from the small-angle limit cos^2(s d/2) ~ exp(-s^2 d^2 / 4) (Eq. smallangle); the
                  same rule is used for fidelity and projected kernels.  Value used as is (off-grid allowed).
  A1m 'metric_map' (added after a pilot on 3 datasets x 3 folds, before the full run; disclosed as such): s =
                  s*_RBF / kappa_k, where kappa_k^2 = (1/n) sum_q lim (1 - K(x, x + e e_q)) / (s e)^2 is the small-bandwidth
                  metric of encoding k (data-free, one statevector calculation per encoding and n; it does not depend
                  on x).  kappa^2 = 1/4 for angle and IQP, so A1m = A1 there; for ZZ kappa^2 is much larger (about 37 at
                  n = 4) because its pair phases depend linearly on the features.  The pilot showed that A1 sends ZZ
                  to s ~ 1, where ZZ is known to lose accuracy (fixed-bandwidth control).
  A2 'angle'    : s = s* of the angle kernel of the same construction (angle fidelity for fidelity kernels, angle
                  projected for projected kernels), tuned on the full grid through its classical closed form.
  Quantum kernels evaluated: IQP-F, IQP-P, ZZ-F, ZZ-P (both surrogates) and Angle-F, Angle-P (A1 only; A2 is the
  identity for angle).
  Primary endpoint: pooled paired accuracy difference (surrogate - full grid), corrected resampled t, two-sided 95% CI;
  surrogate is NON-INFERIOR if the lower bound > -0.02.  Secondary: difference to the tuned RBF kernel (TOST +/-0.02),
  per-dataset results, balanced accuracy and macro-F1, and the bandwidth-search circuits (|S| -> 1).
  Prediction stated in advance: the surrogates should work better for IQP than for ZZ, because at the tuned bandwidth
  IQP is close to stationary and ZZ is not (stationarity_study.py).

Part B - Nystrom landmarks (lever, not a method of this paper; Srikumar et al. 2024, Coelho et al. 2025).  For the
fidelity kernels at the full-grid bandwidth s*, m landmarks are drawn uniformly at random from the training points
(fixed seed per resample), and K ~ C W^+ C^T.  m = 25% and 50% of the training points.  C is re-tuned by inner CV on the
approximated training Gram matrix.  Endpoint: paired accuracy difference (Nystrom - exact), non-inferiority at -0.02.
Circuits per Gram matrix: exact M(M-1)/2 + n_test*M, Nystrom m(m-1)/2 + (M-m)*m + n_test*m.

Outputs (--out): sur_cells.csv, sur_paired.csv, sur_circuits.csv, sur_circuits_ref.csv, sur_kappa.csv, sur_report.txt, surrogate_design.txt,
sur_check_vs_mds.csv (consistency of the recomputed full-grid cells with mds_cells.csv, if present).

Usage:  python surrogate_study.py --out results_v2 --n-jobs 10
        python surrogate_study.py --out test_out --quick
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.svm import SVC

from datasets_cv import ALL_DATASETS, BASE_SEED, DatasetSkip, load_full, preprocess, resamples
from extended_study import DELTA, S_WIDE, corrected
from gram_spread import tie_rule, tune
from qkcommon import KINDS, bloch, fidelity_gram, projected_gamma, projected_gram, states

RBF_FACTOR = 2.0
NYSTROM_FRACS = [0.25, 0.5]
PILOT = ["moons", "circles", "breast_cancer"]   # datasets of the pilot run (3 folds) that preceded the metric-matched surrogate
_KAPPA = {}


def kappa2(kind, n, s=1e-3, h=1e-3):
    """Small-bandwidth metric of an encoding, kappa^2 = (1/n) sum_q g_qq / s^2, with the Fubini-Study metric
    g_qq = <d_q psi|d_q psi> - |<psi|d_q psi>|^2 (central differences in x_q) at x = pi/2.  For a fidelity kernel,
    1 - K(x, x + d) = d^T g d + O(d^3), so to leading order in s, K ~ exp(-kappa^2 s^2 |d|^2) for an isotropic g."""
    if (kind, n) in _KAPPA:
        return _KAPPA[(kind, n)]
    x = np.full(n, np.pi / 2)
    psi = states(x[None], s, kind, n)[0]
    tot = 0.0
    for q in range(n):
        up, dn = x.copy(), x.copy()
        up[q] += h
        dn[q] -= h
        dpsi = (states(up[None], s, kind, n)[0] - states(dn[None], s, kind, n)[0]) / (2 * h)
        tot += (np.vdot(dpsi, dpsi) - abs(np.vdot(psi, dpsi)) ** 2).real / s ** 2
    _KAPPA[(kind, n)] = tot / n
    return tot / n


def scores(K, ytr, yte, ntr, C):
    pred = SVC(kernel="precomputed", C=C).fit(K[:ntr, :ntr], ytr).predict(K[ntr:, :ntr])
    return (float((pred == yte).mean()), float(balanced_accuracy_score(yte, pred)),
            float(f1_score(yte, pred, average="macro")))


def gram(X, s, kind, cons, ntr, n):
    psi = states(X, s, kind, n)
    if cons == "fidelity":
        return fidelity_gram(psi, psi)
    F = bloch(psi, n).reshape(len(X), -1)
    return projected_gram(F, F, projected_gamma(F[:ntr]))


def nystrom(K, ntr, m, rng):
    """Nystrom approximation of the full (train+test) Gram matrix from m training landmarks."""
    L = np.sort(rng.choice(ntr, size=m, replace=False))
    C = K[:, L]
    W = K[np.ix_(L, L)]
    Wp = np.linalg.pinv((W + W.T) / 2, rcond=1e-10, hermitian=True)
    return C @ Wp @ C.T


def evaluate_split(name, ri, Xtr_raw, Xte_raw, ytr, yte, n_max=4):
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw, n_max=n_max)
    n, ntr = Xtr.shape[1], len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + ri
    rows = []

    def rec(kernel, method, K, s, C, cv, extra=None):
        acc, bacc, f1 = scores(K, ytr, yte, ntr, C)
        rows.append(dict(dataset=name, resample=ri, kernel=kernel, method=method, s_used=s, C_star=C, cv_acc=cv,
                         test_acc=acc, test_bacc=bacc, test_f1=f1, n_qubits=n, n_train=ntr, n_test=len(yte),
                         **(extra or {})))

    # classical RBF on the shared grid (the A1 surrogate and the reference)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    cv, s_rbf, C = tune({s: np.exp(-(s ** 2) * d2) for s in S_WIDE}, ytr, ntr, seed)
    rec("classical/rbf", "full_grid", np.exp(-(s_rbf ** 2) * d2), s_rbf, C, cv)

    full = {}
    for kind in KINDS:                       # angle first (KINDS order), so its s* is known for A2
        for cons in ("fidelity", "projected"):
            Kd = {s: gram(X, s, kind, cons, ntr, n) for s in S_WIDE}
            cv, s_star, C = tune(Kd, ytr, ntr, seed)
            k = f"{kind}/{cons}"
            full[k] = (s_star, Kd[s_star])
            rec(k, "full_grid", Kd[s_star], s_star, C, cv)
            surr = {"rbf_map": RBF_FACTOR * s_rbf, "metric_map": s_rbf / np.sqrt(kappa2(kind, n))}
            if kind != "angle":
                surr["angle"] = full[f"angle/{cons}"][0]
            for meth, s_sur in surr.items():
                K1 = Kd[s_sur] if s_sur in Kd else gram(X, s_sur, kind, cons, ntr, n)
                cv1, _, C1 = tune({s_sur: K1}, ytr, ntr, seed)
                rec(k, meth, K1, s_sur, C1, cv1)
            if cons == "fidelity":
                rng = np.random.default_rng(seed + 101)
                for frac in NYSTROM_FRACS:
                    m = max(2, int(round(frac * ntr)))
                    Kn = nystrom(Kd[s_star], ntr, m, rng)
                    cv2, _, C2 = tune({s_star: Kn}, ytr, ntr, seed)
                    rec(k, f"nystrom_{frac:g}", Kn, s_star, C2, cv2, dict(landmarks=m))
    return rows


REF_M, REF_T = 100, 50   # reference size of the loading bill (frontier.csv, lever_bill.csv)


def circuits(cells, ref=False):
    """Circuits of one tuned model per kernel and method, with the accounting of the loading bill (lever_table.py):
    tuning builds the training block of the Gram matrix once per bandwidth tried (|S| for the full grid, 1 for a
    surrogate); the test block is built once, at the chosen bandwidth.  gram_circuits is one complete Gram matrix
    (training + test), exact or, for Nystrom, from m landmarks.  Sizes: median main-study cell, or (ref=True) the
    reference size of the bill, M = 100 training and 50 test points."""
    out = []
    S = len(S_WIDE)
    for (k, meth), g in cells[cells.kernel != "classical/rbf"].groupby(["kernel", "method"]):
        M, T = (REF_M, REF_T) if ref else (g.n_train.median(), g.n_test.median())
        fid = k.endswith("fidelity")
        train = M * (M - 1) / 2 if fid else 3 * M
        test = T * M if fid else 3 * T
        if meth.startswith("nystrom"):
            m = round(g.landmarks.median() / g.n_train.median() * M)
            gram_c = m * (m - 1) / 2 + (M - m) * m + T * m
            tuning, test_c = S * train, T * m          # bandwidth still tuned on the full grid
        else:
            gram_c = train + test
            tuning, test_c = (S if meth == "full_grid" else 1) * train, test
        out.append(dict(kernel=k, method=meth, M_train=M, n_test=T, grid_size=S, tuning_circuits=tuning,
                        test_circuits=test_c, total_circuits=tuning + test_c, gram_circuits=gram_c))
    c = pd.DataFrame(out)
    base = c[c.method == "full_grid"].set_index("kernel")
    c["saving_vs_full_grid"] = c.apply(lambda r: base.total_circuits[r.kernel] / r.total_circuits, axis=1)
    c["gram_saving"] = c.apply(lambda r: base.gram_circuits[r.kernel] / r.gram_circuits, axis=1)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--n-max", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyse-only", action="store_true", help="recompute statistics from an existing sur_cells.csv")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.quick:
        a.datasets, a.folds, a.repeats = a.datasets[:3], 3, 1
    if a.analyse_only:
        analyse(pd.read_csv(out / "sur_cells.csv"), out, a)
        return
    (out / "surrogate_design.txt").write_text(__doc__.split("Outputs")[0] +
                                              f"\nRUN: tie rule = {tie_rule()}, grid = {S_WIDE}, n_max = {a.n_max}, "
                                              f"{a.folds}-fold x {a.repeats}\n")
    jobs = []
    for name in a.datasets:
        try:
            X, y, _ = load_full(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            continue
        for ri, (tr, te) in enumerate(resamples(y, a.folds, a.repeats)):
            jobs.append((name, ri, X[tr], X[te], y[tr], y[te]))
    print(f"{len(jobs)} jobs, n_jobs={a.n_jobs}")
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(
        delayed(evaluate_split)(*j, a.n_max) for j in jobs)
    cells = pd.DataFrame([r for rs in res for r in rs])
    cells.to_csv(out / "sur_cells.csv", index=False)
    print(f"compute time {time.time() - t0:.0f}s")
    analyse(cells, out, a)


def analyse(cells, out, a):
    ratio = 1.0 / (a.folds - 1)
    key = ["dataset", "resample"]
    st = []
    for metric in ("test_acc", "test_bacc", "test_f1"):
        piv = cells.pivot_table(index=key, columns=["kernel", "method"], values=metric)
        rbf = piv[("classical/rbf", "full_grid")]
        for (k, meth) in piv.columns:
            if k == "classical/rbf":
                continue
            for ref_name, ref in (("full_grid", piv[(k, "full_grid")]), ("tuned_rbf", rbf)):
                if meth == "full_grid" and ref_name == "full_grid":
                    continue
                diff = (piv[(k, meth)] - ref).unstack("dataset")
                pooled = corrected(diff.mean(axis=1).values, 1.0, ratio)
                st.append(dict(metric=metric, kernel=k, method=meth, reference=ref_name, dataset="POOLED", **pooled,
                               noninferior=bool(pooled["lo"] > -DELTA)))
                rest = [c for c in diff.columns if c not in PILOT]
                if 0 < len(rest) < len(diff.columns):
                    r = corrected(diff[rest].mean(axis=1).values, 1.0, ratio)
                    st.append(dict(metric=metric, kernel=k, method=meth, reference=ref_name,
                                   dataset="POOLED_EXCL_PILOT", **r, noninferior=bool(r["lo"] > -DELTA)))
                for ds in diff.columns:
                    r = corrected(diff[ds].dropna().values, 1.0, ratio)
                    st.append(dict(metric=metric, kernel=k, method=meth, reference=ref_name, dataset=ds, **r,
                                   noninferior=bool(r["lo"] > -DELTA)))
    st = pd.DataFrame(st)
    st.to_csv(out / "sur_paired.csv", index=False)
    circ = circuits(cells)
    circ.to_csv(out / "sur_circuits.csv", index=False)
    circ_ref = circuits(cells, ref=True)
    circ_ref.to_csv(out / "sur_circuits_ref.csv", index=False)
    kap = pd.DataFrame([dict(encoding=k, n=n, kappa2=kappa2(k, n), factor_vs_rbf=1 / np.sqrt(kappa2(k, n)))
                        for k in KINDS for n in range(2, 13)])
    kap.to_csv(out / "sur_kappa.csv", index=False)

    mds = out / "mds_cells.csv"
    if mds.exists() and not a.quick:
        m = pd.read_csv(mds)
        f = cells[cells.method == "full_grid"].merge(m, on=["dataset", "resample", "kernel"], suffixes=("", "_mds"))
        chk = f.groupby("kernel").apply(lambda g: pd.Series(dict(
            cells=len(g), same_s=float(np.isclose(g.s_used, g.s_star).mean()),
            mean_acc_here=g.test_acc.mean(), mean_acc_mds=g.test_acc_mds.mean())), include_groups=False)
        chk.to_csv(out / "sur_check_vs_mds.csv")
    else:
        chk = None

    p = st[(st.metric == "test_acc") & (st.dataset == "POOLED")]
    sel = p[["kernel", "method", "reference", "mean", "lo", "hi", "noninferior", "equivalent_95ci"]]
    nd = cells.groupby(["kernel", "method"]).dataset.nunique()
    perds = st[(st.metric == "test_acc") & ~st.dataset.str.startswith("POOLED") & (st.reference == "full_grid")]
    share_ni = perds.groupby(["kernel", "method"]).agg(datasets=("noninferior", "size"),
                                                       share_noninferior=("noninferior", "mean"),
                                                       significantly_worse=("hi", lambda h: int((h < 0).sum())))
    excl = st[(st.metric == "test_acc") & (st.dataset == "POOLED_EXCL_PILOT") & (st.reference == "full_grid")]
    lines = [f"DATASETS: {', '.join(sorted(cells.dataset.unique()))}", f"tie rule = {tie_rule()}, grid = {S_WIDE}", "",
             "POOLED PAIRED ACCURACY DIFFERENCE (corrected resampled t, 95% CI; non-inferior if lo > -0.02)",
             sel.round(4).to_string(index=False), "",
             "POOLED OVER THE DATASETS OUTSIDE THE PILOT (" + ", ".join(PILOT) + " excluded)",
             excl[["kernel", "method", "mean", "lo", "hi", "noninferior"]].round(4).to_string(index=False), "",
             "PER DATASET AGAINST FULL-GRID TUNING: share non-inferior and number significantly worse (upper bound < 0)",
             share_ni.round(3).to_string(), "",
             "CIRCUITS (median cell): tuning + final Gram matrix", circ.round(1).to_string(index=False), "",
             f"CIRCUITS AT THE REFERENCE SIZE OF THE BILL (M = {REF_M}, {REF_T} test points)",
             circ_ref.round(1).to_string(index=False), "",
             "SMALL-BANDWIDTH METRIC kappa^2 (metric-matched s = s*_RBF / kappa)",
             kap.pivot_table(index="n", columns="encoding", values="kappa2").round(3).to_string(), "",
             "MEDIAN BANDWIDTH USED", cells.groupby(["kernel", "method"]).s_used.median().round(4).to_string()]
    if chk is not None:
        lines += ["", "CONSISTENCY WITH mds_cells.csv (full-grid cells)", chk.round(4).to_string()]
    (out / "sur_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
