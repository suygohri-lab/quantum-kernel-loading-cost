"""Larger study: all kernels on up to 16 datasets, N <= 500, repeated stratified 10-fold x 2 (20 resamples).

Models (equal tuning budget, 3-fold CV inside each training part): angle / IQP / ZZ fidelity and projected quantum
kernels on the wide bandwidth grid (10 values x 4 C), tuned RBF (same grid, gamma = s^2), RBF with default gamma,
linear, amplitude fidelity kernel (closed form), tuned normalised degree-2 polynomial; optional kNN / gradient
boosting / MLP (--extra-baselines).  n_qubits = min(features after PCA, 4).

Statistics: paired difference to tuned RBF per resample; corrected resampled t (Nadeau & Bengio) with n_test/n_train
of the design (about 1/9); 95% CI, TOST equivalence at +/-0.02; pooled over datasets; Friedman across datasets with
Holm-corrected Wilcoxon vs tuned RBF.

Outputs (--out): mds_dataset_info.csv, mds_cells.csv, mds_paired_vs_rbf.csv, mds_report.txt

Usage:
  python multi_dataset_study.py --out results --n-jobs 4
  python multi_dataset_study.py --out results --quick                 # 3 datasets, 2 resamples
  python multi_dataset_study.py --out results --csv-dir my_csvs       # CSV fallback for downloads
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from datasets_cv import ALL_DATASETS, BASE_SEED, DatasetSkip, load_full, preprocess, resamples
from extended_study import LAMBDAS, POLY_C0, S_WIDE, cka, corrected, geo_diff
from gram_spread import FIXED_EPS, SPREAD_C, mc_shots_projected, offdiag, test_acc, tie_rule, tune
from qkcommon import KINDS, bloch, fidelity_gram, projected_gamma, projected_gram, states


def scores(K, ytr, yte, ntr, C):
    """accuracy, balanced accuracy, macro-F1 of an SVM on a precomputed Gram matrix."""
    pred = SVC(kernel="precomputed", C=C).fit(K[:ntr, :ntr], ytr).predict(K[ntr:, :ntr])
    return (float((pred == yte).mean()), float(balanced_accuracy_score(yte, pred)),
            float(f1_score(yte, pred, average="macro")))


def kta(K, y):
    """Centred kernel-target alignment (Cortes et al. 2012) with the one-hot label kernel Y Y^T (works for multiclass)."""
    Y = (np.asarray(y)[:, None] == np.unique(y)[None, :]).astype(float)
    return cka(K, Y @ Y.T)


SHOT_SUBSAMPLE = 100  # training points used for the shot rule; matches the M = 100 cost model of the frontier


def evaluate_split(name, ri, Xtr_raw, Xte_raw, ytr, yte, extra, n_max=4, diagnostics=False):
    """n_max = largest number of features/qubits (PCA to n_max components when there are more features).
    diagnostics=True also returns, for every tuned quantum kernel: geometric difference g(K_RBF || K_Q) and the
    reverse direction (Huang et al. 2021 form, trace-normalised, ridge lam in LAMBDAS), CKA to the tuned RBF,
    kernel-target alignment (also for the classical kernels), and the shots per circuit of gram_spread.py's rules
    evaluated on the first SHOT_SUBSAMPLE training points of this split.
    Returns rows if diagnostics is False, else (rows, geometry_rows, shot_rows)."""
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw, n_max=n_max)
    n = Xtr.shape[1]
    ntr = len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + ri
    rows, geo_rows, shot_rows = [], [], []
    tuned = {}   # kernel -> (training Gram block, Bloch features of the tuned s or None, projected gamma or None)

    def rec(kernel, K, s, C, cv, grid=None, feats=None, gam=None):
        tuned[kernel] = (K[:ntr, :ntr], feats, gam)
        o = offdiag(K[:ntr, :ntr])
        acc, bacc, f1 = scores(K, ytr, yte, ntr, C)
        rows.append(dict(dataset=name, resample=ri, kernel=kernel, s_star=s, C_star=C, cv_acc=cv,
                         test_acc=acc, test_bacc=bacc, test_f1=f1,
                         at_lower_edge=bool(grid is not None and s == grid[0]),
                         at_upper_edge=bool(grid is not None and s == grid[-1]),
                         offdiag_mean=float(o.mean()), offdiag_sd=float(o.std()), n_qubits=n))

    for kind in KINDS:
        Kf, Kp, Fd, Gd = {}, {}, {}, {}
        for s in S_WIDE:
            psi = states(X, s, kind, n)
            Kf[s] = fidelity_gram(psi, psi)
            F = bloch(psi, n).reshape(len(X), -1)
            Gd[s] = projected_gamma(F[:ntr])
            Kp[s] = projected_gram(F, F, Gd[s])
            Fd[s] = F[:ntr]
        for cons, Kd in (("fidelity", Kf), ("projected", Kp)):
            cv, s, C = tune(Kd, ytr, ntr, seed)
            rec(f"{kind}/{cons}", Kd[s], s, C, cv, S_WIDE, feats=Fd[s] if cons == "projected" else None,
                gam=Gd[s] if cons == "projected" else None)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Kr = {s: np.exp(-(s ** 2) * d2) for s in S_WIDE}
    cv, s, C = tune(Kr, ytr, ntr, seed)
    rec("classical/rbf", Kr[s], s, C, cv, S_WIDE)
    g = 1.0 / (X.shape[1] * Xtr.var())
    Ks = np.exp(-g * d2)
    cv, _, C = tune({"scale": Ks}, ytr, ntr, seed)
    rec("classical/rbf_scale", Ks, g, C, cv)
    Kl = X @ X.T
    cv, _, C = tune({"linear": Kl}, ytr, ntr, seed)
    rec("classical/linear", Kl, np.nan, C, cv)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    G = Xn @ Xn.T
    cv, _, C = tune({"amp": G ** 2}, ytr, ntr, seed)
    rec("amplitude/fidelity", G ** 2, np.nan, C, cv)
    Kpoly = {c: (c + G) ** 2 for c in POLY_C0}
    cv, c0, C = tune(Kpoly, ytr, ntr, seed)
    rec("classical/poly2_normalised", Kpoly[c0], c0, C, cv, POLY_C0)
    if extra:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        folds = StratifiedKFold(3, shuffle=True, random_state=seed)
        specs = {"classical/knn": (KNeighborsClassifier(), {"n_neighbors": [1, 3, 5, 7, 9, 11, 15]}),
                 "classical/gbm": (GradientBoostingClassifier(random_state=seed),
                                   {"n_estimators": [50, 100, 200], "max_depth": [1, 2, 3]}),
                 "classical/rf": (RandomForestClassifier(n_estimators=200, random_state=seed),
                                  {"max_depth": [2, 4, 8, None], "min_samples_leaf": [1, 5]}),
                 "classical/mlp": (MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed),
                                   {"alpha": [1e-4, 1e-2, 1.0]})}
        for k, (est, grid) in specs.items():
            gs = GridSearchCV(est, grid, cv=folds).fit(Xtr, ytr)
            rows.append(dict(dataset=name, resample=ri, kernel=k, s_star=np.nan, C_star=np.nan, cv_acc=gs.best_score_,
                             test_acc=float(gs.score(Xte, yte)),
                             test_bacc=float(balanced_accuracy_score(yte, gs.predict(Xte))),
                             test_f1=float(f1_score(yte, gs.predict(Xte), average="macro")), at_lower_edge=False, at_upper_edge=False,
                             offdiag_mean=np.nan, offdiag_sd=np.nan, n_qubits=n))
    if not diagnostics:
        return rows

    # ---------------- diagnostics on the training block of this split
    Kref = tuned["classical/rbf"][0]
    for k, (Ktr, _, _) in tuned.items():
        row = dict(dataset=name, resample=ri, kernel=k, n_qubits=n, kta=kta(Ktr, ytr))
        if k != "classical/rbf" and k.split("/")[1] in ("fidelity", "projected"):
            row["cka_to_rbf"] = cka(Ktr, Kref)
            for lam in LAMBDAS:
                row[f"g_rbf_to_k_lam{lam:g}"] = geo_diff(Kref, Ktr, lam)   # can the tuned RBF reproduce K? (Huang g_CQ)
                row[f"g_k_to_rbf_lam{lam:g}"] = geo_diff(Ktr, Kref, lam)   # can K reproduce the tuned RBF?
        geo_rows.append(row)
    rng = np.random.default_rng(seed + 17)
    m = min(SHOT_SUBSAMPLE, ntr)
    for k, (Ktr, feats, gam) in tuned.items():
        emb, cons = k.split("/")
        if emb not in KINDS:
            continue
        o = offdiag(Ktr[:m, :m])
        sd = float(o.std())
        if cons == "fidelity":
            mk = float(np.mean(o * (1 - o)))
            for rule, params in (("fixed", FIXED_EPS), ("spread", SPREAD_C)):
                for p in params:
                    eps = p if rule == "fixed" else sd / p
                    shot_rows.append(dict(dataset=name, resample=ri, kernel=k, rule=rule, param=p, eps=eps,
                                          shots=int(np.ceil(max(mk, 1e-12) / eps ** 2)),
                                          shots_worstcase=int(np.ceil(1 / (4 * eps ** 2)))))
        else:
            R = feats[:m]
            for rule, params in (("fixed", FIXED_EPS), ("spread", SPREAD_C)):
                for p in params:
                    eps = p if rule == "fixed" else sd / p
                    shot_rows.append(dict(dataset=name, resample=ri, kernel=k, rule=rule, param=p, eps=eps,
                                          shots=mc_shots_projected(R, gam, eps, rng), shots_worstcase=np.nan))
    return rows, geo_rows, shot_rows


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
    ap.add_argument("--extra-baselines", action="store_true")
    ap.add_argument("--n-max", type=int, default=4, help="max features/qubits (PCA components)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also write mds_geometry.csv (geometric difference, CKA, KTA) and mds_shots.csv")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.quick:
        a.datasets = a.datasets[:3]
        a.folds, a.repeats = 2, 1

    jobs, info = [], []
    for name in a.datasets:
        try:
            X, y, inf = load_full(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            info.append(dict(dataset=name, status=f"skipped: {e}"))
            continue
        inf["status"] = "used"
        info.append(inf)
        print(f"loaded {name}: {inf['n_used']} of {inf['n_full']} points, {inf['n_features']} features, "
              f"{inf['n_classes']} classes")
        for ri, (tr, te) in enumerate(resamples(y, a.folds, a.repeats)):
            jobs.append((name, ri, X[tr], X[te], y[tr], y[te]))
    pd.DataFrame(info).to_csv(out / "mds_dataset_info.csv", index=False)
    if not jobs:
        raise SystemExit("no dataset loaded")
    print(f"{len(jobs)} jobs ({len(set(j[0] for j in jobs))} datasets x resamples); running with n_jobs={a.n_jobs}")
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(
        delayed(evaluate_split)(*j, a.extra_baselines, a.n_max, a.diagnostics) for j in jobs)
    if a.diagnostics:
        pd.DataFrame([r for rs in res for r in rs[1]]).to_csv(out / "mds_geometry.csv", index=False)
        pd.DataFrame([r for rs in res for r in rs[2]]).to_csv(out / "mds_shots.csv", index=False)
        res = [rs[0] for rs in res]
    cells = pd.DataFrame([r for rs in res for r in rs])
    cells.to_csv(out / "mds_cells.csv", index=False)
    print(f"compute time {time.time() - t0:.0f}s")

    # ---------------- statistics
    ratio = 1.0 / (a.folds - 1)  # n_test / n_train for k-fold
    allst = []
    for metric in ("test_acc", "test_bacc", "test_f1"):
        piv = cells.pivot_table(index=["dataset", "resample"], columns="kernel", values=metric)
        for k in [c for c in piv.columns if c != "classical/rbf"]:
            diff = (piv[k] - piv["classical/rbf"]).unstack("dataset")
            allst.append(dict(metric=metric, kernel=k, dataset="POOLED",
                              **corrected(diff.mean(axis=1).values, 1.0, ratio)))
            for ds in diff.columns:
                allst.append(dict(metric=metric, kernel=k, dataset=ds, **corrected(diff[ds].values, 1.0, ratio)))
    st_all = pd.DataFrame(allst)
    st_all.to_csv(out / "mds_paired_vs_rbf.csv", index=False)
    st = st_all[st_all.metric == "test_acc"]

    means = cells.groupby(["dataset", "kernel"]).test_acc.mean().unstack()
    names = [k for k in means.columns if k != "classical/rbf_scale"]
    fr = stats.friedmanchisquare(*[means[k].values for k in names])
    ranks = means[names].rank(axis=1, ascending=False).mean().sort_values()
    pv = {k: stats.wilcoxon(means[k].values, means["classical/rbf"].values).pvalue for k in names if k != "classical/rbf"}
    order = sorted(pv, key=pv.get)
    holm, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * pv[k]))
        holm[k] = run
    ge = cells[cells.kernel.str.contains("fidelity|projected|rbf$|poly2")].groupby("kernel").agg(
        at_lower_edge=("at_lower_edge", "mean"), at_upper_edge=("at_upper_edge", "mean"), median_s=("s_star", "median"))

    lines = [f"DATASETS USED: {', '.join(sorted(set(cells.dataset)))}",
             "SKIPPED: " + ("; ".join(f"{r['dataset']} ({r['status']})" for r in info if r['status'] != 'used') or "none"),
             f"DESIGN: cap {a.cap}, repeated stratified {a.folds}-fold x {a.repeats} = {a.folds * a.repeats} resamples, "
             f"n_test/n_train = {ratio:.3f}",
             f"TUNING: tie rule = {tie_rule()}, bandwidth grid s = {S_WIDE}", "",
             "MEAN TEST ACCURACY PER KERNEL (mean of dataset means)", means.mean().round(4).sort_values(ascending=False).to_string(),
             "", f"FRIEDMAN across {len(means)} datasets, {len(names)} models: chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.4g}",
             "Average ranks (1 = best):", ranks.round(2).to_string(),
             "", "Holm-adjusted Wilcoxon vs tuned RBF:"] + [f"  {k}: {holm[k]:.4f}" for k in order]
    lines += ["", "POOLED PAIRED DIFFERENCE VS TUNED RBF (corrected resampled t; equivalence margin 0.02)",
              st[st.dataset == "POOLED"][["kernel", "mean", "lo", "hi", "p", "tost_p", "equivalent_95ci"]].round(4).to_string(index=False),
              "", "POOLED PAIRED DIFFERENCE VS TUNED RBF, BALANCED ACCURACY AND MACRO-F1",
              st_all[(st_all.dataset == "POOLED") & (st_all.metric != "test_acc")][["metric", "kernel", "mean", "lo", "hi", "equivalent_95ci"]].round(4).to_string(index=False),
              "", "GRID EDGES (wide grid 0.002 ... 2)", ge.round(3).to_string()]
    (out / "mds_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
