"""Positive control: can the protocol detect a quantum-kernel advantage when the labels are built to contain one, and
what does that advantage cost to load?

PRE-SPECIFIED DESIGN (fixed on 2026-10-31, before any result of this script was seen; written to relabel_design.txt)
Datasets: four real (breast_cancer, ionosphere, sonar, spambase; >= 30 features, capped at 500 points) and two synthetic
(synth_r16, synth_r64 of reduction_study.py; 500 points, 64 features).  Qubits n = 6 and 8 (n = 4 added after the main analysis, to match the main study).
Features: standardise, PCA to n components and min-max to [0, pi], fitted once on the whole dataset; these n features are
the new dataset.  The standard protocol then re-applies its own train-only preprocessing in every resample.
Engineered labels (Huang et al. 2021, Nat. Commun. 12:2631): labelling kernel K_Q = IQP or ZZ fidelity kernel at the
unscaled bandwidth s = 1; classical reference K_C = RBF with the default bandwidth gamma = 1/(n Var X); both
trace-normalised to Tr = N.  With S = sqrt(K_Q) and M = S (K_C + lam I)^-1 S (lam = 0.01), v = top eigenvector of M,
the label of point i is 1 if (S v)_i > median(S v), else 0 (balanced classes).  g_label = sqrt(top eigenvalue of M).
Protocol: exactly the main study (repeated stratified 10-fold x 2 = 20 resamples, shared grid of 12 bandwidths x 4 C,
inner 3-fold CV, tie rule from QK_TIE): angle/IQP/ZZ fidelity and projected kernels tuned on the grid, tuned RBF,
default-bandwidth RBF, random forest and gradient boosting (same grids as the main study), and the labelling kernel
fixed at s = 1 with C tuned ('oracle': it knows the bandwidth used to make the labels).
Endpoints (accuracy; corrected resampled t, 95% CI), for each labelling kernel and n:
  E1  pooled difference of the labelling kernel, tuned on the grid, to the tuned RBF kernel; ADVANTAGE DETECTED if the
      lower 95% bound > 0.  Also per dataset and for real datasets only.
  E2  the same for the oracle (s = 1): how much advantage exists when the bandwidth is known.
  E3  E2 - E1: how much of the engineered advantage the shared tuning grid keeps.
  E4  the other quantum kernels against the tuned RBF kernel (does a cheaper encoding pick up the advantage?).
  Cost: two-qubit gates per fidelity circuit of each encoding at n (gate_counts.csv, routed, optimisation level 3).
  Diagnostics: geometric difference g(K_RBF,tuned || K) on the training block (lam = 0.01) for the tuned and oracle
  labelling kernel, and the tuned bandwidth of each kernel.
Expectation stated in advance: a clear advantage for the oracle; part of it kept by tuning, because s = 1 is on the grid;
little or none for the angle kernels, which are stationary and close to the RBF kernel.

Outputs (--out): rel_labels.csv (label summary per labelling kernel, n, dataset), rel_cells.csv, rel_geometry.csv,
rel_paired.csv, rel_report.txt, relabel_design.txt
Usage:  python relabel_study.py --out results_v2 --n-jobs 10
        python relabel_study.py --out test_out --quick
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from datasets_cv import BASE_SEED, DatasetSkip, load_full, preprocess, resamples
from extended_study import S_WIDE, corrected, geo_diff, psd_sqrt, trace_norm
from gram_spread import tie_rule, tune
from qkcommon import KINDS, bloch, fidelity_gram, projected_gamma, projected_gram, states

REAL = ["breast_cancer", "ionosphere", "sonar", "spambase"]
SYNTH = ["synth_r16", "synth_r64"]
LABELLERS = ["iqp", "zz"]
N_QUBITS = [4, 6, 8]   # n = 4 added after the main analysis (matches the main study); 6 and 8 unchanged
S_LABEL = 1.0
LAM = 0.01


def raw_dataset(name, cap, cache, csv_dir):
    if name.startswith("synth_r"):
        from sklearn import datasets as skd
        return skd.make_classification(n_samples=500, n_features=64, n_informative=int(name[7:]), n_redundant=0,
                                       n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=2.0, flip_y=0.02,
                                       shuffle=True, random_state=BASE_SEED)[0]
    return load_full(name, cap, cache, csv_dir)[0]


def global_features(X, n):
    Z = StandardScaler().fit_transform(X)
    if Z.shape[1] > n:
        Z = PCA(n_components=n, random_state=BASE_SEED).fit_transform(Z)
    return MinMaxScaler(feature_range=(0.0, np.pi)).fit_transform(Z)


def engineered_labels(Z, kind, n):
    psi = states(Z, S_LABEL, kind, n)
    KQ = trace_norm(fidelity_gram(psi, psi))
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    KC = trace_norm(np.exp(-d2 / (n * Z.var())))
    S = psd_sqrt(KQ)
    M = S @ np.linalg.solve(KC + LAM * np.eye(len(Z)), S)
    w, V = np.linalg.eigh((M + M.T) / 2)
    f = S @ V[:, -1]
    y = (f > np.median(f)).astype(int)
    return y, float(np.sqrt(max(w[-1], 0.0)))


def evaluate_split(labeller, n, name, ri, Xtr_raw, Xte_raw, ytr, yte):
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw, n_max=n)
    ntr = len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + ri
    rows, geo = [], []
    base = dict(labeller=labeller, n_qubits=n, dataset=name, resample=ri)
    kept = {}

    def rec(kernel, K, s, C, cv):
        pred = SVC(kernel="precomputed", C=C).fit(K[:ntr, :ntr], ytr).predict(K[ntr:, :ntr])
        rows.append(dict(**base, kernel=kernel, s_star=s, C_star=C, cv_acc=cv, test_acc=float((pred == yte).mean())))
        kept[kernel] = K[:ntr, :ntr]

    for kind in KINDS:
        Kf, Kp = {}, {}
        for s in S_WIDE:
            psi = states(X, s, kind, n)
            Kf[s] = fidelity_gram(psi, psi)
            F = bloch(psi, n).reshape(len(X), -1)
            Kp[s] = projected_gram(F, F, projected_gamma(F[:ntr]))
        for cons, Kd in (("fidelity", Kf), ("projected", Kp)):
            cv, s, C = tune(Kd, ytr, ntr, seed)
            rec(f"{kind}/{cons}", Kd[s], s, C, cv)
        if kind == labeller:
            K1 = Kf[S_LABEL] if S_LABEL in Kf else fidelity_gram(states(X, S_LABEL, kind, n), states(X, S_LABEL, kind, n))
            cv, _, C = tune({S_LABEL: K1}, ytr, ntr, seed)
            rec(f"{kind}/fidelity_oracle_s1", K1, S_LABEL, C, cv)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    cv, s, C = tune({s: np.exp(-(s ** 2) * d2) for s in S_WIDE}, ytr, ntr, seed)
    rec("classical/rbf", np.exp(-(s ** 2) * d2), s, C, cv)
    g = 1.0 / (n * Xtr.var())
    cv, _, C = tune({"scale": np.exp(-g * d2)}, ytr, ntr, seed)
    rec("classical/rbf_scale", np.exp(-g * d2), g, C, cv)
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)
    for k, est, grid in (("classical/rf", RandomForestClassifier(n_estimators=200, random_state=seed),
                          {"max_depth": [2, 4, 8, None], "min_samples_leaf": [1, 5]}),
                         ("classical/gbm", GradientBoostingClassifier(random_state=seed),
                          {"n_estimators": [50, 100, 200], "max_depth": [1, 2, 3]})):
        gs = GridSearchCV(est, grid, cv=folds).fit(Xtr, ytr)
        rows.append(dict(**base, kernel=k, s_star=np.nan, C_star=np.nan, cv_acc=gs.best_score_,
                         test_acc=float(gs.score(Xte, yte))))
    Kref = kept["classical/rbf"]
    for k in (f"{labeller}/fidelity", f"{labeller}/fidelity_oracle_s1", "angle/fidelity"):
        geo.append(dict(**base, kernel=k, g_rbf_to_k=geo_diff(Kref, kept[k], LAM)))
    return rows, geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="+", default=REAL + SYNTH)
    ap.add_argument("--n", nargs="+", type=int, default=N_QUBITS)
    ap.add_argument("--labellers", nargs="+", default=LABELLERS)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.quick:
        a.datasets, a.n, a.folds, a.repeats = ["breast_cancer", "synth_r16"], [6], 3, 1
    (out / "relabel_design.txt").write_text(__doc__.split("Outputs")[0] +
                                            f"\nRUN: tie rule = {tie_rule()}, grid = {S_WIDE}, datasets = {a.datasets}, "
                                            f"n = {a.n}, labellers = {a.labellers}, {a.folds}-fold x {a.repeats}\n")
    jobs, lab = [], []
    for name in a.datasets:
        try:
            Xraw = raw_dataset(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            continue
        for n in a.n:
            Z = global_features(Xraw, n)
            for labeller in a.labellers:
                y, g_label = engineered_labels(Z, labeller, n)
                lab.append(dict(labeller=labeller, n_qubits=n, dataset=name, points=len(y), share_ones=y.mean(),
                                g_label=g_label))
                for ri, (tr, te) in enumerate(resamples(y, a.folds, a.repeats)):
                    jobs.append((labeller, n, name, ri, Z[tr], Z[te], y[tr], y[te]))
    lab = pd.DataFrame(lab)
    lab.to_csv(out / "rel_labels.csv", index=False)
    print(f"{len(jobs)} jobs, n_jobs={a.n_jobs}")
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(evaluate_split)(*j) for j in jobs)
    cells = pd.DataFrame([r for rs in res for r in rs[0]])
    geo = pd.DataFrame([r for rs in res for r in rs[1]])
    cells.to_csv(out / "rel_cells.csv", index=False)
    geo.to_csv(out / "rel_geometry.csv", index=False)
    print(f"compute time {time.time() - t0:.0f}s")

    ratio = 1.0 / (a.folds - 1)
    st = []
    for (labeller, n), grp in cells.groupby(["labeller", "n_qubits"]):
        piv = grp.pivot_table(index=["dataset", "resample"], columns="kernel", values="test_acc")
        for k in [c for c in piv.columns if c != "classical/rbf"]:
            diff = (piv[k] - piv["classical/rbf"]).unstack("dataset")
            for scope, cols in (("POOLED", list(diff.columns)), ("POOLED_REAL", [c for c in diff.columns if c in REAL])):
                if cols:
                    r = corrected(diff[cols].mean(axis=1).values, 1.0, ratio)
                    st.append(dict(labeller=labeller, n_qubits=n, kernel=k, dataset=scope, **r,
                                   advantage=bool(r["lo"] > 0)))
            for ds in diff.columns:
                r = corrected(diff[ds].dropna().values, 1.0, ratio)
                st.append(dict(labeller=labeller, n_qubits=n, kernel=k, dataset=ds, **r, advantage=bool(r["lo"] > 0)))
        # E3: oracle - tuned labelling kernel
        d = (piv[f"{labeller}/fidelity_oracle_s1"] - piv[f"{labeller}/fidelity"]).unstack("dataset")
        r = corrected(d.mean(axis=1).values, 1.0, ratio)
        st.append(dict(labeller=labeller, n_qubits=n, kernel="E3: oracle minus tuned", dataset="POOLED", **r,
                       advantage=None))
    st = pd.DataFrame(st)
    st.to_csv(out / "rel_paired.csv", index=False)

    gates = None
    gc_file = out / "gate_counts.csv"
    if gc_file.exists():
        gc = pd.read_csv(gc_file)
        gates = gc[(gc["mode"] == "routed") & (gc.opt_level == 3) & (gc.circuit == "fidelity") & gc.n.isin(a.n)] \
            .pivot_table(index="embedding", columns="n", values="twoq_median")
    pooled = st[st.dataset.isin(["POOLED", "POOLED_REAL"])]
    sstar = cells.groupby(["labeller", "n_qubits", "kernel"]).s_star.median().unstack("kernel")
    gmed = geo.groupby(["labeller", "n_qubits", "kernel"]).g_rbf_to_k.median().unstack("kernel")
    per_ds = st[~st.dataset.isin(["POOLED", "POOLED_REAL"]) & st.kernel.str.endswith("/fidelity")
                & st.apply(lambda r: r.kernel.startswith(r.labeller), axis=1)]
    lines = [f"tie rule = {tie_rule()}, grid = {S_WIDE}", "", "ENGINEERED LABELS", lab.round(3).to_string(index=False), "",
             "PAIRED ACCURACY DIFFERENCE TO THE TUNED RBF KERNEL (advantage if lo > 0)",
             pooled[["labeller", "n_qubits", "dataset", "kernel", "mean", "lo", "hi", "advantage"]].round(4)
             .to_string(index=False), "",
             "PER DATASET, LABELLING KERNEL TUNED ON THE GRID",
             per_ds[["labeller", "n_qubits", "dataset", "mean", "lo", "hi", "advantage"]].round(4).to_string(index=False),
             "", "MEDIAN TUNED BANDWIDTH", sstar.round(3).to_string(), "",
             "MEDIAN GEOMETRIC DIFFERENCE g(K_RBF,tuned || K), lam = 0.01", gmed.round(2).to_string()]
    if gates is not None:
        lines += ["", "TWO-QUBIT GATES PER FIDELITY CIRCUIT (routed, opt. level 3)", gates.to_string()]
    (out / "rel_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
