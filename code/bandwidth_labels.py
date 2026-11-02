"""Does the surrogate result depend on the labelling bandwidth?  Also: labels from a kernel outside
the tested set.

DESIGN (added after the main analysis, 2026-11-01; nothing else changed)
The positive control built its labels from the IQP or ZZ fidelity kernel at s = 1, a value on the tuning grid.  Here the
labels are built in the same way (relabel_study.py: same datasets, preprocessing, lam = 0.01, median threshold) but at
labelling bandwidths s_L in {0.15, 0.3, 0.7, 1.0}; 0.15, 0.3 and 0.7 are NOT on the grid.  In addition (F2), labels are
built from IQP circuits with L = 1 and L = 3 layers at s_L = 1, kernels that none of the tested models (all L = 2) is.
Per cell: the IQP/ZZ fidelity kernel of the tested family (L = 2) tuned on the full grid, the same kernel at the labelling
bandwidth (oracle; for L = 1, 3 labels the L = 2 kernel at s_L), the tuned RBF kernel, the tuned angle fidelity kernel,
and the three surrogates of surrogate_study.py (metric_map s = s*_RBF / kappa, rbf_map s = 2 s*_RBF, angle s = s*_angle-F)
applied to the L = 2 kernel.  n in {4, 6}; 20 resamples.
Endpoints (pooled over datasets, corrected resampled t): full-grid kernel - tuned RBF (advantage detected if lo > 0);
oracle - full-grid kernel; each surrogate - full grid (non-inferior if lo > -0.02) and - tuned RBF (advantage kept if
lo > 0); share of cells whose full-grid bandwidth equals the labelling bandwidth.

Outputs (--out): bwl_labels.csv, bwl_cells.csv, bwl_paired.csv, bwl_report.txt
Usage:  python bandwidth_labels.py --out results_v2 --n-jobs 10
        python bandwidth_labels.py --out test_out --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.svm import SVC

from datasets_cv import BASE_SEED, DatasetSkip, preprocess, resamples
from extended_study import DELTA, S_WIDE, corrected, psd_sqrt, trace_norm
from gram_spread import tie_rule, tune
from qkcommon import fidelity_gram, states
from relabel_study import LAM, REAL, SYNTH, global_features, raw_dataset
from surrogate_study import RBF_FACTOR, kappa2

S_LABELS = [0.15, 0.3, 0.7, 1.0]
# (setting name, encoding, layers of the labelling circuit, labelling bandwidths)
SETTINGS = [("iqp", "iqp", 2, S_LABELS), ("zz", "zz", 2, S_LABELS), ("iqp_L1", "iqp", 1, [1.0]), ("iqp_L3", "iqp", 3, [1.0])]


def labels(Z, kind, n, s, L):
    psi = states(Z, s, kind, n, L=L)
    KQ = trace_norm(fidelity_gram(psi, psi))
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    KC = trace_norm(np.exp(-d2 / (n * Z.var())))
    S = psd_sqrt(KQ)
    M = S @ np.linalg.solve(KC + LAM * np.eye(len(Z)), S)
    w, V = np.linalg.eigh((M + M.T) / 2)
    f = S @ V[:, -1]
    return (f > np.median(f)).astype(int), float(np.sqrt(max(w[-1], 0.0)))


def one(setting, kind, s_lab, n, name, ri, Xtr_raw, Xte_raw, ytr, yte):
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw, n_max=n)
    ntr = len(ytr)
    X = np.vstack([Xtr, Xte])
    seed = BASE_SEED + ri
    rows = []

    def fid(s, k=kind):
        psi = states(X, s, k, n)
        return fidelity_gram(psi, psi)

    def rec(model, Kd):
        cv, s, C = tune(Kd, ytr, ntr, seed)
        K = Kd[s]
        acc = float((SVC(kernel="precomputed", C=C).fit(K[:ntr, :ntr], ytr).predict(K[ntr:, :ntr]) == yte).mean())
        rows.append(dict(setting=setting, s_label=s_lab, n_qubits=n, dataset=name, resample=ri, model=model, s_used=s,
                         test_acc=acc))
        return s

    q = {s: fid(s) for s in S_WIDE}
    rec("full_grid", q)
    rec("oracle", {s_lab: q[s_lab] if s_lab in q else fid(s_lab)})
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    s_rbf = rec("rbf", {s: np.exp(-(s ** 2) * d2) for s in S_WIDE})
    s_ang = rec("angle", {s: fid(s, "angle") for s in S_WIDE})
    for meth, s in (("metric_map", s_rbf / np.sqrt(kappa2(kind, n))), ("rbf_map", RBF_FACTOR * s_rbf), ("angle_transfer", s_ang)):
        rec(meth, {s: q[s] if s in q else fid(s)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--datasets", nargs="+", default=REAL + SYNTH)
    ap.add_argument("--n", nargs="+", type=int, default=[4, 6])
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = SETTINGS
    if a.quick:
        a.datasets, a.n, a.folds, a.repeats = ["breast_cancer", "synth_r16"], [4], 3, 1
        settings = [("iqp", "iqp", 2, [0.3, 1.0]), ("iqp_L3", "iqp", 3, [1.0])]
    jobs, lab = [], []
    for name in a.datasets:
        try:
            Xraw = raw_dataset(name, a.cap, a.cache, a.csv_dir)
        except DatasetSkip as e:
            print(f"SKIPPED {name}: {e}")
            continue
        for n in a.n:
            Z = global_features(Xraw, n)
            for setting, kind, L, sl in settings:
                for s_lab in sl:
                    y, g = labels(Z, kind, n, s_lab, L)
                    lab.append(dict(setting=setting, layers=L, s_label=s_lab, n_qubits=n, dataset=name, g_label=g))
                    for ri, (tr, te) in enumerate(resamples(y, a.folds, a.repeats)):
                        jobs.append((setting, kind, s_lab, n, name, ri, Z[tr], Z[te], y[tr], y[te]))
    pd.DataFrame(lab).to_csv(out / "bwl_labels.csv", index=False)
    print(f"{len(jobs)} jobs, tie rule {tie_rule()}")
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(one)(*j) for j in jobs)
    cells = pd.DataFrame([r for rs in res for r in rs])
    cells.to_csv(out / "bwl_cells.csv", index=False)

    ratio = 1.0 / (a.folds - 1)
    st = []
    for (setting, s_lab, n), g in cells.groupby(["setting", "s_label", "n_qubits"]):
        piv = g.pivot_table(index=["dataset", "resample"], columns="model", values="test_acc")
        comps = [("full_grid", "rbf"), ("oracle", "full_grid"), ("angle", "rbf")] + \
                [(m, ref) for m in ("metric_map", "rbf_map", "angle_transfer") for ref in ("full_grid", "rbf")]
        for m, ref in comps:
            diff = (piv[m] - piv[ref]).unstack("dataset")
            r = corrected(diff.mean(axis=1).values, 1.0, ratio)
            st.append(dict(setting=setting, s_label=s_lab, n_qubits=n, model=m, reference=ref, **r,
                           advantage=bool(r["lo"] > 0), noninferior=bool(r["lo"] > -DELTA)))
    st = pd.DataFrame(st)
    st.to_csv(out / "bwl_paired.csv", index=False)
    fg = cells[cells.model == "full_grid"]
    hit = fg.assign(hit=np.isclose(fg.s_used, fg.s_label)).groupby(["setting", "s_label", "n_qubits"]).agg(
        median_s_full_grid=("s_used", "median"), share_at_label_s=("hit", "mean"))
    sm = cells[cells.model.isin(["metric_map", "rbf_map", "angle_transfer"])].pivot_table(
        index=["setting", "s_label", "n_qubits"], columns="model", values="s_used", aggfunc="median")
    key = st.set_index(["setting", "s_label", "n_qubits", "model", "reference"])

    def cell(m, ref):
        r = key.xs((m, ref), level=("model", "reference"))
        return r["mean"].round(3).astype(str) + " [" + r.lo.round(3).astype(str) + ", " + r.hi.round(3).astype(str) + "]"
    summ = pd.DataFrame({"full grid - RBF": cell("full_grid", "rbf"), "oracle - full grid": cell("oracle", "full_grid"),
                         "metric_map - full grid": cell("metric_map", "full_grid"),
                         "metric_map - RBF": cell("metric_map", "rbf"), "angle - RBF": cell("angle", "rbf")})
    lines = [f"Labelling bandwidths {S_LABELS} (0.15, 0.3, 0.7 off the grid); IQP labels with L = 1 and 3 at s = 1",
             "", "SUMMARY (pooled over datasets, 95% interval)", summ.to_string(), "",
             "ADVANTAGE DETECTED (full grid - RBF, lo > 0) AND KEPT BY EACH SURROGATE (surrogate - RBF, lo > 0)",
             st[st.reference == "rbf"].pivot_table(index=["setting", "s_label", "n_qubits"], columns="model",
                                                   values="advantage").to_string(), "",
             "SURROGATE NON-INFERIOR TO FULL GRID (lo > -0.02)",
             st[(st.reference == "full_grid") & (st.model != "oracle")].pivot_table(
                 index=["setting", "s_label", "n_qubits"], columns="model", values="noninferior").to_string(), "",
             "BANDWIDTHS: full-grid median, share of cells at the labelling bandwidth, surrogate medians",
             pd.concat([hit, sm], axis=1).round(3).to_string(), "",
             "LABELS: median geometric difference used to build them",
             pd.DataFrame(lab).groupby(["setting", "s_label", "n_qubits"]).g_label.median().round(2).to_string()]
    (out / "bwl_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
