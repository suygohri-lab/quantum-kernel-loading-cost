"""Additional analysis of the main study (reads the outputs of multi_dataset_study.py; computes nothing new on data).

1. Per-dataset equivalence: for every quantum kernel and dataset, the corrected-t 95% interval of the paired
   difference to the tuned RBF (already in mds_paired_vs_rbf.csv) is classified as
     equivalent    interval inside (-delta, +delta)
     lower/higher  interval entirely below / above 0
     inconclusive  otherwise
   (a kernel can be both equivalent and lower).  No multiplicity correction across datasets: counts are descriptive.
2. Paired differences of every quantum kernel to other classical references present in mds_cells.csv
   (default-gamma RBF, gradient boosting, random forest, kNN, MLP), pooled and per dataset, same statistics.
3. Geometry summary from mds_geometry.csv: geometric difference g(K_RBF || K_Q) (Huang et al. 2021) per lambda,
   CKA to the tuned RBF and kernel-target alignment, as median over cells, with sqrt(N_train) for reference
   (Huang et al.: a prediction advantage of K_Q over K_RBF needs g of order sqrt(N)).

4. Post-hoc pooled analysis of the reduction study (H1), from reduction_cells.csv: for every reduction setting that
   exists on at least two datasets, the paired accuracy change of the amplitude kernel against the unreduced amplitude
   kernel is averaged over datasets within each resample, then tested with the corrected resampled t (non-inferiority
   at -delta).  Pooled over all datasets and over the three real datasets only.  This is a post-hoc re-analysis; the
   pre-specified per-configuration analysis in reduction_table.csv stays primary.

Outputs (--results): v2_per_dataset_equivalence.csv, v2_equivalence_counts.csv, v2_paired_vs_refs.csv,
                     v2_geometry_summary.csv, v2_reduction_pooled.csv, v2_report.txt
Usage:
  python v2_analysis.py --results results_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extended_study import DELTA, corrected

QK = ["angle/fidelity", "angle/projected", "iqp/fidelity", "iqp/projected", "zz/fidelity", "zz/projected"]
OTHER_REFS = ["classical/rbf_scale", "classical/gbm", "classical/rf", "classical/knn", "classical/mlp"]


def classify(r, delta=DELTA):
    eq = (r.lo > -delta) and (r.hi < delta)
    direction = "lower" if r.hi < 0 else ("higher" if r.lo > 0 else "no difference shown")
    return pd.Series(dict(equivalent=eq, direction=direction,
                          verdict="equivalent" if eq else ("lower" if r.hi < 0 else ("higher" if r.lo > 0 else "inconclusive"))))


REAL = ["breast_cancer", "wine", "digits38"]
CEILING = 0.99


def reduction_pooled(c, ratio, delta=DELTA):
    c = c[~c.exploratory_post_hoc.astype(bool)].copy()
    base = c[(c.reduction == "none") & (c["sample"] == "full")].set_index(["dataset", "resample"])
    keys = ["reduction", "k", "sample", "frac"]
    c["setting"] = c.apply(lambda r: (f"{r.reduction} k={r.k}" if r.reduction != "none" else "no dimension reduction")
                           + ("" if r["sample"] == "full" else f", {r['sample']} {r.frac:g} of points"), axis=1)
    c = c[~((c.reduction == "none") & (c["sample"] == "full"))]
    # sample-only reductions are pooled across datasets regardless of the dataset's own dimension k
    c["group"] = c.apply(lambda r: r.setting.replace(f" k={r.k}", "") if r.reduction == "none" else r.setting, axis=1)
    rows = []
    for g, sub in c.groupby("group"):
        for scope, ds in (("all datasets", sorted(set(sub.dataset))), ("real datasets", [d for d in REAL if d in set(sub.dataset)])):
            if len(ds) < 2:
                continue
            s2 = sub[sub.dataset.isin(ds)].set_index(["dataset", "resample"])
            d = (s2.acc_amp - base.acc_amp.reindex(s2.index)).unstack("dataset")
            st = corrected(d.mean(axis=1).values, 1.0, ratio, delta)
            gr = (s2.groupby("dataset").twoq_total.mean() / base.groupby("dataset").twoq_total.mean().reindex(ds)).replace(0, np.nan)
            rows.append(dict(setting=g, scope=scope, n_datasets=len(ds), datasets=" ".join(ds), mean=st["mean"], lo=st["lo"],
                             hi=st["hi"], noninferior=bool(st["lo"] > -delta), lower=bool(st["hi"] < 0),
                             median_gate_saving=float(np.nanmedian(1.0 / gr.values)) if gr.notna().any() else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
    res = Path(a.results)
    cells = pd.read_csv(res / "mds_cells.csv")
    st = pd.read_csv(res / "mds_paired_vs_rbf.csv")
    ratio = 1.0 / (a.folds - 1)
    lines = []

    # 1. per-dataset equivalence vs tuned RBF
    pdq = st[(st.dataset != "POOLED") & st.kernel.isin(QK)].copy()
    pdq = pd.concat([pdq.reset_index(drop=True), pdq.apply(classify, axis=1).reset_index(drop=True)], axis=1)
    pdq.to_csv(res / "v2_per_dataset_equivalence.csv", index=False)
    acc = pdq[pdq.metric == "test_acc"]
    counts = acc.groupby("kernel").agg(datasets=("dataset", "size"), equivalent=("equivalent", "sum"),
                                       lower=("direction", lambda x: int((x == "lower").sum())),
                                       higher=("direction", lambda x: int((x == "higher").sum())))
    counts["not_shown_equivalent"] = counts.datasets - counts.equivalent
    counts = counts.reindex(QK)
    counts.to_csv(res / "v2_equivalence_counts.csv")
    lines += [f"1. PER-DATASET EQUIVALENCE TO TUNED RBF (accuracy, margin +/-{DELTA}, corrected-t 95% CI, no correction across datasets)",
              counts.to_string(), "",
              "Datasets where a quantum kernel was NOT shown equivalent (accuracy):",
              acc[~acc.equivalent][["kernel", "dataset", "mean", "lo", "hi", "verdict"]].round(4).to_string(index=False)
              if (~acc.equivalent).any() else "  none", ""]

    # 2. other classical references
    refs = [r for r in OTHER_REFS if r in set(cells.kernel)]
    rows = []
    for metric in ("test_acc", "test_bacc", "test_f1"):
        piv = cells.pivot_table(index=["dataset", "resample"], columns="kernel", values=metric)
        for ref in refs:
            for k in QK:
                d = (piv[k] - piv[ref]).unstack("dataset")
                rows.append(dict(metric=metric, reference=ref, kernel=k, dataset="POOLED",
                                 **corrected(d.mean(axis=1).values, 1.0, ratio)))
                for ds in d.columns:
                    rows.append(dict(metric=metric, reference=ref, kernel=k, dataset=ds, **corrected(d[ds].values, 1.0, ratio)))
    pr = pd.DataFrame(rows)
    pr.to_csv(res / "v2_paired_vs_refs.csv", index=False)
    means = cells.groupby(["dataset", "kernel"]).test_acc.mean().unstack().mean().sort_values(ascending=False)
    lines += ["2. MEAN TEST ACCURACY OF ALL MODELS (mean of dataset means)", means.round(4).to_string(), ""]
    if len(pr):
        p = pr[(pr.metric == "test_acc") & (pr.dataset == "POOLED")]
        lines += ["POOLED PAIRED DIFFERENCE, QUANTUM KERNEL MINUS OTHER CLASSICAL REFERENCE (accuracy)",
                  p[["reference", "kernel", "mean", "lo", "hi", "p", "equivalent_95ci"]].round(4).to_string(index=False), ""]
    else:
        lines += ["(no extra classical references in mds_cells.csv; run multi_dataset_study.py with --extra-baselines)", ""]

    # 3. geometry
    gf = res / "mds_geometry.csv"
    if gf.exists():
        g = pd.read_csv(gf)
        info = pd.read_csv(res / "mds_dataset_info.csv")
        ntr = (info.set_index("dataset").n_used * (1 - 1.0 / a.folds))
        g["sqrt_ntrain"] = np.sqrt(g.dataset.map(ntr))
        gcols = [c for c in g.columns if c.startswith(("g_", "cka", "kta"))] + ["sqrt_ntrain"]
        summ = g.groupby("kernel")[gcols].median()
        summ.to_csv(res / "v2_geometry_summary.csv")
        g.groupby(["dataset", "kernel"])[gcols].median().to_csv(res / "v2_geometry_per_dataset.csv")
        lines += ["3. GEOMETRY ON THE TRAINING BLOCK (median over dataset-resample cells)",
                  "g_rbf_to_k = g(K_RBF || K_Q): can the tuned RBF reproduce the quantum kernel? Compare with sqrt(N_train).",
                  summ.round(3).to_string(), ""]
    else:
        lines += ["(mds_geometry.csv not found; run multi_dataset_study.py with --diagnostics)", ""]
    # 5. ceiling sensitivity: drop datasets on which the tuned RBF reaches >= CEILING mean accuracy
    rbf_mean = cells[cells.kernel == "classical/rbf"].groupby("dataset").test_acc.mean()
    ceil_ds = sorted(rbf_mean[rbf_mean >= CEILING].index)
    piv = cells.pivot_table(index=["dataset", "resample"], columns="kernel", values="test_acc")
    crow = []
    for k in QK + ["amplitude/fidelity"]:
        d = (piv[k] - piv["classical/rbf"]).unstack("dataset")
        d = d[[c for c in d.columns if c not in ceil_ds]]
        crow.append(dict(kernel=k, n_datasets=d.shape[1], excluded=" ".join(ceil_ds),
                         **corrected(d.mean(axis=1).values, 1.0, ratio)))
    cdf = pd.DataFrame(crow)
    cdf.to_csv(res / "v2_ceiling_sensitivity.csv", index=False)
    lines += [f"5. CEILING SENSITIVITY: pooled difference to tuned RBF without datasets where it reaches >= {CEILING} "
              f"(excluded: {', '.join(ceil_ds)})",
              cdf[["kernel", "n_datasets", "mean", "lo", "hi", "equivalent_95ci"]].round(4).to_string(index=False), ""]

    # 4. pooled reduction study (post hoc)
    rf = res / "reduction_cells.csv"
    if rf.exists():
        rp = reduction_pooled(pd.read_csv(rf), ratio)
        rp.to_csv(res / "v2_reduction_pooled.csv", index=False)
        lines += ["4. REDUCTION BEFORE LOADING, POOLED OVER DATASETS (post hoc; primary analysis is per configuration)",
                  "delta_acc = reduced minus unreduced amplitude kernel; non-inferior if lo > -" + f"{DELTA}",
                  rp.round(4).to_string(index=False), ""]
    (res / "v2_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
