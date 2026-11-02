"""The loading bill and what each lever does to it (one factor at a time from the main-study baseline).

Total loading cost of one quantum-kernel model
    = [(bandwidths tried) x (training circuits) + (test circuits)] x (shots per circuit) x (two-qubit gates per circuit)
(tuning needs the training block of the Gram matrix at every bandwidth, the test block only at the chosen one)
      \\_______________________ number of loads _______________________/   \\________ cost per load ________/

For every lever this script reports the cost before and after, the saving factor, and the paired accuracy change with
its 95% interval (corrected resampled t) where the design allows it; non-inferior = lower bound > -0.02.
Baseline: main study, n = 4, tuned on the full grid, fidelity construction unless stated, spread shot rule c = 2.

Sources (all in --results):
  encoding, construction : mds_cells.csv (accuracy, paired) and frontier.csv (gates, circuits, shots; M = 100)
  bandwidth search       : fb_gain.csv (unscaled s = 1 against full-grid tuning) and the grid size from design_v2.txt
  surrogate, Nystrom     : sur_paired.csv and sur_circuits.csv (surrogate_study.py), if present
  shots                  : shots_summary.csv (finite-shot study; means only, no paired interval)
  reduction (H1)         : v2_reduction_pooled.csv (pooled post-hoc analysis)

Outputs: lever_table.csv, lever_bill.csv, lever_report.txt
Usage:  python lever_table.py --results results_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extended_study import DELTA, corrected

NAMES = {"angle/fidelity": "Angle-F", "angle/projected": "Angle-P", "iqp/fidelity": "IQP-F", "iqp/projected": "IQP-P",
         "zz/fidelity": "ZZ-F", "zz/projected": "ZZ-P"}


def paired(cells, a, b, ratio):
    """Pooled paired accuracy change b - a over datasets (corrected resampled t)."""
    piv = cells.pivot_table(index=["dataset", "resample"], columns="kernel", values="test_acc")
    d = (piv[b] - piv[a]).unstack("dataset").mean(axis=1).values
    return corrected(d, 1.0, ratio)


def row(factor, lever, kernel, frm, to, c0, c1, unit, acc=None, source="", note=""):
    r = dict(factor=factor, lever=lever, kernel=kernel, from_=frm, to=to, cost_from=c0, cost_to=c1, cost_unit=unit,
             saving_factor=(c0 / c1) if (c1 and np.isfinite(c1) and c1 > 0) else np.inf,
             acc_change=np.nan, lo=np.nan, hi=np.nan, noninferior=None, source=source, note=note)
    if acc is not None:
        r.update(acc_change=acc["mean"], lo=acc.get("lo", np.nan), hi=acc.get("hi", np.nan))
        if np.isfinite(r["lo"]):
            r["noninferior"] = bool(r["lo"] > -DELTA)
    return r


def grid_size(res):
    f = res / "design_v2.txt"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith("s grid"):
                return len([v for v in line.split("=", 1)[1].split(",") if v.strip()])
    return 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
    res = Path(a.results)
    ratio = 1.0 / (a.folds - 1)
    cells = pd.read_csv(res / "mds_cells.csv")
    fr = pd.read_csv(res / "frontier.csv").set_index("kernel")
    S = grid_size(res)
    rows = []

    # the bill at the baseline (M = 100 training points, 50 test points, as in frontier.csv), per kernel
    bill = fr[["twoq_per_circuit", "circuits_per_gram", "shots_per_circuit"]].copy()
    bill["bandwidths_tried"] = S
    # tuning needs the training block of the Gram matrix at every bandwidth; the test block only at the chosen one
    cg = pd.read_csv(res / "circuits_per_gram.csv").set_index("construction")
    cons = pd.Series(bill.index.str.endswith("-F"), index=bill.index).map({True: "fidelity", False: "projected"})
    bill["circuits_train"] = cons.map(cg.circuits_train)
    bill["circuits_test"] = cons.map(cg.circuits_test)
    bill["circuits_model"] = bill.bandwidths_tried * bill.circuits_train + bill.circuits_test
    bill["tuning_factor"] = bill.circuits_model / bill.circuits_per_gram
    bill["loads"] = bill.circuits_model * bill.shots_per_circuit
    bill["twoq_gate_executions"] = bill.loads * bill.twoq_per_circuit
    bill.to_csv(res / "lever_bill.csv")

    # 1. cost per load: encoding (entangling -> angle), same construction
    for cons, tag in (("fidelity", "F"), ("projected", "P")):
        for emb in ("iqp", "zz"):
            k0, k1 = f"{emb}/{cons}", f"angle/{cons}"
            rows.append(row("gates per load", "encoding", f"{NAMES[k0]} -> {NAMES[k1]}", NAMES[k0], NAMES[k1],
                            fr.loc[NAMES[k0], "twoq_per_circuit"], fr.loc[NAMES[k1], "twoq_per_circuit"],
                            "two-qubit gates per circuit", paired(cells, k0, k1, ratio), "mds_cells.csv, frontier.csv"))
    # 2. number of loads: construction (fidelity -> projected), same encoding
    for emb in ("angle", "iqp", "zz"):
        k0, k1 = f"{emb}/fidelity", f"{emb}/projected"
        rows.append(row("circuits per Gram matrix", "construction", f"{NAMES[k0]} -> {NAMES[k1]}", NAMES[k0], NAMES[k1],
                        fr.loc[NAMES[k0], "circuits_per_gram"], fr.loc[NAMES[k1], "circuits_per_gram"],
                        "circuits per Gram matrix (M = 100)", paired(cells, k0, k1, ratio), "mds_cells.csv, frontier.csv",
                        note="shots per circuit also change, see lever_bill.csv"))
    # 3. bandwidths tried: skip the search (s = 1)
    fb = pd.read_csv(res / "fb_gain.csv")
    fb = fb[(fb.metric == "test_acc") & (fb.dataset == "POOLED")].set_index("kernel")
    for k in NAMES:
        if k in fb.index:
            r = fb.loc[k]   # mean = tuned - fixed  ->  change from skipping = fixed - tuned
            rows.append(row("bandwidths tried", "skip tuning (s = 1)", NAMES[k], f"grid of {S}", "s = 1", S, 1,
                            "Gram matrices", dict(mean=-r["mean"], lo=-r["hi"], hi=-r["lo"]), "fb_gain.csv"))
    # 4. surrogate tuning and Nystrom (if run)
    sp, sc = res / "sur_paired.csv", res / "sur_circuits_ref.csv"
    if sp.exists() and sc.exists():
        p = pd.read_csv(sp)
        p = p[(p.metric == "test_acc") & (p.dataset == "POOLED") & (p.reference == "full_grid")]
        c = pd.read_csv(sc).set_index(["kernel", "method"])
        for _, r in p.iterrows():
            base = c.loc[(r.kernel, "full_grid")]
            alt = c.loc[(r.kernel, r.method)]
            if r.method.startswith("nystrom"):
                rows.append(row("circuits per Gram matrix", f"Nystrom {r.method.split('_')[1]} landmarks", NAMES[r.kernel],
                                "exact Gram", r.method, base.gram_circuits, alt.gram_circuits,
                                "circuits per Gram matrix (M = 100, 50 test)", r, "sur_paired.csv, sur_circuits_ref.csv"))
            else:
                rows.append(row("bandwidths tried", f"surrogate tuning ({r.method})", NAMES[r.kernel], f"grid of {S}",
                                "1 Gram matrix", base.total_circuits, alt.total_circuits,
                                "circuits per tuned model (M = 100, 50 test)", r, "sur_paired.csv, sur_circuits_ref.csv"))
    # 5. shots per circuit
    sh = pd.read_csv(res / "shots_summary.csv")
    for k, g in sh.groupby("kernel"):
        g = g.set_index("shots")
        top = g.index.max()
        for sv in sorted(g.index):
            if sv == top:
                continue
            rows.append(row("shots per circuit", f"{top} -> {sv} shots", NAMES.get(k, k), top, sv, top, sv,
                            "shots per circuit", dict(mean=g.loc[sv, "acc_shots"] - g.loc[top, "acc_shots"]),
                            "shots_summary.csv", note="finite-shot study subset; mean only"))
    # 6. reduction before amplitude loading (H1, pooled)
    rp = pd.read_csv(res / "v2_reduction_pooled.csv")
    for _, r in rp.iterrows():
        dim = "pca" in r.setting or "randproj" in r.setting
        pts = "kmeans" in r.setting or "uniform" in r.setting
        rows.append(row("gates per load and circuits" if dim and pts else ("gates per load" if dim else "circuits per Gram matrix"),
                        f"reduction: {r.setting} ({r.scope})", "Amplitude", "unreduced", r.setting,
                        r.median_gate_saving, 1.0, "saving factor (median, two-qubit gates)",
                        dict(mean=r["mean"], lo=r["lo"], hi=r["hi"]), "v2_reduction_pooled.csv",
                        note=f"{r.n_datasets} datasets; post hoc"))
    t = pd.DataFrame(rows).rename(columns={"from_": "from"})
    t.to_csv(res / "lever_table.csv", index=False)
    show = t[["factor", "lever", "kernel", "cost_from", "cost_to", "saving_factor", "acc_change", "lo", "hi", "noninferior"]]
    lines = ["THE BILL AT THE BASELINE (per kernel; loads = (bandwidths x training circuits + test circuits) x shots)", bill.round(1).to_string(), "",
             "LEVERS (saving factor and paired accuracy change; non-inferior if lo > -0.02)",
             show.round(4).to_string(index=False)]
    (res / "lever_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
