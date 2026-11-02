"""Joint accuracy-vs-cost frontier for all tested kernels.

Inputs (all produced by this package):
  results/gate_counts.csv        from gate_counts.py        (two-qubit gates per circuit)
  results/circuits_per_gram.csv  from gate_counts.py        (circuits per train+test Gram)
  results/mds_shots.csv          from multi_dataset_study.py --diagnostics (shots per circuit, per rule)
  results/mds_cells.csv          from multi_dataset_study.py (test accuracy of every model)

Cost definition (stated once, used everywhere):
  circuit executions per Gram = circuits per Gram x shots per circuit
  two-qubit gate executions   = circuit executions x two-qubit gates per circuit
  classical operations        = Gram entries x operations per entry   (different unit: reported separately,
                                never divided by the quantum numbers)
  Counted: kernel evaluation for one train+test Gram matrix.  Not counted: SVM fitting, hyper-parameter
  search repetitions (x84 for every model alike), compilation time, classical post-processing.

Usage:
  python cost_frontier.py --results results --acc-source mds --shots-file mds_shots.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {"angle/fidelity": "Angle-F", "angle/projected": "Angle-P", "iqp/fidelity": "IQP-F",
         "iqp/projected": "IQP-P", "zz/fidelity": "ZZ-F", "zz/projected": "ZZ-P"}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--shots-file", default="shots.csv",
                    help="shots per circuit, long format: shots.csv (six-dataset gram_spread.py run) or mds_shots.csv "
                         "(16-dataset run of multi_dataset_study.py --diagnostics, same cells as the accuracy)")
    ap.add_argument("--n", type=int, default=4, help="qubits / features of the main study")
    ap.add_argument("--acc-source", default="mds", choices=["rerun", "mds"],
                    help="mds = mean test accuracy in <results>/mds_cells.csv; rerun = mean test accuracy in <results>/cells.csv")
    ap.add_argument("--mode", default="routed", choices=["routed", "all_to_all"])
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--rule", default="spread", choices=["spread", "fixed"])
    ap.add_argument("--param", type=float, default=2.0, help="c for rule=spread (eps=SD/c) or eps for rule=fixed")
    ap.add_argument("--classical-ops-rbf", type=int, default=None, help="ops per RBF entry (default 3d+1)")
    ap.add_argument("--classical-ops-linear", type=int, default=None, help="ops per linear entry (default 2d)")
    ap.add_argument("--classical-ops-angle-closed", type=int, default=None, help="ops per closed-form angle-F entry (default 4d)")
    a = ap.parse_args()
    res = Path(a.results)
    gc = pd.read_csv(res / "gate_counts.csv")
    cg = pd.read_csv(res / "circuits_per_gram.csv").set_index("construction")
    sh = pd.read_csv(res / a.shots_file)
    if a.acc_source == "rerun":
        acc_mean = pd.read_csv(res / "cells.csv").groupby("kernel").test_acc.mean()
    elif a.acc_source == "mds":  # larger study: mean over datasets of the per-dataset mean test accuracy
        acc_mean = pd.read_csv(res / "mds_cells.csv").groupby(["dataset", "kernel"]).test_acc.mean().groupby("kernel").mean()
    d = a.n
    M, nte = int(cg.M_train.iloc[0]), int(cg.n_test.iloc[0])
    entries = M * (M - 1) // 2 + nte * M

    def twoq(emb, construction):
        circ = "fidelity" if construction == "fidelity" else "state"
        r = gc[(gc.n == a.n) & (gc.embedding == emb) & (gc.circuit == circ) & (gc["mode"] == a.mode)
               & (gc.opt_level == a.opt_level)]
        if r.empty:
            raise SystemExit(f"no gate count row for {emb}/{circ}/{a.mode}/opt{a.opt_level}")
        return float(r.twoq_median.iloc[0])

    rows, sens = [], []
    for kern in NAMES:
        emb, cons = kern.split("/")
        g = twoq(emb, cons)
        ncirc = int(cg.loc[cons, "circuits_total"])
        sub = sh[sh.kernel == kern]
        for (rule, param), grp in sub.groupby(["rule", "param"]):
            S = grp.shots.median()
            ce = ncirc * S
            sens.append(dict(kernel=NAMES[kern], rule=rule, param=param, median_eps=grp.eps.median(),
                             shots_per_circuit=S, share_cells_unresolved=float(grp.shots.isna().mean()),
                             circuits_per_gram=ncirc, circuit_executions=ce, twoq_per_circuit=g,
                             twoq_gate_executions=ce * g))
        pick = sub[(sub.rule == a.rule) & (np.isclose(sub.param, a.param))]
        if pick.empty:
            raise SystemExit("rule/param not found in shots.csv; available: " + str(sorted(set(zip(sub.rule, sub.param)))))
        S = pick.shots.median()
        rows.append(dict(kernel=NAMES[kern], accuracy=acc_mean[kern], twoq_per_circuit=g, circuits_per_gram=ncirc,
                         shots_per_circuit=S, circuit_executions=ncirc * S, twoq_gate_executions=ncirc * S * g,
                         median_eps=pick.eps.median(), unit="quantum"))
    ops_rbf = a.classical_ops_rbf or (3 * d + 1)
    ops_lin = a.classical_ops_linear or (2 * d)
    ops_ang = a.classical_ops_angle_closed or (4 * d)
    classical = [dict(kernel="RBF (tuned)", accuracy=acc_mean["classical/rbf"], classical_ops=entries * ops_rbf,
                      ops_per_entry=ops_rbf),
                 dict(kernel="Linear", accuracy=acc_mean["classical/linear"], classical_ops=entries * ops_lin,
                      ops_per_entry=ops_lin),
                 dict(kernel="Angle-F closed form (classical)", accuracy=acc_mean["angle/fidelity"],
                      classical_ops=entries * ops_ang, ops_per_entry=ops_ang)]
    fr = pd.DataFrame(rows)
    cl = pd.DataFrame(classical)
    fr.to_csv(res / "frontier.csv", index=False)
    cl.to_csv(res / "frontier_classical.csv", index=False)
    pd.DataFrame(sens).to_csv(res / "frontier_sensitivity.csv", index=False)

    # LaTeX table
    def sci(v):
        return "--" if pd.isna(v) else (f"{v:.2e}" if v >= 1e4 else f"{v:.0f}")
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Accuracy and kernel-evaluation cost of one train+test Gram matrix "
             rf"($M={M}$, {nte} test points, $n={a.n}$). Shots per circuit follow the rule "
             + (rf"$\varepsilon=\mathrm{{SD}}/{a.param:g}$ of the tuned off-diagonal entries" if a.rule == "spread"
                else rf"$\varepsilon={a.param:g}$") + r".}", r"\label{tab:frontier}",
             r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Kernel & Acc. & 2q gates/circuit & Circuits & Shots/circuit & 2q-gate executions \\", r"\hline"]
    for _, r in fr.iterrows():
        lines.append(f'{r.kernel} & {r.accuracy:.3f} & {r.twoq_per_circuit:.0f} & {sci(r.circuits_per_gram)} & '
                     f'{sci(r.shots_per_circuit)} & {sci(r.twoq_gate_executions)} \\\\')
    lines.append(r"\hline")
    for _, r in cl.iterrows():
        lines.append(f'{r.kernel} & {r.accuracy:.3f} & \\multicolumn{{3}}{{c}}{{{r.ops_per_entry:.0f} ops/entry}} & '
                     f'{sci(r.classical_ops)} ops \\\\')
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (res / "tab_frontier.tex").write_text("\n".join(lines) + "\n")

    # Figure
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    mk = {"Angle": "o", "IQP": "s", "ZZ": "^"}
    col = {"Angle": "#1b6ca8", "IQP": "#d9822b", "ZZ": "#3a9d5d"}
    s_df = pd.DataFrame(sens)
    for _, r in fr.iterrows():
        e = r.kernel.split("-")[0]
        filled = r.kernel.endswith("F")
        rng_ = s_df[(s_df.kernel == r.kernel) & (s_df.rule == a.rule)]
        lo, hi = rng_.circuit_executions.min(), rng_.circuit_executions.max()
        ax.plot([lo, hi], [r.accuracy] * 2, color=col[e], alpha=0.35, lw=2)
        ax.scatter(r.circuit_executions, r.accuracy, marker=mk[e], s=70, facecolor=col[e] if filled else "white",
                   edgecolor=col[e], linewidth=1.6, zorder=3)
        ax.annotate(f"{r.kernel} ({r.twoq_per_circuit:.0f} 2q/circ)", (r.circuit_executions, r.accuracy),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    for _, r in cl.iterrows():
        if r.kernel.startswith("RBF"):
            ax.axhline(r.accuracy, color="k", ls="--", lw=1)
            ax.text(0.01, r.accuracy, f" tuned RBF ({a.acc_source} accuracy): {r.classical_ops:.1e} classical ops per Gram "
                    "(different unit)", fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("circuit executions per Gram matrix (circuits x shots), log scale")
    ax.set_ylabel("mean tuned test accuracy")
    ax.set_title("Accuracy vs. quantum evaluation cost", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(res / "fig_frontier.pdf")
    fig.savefig(res / "fig_frontier.png", dpi=200)
    print(fr.to_string(index=False))
    print(cl.to_string(index=False))
    print(f"shots from {a.shots_file}, accuracy from {a.acc_source}")
    print("Written: frontier.csv, frontier_classical.csv, frontier_sensitivity.csv, tab_frontier.tex, fig_frontier.pdf/png")


if __name__ == "__main__":
    main()
