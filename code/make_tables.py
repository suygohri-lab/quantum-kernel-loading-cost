"""LaTeX tables for the paper, generated from the result files (nothing is typed in by hand).

Reads --results (default results) and writes --out (default results/tables) one .tex file per table.  A table is skipped
if its input is missing.  Minus signs are typeset as math minus; intervals are 95% corrected-resampled-t intervals as
written by the study scripts.

Usage: python make_tables.py --results results --out results/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OFFLINE = {"moons", "circles", "breast_cancer", "iris", "wine", "digits38", "digits17", "digits49"}
QKERN = ["angle/fidelity", "angle/projected", "iqp/fidelity", "iqp/projected", "zz/fidelity", "zz/projected"]
NAME = {"angle/fidelity": "Angle-F", "angle/projected": "Angle-P", "iqp/fidelity": "IQP-F", "iqp/projected": "IQP-P",
        "zz/fidelity": "ZZ-F", "zz/projected": "ZZ-P", "classical/rbf": "RBF (tuned)", "classical/rbf_scale": "RBF (default $\\gamma$)",
        "classical/linear": "Linear", "classical/poly2_normalised": "Poly-2 (tuned)", "amplitude/fidelity": "Amplitude-F",
        "classical/gbm": "Gradient boosting", "classical/rf": "Random forest", "classical/knn": "$k$NN", "classical/mlp": "MLP"}
EXTRA = ["classical/gbm", "classical/rf", "classical/knn", "classical/mlp"]
ORDER = QKERN + ["classical/rbf", "classical/rbf_scale", "classical/linear", "classical/poly2_normalised", "amplitude/fidelity"]
DSNAME = {"breast_cancer": "breast cancer", "digits38": "digits 3/8", "digits17": "digits 1/7", "digits49": "digits 4/9",
          "ilpd": "ILPD", "pima": "Pima", "blood": "blood transf.", "haberman": "Haberman", "banknote": "banknote"}


def num(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if round(x, nd) == 0:
        x = 0.0                     # avoid "-0.000"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def ci(m, lo, hi, nd=3):
    return f"{num(m, nd, True)} [{num(lo, nd, True)}, {num(hi, nd, True)}]"


def ci2(m, lo, hi, nd=3, bold=False):
    """Two-line interval for narrow columns: mean over [lo, hi]."""
    top = num(m, nd, True)
    if bold:
        top = "\\textbf{" + top + "}"
    return "\\mstack{" + top + "\\\\[-1pt][" + num(lo, nd, True) + ", " + num(hi, nd, True) + "]}"


def sci(x):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** e:.2f}\\times10^{{{e}}}$"


MSTACK = "\\providecommand{\\mstack}[1]{\\begin{tabular}[c]{@{}c@{}}#1\\end{tabular}}"


def restyle(text, name=""):
    """Readable layout for the single-column Springer Nature template (text width 5.15 in): \\footnotesize, tight column
    spacing, upright tables (wide tables are built as stacked panels instead of being rotated)."""
    import re
    text = text.replace("\\begin{table*}[t]", "\\begin{table}[t]").replace("\\end{table*}", "\\end{table}")
    text = re.sub(r"\\(scriptsize|footnotesize)\n(\\setlength\{\\tabcolsep\}\{[0-9.]+pt\}\n)?", "", text)
    text = text.replace("\\setlength{\\tabcolsep}{2pt}", "").replace("\\scriptsize", "").replace("\\footnotesize", "")
    text = text.replace("\\centering\n", "\\centering\n\\footnotesize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.15}\n", 1)
    if name == "bwl":
        text = text.replace("\\centering\n\\footnotesize\\setlength{\\tabcolsep}{3pt}", "\\centering\n\\scriptsize\\setlength{\\tabcolsep}{2.5pt}", 1)
    if name == "positive":
        text = text.replace("\\centering\n\\footnotesize\\setlength{\\tabcolsep}{3pt}", "\\centering\n\\scriptsize\\setlength{\\tabcolsep}{1.6pt}", 1)
    if name in ("surrogate", "positive", "bill", "kernelclass"):
        text = text.replace("\\centering\n\\footnotesize\\setlength{\\tabcolsep}{3pt}", "\\centering\n\\scriptsize\\setlength{\\tabcolsep}{2.5pt}", 1)
    if name in ("datasets", "conc", "gates", "gridedge", "shots", "frontier", "fixed"):
        text = text.replace("\\setlength{\\tabcolsep}{3pt}", "\\setlength{\\tabcolsep}{6pt}", 1)
    return text


def write(out, name, text):
    (out / f"{name}.tex").write_text(restyle(text, name.replace("tab_", "")))
    print("wrote", name)


def tabular(cols, rows, spec=None, header=None):
    spec = spec or ("l" + "r" * (cols - 1))
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header, "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def tab_datasets(res, out):
    f = res / "mds_dataset_info.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d.status == "used"]
    nq = pd.read_csv(res / "mds_cells.csv").groupby("dataset").n_qubits.first()
    from datasets_cv import OPENML
    rows = []
    for _, r in d.iterrows():
        src = "scikit-learn" if r.dataset in OFFLINE else f"OpenML {OPENML[r.dataset][0]}"
        rows.append(f"{DSNAME.get(r.dataset, r.dataset)} & {src} & "
                    f"{int(r.n_used)}/{int(r.n_full)} & {int(r.n_features)} & {int(r.n_classes)} & {r.majority_share:.2f} & {int(nq[r.dataset])} \\\\")
    hdr = "Dataset & Source & $N_{\\mathrm{used}}/N$ & $d$ & Classes & Majority & $n$ \\\\"
    body = tabular(7, rows, "llrrrrr", hdr)
    write(out, "tab_datasets", "\\begin{table}[t]\n\\centering\n\\caption{Datasets. $N_{\\mathrm{used}}$ is the stratified subsample (cap 500); "
          "$d$ the number of raw features; Majority the share of the largest class; $n$ the number of qubits after PCA. OpenML entries give the data identifier.}\n"
          "\\label{tab:datasets}\n\\footnotesize\n" + body + "\n\\end{table}\n")


def tab_accuracy(res, out):
    f = res / "mds_cells.csv"
    if not f.exists():
        return
    c = pd.read_csv(f)
    m = c.groupby(["dataset", "kernel"]).test_acc.mean().unstack()[ORDER]
    ranks = m.rank(axis=1, ascending=False)
    qk = [k for k in ORDER if k.split("/")[0] in ("angle", "iqp", "zz")] + ["classical/rbf"]
    ck = [k for k in ORDER if k not in qk]

    def panel(keys, title):
        rr = []
        for ds, r in m.iterrows():
            best = r.max()
            cells = [("\\textbf{" + f"{r[k]:.3f}" + "}") if abs(r[k] - best) < 5e-4 else f"{r[k]:.3f}" for k in keys]
            rr.append(f"{DSNAME.get(ds, ds)} & " + " & ".join(cells) + " \\\\")
        rr.append("\\midrule")
        rr.append("Mean & " + " & ".join(f"{m[k].mean():.3f}" for k in keys) + " \\\\")
        h = "Dataset & " + " & ".join(NAME[k].replace("(default $\\gamma$)", "(def.)") for k in keys) + " \\\\"
        return f"\\textit{{{title}}}\\\\[0.4ex]\n" + tabular(len(keys) + 1, rr, "l" + "r" * len(keys), h)
    body = panel(qk, "(a) Quantum kernels and tuned RBF") + "\n\\medskip\n" + panel(ck, "(b) Other classical kernels and the amplitude encoding")
    write(out, "tab_accuracy", "\\begin{table}[t]\n\\centering\n\\caption{Test accuracy per dataset (mean over 20 paired resamples; best per row in bold). "
          "Intervals for every dataset are in Online Resource 1.}\n\\label{tab:acc}\n"
          + body + "\n\\end{table}\n")


def tab_pooled(res, out):
    f, g = res / "mds_paired_vs_rbf.csv", res / "mds_cells.csv"
    if not f.exists():
        return
    p = pd.read_csv(f)
    p = p[p.dataset == "POOLED"]
    c = pd.read_csv(g)
    means = c.groupby(["dataset", "kernel"]).test_acc.mean().unstack()
    ranks = means[[k for k in ORDER if k != "classical/rbf_scale"]].rank(axis=1, ascending=False).mean()
    rows = []
    for k in ORDER:
        if k == "classical/rbf":
            rows.append(f"{NAME[k]} & {means[k].mean():.3f} & \\multicolumn{{3}}{{c}}{{reference}} & {ranks[k]:.2f} \\\\")
            continue
        cols = []
        for met in ("test_acc", "test_bacc", "test_f1"):
            r = p[(p.metric == met) & (p.kernel == k)].iloc[0]
            cols.append(ci(r["mean"], r.lo, r.hi) + ("$^{\\ddagger}$" if bool(r.equivalent_95ci) else ""))
        rk = f"{ranks[k]:.2f}" if k in ranks else "--"
        rows.append(f"{NAME[k]} & {means[k].mean():.3f} & " + " & ".join(cols) + f" & {rk} \\\\")
    rowsA, rowsB = [], []
    for k in ORDER + [e for e in EXTRA if e in means.columns]:
        if k == "classical/rbf":
            rowsA.append(f"{NAME[k]} & {means[k].mean():.3f} & reference \\\\")
            rowsB.append(f"{NAME[k]} & reference & reference \\\\")
            continue
        cols = []
        for met in ("test_acc", "test_bacc", "test_f1"):
            r = p[(p.metric == met) & (p.kernel == k)].iloc[0]
            cols.append(ci(r["mean"], r.lo, r.hi) + ("$^{\\ddagger}$" if bool(r.equivalent_95ci) else ""))
        rk = f"{ranks[k]:.2f}" if k in ranks else "--"
        rowsA.append(f"{NAME[k]} & {means[k].mean():.3f} & {cols[0]} \\\\")
        rowsB.append(f"{NAME[k]} & {cols[1]} & {cols[2]} \\\\")
    ta = tabular(3, rowsA, "lrc", "Kernel & Mean acc. & $\\Delta$ accuracy [95\\% CI] \\\\")
    tb = tabular(3, rowsB, "lcc", "Kernel & $\\Delta$ balanced accuracy [95\\% CI] & $\\Delta$ macro-F1 [95\\% CI] \\\\")
    write(out, "tab_pooled", "\\begin{table}[t]\n\\centering\n\\caption{Pooled paired difference to tuned RBF (16 datasets, mean over 20 resamples, 95\\% interval "
          "from the corrected resampled $t$). $^{\\ddagger}$: interval inside the $\\pm0.02$ margin (equivalent). "
          "}\n\\label{tab:pooled}\n"
          "\\textit{(a) Accuracy}\\\\[0.4ex]\n" + ta + "\n\\medskip\n\\textit{(b) Balanced accuracy and macro-F1}\\\\[0.4ex]\n" + tb + "\n\\end{table}\n")


def tab_gridedge(res, out):
    f = res / "mds_cells.csv"
    if not f.exists():
        return
    c = pd.read_csv(f)
    ks = QKERN + ["classical/rbf", "classical/poly2_normalised"]
    g = c[c.kernel.isin(ks)].groupby("kernel").agg(lo=("at_lower_edge", "mean"), up=("at_upper_edge", "mean"), med=("s_star", "median"))
    rows = [f"{NAME[k]} & {100 * g.loc[k, 'lo']:.1f} & {100 * g.loc[k, 'up']:.1f} & {g.loc[k, 'med']:.3g} \\\\" for k in ks]
    hdr = "Kernel & Lower edge (\\%) & Upper edge (\\%) & Median $s$ \\\\"
    from extended_study import S_WIDE
    ncell = int((c.kernel == "classical/rbf").sum())
    write(out, "tab_gridedge", "\\begin{table}[t]\n\\centering\n\\caption{Where the tuned bandwidth lands on the grid "
          f"$s\\in[{S_WIDE[0]:g},{S_WIDE[-1]:g}]$ ({len(S_WIDE)} values; share of the {ncell} dataset--resample cells; "
          "poly-2 searches an offset instead of $s$).}\n\\label{tab:gridedge}\n\\footnotesize\n"
          + tabular(4, rows, "lrrr", hdr) + "\n\\end{table}\n")


def tab_qubits(res, out):
    f = res / "qs_paired_vs_rbf.csv"
    if not f.exists():
        return
    q = pd.read_csv(f)
    q = q[q.dataset == "POOLED"] if "dataset" in q else q
    if "metric" in q:
        q = q[q.metric == "test_acc"]
    ns = sorted(q.n_qubits.unique()) if "n_qubits" in q else sorted(q.n.unique())
    nc = "n_qubits" if "n_qubits" in q else "n"
    ks = QKERN + ["amplitude/fidelity", "classical/poly2_normalised", "classical/linear"]
    rows = []
    for k in ks:
        cells = []
        for n in ns:
            r = q[(q.kernel == k) & (q[nc] == n)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            hw = (r.hi - r.lo) / 2
            mark = "$^{\\ddagger}$" if bool(r.equivalent_95ci) else ""
            cells.append(f"{num(r['mean'], 3, True)} ({hw:.3f}){mark}")
        rows.append(f"{NAME[k]} & " + " & ".join(cells) + " \\\\")
    hdr = "Kernel & " + " & ".join(f"$n={n}$" for n in ns) + " \\\\"
    write(out, "tab_qubits", "\\begin{table*}[t]\n\\centering\n\\caption{Accuracy difference to the tuned RBF on the same $n$-dimensional data, pooled over 8 datasets "
          "(10-fold, one repeat), as mean (95\\% half-width). $^{\\ddagger}$: interval inside the $\\pm0.02$ margin. "
          "All values are exact statevector (infinite-shot) kernels.}\n\\label{tab:qubits}\n\\footnotesize\n"
          + tabular(len(ns) + 1, rows, "l" + "r" * len(ns), hdr) + "\n\\end{table*}\n")


def tab_fixed(res, out):
    f = res / "fb_gain.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    g = g[(g.metric == "test_acc") & (g.dataset == "POOLED")]
    rows = []
    for k in QKERN:
        a = g[g.kernel == k].iloc[0]
        b = g[g.kernel == f"{k} [fixed s=1] - tuned RBF"].iloc[0]
        rows.append(f"{NAME[k]} & {a.fixed_mean:.3f} & {a.tuned_mean:.3f} & {ci(a['mean'], a.lo, a.hi)} & {ci(b['mean'], b.lo, b.hi)} \\\\")
    hdr = "Kernel & Fixed $s{=}1$ & Tuned & Gain (tuned $-$ fixed) & Fixed $-$ tuned RBF \\\\"
    write(out, "tab_fixed", "\\begin{table*}[t]\n\\centering\n\\caption{Effect of tuning the bandwidth (16 datasets, 20 resamples, accuracy). "
          "Fixed: $s=1$ and projected width $\\gamma=1/(3n\\,\\mathrm{Var}(F))$, only $C$ tuned. The last column is what a benchmark "
          "without bandwidth tuning would have reported against the tuned RBF.}\n\\label{tab:fixed}\n\\footnotesize\n"
          + tabular(5, rows, "lrrcc", hdr) + "\n\\end{table*}\n")


def tab_conc(res, out):
    f = res / "fb_concentration.csv"
    if not f.exists():
        return
    c = pd.read_csv(f)
    ns = sorted(c.n.unique())
    rows = []
    for n in ns:
        r = c[c.n == n].set_index("kind")
        rows.append(f"{n} & {1 / (2 ** n + 1):.4f} & " + " & ".join(f"{r.loc[k, 'offdiag_mean']:.4f}" for k in ("angle", "iqp", "zz")) + " \\\\")
    hdr = "$n$ & Haar $1/(2^n{+}1)$ & Angle & IQP & ZZ \\\\"
    write(out, "tab_conc", "\\begin{table}[t]\n\\centering\n\\caption{Mean off-diagonal entry of the fidelity Gram matrix at the untuned bandwidth $s=1$, "
          "against the Haar value (8 datasets, 3 resamples each).}\n\\label{tab:conc}\n\\footnotesize\n"
          + tabular(5, rows, "rrrrr", hdr) + "\n\\end{table}\n")


def tab_gates(res, out):
    f = res / "gate_counts.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    g = g[g.embedding == "zz"]
    ns = [4, 6, 8, 10, 12]

    def get(n, circ, mode, opt):
        r = g[(g.n == n) & (g.circuit == circ) & (g["mode"] == mode) & (g.opt_level == opt)]
        return "--" if r.empty else f"{int(r.twoq_median.iloc[0])}"
    rows = []
    for n in ns:
        rows.append(f"{n} & {get(n, 'state', 'all_to_all', 1)} & {get(n, 'state', 'routed', 1)} & {get(n, 'state', 'routed', 3)} & "
                    f"{get(n, 'fidelity', 'all_to_all', 1)} & {get(n, 'fidelity', 'routed', 1)} & {get(n, 'fidelity', 'routed', 3)} \\\\")
    hdr = ("$n$ & \\multicolumn{3}{c}{State circuit} & \\multicolumn{3}{c}{Fidelity circuit} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n & all-to-all & routed O1 & routed O3 & all-to-all & routed O1 & routed O3 \\\\")
    write(out, "tab_gates", "\\begin{table}[t]\n\\centering\n\\caption{Two-qubit gates per circuit for IQP and ZZ embeddings ($L=2$; IQP and ZZ give the same counts; "
          "the angle embedding needs none). Median over transpiler seeds; basis $\\{\\mathrm{cx},\\mathrm{rz},\\mathrm{sx},\\mathrm{x}\\}$; "
          "routed = heavy-hex coupling map of the stored FakeSherbrooke snapshot; O1/O3 = optimisation level.}\n\\label{tab:gates}\n\\footnotesize\n"
          + tabular(7, rows, "rrrrrrr", hdr).replace("\\midrule\n\\cmidrule", "\\cmidrule") + "\n\\end{table}\n")


def tab_loading(res, out):
    f, r = res / "loading_counts.csv", res / "loading_ratio.csv"
    if not (f.exists() and r.exists()):
        return
    d = pd.read_csv(f)
    d = d[d.family == "dense_real"].groupby(["toolchain", "n"]).twoq.median().unstack(0)
    rt = pd.read_csv(r)
    rt = rt[rt.toolchain == "qiskit_opt1"].pivot_table(index="n", columns="eps", values="ratio")
    rows = []
    for n in sorted(d.index):
        if n < 2:
            continue

        def gv(c):
            return f"{int(d.loc[n, c])}" if (c in d.columns and not np.isnan(d.loc[n, c])) else "--"
        rr = rt.loc[n] if n in rt.index else None
        rcells = " & ".join(f"{rr[e]:.1f}" for e in (0.1, 0.05, 0.01)) if rr is not None else "-- & -- & --"
        rows.append(f"{n} & {2 ** n} & {gv('pennylane')} & {gv('qiskit_opt1')} & {gv('qiskit_routed_opt1')} & {rcells} \\\\")
    hdr = ("$n$ & $N$ & PennyLane & Qiskit & Qiskit routed & \\multicolumn{3}{c}{$Q_{\\mathrm{entry}}/C_{\\mathrm{cl}}$ at $\\varepsilon=$} \\\\\n"
           "\\cmidrule(lr){6-8}\n & & CNOT & CX & CX & 0.1 & 0.05 & 0.01 \\\\")
    write(out, "tab_loading", "\\begin{table*}[t]\n\\centering\n\\caption{Two-qubit gates of exact amplitude loading of a dense real vector and the resulting cost ratio "
          "$Q_{\\mathrm{entry}}/C_{\\mathrm{cl}}=S(\\varepsilon)G_{\\mathrm{load}}/N$ (Qiskit optimisation level 1 counts). Routed: heavy-hex map of the stored FakeSherbrooke snapshot.}\n"
          "\\label{tab:load}\n\\footnotesize\n" + tabular(8, rows, "rrrrrrrr", hdr).replace("\\midrule\n\\cmidrule", "\\cmidrule") + "\n\\end{table*}\n")


def tab_frontier(res, out):
    f, c = res / "frontier.csv", res / "frontier_classical.csv"
    if not (f.exists() and c.exists()):
        return
    d, cl = pd.read_csv(f), pd.read_csv(c)
    rows = []
    for _, r in d.iterrows():
        rows.append(f"{r.kernel} & {r.accuracy:.3f} & {int(r.twoq_per_circuit)} & {int(r.circuits_per_gram)} & {r.shots_per_circuit:.0f} & "
                    f"{sci(r.circuit_executions)} & {sci(r.twoq_gate_executions)} \\\\")
    rows.append("\\midrule")
    for _, r in cl.iterrows():
        rows.append(f"{r.kernel} & {r.accuracy:.3f} & \\multicolumn{{4}}{{c}}{{{int(r.ops_per_entry)} operations per entry}} & {sci(r.classical_ops)} ops \\\\")
    hdr = "Kernel & Acc. & 2q/circ. & Circuits & Shots & Executions & 2q-gate exec. \\\\"
    write(out, "tab_frontier", "\\begin{table*}[t]\n\\centering\n\\caption{Accuracy and kernel-evaluation cost of one Gram matrix ($M=100$ training, 50 test points, $n=4$). "
          "Accuracy is the 16-dataset mean (20 resamples). "
          + ("Shots per circuit are the median over the same 16-dataset cells "
             if (res / "mds_shots.csv").exists() else "Circuits and shots come from the six-dataset calibration run ")
          + "(shots per circuit so that the standard error equals half the SD of the off-diagonal entries). Executions = circuits $\\times$ shots; "
          "2q-gate exec. = executions $\\times$ two-qubit gates per circuit (routed, optimisation level 3). Classical costs are counted in arithmetic operations "
          "and are not comparable in unit.}\n\\label{tab:frontier}\n\\footnotesize\n" + tabular(7, rows, "lrrrrrr", hdr) + "\n\\end{table*}\n")


def tab_reduction(res, out):
    f = res / "reduction_table.csv"
    if not f.exists():
        return
    r = pd.read_csv(f)
    r = r[~r.exploratory_post_hoc.astype(bool)] if "exploratory_post_hoc" in r else r
    r = r[r.dataset != "none"]
    ds = ["breast_cancer", "wine", "digits38", "synth_r4", "synth_r16", "synth_r64"]
    ds = [d for d in ds if d in set(r.dataset)]
    rows = []
    for d in ds:
        s = r[r.dataset == d]
        full = s[(s["sample"] == "full") & (s.reduction != "none")]
        ks = sorted(full[full.reduction == "pca"].k.unique())
        kmid = [k for k in ks if k >= 8][:1] or ks[-1:]
        kmax = ks[-1]

        def cell(red, k):
            q = full[(full.reduction == red) & (full.k == k)]
            if q.empty:
                return "--"
            q = q.iloc[0]
            gr = "" if np.isnan(q.gate_ratio) else f", {q.gate_ratio:.1f}$\\times$"
            return f"\\mstack{{{num(q.delta, 3, True)}\\\\($\\pm${q.ci_half_width:.3f}{gr})}}"
        km = s[(s.reduction == "none") & (s["sample"] == "kmeans") & (s.frac == 0.5)]
        kcell = "--" if km.empty else f"\\mstack{{{num(km.iloc[0].delta, 3, True)}\\\\($\\pm${km.iloc[0].ci_half_width:.3f}, {km.iloc[0].gate_ratio:.1f}$\\times$)}}"
        rows.append(f"{DSNAME.get(d, d.replace('_', ' '))} & {cell('pca', kmid[0])} & {cell('pca', kmax)} & {cell('randproj', kmax)} & {kcell} \\\\")
    hdr = "Dataset & \\mstack{PCA\\\\$k=8$} & \\mstack{PCA\\\\largest $k$} & \\mstack{Random proj.\\\\largest $k$} & \\mstack{$k$-means\\\\50\\% of points} \\\\"
    n_all = int((r.reduction != "none").sum() + ((r.reduction == "none") & (r["sample"] != "full")).sum())
    write(out, "tab_reduction", "\\begin{table}[t]\n\\centering\n\\caption{Accuracy change of the amplitude kernel after reduction before loading, relative to the unreduced amplitude kernel "
          "(mean paired difference; in brackets: 95\\% half-width and two-qubit-gate saving). Non-inferiority at $\\delta=0.02$ requires the lower bound above $-0.02$. "
          "All 84 pre-specified configurations are in Online Resource 2.}\n\\label{tab:reduction}\n"
          + tabular(5, rows, "lcccc", hdr) + "\n\\end{table}\n")


def tab_shots(res, out):
    f = res / "shots_summary.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    rows = []
    for k in QKERN:
        z = s[s.kernel == k].set_index("shots")
        rows.append(f"{NAME[k]} & " + " & ".join(f"{z.loc[n, 'loss']:.3f}" for n in (100, 1000, 10000)) + " & "
                    + " & ".join(f"{z.loc[n, 'gram_rms_err']:.4f}" for n in (100, 1000, 10000)) + " \\\\")
    hdr = ("Kernel & \\multicolumn{3}{c}{Accuracy loss at $N=$} & \\multicolumn{3}{c}{Gram RMS error at $N=$} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
           " & $10^2$ & $10^3$ & $10^4$ & $10^2$ & $10^3$ & $10^4$ \\\\")
    write(out, "tab_shots", "\\begin{table}[t]\n\\centering\n\\caption{Finite-shot sampling at fixed ideal hyper-parameters (6 datasets, 10 resamples): mean accuracy loss versus the exact kernel "
          "and RMS error of the off-diagonal Gram entries.}\n\\label{tab:shots}\n\\footnotesize\n"
          + tabular(7, rows, "lrrrrrr", hdr).replace("\\midrule\n\\cmidrule", "\\cmidrule") + "\n\\end{table}\n")


def tab_calib(res, out):
    f = res / "device_backends.csv"
    if not f.exists():
        return
    b = pd.read_csv(f)
    rows = []
    for _, r in b.iterrows():
        rows.append(f"{r.backend} & {int(r.n_qubits)} & {r.snapshot[:10]} & {r.two_qubit_gate} & {r.median_T1_us:.0f} & {r.median_T2_us:.0f} & "
                    f"{100 * r.median_readout_err:.2f} & {100 * r.median_2q_err:.2f} \\\\")
    hdr = "Backend & Qubits & Snapshot & 2q gate & $T_1$ ($\\mu$s) & $T_2$ ($\\mu$s) & Readout err. (\\%) & 2q err. (\\%) \\\\"
    write(out, "tab_calib", "\\begin{table*}[t]\n\\centering\n\\caption{Stored calibration snapshots used by the device-model replay (medians over the device). "
          "These are fake-backend snapshots shipped with \\texttt{qiskit-ibm-runtime}; no job was sent to a real device.}\n\\label{tab:calib}\n\\footnotesize\n"
          + tabular(8, rows, "lrllrrrr", hdr) + "\n\\end{table*}\n")


def tab_device(res, out):
    f = res / "device_summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    rowsA, rowsB = [], []
    for b in ("FakeFez", "FakeSherbrooke", "FakeCusco"):
        z = d[d.backend == b].set_index("kernel")
        for k in QKERN:
            r = z.loc[k]
            first = b if k == QKERN[0] else ""
            rowsA.append(f"{first} & {NAME[k]} & {r.acc_ideal:.3f} & {r.acc_shotonly:.3f} & {r.acc_device:.3f} & {num(r.gram_bias, 2, True)} \\\\")
            rowsB.append(f"{first} & {NAME[k]} & {ci(r.change_shot_mean, r.change_shot_lo, r.change_shot_hi, 3)} & "
                         f"{ci(r.change_noise_mean, r.change_noise_lo, r.change_noise_hi, 3)} \\\\")
        rowsA.append("\\midrule"); rowsB.append("\\midrule")
    rowsA, rowsB = rowsA[:-1], rowsB[:-1]
    ta = tabular(6, rowsA, "llrrrr", "Backend & Kernel & Ideal & Shots only & Device & Gram bias \\\\")
    tb = tabular(4, rowsB, "llcc", "Backend & Kernel & $\\Delta$ shot noise [95\\% CI] & $\\Delta$ device noise [95\\% CI] \\\\")
    write(out, "tab_device", "\\begin{table}[t]\n\\centering\n\\caption{Device-model replay at fixed ideal hyper-parameters (5 datasets, 10 resamples, 40 train / 20 test points, 1000 shots). "
          "(a) Mean accuracy; Gram bias = mean (noisy $-$ exact) off-diagonal entry. (b) $\\Delta$ shot noise = shots-only minus exact; $\\Delta$ device noise = device replay minus shots-only "
          "(the effect of the noise model beyond sampling); intervals use $n_{\\mathrm{test}}/n_{\\mathrm{train}}=0.5$.}\n\\label{tab:device}\n"
          "\\textit{(a) Accuracy and Gram bias}\\\\[0.4ex]\n" + ta + "\n\\medskip\n\\textit{(b) Accuracy changes}\\\\[0.4ex]\n" + tb + "\n\\end{table}\n")


def tab_bill(res, out):
    f = res / "lever_bill.csv"
    if not f.exists():
        return
    b = pd.read_csv(f).set_index("kernel")
    if "circuits_model" not in b:
        return print("skip tab_bill (old lever_bill.csv; rerun lever_table.py)")
    rows = []
    for k in ["Angle-F", "Angle-P", "IQP-F", "IQP-P", "ZZ-F", "ZZ-P"]:
        r = b.loc[k]
        rows.append(f"{k} & {r.twoq_per_circuit:.0f} & {r.circuits_per_gram:,.0f} & {r.circuits_model:,.0f} ({r.tuning_factor:.1f}$\\times$) & "
                    f"{r.shots_per_circuit:g} & {sci(r.loads)} & {sci(r.twoq_gate_executions)} \\\\".replace(",", "\\,"))
    hdr = ("Kernel & \\mstack{2q gates\\\\per circuit} & \\mstack{Circuits per\\\\Gram matrix} & \\mstack{Circuits per\\\\tuned model} & "
           "\\mstack{Shots per\\\\circuit} & Loads & \\mstack{2q gate\\\\executions} \\\\")
    write(out, "tab_bill", "\\begin{table}[t]\n\\centering\n" + MSTACK + "\n\\caption{The loading bill of one tuned quantum-kernel "
          "model at the baseline ($n=4$, $M=100$ training and 50 test points, routed circuits at optimisation level 3, shot rule "
          "$c=2$). A tuned model builds the training block of the Gram matrix for each of the 12 bandwidths and the test block "
          "once; the factor in brackets is the increase over a single Gram matrix. Loads are circuits $\\times$ shots; two-qubit "
          "(2q) gate executions are loads $\\times$ gates per circuit.}\n\\label{tab:bill}\n\\footnotesize\n"
          + tabular(7, rows, "lrrrrrr", hdr) + "\n\\end{table}\n")


def tab_kernelclass(res, out):
    fs, fr, fd = res / "stat_summary.csv", res / "rank_check.csv", res / "rank_data.csv"
    if not (fs.exists() and fr.exists() and fd.exists()):
        return
    st = pd.read_csv(fs).set_index(["kernel", "bandwidth"])
    rk = pd.read_csv(fr)
    rd = pd.read_csv(fd)
    if "eff_dim_s1" not in rd:
        return print("skip tab_kernelclass (old rank_data.csv; rerun rank_check.py)")
    rd = rd.groupby("kernel")[["eff_dim", "eff_dim_s1", "n_train"]].median()
    rows = []
    for k in QKERN + ["amplitude/fidelity"]:
        if k == "amplitude/fidelity":
            t = st.loc[(k, "none")]
            rows.append(f"{NAME[k]} & \\multicolumn{{2}}{{c}}{{{num(t['50%'], 2)} (no bandwidth)}} & -- & -- \\\\")
            continue
        t, u = st.loc[(k, "tuned")], st.loc[(k, "s=1")]
        if k.endswith("fidelity"):
            e_t = f"{rd.loc[k, 'eff_dim']:.1f}"
            e_1 = f"{rd.loc[k, 'eff_dim_s1']:.1f}"
        else:
            e_t = e_1 = "--"
        rows.append(f"{NAME[k]} & {num(t['50%'], 2)} [{num(t['25%'], 2)}, {num(t['75%'], 2)}] & "
                    f"{num(u['50%'], 2)} [{num(u['25%'], 2)}, {num(u['75%'], 2)}] & {e_t} & {e_1} \\\\")
    hdr = ("Kernel & \\mstack{Stationarity index\\\\tuned $s^*$} & \\mstack{Stationarity index\\\\$s=1$} & "
           "\\mstack{Eff. dimension\\\\tuned $s^*$} & \\mstack{Eff. dimension\\\\$s=1$} \\\\")
    write(out, "tab_kernelclass", "\\begin{table}[t]\n\\centering\n" + MSTACK + "\n\\caption{Kernel class at the tuned and at the "
          "unscaled bandwidth. Stationarity index: median [quartiles] over the 320 main-study cells of "
          "$\\mathrm{mean}|K(\\mathbf{x},\\mathbf{z})-K(\\mathbf{x}+\\mathbf{t},\\mathbf{z}+\\mathbf{t})|$ relative to the spread of $K$ "
          "(0: stationary). Effective dimension $(\\sum\\lambda)^2/\\sum\\lambda^2$ of the fidelity Gram matrix on the same "
          f"main-study training blocks (median over datasets, {rd.n_train.median():.0f} points, first resample) at $s^*$ and at $s=1$." + "}\n\\label{tab:kernelclass}\n\\footnotesize\n" + tabular(5, rows, "lcccc", hdr) + "\n\\end{table}\n")


def tab_surrogate(res, out):
    fp, fc = res / "sur_paired.csv", res / "sur_cells.csv"
    if not (fp.exists() and fc.exists()):
        return
    p = pd.read_csv(fp)
    p = p[(p.metric == "test_acc") & (p.dataset == "POOLED") & (p.reference == "full_grid")].set_index(["kernel", "method"])
    c = pd.read_csv(fc)
    sm = c.groupby(["kernel", "method"]).s_used.median()
    rows = []
    for k in QKERN:
        def cell(m):
            if (k, m) not in p.index:
                return "--"
            r = p.loc[(k, m)]
            return ci2(r['mean'], r.lo, r.hi, bold=bool(r.noninferior))
        rows.append(f"{NAME[k]} & {sm[(k, 'full_grid')]:.2f} & {sm[(k, 'metric_map')]:.3f} & {cell('rbf_map')} & "
                    f"{cell('metric_map')} & {cell('angle')} \\\\")
    hdr = ("Kernel & \\mstack{$s^*$ (full\\\\grid)} & \\mstack{$s$ metric-\\\\matched} & RBF map & Metric-matched & "
           "Angle transfer \\\\")
    write(out, "tab_surrogate", "\\begin{table}[t]\n\\centering\n" + MSTACK + "\n\\caption{Surrogate tuning: accuracy change "
          "against full-grid tuning (16 datasets, 20 resamples, pooled, 95\\% interval; bold: non-inferior, lower bound "
          "$>-0.02$). A surrogate builds one quantum Gram matrix instead of 12. Bandwidths are medians over cells. The angle "
          "transfer is the identity for the angle kernels.}\n\\label{tab:surrogate}\n\\footnotesize\n"
          + tabular(6, rows, "lrrccc", hdr) + "\n\\end{table}\n")


def tab_positive(res, out):
    fp, fl, fe, fg = (res / "rel_paired.csv", res / "rel_labels.csv", res / "rel_edge_paired.csv",
                      res / "gate_counts.csv")
    if not (fp.exists() and fl.exists()):
        return
    p = pd.read_csv(fp)
    lab = pd.read_csv(fl)
    e = pd.read_csv(fe) if fe.exists() else None
    fs = res / "relsur_paired.csv"
    sur = pd.read_csv(fs) if fs.exists() else None
    gc = pd.read_csv(fg) if fg.exists() else None
    rows = []
    for (L, n), _ in p.groupby(["labeller", "n_qubits"]):
        q = p[(p.labeller == L) & (p.n_qubits == n)]
        pool = q[q.dataset == "POOLED"].set_index("kernel")
        k = f"{L}/fidelity"
        a = pool.loc[k]
        o = pool.loc["E3: oracle minus tuned"]
        ang = pool.loc["angle/fidelity"]
        rf = pool.loc["classical/rf"]
        per = q[(q.kernel == k) & ~q.dataset.isin(["POOLED", "POOLED_REAL"])]
        ext = "--"
        if e is not None:
            r = e[(e.labeller == L) & (e.n_qubits == n) & (e.kernel == k) & (e.dataset == "POOLED")]
            if len(r):
                ext = num(r["mean"].iloc[0], 3, True)
        surv = "--"
        if sur is not None:
            r = sur[(sur.labeller == L) & (sur.n_qubits == n) & (sur.kernel == k) & (sur.method == "metric_map")
                    & (sur.reference == "tuned_rbf") & (sur.dataset == "POOLED")]
            if len(r):
                surv = num(r["mean"].iloc[0], 3, True)
        gates = "--"
        if gc is not None:
            r = gc[(gc.n == n) & (gc.embedding == L) & (gc.circuit == "fidelity") & (gc["mode"] == "routed") & (gc.opt_level == 3)]
            if len(r):
                gates = f"{r.twoq_median.iloc[0]:.0f}"
        gl = lab[(lab.labeller == L) & (lab.n_qubits == n)].g_label.median()
        rows.append(f"{L.upper()} & {n} & {ci2(a['mean'], a.lo, a.hi)} & {int(per.advantage.sum())}/{len(per)} & "
                    f"{num(o['mean'], 3, True)} & {ext} & {surv} & {num(ang['mean'], 3, True)} & {gates} \\\\")
    hdr = ("Labels & $n$ & \\mstack{Labelling\\\\kernel $-$ RBF} & \\mstack{Datasets\\\\with adv.} & "
           "\\mstack{Oracle\\\\$-$ tuned} & \\mstack{vs RBF,\\\\$s\\le64$} & \\mstack{Surrogate-\\\\tuned $-$ RBF} & \\mstack{Angle-F\\\\$-$ RBF} & "
           "\\mstack{2q gates\\\\per circuit} \\\\")
    write(out, "tab_positive", "\\begin{table}[t]\n\\centering\n" + MSTACK + "\n\\caption{Positive control on engineered labels "
          "(four real and two synthetic datasets, 20 resamples each; $n=4$ added after the main analysis). Differences in accuracy, pooled over datasets (95\\% interval for the labelling kernel). Datasets "
          "with adv.: lower bound $>0$. Oracle: labelling kernel at $s=1$. vs RBF, $s\\le64$: against the RBF kernel re-tuned on "
          "the grid extended to $s=64$. Surrogate-tuned: labelling kernel at the metric-matched bandwidth. Gates: routed fidelity circuit, optimisation level 3.}\n"
          "\\label{tab:positive}\n\\footnotesize\n" + tabular(9, rows, "lrcrrrrrr", hdr) + "\n\\end{table}\n")


def tab_bwl(res, out):
    f = res / "bwl_paired.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    key = d.set_index(["setting", "s_label", "n_qubits", "model", "reference"])
    name = {"iqp": "IQP", "zz": "ZZ", "iqp_L1": "IQP, $L{=}1$", "iqp_L3": "IQP, $L{=}3$"}
    rows = []
    for (st, sl, n), _ in d.groupby(["setting", "s_label", "n_qubits"], sort=False):
        def c(m, ref, bold_if=None):
            r = key.loc[(st, sl, n, m, ref)]
            txt = ci2(r["mean"], r.lo, r.hi, bold=bool(bold_if and r[bold_if]))
            return txt
        rows.append(f"{name.get(st, st)} & {sl:g} & {n} & {c('full_grid', 'rbf', 'advantage')} & "
                    f"{c('oracle', 'full_grid')} & {c('metric_map', 'rbf', 'advantage')} \\\\")
    hdr = ("Labels & $s_L$ & $n$ & \\mstack{Grid-tuned kernel\\\\$-$ tuned RBF} & \\mstack{Oracle at $s_L$\\\\$-$ grid-tuned} & "
           "\\mstack{Metric-matched\\\\surrogate $-$ RBF} \\\\")
    write(out, "tab_bwl", "\\begin{table}[t]\n\\centering\n" + MSTACK + "\n\\caption{Engineered labels built at several labelling "
          "bandwidths $s_L$ (0.15, 0.3 and 0.7 lie between grid points) and from IQP circuits with $L=1$ or 3 layers, which no "
          "tested kernel uses ($L=2$). Pooled accuracy differences over six datasets with 95\\% intervals; bold: lower bound "
          "above zero (advantage detected or kept). Oracle: the tested kernel at the labelling bandwidth.}\n"
          "\\label{tab:bwl}\n\\footnotesize\n" + tabular(6, rows, "lrrccc", hdr) + "\n\\end{table}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (tab_datasets, tab_accuracy, tab_pooled, tab_gridedge, tab_qubits, tab_fixed, tab_conc, tab_gates, tab_loading,
               tab_frontier, tab_reduction, tab_shots, tab_calib, tab_device, tab_bill, tab_kernelclass,
               tab_surrogate, tab_positive, tab_bwl):
        fn(res, out)


if __name__ == "__main__":
    main()
