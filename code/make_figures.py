"""Paper figures from the result files (no number is typed in by hand; a figure is skipped if its input is missing).

  fig_pipeline    study design / flow diagram (no data)
  fig_motivation  introduction diagram: data loading inside a quantum machine-learning model (no data)
  fig_forest      pooled accuracy difference to the tuned RBF with 95% CI and the +/-0.02 equivalence band
  fig_heatmap     the same difference per dataset (diverging, neutral midpoint)
  fig_qubits      difference to RBF and kernel concentration versus number of qubits (4..12)
  fig_gates       two-qubit gate count versus qubits (state and fidelity circuits)
  fig_frontier    accuracy minus tuned RBF against circuit executions and two-qubit gate executions
  fig_reduction   accuracy change versus gate saving for dimension / sample reduction
  fig_shots       accuracy loss and Gram error versus shots per circuit
  fig_device      accuracy change on the replayed device models
  fig_loading     central figure: two-qubit gates to load one data point vs accuracy (every encoding, n = 4..12),
                  and accuracy change vs gate saving when loading is made cheaper (pooled reduction study)
  fig_surrogate   surrogate tuning: chosen bandwidths and accuracy change against full-grid tuning
  fig_positive    accuracy difference to the tuned RBF on real labels and on engineered labels (positive control)
  fig_dose        detection limit: advantage against the share of flipped engineered labels

Usage: python make_figures.py --results results --out results/figs
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# categorical slots 1-3 of the reference palette (validated); fixed per embedding, never per rank
COL = {"angle": "#2a78d6", "iqp": "#eb6834", "zz": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
NAME = {"angle": "Angle", "iqp": "IQP", "zz": "ZZ"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})
W1, W2 = 4.4, 5.15  # Springer Nature text width is 5.15 in; figures are drawn at final size


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def label(k):
    if "/" not in k:
        return k
    e, c = k.split("/")
    return {"angle": "Angle", "iqp": "IQP", "zz": "ZZ", "amplitude": "Amplitude", "classical": ""}[e] + \
        {"fidelity": "-F", "projected": "-P", "linear": "Linear", "rbf_scale": "RBF (default $\\gamma$)",
         "poly2_normalised": "Poly-2 (tuned)", "rbf": "RBF (tuned)", "gbm": "Gradient boosting",
         "rf": "Random forest", "knn": "kNN", "mlp": "MLP"}[c] if e != "amplitude" else "Amplitude-F"


def style_of(k):
    e = k.split("/")[0]
    return COL.get(e, "#6b6a66"), ("-" if k.endswith("fidelity") else "--"), ("o" if k.endswith("fidelity") else "s")


# ---------------------------------------------------------------- pipeline diagram
def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(W2, 5.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.grid(False)
    left = [("Datasets", "16 datasets (8 scikit-learn,\n8 OpenML), $N\\leq500$", "#eef3fb"),
            ("Resampling", "stratified 10-fold,\n2 repeats = 20 paired splits", "#eef3fb"),
            ("Preprocessing", "standardise, PCA to $n$, min-max\nto $[0,\\pi]$ (fit on train only)", "#eef3fb"),
            ("Equal tuning budget", "3-fold CV, same bandwidth and\n$C$ grid for every kernel", "#fdf0ea"),
            ("Scores", "accuracy, balanced accuracy,\nmacro-F1 (paired per split)", "#fdf0ea")]
    right = [("Loading circuits", "angle, IQP, ZZ, amplitude;\nfidelity and projected; $n=4$ to 12", "#e9f6f1"),
             ("Classical baselines", "tuned RBF, linear, poly-2,\ntree ensembles, $k$NN, MLP", "#e9f6f1"),
             ("Loading bill and levers", "gates, circuits, shots, grid;\nreduction, surrogate, Nystrom", "#e9f6f1"),
             ("Controls", "fixed bandwidth; positive control\nand detection limit; device replay", "#e9f6f1"),
             ("Statistics and diagnostics", "corrected $t$, TOST $\\pm0.02$;\ngeometry, stationarity, rank", "#fdf0ea")]
    w, h, pitch = 44, 13, 19
    for col, (rows, x0, title) in enumerate(((left, 2, "Protocol"), (right, 54, "Models, cost and inference"))):
        ax.text(x0, 98, title, fontsize=9, color=INK2, va="top", fontweight="bold")
        for i, (head, body, c) in enumerate(rows):
            y = 80 - i * pitch
            ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=c, ec=INK2, lw=0.8))
            ax.text(x0 + w / 2, y + h - 3.2, head, ha="center", va="center", fontsize=9, fontweight="bold")
            ax.text(x0 + w / 2, y + 4.4, body, ha="center", va="center", fontsize=8, linespacing=1.3)
            if i < 4:
                ax.annotate("", xy=(x0 + w / 2, y - pitch + h + 0.8 - 0.3), xytext=(x0 + w / 2, y - 0.6),
                            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    save(fig, out, "fig_pipeline")


# ---------------------------------------------------------------- forest
def fig_forest(res, out):
    f = res / "mds_paired_vs_rbf.csv"
    if not f.exists():
        return print("skip fig_forest (mds_paired_vs_rbf.csv missing)")
    d = pd.read_csv(f)
    if "metric" in d:
        d = d[d.metric == "test_acc"]
    d = d[(d.dataset == "POOLED") & (d.kernel != "classical/rbf_scale")].sort_values("mean")
    fig, ax = plt.subplots(figsize=(W2 * 0.92, 0.3 * len(d) + 1.0))
    ax.axvspan(-0.02, 0.02, color="#e4ece6", zorder=0)
    ax.axvline(0, color=INK2, lw=0.8)
    for i, (_, r) in enumerate(d.iterrows()):
        c, ls, mk = style_of(r.kernel)
        eq = bool(r.equivalent_95ci)
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(r["mean"], i, mk, ms=6, mfc=c if eq else "white", mec=c, mew=1.4, zorder=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([label(k) for k in d.kernel])
    ax.set_xlabel("accuracy difference to tuned RBF (pooled, 95% CI)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_forest")


def fig_heatmap(res, out):
    f = res / "mds_paired_vs_rbf.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    if "metric" in d:
        d = d[d.metric == "test_acc"]
    d = d[(d.dataset != "POOLED") & d.kernel.str.contains("fidelity|projected") & ~d.kernel.str.startswith("amplitude")]
    piv = d.pivot_table(index="dataset", columns="kernel", values="mean")
    order = [f"{e}/{c}" for e in ("angle", "iqp", "zz") for c in ("fidelity", "projected")]
    piv = piv[[c for c in order if c in piv.columns]]
    cm = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f1f0ec", "#2a78d6"])
    lim = max(0.05, float(np.nanmax(np.abs(piv.values))))
    fig, ax = plt.subplots(figsize=(W2, 0.27 * len(piv) + 1.2))
    im = ax.imshow(piv.values, cmap=cm, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([label(k) for k in piv.columns])
    dsname = {"breast_cancer": "breast cancer", "digits38": "digits 3/8", "digits17": "digits 1/7", "digits49": "digits 4/9",
              "ilpd": "ILPD", "pima": "Pima", "blood": "blood transf.", "haberman": "Haberman"}
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([dsname.get(x, x) for x in piv.index])
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.3f}", ha="center", va="center", fontsize=7, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("accuracy minus tuned RBF")
    cb.outline.set_visible(False)
    save(fig, out, "fig_heatmap")


# ---------------------------------------------------------------- qubits
def fig_qubits(res, out):
    f, g = res / "qs_paired_vs_rbf.csv", res / "qs_concentration.csv"
    if not f.exists():
        return print("skip fig_qubits (qs_paired_vs_rbf.csv missing)")
    d = pd.read_csv(f)
    d = d[d.metric == "test_acc"]
    fig = plt.figure(figsize=(W2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    for ax, cons, title in ((axs[0], "fidelity", "fidelity kernels"), (axs[1], "projected", "projected kernels")):
        ax.axhspan(-0.02, 0.02, color="#e4ece6", zorder=0)
        ax.axhline(0, color=INK2, lw=0.8)
        for e in ("angle", "iqp", "zz"):
            s = d[d.kernel == f"{e}/{cons}"].sort_values("n_qubits")
            if s.empty:
                continue
            ax.fill_between(s.n_qubits, s.lo, s.hi, color=COL[e], alpha=0.15, lw=0)
            ax.plot(s.n_qubits, s["mean"], "-o" if cons == "fidelity" else "--s", color=COL[e], ms=3.5, lw=1.4,
                    label=NAME[e])
        ax.set_xlabel("qubits $n$"); ax.set_title(title)
        ax.set_xticks(sorted(d.n_qubits.unique()))
        ax.set_ylim(-0.045, 0.03)
    axs[0].set_ylabel("accuracy minus tuned RBF")
    axs[1].set_ylabel("")
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color=COL[e], lw=1.6, label=NAME[e]) for e in COL] + \
         [Line2D([], [], color=INK2, lw=1.2, ls="-", marker="o", ms=3.5, label="fidelity"),
          Line2D([], [], color=INK2, lw=1.2, ls="--", marker="s", ms=3.5, label="projected")]
    ax = axs[2]
    if g.exists():
        c = pd.read_csv(g)
        for k, s in c.groupby("kernel"):
            if k.startswith("amplitude"):
                continue
            col, ls, mk = style_of(k)
            s = s.sort_values("n_max")
            ax.plot(s.n_max, s.offdiag_sd, ls, marker=mk, color=col, ms=3.5, lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("qubits $n$"); ax.set_title("kernel spread (concentration), tuned bandwidth")
        ax.set_ylabel("SD of off-diag. Gram entries")
        ax.set_xticks(sorted(c.n_max.unique()))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=hl, ncol=5, loc="lower center", fontsize=8, bbox_to_anchor=(0.5, -0.01))
    save(fig, out, "fig_qubits")


# ---------------------------------------------------------------- gates
def fig_gates(res, out):
    f = res / "gate_counts.csv"
    if not f.exists():
        return print("skip fig_gates")
    g = pd.read_csv(f)
    g = g[(g["mode"] == "routed") & (g.opt_level == 3)]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.6), sharey=True)
    for ax, circ, title in ((axs[0], "state", "state-preparation circuit"), (axs[1], "fidelity", "fidelity circuit")):
        for e, ls_, mk_, z_ in (("angle", "-", "o", 2), ("zz", "-", "o", 2), ("iqp", "--", "s", 3)):
            s = g[(g.embedding == e) & (g.circuit == circ)].sort_values("n")
            ax.plot(s.n, s.twoq_median.clip(lower=0.8), ls_, marker=mk_, color=COL[e], ms=3.5, lw=1.4, zorder=z_,
                    label=NAME[e] + (" (0 gates, drawn at 0.8)" if e == "angle" else (" (same count as ZZ)" if e == "iqp" else "")))
        n = np.arange(g.n.min(), g.n.max() + 1)
        ax.plot(n, 2 ** n - n - 1, ":", color=INK2, lw=1, label="exact amplitude loading $2^n-n-1$" if circ == "state" else None)
        ax.set_yscale("log"); ax.set_xlabel("qubits $n$"); ax.set_title(title)
    axs[0].set_ylabel("two-qubit gates (routed, opt. level 3)")
    axs[0].set_ylim(0.5, g.twoq_median.max() * 12)
    for ax in axs:
        ax.set_xticks(range(2, 13, 2))
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, out, "fig_gates")


# ---------------------------------------------------------------- reduction
def fig_reduction(res, out):
    f = res / "reduction_table.csv"
    if not f.exists():
        return print("skip fig_reduction")
    t = pd.read_csv(f)
    t = t[(t.reduction != "none") | (t["sample"] != "full")].dropna(subset=["gate_ratio"])
    t = t[~t.exploratory_post_hoc]
    method = {"pca": ("#2a78d6", "PCA"), "randproj": ("#eb6834", "random projection"), "none": ("#1baf7a", "sample reduction only")}
    fig, ax = plt.subplots(figsize=(W2 * 0.95, 3.5))
    ax.axhspan(-0.02, 0.02, color="#e4ece6", zorder=0)
    ax.axhline(0, color=INK2, lw=0.8)
    for m, (c, lab) in method.items():
        s = t[t.reduction == m]
        if s.empty:
            continue
        ax.errorbar(s.gate_ratio, s.delta, yerr=[s.delta - s.ci_lo, s.ci_hi - s.delta], fmt="o", color=c, ms=3.5, lw=0.7,
                    alpha=0.8, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    lo_, hi_ = float(t.delta.min()), float(t.delta.max())
    ax.set_ylim(max(-0.45, lo_ - 0.06), hi_ + 0.06)
    ax.set_xlabel("two-qubit-gate saving (unreduced / reduced)")
    ax.set_ylabel("accuracy change vs. unreduced")
    ax.legend(loc="lower left")
    save(fig, out, "fig_reduction")


# ---------------------------------------------------------------- shots
def fig_shots(res, out):
    f = res / "shots_summary.csv"
    if not f.exists():
        return print("skip fig_shots")
    s = pd.read_csv(f)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.1))
    for k, g in s.groupby("kernel"):
        col, ls, mk = style_of(k)
        g = g.sort_values("shots")
        axs[0].plot(g.shots, g.loss, ls, marker=mk, color=col, ms=3.5, lw=1.2, label=label(k))
        axs[1].plot(g.shots, g.gram_rms_err, ls, marker=mk, color=col, ms=3.5, lw=1.2)
    x = np.array([s.shots.min(), s.shots.max()], float)
    ref = s[s.shots == s.shots.min()].gram_rms_err.median() * np.sqrt(s.shots.min() / x)
    axs[1].plot(x, ref, ":", color=INK2, lw=1)
    axs[1].text(x[1] * 0.45, ref[1] * 1.25, "$\\propto N^{-1/2}$", color=INK2, fontsize=8, ha="right")
    axs[1].set_yscale("log")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("shots per circuit $N$")
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[0].set_ylabel("accuracy loss vs. exact kernel"); axs[1].set_ylabel("RMS Gram error (off-diagonal)")
    axs[0].legend(ncol=1, loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_shots")


# ---------------------------------------------------------------- device
def fig_device(res, out):
    f = res / "device_summary.csv"
    if not f.exists():
        return print("skip fig_device")
    s = pd.read_csv(f)
    backs = list(dict.fromkeys(s.backend))
    kern = [f"{e}/{c}" for e in ("angle", "iqp", "zz") for c in ("fidelity", "projected")]
    marks = ["o", "s", "^", "D"]
    from matplotlib.lines import Line2D
    panels = [("change_shot", "shot noise only (control)\naccuracy: sampled minus exact"),
              ("change_noise", "device noise beyond shot noise\naccuracy: device replay minus sampled")]
    panels = [p_ for p_ in panels if f"{p_[0]}_mean" in s]
    if not panels:
        panels = [("change", "accuracy: device replay minus exact")]
    fig, axs = plt.subplots(len(panels), 1, figsize=(W2, 5.2), sharex=True, squeeze=False)
    axs = axs.T
    off = 0.22
    for ax, (col, title) in zip(axs[0], panels):
        ax.axhline(0, color=INK2, lw=0.8)
        for bi, b in enumerate(backs):
            for ki, k in enumerate(kern):
                r = s[(s.backend == b) & (s.kernel == k)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = ki + (bi - (len(backs) - 1) / 2) * off
                c = COL[k.split("/")[0]]
                lo, hi = r[f"{col}_lo"], r[f"{col}_hi"]
                if np.isfinite(lo):
                    ax.plot([x, x], [lo, hi], color=c, lw=1.2)
                ax.plot(x, r[f"{col}_mean"], marks[bi % 4], color=c, ms=4.5, mfc=c if k.endswith("fidelity") else "white", mew=1.1)
        ax.set_xticks(range(len(kern))); ax.set_xticklabels([label(k) for k in kern], rotation=0)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("accuracy change (95% CI)")
        ax.grid(axis="x", visible=False)
    fig.legend(handles=[Line2D([], [], marker=marks[i % 4], color=INK2, ls="", ms=5, mfc="none", label=b)
                               for i, b in enumerate(backs)], title="device model (2025 snapshot)", fontsize=8,
                      title_fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, out, "fig_device")


# ---------------------------------------------------------------- frontier (accuracy vs. quantum evaluation cost)
def fig_frontier(res, out):
    f, c, sf = res / "frontier.csv", res / "frontier_classical.csv", res / "frontier_sensitivity.csv"
    if not (f.exists() and c.exists()):
        return print("skip fig_frontier (frontier.csv missing)")
    d = pd.read_csv(f)
    cl = pd.read_csv(c)
    rbf = float(cl[cl.kernel.str.startswith("RBF")].accuracy.iloc[0])
    sens = pd.read_csv(sf) if sf.exists() else None
    fig, axs = plt.subplots(2, 1, figsize=(W2, 6.0), gridspec_kw=dict(height_ratios=[1.15, 1]))
    ax = axs[0]
    for i, (_, r) in enumerate(d.iterrows()):
        e = r.kernel.split("-")[0].lower()
        fid = r.kernel.endswith("F")
        col = COL[e]
        y = len(d) - 1 - i
        if sens is not None:
            sub = sens[(sens.kernel == r.kernel) & (sens.rule == "spread") & sens.param.isin([1.0, 4.0])]
            if len(sub) == 2:
                ax.plot(sub.circuit_executions, [y, y], color=col, alpha=0.4, lw=2.4, solid_capstyle="round")
        ax.plot(r.circuit_executions, y, "o" if fid else "s", ms=6, mfc=col if fid else "white", mec=col, mew=1.3, zorder=3)
        ax.text(4e6, y, f"{r.accuracy - rbf:+.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{k} ({g:.0f})" for k, g in zip(d.kernel[::-1], d.twoq_per_circuit[::-1])])
    ax.set_xscale("log")
    ax.set_xlim(2.5e4, 3e7)
    ax.set_ylim(-0.6, len(d) + 0.2)
    ax.text(4e6, len(d) - 0.45, "acc. minus RBF", fontsize=8, color=INK2)
    ax.set_xlabel("circuit executions per Gram matrix (bar: shots at $c=1$ to $4$)", fontsize=8.5)
    ax.set_title("(a) evaluation cost; label: two-qubit gates per circuit", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax2 = axs[1]
    names = list(d.kernel)
    x = np.arange(len(names))
    vals = d.twoq_gate_executions.values.astype(float)
    floor = 1e5
    ax2.bar(x, np.where(vals > 0, vals, floor), color=[COL[k.split("-")[0].lower()] if v > 0 else "#cfcdc8" for k, v in zip(names, vals)],
            width=0.65, bottom=0)
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right")
    for xi, v in zip(x, vals):
        ax2.text(xi, (v if v > 0 else floor) * 1.15, "0" if v == 0 else f"{v:.1e}", ha="center", fontsize=8)
    ax2.set_ylim(floor, vals.max() * 6)
    ax2.set_ylabel("two-qubit gate executions")
    ax2.set_title("(b) entangling-gate cost", fontsize=9)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "fig_frontier")


# ---------------------------------------------------------------- loading cost vs accuracy (central figure)
def fig_loading(res, out):
    fq, fg, fl = res / "qs_paired_vs_rbf.csv", res / "gate_counts.csv", res / "loading_counts.csv"
    if not (fq.exists() and fg.exists() and fl.exists()):
        return print("skip fig_loading")
    q = pd.read_csv(fq)
    q = q[q.metric == "test_acc"] if "metric" in q else q
    g = pd.read_csv(fg)
    g = g[(g["mode"] == "routed") & (g.opt_level == 3) & (g.circuit == "state")].set_index(["embedding", "n"]).twoq_median
    l = pd.read_csv(fl)
    l = l[(l.family == "dense_real") & (l.toolchain == "qiskit_routed_opt3")].groupby("n").twoq.median()
    fr = res / "v2_reduction_pooled.csv"
    two = fr.exists()
    fig, axs = plt.subplots(1, 2 if two else 1, figsize=(W2, 3.4) if two else (W1, 3.4),
                            gridspec_kw={"width_ratios": [1.35, 1]} if two else None)
    ax = axs[0] if two else axs
    ax.axhspan(-0.02, 0.02, color="#e4ece6", zorder=0)
    ax.axhline(0, color=INK2, lw=0.8)
    AMP = "#7a5195"
    for k in ["angle/fidelity", "angle/projected", "iqp/fidelity", "iqp/projected", "zz/fidelity", "zz/projected",
              "amplitude/fidelity"]:
        s = q[q.kernel == k].sort_values("n_qubits")
        if s.empty:
            continue
        e, cons = k.split("/")
        if e == "amplitude":
            m = np.ceil(np.log2(s.n_qubits.values)).astype(int)
            x = np.array([l.get(int(v), np.nan) for v in m], float)
            c, mk, lab = AMP, "D", "Amplitude"
        else:
            x = np.array([g.get((e, int(v)), np.nan) for v in s.n_qubits], float)
            c, mk = COL[e], ("o" if cons == "fidelity" else "s")
            lab = f"{NAME[e]}-{'F' if cons == 'fidelity' else 'P'}"
        off = {"fidelity": 0.0, "projected": 0.0}
        y = s["mean"].values
        ax.plot(x, y, "-", color=c, lw=0.7, alpha=0.5)
        ax.errorbar(x, y, yerr=[y - s.lo.values, s.hi.values - y], fmt=mk, color=c, ms=3.6, lw=0.7, capsize=0,
                    mfc=c if cons == "fidelity" else "white", label=lab)
        if e == "amplitude":
            for xi, yi, n in zip(x, y, s.n_qubits):
                ax.annotate(f"d={n}", (xi, yi), textcoords="offset points", xytext=(-22, -3), fontsize=6.5, color=c)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, max(1.0, np.nanmax(g.values)) * 1.6)
    ax.set_xlabel("two-qubit gates to load one point\n(routed, opt. level 3)")
    ax.set_ylabel("accuracy minus tuned RBF")
    ax.set_title("(a) what extra loading cost buys")
    h_, l_ = ax.get_legend_handles_labels()
    if two:
        r = pd.read_csv(fr)
        scope = "real datasets" if (r.scope == "real datasets").any() else "all datasets"
        r = r[r.scope == scope]
        ax2 = axs[1]
        ax2.axhspan(-0.02, 0.02, color="#e4ece6", zorder=0)
        ax2.axhline(0, color=INK2, lw=0.8)
        meth = {"pca": ("#2a78d6", "PCA"), "randproj": ("#eb6834", "random projection"), "no dim": ("#1baf7a", "fewer points only")}
        for key, (c, lab) in meth.items():
            sub = r[r.setting.str.startswith(key)].dropna(subset=["median_gate_saving"])
            if sub.empty:
                continue
            ax2.errorbar(sub.median_gate_saving, sub["mean"], yerr=[sub["mean"] - sub.lo, sub.hi - sub["mean"]], fmt="o",
                         color=c, ms=3.6, lw=0.7, label=lab)
        ax2.set_xscale("log")
        ax2.set_xticks([1, 2, 5, 10, 20]); ax2.set_xticklabels(["1", "2", "5", "10", "20"]); ax2.minorticks_off()
        ax2.set_xlabel("two-qubit-gate saving\n(unreduced / reduced)")
        ax2.set_ylabel("accuracy change vs. unreduced")
        ax2.set_title("(b) what cheaper loading costs")
        ax2.legend(fontsize=6.5, loc="lower left")
    fig.legend(h_, l_, loc="lower center", ncol=7, fontsize=6.5, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3,
               columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "fig_loading")


def fig_surrogate(res, out):
    fc, fp = res / "sur_cells.csv", res / "sur_paired.csv"
    if not (fc.exists() and fp.exists()):
        return print("skip fig_surrogate (sur_cells.csv / sur_paired.csv missing)")
    c = pd.read_csv(fc)
    p = pd.read_csv(fp)
    p = p[(p.metric == "test_acc") & (p.reference == "full_grid")]
    kern = ["iqp/fidelity", "iqp/projected", "zz/fidelity", "zz/projected"]
    meths = [("metric_map", "metric-matched", "o"), ("rbf_map", "RBF map", "s"), ("angle", "angle transfer", "D")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.08),
                                 sharey=True)
    rng = np.random.default_rng(0)
    for i, k in enumerate(kern):
        col = COL[k.split("/")[0]]
        g = c[(c.kernel == k) & (c.method == "full_grid")].s_used.values
        ys = i + rng.uniform(-0.18, 0.18, len(g))
        a1.scatter(g * np.exp(rng.uniform(-0.12, 0.12, len(g))), ys, s=4, color=col, alpha=0.25, lw=0)
        a1.plot(np.median(g), i, "|", ms=14, mew=2.2, color=INK)
        for j, (m, lab, mk) in enumerate(meths):
            v = c[(c.kernel == k) & (c.method == m)].s_used
            if len(v):
                a1.plot(v.median(), i + 0.26 + 0.12 * j, mk, ms=4.5, mfc="white" if m != "metric_map" else col, mec=col,
                        mew=1.1)
        for j, (m, lab, mk) in enumerate(meths):
            r = p[(p.kernel == k) & (p.method == m) & (p.dataset == "POOLED")]
            if not len(r):
                continue
            r = r.iloc[0]
            y = i + (j - 1) * 0.24
            if m == "metric_map":
                d = p[(p.kernel == k) & (p.method == m) & ~p.dataset.str.startswith("POOLED")]
                a2.scatter(d["mean"], y + rng.uniform(-0.05, 0.05, len(d)), s=5, color=col, alpha=0.35, lw=0, zorder=2)
            a2.plot([r.lo, r.hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=3)
            a2.plot(r["mean"], y, mk, ms=5, mfc=col if r.noninferior else "white", mec=col, mew=1.3, zorder=4)
    a1.set_xscale("log")
    a1.set_xlabel("bandwidth $s$")
    a1.set_yticks(range(len(kern)))
    a1.set_yticklabels([label(k) for k in kern])
    a1.set_ylim(-0.5, len(kern) - 0.25)
    a1.invert_yaxis()
    a1.grid(axis="y", visible=False)
    a1.set_title("(a) bandwidth chosen", loc="left", fontsize=8)
    a2.axvspan(-0.02, 0.02, color="#e4ece6", zorder=0)
    a2.axvline(0, color=INK2, lw=0.8)
    a2.set_xlabel("accuracy change vs full-grid tuning")
    a2.grid(axis="y", visible=False)
    a2.set_title("(b) accuracy change", loc="left", fontsize=8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk, ls="", mfc="white" if m != "metric_map" else INK2, mec=INK2, label=lab)
         for m, lab, mk in meths]
    a2.legend(handles=h, loc="lower left", fontsize=7, handletextpad=0.2)
    save(fig, out, "fig_surrogate")


def fig_positive(res, out):
    fm, fr = res / "mds_paired_vs_rbf.csv", res / "rel_paired.csv"
    if not (fm.exists() and fr.exists()):
        return print("skip fig_positive (mds_paired_vs_rbf.csv / rel_paired.csv missing)")
    m = pd.read_csv(fm)
    if "metric" in m:
        m = m[m.metric == "test_acc"]
    m = m[m.dataset == "POOLED"].set_index("kernel")
    r = pd.read_csv(fr)
    r = r[r.dataset == "POOLED"]
    ns = sorted(r.n_qubits.unique())
    groups = [("real\nlabels", None, None)] + [(f"{L.upper()}\n$n={n}$", L, n) for L in ("iqp", "zz") for n in ns]
    fs = res / "relsur_paired.csv"
    sur = pd.read_csv(fs) if fs.exists() else None
    if sur is not None:
        sur = sur[(sur.dataset == "POOLED") & (sur.reference == "tuned_rbf") & (sur.method == "metric_map")]
    kern = ["angle/fidelity", "iqp/fidelity", "zz/fidelity", "classical/rf"]
    fig, ax = plt.subplots(figsize=(W2, 2.8))
    ax.axhspan(-0.02, 0.02, color="#e4ece6", zorder=0)
    ax.axhline(0, color=INK2, lw=0.8)
    off = np.linspace(-0.3, 0.24, len(kern))
    for gi, (gl, L, n) in enumerate(groups):
        if gi:
            ax.axvline(gi - 0.5, color=GRID, lw=0.8)
        for k, o in zip(kern, off):
            if L is None:
                if k not in m.index:
                    continue
                row = m.loc[k]
            else:
                q = r[(r.labeller == L) & (r.n_qubits == n) & (r.kernel == k)]
                if not len(q):
                    continue
                row = q.iloc[0]
            col = COL.get(k.split("/")[0], "#6b6a66")
            mk = "o" if k != "classical/rf" else "^"
            sig = row.lo > 0
            ax.plot([gi + o, gi + o], [row.lo, row.hi], color=col, lw=1.6, solid_capstyle="round")
            ax.plot(gi + o, row["mean"], mk, ms=5, mfc=col if sig else "white", mec=col, mew=1.3, zorder=3)
            if sur is not None and L is not None and k == f"{L}/fidelity":
                q = sur[(sur.labeller == L) & (sur.n_qubits == n) & (sur.kernel == k)]
                if len(q):
                    ax.plot(gi + o + 0.09, q["mean"].iloc[0], "x", ms=6, mew=1.6, color=INK, zorder=4)
                    ax.plot([gi + o + 0.09] * 2, [q.lo.iloc[0], q.hi.iloc[0]], color=INK, lw=0.9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("accuracy difference\nto tuned RBF (95% CI)")
    ax.set_ylim(-0.3, 0.36)
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o" if k != "classical/rf" else "^", ls="", mfc=COL.get(k.split("/")[0], "#6b6a66"),
                mec=COL.get(k.split("/")[0], "#6b6a66"), label=label(k)) for k in kern]
    if sur is not None:
        h.append(Line2D([], [], marker="x", ls="", color=INK, label="labelling kernel, surrogate-tuned"))
    ax.legend(handles=h, loc="upper left", ncol=3, fontsize=7, handletextpad=0.2, columnspacing=0.8)
    save(fig, out, "fig_positive")


def fig_dose(res, out):
    f = res / "dose_paired.csv"
    if not f.exists():
        return print("skip fig_dose (dose_paired.csv missing)")
    d = pd.read_csv(f)
    pooled = d[d.dataset == "POOLED"]
    per = d[~d.dataset.str.startswith("POOLED")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw=dict(wspace=0.35))
    a1.axhline(0, color=INK2, lw=0.8)
    for (L, n), g in pooled.groupby(["labeller", "n_qubits"]):
        g = g.sort_values("rho")
        col = COL[L]
        ls = "-" if n == min(pooled.n_qubits) else "--"
        x = g.rho + (0.006 if ls == "--" else -0.006) + (0.003 if L == "zz" else 0)
        a1.plot(x, g["mean"], ls, color=col, lw=1.1)
        for xi, (_, r) in zip(x, g.iterrows()):
            a1.plot([xi, xi], [r.lo, r.hi], color=col, lw=1.2)
            a1.plot(xi, r["mean"], "o", ms=4, mfc=col if r.detected else "white", mec=col, mew=1.1)
    a1.set_xlabel("share of engineered labels flipped")
    a1.set_ylabel("labelling kernel $-$ tuned RBF")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[L], ls=ls, marker="o", ms=4, label=f"{L.upper()}, $n={n}$")
         for L in ("iqp", "zz") for n, ls in zip(sorted(pooled.n_qubits.unique()), ("-", "--"))]
    a1.legend(handles=h, fontsize=6.5, loc="upper right")
    a1.set_title("(a) pooled advantage", loc="left", fontsize=8)
    bins = [-1, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 1]
    labs = ["$\\leq$0", "0–0.02", "0.02–0.04", "0.04–0.06", "0.06–0.08", "0.08–0.1", ">0.1"]
    cut = pd.cut(per["mean"], bins, labels=labs)
    rate = per.groupby(cut, observed=False).detected.agg(["mean", "size"])
    a2.bar(range(len(labs)), rate["mean"].fillna(0), color=INK2, width=0.7)
    for i, (m_, n_) in enumerate(zip(rate["mean"].fillna(0), rate["size"])):
        a2.text(i, m_ + 0.03, f"{int(n_)}", ha="center", fontsize=6.5, color=INK2)
    a2.set_xticks(range(len(labs)))
    a2.set_xticklabels(labs, rotation=45, ha="right", fontsize=6.5)
    a2.set_ylim(0, 1.12)
    a2.set_xlabel("advantage on a single dataset")
    a2.set_ylabel("share detected")
    a2.grid(axis="x", visible=False)
    a2.set_title("(b) single datasets (n above bars)", loc="left", fontsize=8)
    save(fig, out, "fig_dose")


def fig_motivation(out):
    """Introduction diagram: where data loading sits in a quantum machine-learning model and why it is paid many times."""
    from matplotlib.patches import FancyArrowPatch
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, head, body, fc, ec=INK2, lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h - 4.2, head, ha="center", va="center", fontsize=7.8, fontweight="bold", linespacing=1.1)
        ax.text(x + w / 2, y + 6.3, body, ha="center", va="center", fontsize=6.2, linespacing=1.25, color=INK)

    def arrow(x0, y0, x1, y1, col=INK2, **kw):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=col, lw=1.0, **kw))

    yb, hb = 27, 22
    box(0.5, yb, 13.5, hb, "Classical\ndata", "features\n$\\mathbf{x}\\in\\mathbb{R}^d$", "#eef3fb")
    ax.add_patch(FancyBboxPatch((18.5, 13), 61, 39.5, boxstyle="round,pad=0.4,rounding_size=1.5", fc="none",
                                ec=INK2, lw=1.0, ls=(0, (4, 3))))
    ax.text(49, 57.5, "Quantum machine-learning process", ha="center", va="center", fontsize=8.2, color=INK2,
            fontweight="bold")
    box(20.5, yb, 17.5, hb, "Data\nloading", "embedding $U(\\mathbf{x})$\nangle, IQP,\nZZ, amplitude", "#fdf0ea",
        ec=COL["iqp"], lw=1.5)
    box(40.5, yb, 17.5, hb, "Quantum\nprocessing", "overlap\n$U^\\dagger(\\mathbf{x}')U(\\mathbf{x})$\nor Pauli terms", "#e9f6f1")
    box(60.5, yb, 17.5, hb, "Measure-\nment", "quantum $\\rightarrow$\nclassical; shots\ngive $k(\\mathbf{x},\\mathbf{x}')$", "#e9f6f1")
    box(84.5, yb, 15, hb, "Classical\noutput", "Gram matrix,\nSVM,\nprediction", "#eef3fb")
    mid = yb + hb / 2
    arrow(14.2, mid, 20.1, mid)
    arrow(38.2, mid, 40.1, mid)
    arrow(58.2, mid, 60.1, mid)
    arrow(80.2, mid, 84.1, mid)
    ax.add_patch(FancyArrowPatch((69.2, yb - 0.6), (29.2, yb - 0.6), connectionstyle="arc3,rad=-0.28", arrowstyle="-|>",
                                 mutation_scale=9, color=COL["iqp"], lw=1.3))
    ax.text(49, 16.8, "repeated for every kernel entry, shot and bandwidth tried", ha="center", va="center",
            fontsize=6.3, color=COL["iqp"])
    ax.text(49, 6.5, "loading bill = (bandwidths tried $\\times$ circuits) $\\times$ shots $\\times$ two-qubit gates per load",
            ha="center", va="center", fontsize=7.3, color=INK)
    save(fig, out, "fig_motivation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_pipeline(out)
    fig_motivation(out)
    for fn in (fig_forest, fig_heatmap, fig_qubits, fig_gates, fig_reduction, fig_shots, fig_device, fig_frontier, fig_loading,
               fig_surrogate, fig_positive, fig_dose):
        fn(res, out)


if __name__ == "__main__":
    main()
