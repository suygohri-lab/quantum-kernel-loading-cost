"""Dimension and sample reduction before amplitude loading (hypothesis H1 of the paper).

For every dataset: standardise -> reduce (none | PCA | Gaussian random projection to k = dpad/8, dpad/4, dpad/2)
-> min-max to [0, 1] -> zero-pad to a power of two -> amplitude kernel |<x|x'>|^2 = (x.x')^2/(|x|^2|x'|^2) (closed form)
-> SVM with C tuned (3-fold CV).  Sample reduction keeps 50% or 25% of each class (uniform or per-class k-means
representatives), crossed with no dimension reduction and PCA to dpad/2.  An RBF SVM on the same reduced data
(5 width multipliers x 4 C) is the classical baseline.

Gate accounting: total two-qubit gates = 2 x G_load(q) x (M'(M'-1)/2 + n_test x M'), with M' retained training points and
G_load(q) the measured Qiskit count (transpiled to {cx, rz, sx, x}, optimisation level 3) of exact loading of a dense real
q-qubit vector.  Absolute counts are reported next to the ratio (q = 1 needs no entangling gate: ratio undefined).

Non-inferiority: one-sided corrected resampled t test of H0: Delta <= -delta, delta = 0.02, alpha = 0.05, where Delta is the
paired accuracy difference to the unreduced amplitude kernel on the same resample.  The CI half-width is reported so
that an inconclusive result (half-width > delta) is visible as such.

Datasets: synth_r4/r16/r64 (d = 64 features, r informative; sklearn make_classification, 2 clusters per class,
class_sep 2.0, 2% label noise), breast_cancer,
wine, digits38; --latent adds low-rank latent sets lat_r4/r16/r64 (x = zW + 0.1 eta, class shift +/-1.5 along one latent
direction, orthonormal W): EXPLORATORY / post hoc, labelled so in the output.

Design: N <= 500, repeated stratified 10-fold x 2 (same as multi_dataset_study.py).

Outputs (--out): reduction_cells.csv, reduction_table.csv, reduction_report.txt
Usage: python reduction_study.py --out results --n-jobs 4 [--latent] [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn import datasets as skd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from datasets_cv import BASE_SEED, load_full, resamples
from gram_spread import tune
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

DELTA = 0.02
RBF_MULT = [0.1, 0.3, 1.0, 3.0, 10.0]
C_GRID = [0.1, 1.0, 10.0, 100.0]
_LOAD_CACHE: dict[int, int] = {}


def loader_gates(q, trials=10):
    """Two-qubit gates of exact amplitude loading of a dense real q-qubit vector (Qiskit, opt level 3)."""
    if q in _LOAD_CACHE:
        return _LOAD_CACHE[q]
    if q <= 1:
        _LOAD_CACHE[q] = 0
        return 0
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation

    rng = np.random.default_rng(0)
    cnt = []
    for _ in range(trials):
        v = np.abs(rng.normal(size=2 ** q))
        qc = QuantumCircuit(q)
        qc.append(StatePreparation(v / np.linalg.norm(v)), range(q))
        cnt.append(transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3,
                             seed_transpiler=0).num_nonlocal_gates())
    _LOAD_CACHE[q] = int(np.median(cnt))
    return _LOAD_CACHE[q]


def synth(r, n=500):
    return skd.make_classification(n_samples=n, n_features=64, n_informative=r, n_redundant=0, n_repeated=0,
                                   n_classes=2, n_clusters_per_class=2, class_sep=2.0, flip_y=0.02,
                                   shuffle=True, random_state=BASE_SEED)


def latent(r, n=500):
    rng = np.random.default_rng(BASE_SEED + r)
    y = rng.integers(0, 2, n)
    z = rng.normal(size=(n, r))
    z[:, 0] += np.where(y == 1, 1.5, -1.5)
    W = np.linalg.qr(rng.normal(size=(64, r)))[0].T  # r x 64, orthonormal rows
    return z @ W + 0.1 * rng.normal(size=(n, 64)), y


def get_dataset(name):
    if name.startswith("synth_r"):
        return synth(int(name[7:]))
    if name.startswith("lat_r"):
        return latent(int(name[5:]))
    X, y, _ = load_full(name)
    return X, y


def next_pow2(d):
    return 1 << int(np.ceil(np.log2(max(d, 2))))


def configs(dpad):
    c = [dict(reduction="none", k=dpad, sample="full", frac=1.0)]
    for m in ("pca", "randproj"):
        for k in (dpad // 8, dpad // 4, dpad // 2):
            c.append(dict(reduction=m, k=k, sample="full", frac=1.0))
    for samp in ("uniform", "kmeans"):
        for frac in (0.5, 0.25):
            c.append(dict(reduction="none", k=dpad, sample=samp, frac=frac))
            c.append(dict(reduction="pca", k=dpad // 2, sample=samp, frac=frac))
    return c


def reduce_dims(Xtr, Xte, method, k, seed):
    if method == "none":
        return Xtr, Xte
    if method == "pca":
        p = PCA(n_components=k, random_state=seed).fit(Xtr)
    else:
        p = GaussianRandomProjection(n_components=k, random_state=seed).fit(Xtr)
    return p.transform(Xtr), p.transform(Xte)


def pick_samples(Ztr, ytr, sample, frac, seed):
    if sample == "full":
        return np.arange(len(ytr))
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(ytr):
        ci = np.where(ytr == c)[0]
        m = max(2, int(round(frac * len(ci))))
        if sample == "uniform":
            idx += list(rng.choice(ci, m, replace=False))
        else:
            km = KMeans(n_clusters=m, n_init=3, random_state=seed).fit(Ztr[ci])
            for j in range(m):
                d = ((Ztr[ci] - km.cluster_centers_[j]) ** 2).sum(1)
                idx.append(ci[int(np.argmin(d))])
    return np.array(sorted(set(idx)))


def fit_eval(Ktr, Kte, ytr, yte, seed):
    cv, _, C = tune({"k": np.vstack([np.hstack([Ktr, Kte.T]), np.hstack([Kte, np.zeros((len(yte), len(yte)))])])},
                    ytr, len(ytr), seed)
    clf = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)
    return float(clf.score(Kte, yte))


def one_resample(name, ri, Xtr_raw, Xte_raw, ytr, yte, dpad_all):
    seed = BASE_SEED + ri
    sc = StandardScaler().fit(Xtr_raw)
    Xs_tr, Xs_te = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    dpad = dpad_all
    rows = []
    for cfg in configs(dpad):
        Ztr, Zte = reduce_dims(Xs_tr, Xs_te, cfg["reduction"], cfg["k"], seed)
        keep = pick_samples(Ztr, ytr, cfg["sample"], cfg["frac"], seed)
        Ztr_k, ytr_k = Ztr[keep], ytr[keep]
        mm = MinMaxScaler((0.0, 1.0)).fit(Ztr_k)
        A, B = mm.transform(Ztr_k), mm.transform(Zte)
        k_eff = A.shape[1]
        q_dim = next_pow2(k_eff)
        padA = np.zeros((len(A), q_dim)); padA[:, :k_eff] = A
        padB = np.zeros((len(B), q_dim)); padB[:, :k_eff] = B
        nA = padA / np.maximum(np.linalg.norm(padA, axis=1, keepdims=True), 1e-12)
        nB = padB / np.maximum(np.linalg.norm(padB, axis=1, keepdims=True), 1e-12)
        acc_amp = fit_eval((nA @ nA.T) ** 2, (nB @ nA.T) ** 2, ytr_k, yte, seed)
        # RBF baseline on the same reduced (unpadded) data
        g0 = 1.0 / (k_eff * max(A.var(), 1e-12))
        d2tr = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        d2te = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        Kfull = {}
        for m_ in RBF_MULT:
            K = np.exp(-m_ * g0 * np.vstack([np.hstack([d2tr, d2te.T]),
                                             np.hstack([d2te, np.zeros((len(B), len(B)))])]))
            Kfull[m_] = K
        cv, mbest, Cbest = tune(Kfull, ytr_k, len(ytr_k), seed)
        Kb = Kfull[mbest]
        clf = SVC(kernel="precomputed", C=Cbest).fit(Kb[:len(ytr_k), :len(ytr_k)], ytr_k)
        acc_rbf = float(clf.score(Kb[len(ytr_k):, :len(ytr_k)], yte))
        q = int(np.log2(q_dim))
        M, nte = len(ytr_k), len(yte)
        entries = M * (M - 1) // 2 + nte * M
        rows.append(dict(dataset=name, resample=ri, **cfg, M_train=M, qubits=q, acc_amp=acc_amp, acc_rbf=acc_rbf,
                         twoq_per_loader=loader_gates(q), kernel_entries=entries,
                         twoq_total=2 * loader_gates(q) * entries))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--latent", action="store_true", help="add exploratory post-hoc low-rank latent sets")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = ["synth_r4", "synth_r16", "synth_r64", "breast_cancer", "wine", "digits38"]
    if a.latent:
        names += ["lat_r4", "lat_r16", "lat_r64"]
    folds, repeats = (2, 1) if a.quick else (10, 2)
    if a.quick:
        names = names[:2] + names[4:5]
    jobs = []
    for name in names:
        X, y = get_dataset(name)
        dpad = next_pow2(X.shape[1])
        for ri, (tr, te) in enumerate(resamples(y, folds, repeats)):
            jobs.append((name, ri, X[tr], X[te], y[tr], y[te], dpad))
    print(f"{len(jobs)} jobs; n_jobs={a.n_jobs}")
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(one_resample)(*j) for j in jobs)
    cells = pd.DataFrame([r for rs in res for r in rs])
    cells["exploratory_post_hoc"] = cells.dataset.str.startswith("lat_")
    cells.to_csv(out / "reduction_cells.csv", index=False)
    print(f"compute time {time.time() - t0:.0f}s")

    ratio = 1.0 / (folds - 1)
    key = ["dataset", "reduction", "k", "sample", "frac"]
    ref = cells[(cells.reduction == "none") & (cells["sample"] == "full")].set_index(["dataset", "resample"])
    table = []
    for ks, g in cells.groupby(key):
        r0 = ref.loc[list(zip(g["dataset"], g["resample"]))]
        d = g.acc_amp.values - r0.acc_amp.values
        R = len(d)
        m = d.mean()
        sd = d.std(ddof=1) if R > 1 else 0.0
        se = np.sqrt(1.0 / R + ratio) * sd
        t = stats.t.ppf(0.975, max(R - 1, 1))
        p_ni = float(stats.t.sf((m + DELTA) / se, R - 1)) if se > 0 else (0.0 if m > -DELTA else 1.0)
        tot0 = r0.twoq_total.mean()
        tot = g.twoq_total.mean()
        table.append(dict(zip(key, ks), qubits=int(g.qubits.iloc[0]), acc=g.acc_amp.mean(), delta=m,
                          ci_lo=m - t * se, ci_hi=m + t * se, ci_half_width=t * se, p_noninferior=p_ni,
                          noninferior=bool(p_ni < 0.05), inconclusive=bool(t * se > DELTA and p_ni >= 0.05),
                          twoq_total_unreduced=tot0, twoq_total=tot,
                          gate_ratio=(tot0 / tot) if tot > 0 else np.nan, rbf_acc=g.acc_rbf.mean(),
                          exploratory_post_hoc=bool(g.exploratory_post_hoc.iloc[0])))
    tb = pd.DataFrame(table)
    tb.to_csv(out / "reduction_table.csv", index=False)
    lines = [f"DESIGN: N<=500, repeated stratified {folds}-fold x {repeats}; n_test/n_train = {ratio:.3f}; delta = {DELTA}",
             "inconclusive = CI half-width above delta and non-inferiority not shown", "",
             tb[~tb.exploratory_post_hoc].round(4).to_string(index=False)]
    lat = tb[tb.exploratory_post_hoc]
    if len(lat):
        lines += ["", "EXPLORATORY (post hoc, latent generator)", lat.round(4).to_string(index=False)]
    ni = tb[(~tb.exploratory_post_hoc) & ((tb.reduction != "none") | (tb["sample"] != "full"))]
    lines += ["", f"Non-inferior in {int(ni.noninferior.sum())} of {len(ni)} reduced configurations (pre-specified sets); "
              f"inconclusive in {int(ni.inconclusive.sum())}"]
    (out / "reduction_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
