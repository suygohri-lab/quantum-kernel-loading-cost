"""Finite-shot sampling and device-noise (replay) studies (finite-shot results and device-model replay of the paper).

PART 1  --part shots : finite-shot Gram perturbation
  Per dataset and resample the six quantum kernels are tuned on the ideal (exact) Gram (wide grid, 3-fold CV, same rule
  as everywhere else).  With (s*, C*) FIXED, the Gram is then replaced by a shot estimate with N shots per circuit:
    fidelity : every off-diagonal entry ~ Binomial(N, k)/N (train block symmetric, diagonal 1, PSD-clipped by eigenvalue
               clipping; test block not symmetrised or clipped)
    projected: every one-qubit Pauli expectation <P> ~ 2 Binomial(N, (1+<P>)/2)/N - 1, then the kernel is computed from
               the noisy vectors with the ideal gamma (PSD by construction)
  Reported: accuracy ideal vs N, loss, Gram error, and paired loss differences IQP/ZZ minus angle.

PART 2  --part device : device-model replay
  Aer density-matrix simulation with the noise model of the stored FakeSherbrooke calibration snapshot shipped with
  qiskit-ibm-runtime (a 2025 replay of a real device - NOT a measurement on hardware).  Parametrised template circuits
  are transpiled once (opt level 3, routed to the heavy-hex map) and bound per data pair.  A 40-train / 20-test subset
  of each resample, 1000 shots per circuit; hyper-parameters s*, C* from the ideal tuning on the same subset.
  Fidelity kernel: compute-uncompute circuit, entry = frequency of the all-zero string.  Projected kernel: 3 basis
  circuits per point, per-qubit Pauli expectations from counts.

Statistics: corrected resampled t (Nadeau-Bengio, ratio n_test/n_train = 1/(folds-1)), 95% CI, pooled over datasets.

Outputs (--out): shots_cells.csv shots_summary.csv shots_paired.csv  device_cells.csv device_summary.csv device_backends.csv
                 device_paired.csv device_gates.csv  shots_device_report.txt
Usage:
  python shots_device_study.py --out results --part both --R 10 --R-device 5 --n-jobs 4
  python shots_device_study.py --out results --quick
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from datasets_cv import BASE_SEED, load_full, preprocess, resamples
from extended_study import S_WIDE, corrected
from gram_spread import tune
from qkcommon import KINDS, bloch, embed_circuit, fidelity_gram, projected_gamma, projected_gram, states

SHOTS = [100, 1000, 10000]
SHOT_DATASETS = ["moons", "circles", "breast_cancer", "iris", "wine", "digits38"]
DEVICE_DATASETS = ["moons", "breast_cancer", "digits17", "wine", "ionosphere"]
FOLDS = 10


# --------------------------------------------------------------------------- shared helpers
def ideal_models(Xtr, Xte, ytr, seed, s_values=S_WIDE):
    """Tune all six quantum kernels on exact Grams.  Returns dict name -> info incl. s*, C*, states."""
    n = Xtr.shape[1]
    X = np.vstack([Xtr, Xte])
    ntr = len(ytr)
    out = {}
    for kind in KINDS:
        Kf, Kp, psis = {}, {}, {}
        for s in s_values:
            psi = states(X, s, kind, n)
            psis[s] = psi
            Kf[s] = fidelity_gram(psi, psi)
            F = bloch(psi, n).reshape(len(X), -1)
            Kp[s] = projected_gram(F, F, projected_gamma(F[:ntr]))
        for cons, Kd in (("fidelity", Kf), ("projected", Kp)):
            cv, s, C = tune(Kd, ytr, ntr, seed)
            F = bloch(psis[s], n).reshape(len(X), -1)
            out[f"{kind}/{cons}"] = dict(kind=kind, cons=cons, s=s, C=C, K=Kd[s], F=F,
                                         gamma=projected_gamma(F[:ntr]))
    return out


def acc_of(Ktr, Kte, C, ytr, yte):
    return float(SVC(kernel="precomputed", C=C).fit(Ktr, ytr).score(Kte, yte))


def psd_clip(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    return (V * np.clip(w, 0, None)) @ V.T


def sym_from_triu(M, v):
    A = np.eye(M)
    A[np.triu_indices(M, 1)] = v
    return np.triu(A) + np.triu(A, 1).T


def gram_err(Kh, K):
    off = ~np.eye(len(K), dtype=bool)
    return float(np.sqrt(np.mean((Kh - K)[off] ** 2))), float(np.linalg.norm(Kh - K) / np.linalg.norm(K))


# --------------------------------------------------------------------------- PART 1
def shots_job(name, ri, Xtr_raw, Xte_raw, ytr, yte):
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw)
    ntr = len(ytr)
    seed = BASE_SEED + ri
    rng = np.random.default_rng(seed + 991)
    models = ideal_models(Xtr, Xte, ytr, seed)
    rows = []
    for kname, m in models.items():
        K = m["K"]
        acc0 = acc_of(K[:ntr, :ntr], K[ntr:, :ntr], m["C"], ytr, yte)
        for N in SHOTS:
            if m["cons"] == "fidelity":
                Ktr = psd_clip(sym_from_triu(ntr, rng.binomial(N, np.clip(K[:ntr, :ntr][np.triu_indices(ntr, 1)], 0, 1)) / N))
                Kte = rng.binomial(N, np.clip(K[ntr:, :ntr], 0, 1)) / N
                Ktr_ideal, Kte_ideal = K[:ntr, :ntr], K[ntr:, :ntr]
            else:
                F = m["F"]
                Fh = 2.0 * rng.binomial(N, np.clip((1 + F) / 2, 0, 1)) / N - 1.0
                Ktr = projected_gram(Fh[:ntr], Fh[:ntr], m["gamma"])
                Kte = projected_gram(Fh[ntr:], Fh[:ntr], m["gamma"])
                Ktr_ideal, Kte_ideal = K[:ntr, :ntr], K[ntr:, :ntr]
            rms, rel = gram_err(Ktr, Ktr_ideal)
            rows.append(dict(dataset=name, resample=ri, kernel=kname, shots=N, s_star=m["s"], C_star=m["C"],
                             acc_ideal=acc0, acc_shots=acc_of(Ktr, Kte, m["C"], ytr, yte),
                             gram_rms_err=rms, gram_rel_frob=rel))
    return rows


def paired_table(cells, acc_col, group_cols, ratio, label):
    """Loss = acc_ideal - acc_noisy.  Paired difference loss(kernel) - loss(angle, same construction)."""
    c = cells.copy()
    c["loss"] = c["acc_ideal"] - c[acc_col]
    c["emb"] = c.kernel.str.split("/").str[0]
    c["cons"] = c.kernel.str.split("/").str[1]
    rows = []
    for gk, g in c.groupby(group_cols + ["cons"]):
        piv = g.pivot_table(index=["dataset", "resample"], columns="emb", values="loss")
        for emb in ("iqp", "zz"):
            d = (piv[emb] - piv["angle"]).unstack("dataset")
            rows.append(dict(zip(group_cols + ["cons"], gk), contrast=f"{emb} - angle (loss)", dataset="POOLED",
                             **corrected(d.mean(axis=1).values, 1.0, ratio)))
            for ds in d.columns:
                rows.append(dict(zip(group_cols + ["cons"], gk), contrast=f"{emb} - angle (loss)", dataset=ds,
                                 **corrected(d[ds].values, 1.0, ratio)))
    return pd.DataFrame(rows)


def shot_only_acc(m, ntr, N, rng, ytr, yte):
    """Control for the device replay: the same N shots per circuit but NO device noise (ideal Gram + binomial sampling)."""
    K = m["K"]
    if m["cons"] == "fidelity":
        Ktr = psd_clip(sym_from_triu(ntr, rng.binomial(N, np.clip(K[:ntr, :ntr][np.triu_indices(ntr, 1)], 0, 1)) / N))
        Kte = rng.binomial(N, np.clip(K[ntr:, :ntr], 0, 1)) / N
    else:
        F = m["F"]
        Fh = 2.0 * rng.binomial(N, np.clip((1 + F) / 2, 0, 1)) / N - 1.0
        Ktr = projected_gram(Fh[:ntr], Fh[:ntr], m["gamma"])
        Kte = projected_gram(Fh[ntr:], Fh[:ntr], m["gamma"])
    return acc_of(Ktr, Kte, m["C"], ytr, yte)


# --------------------------------------------------------------------------- PART 2
def template(kind, n, prefix):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    a = ParameterVector(prefix, n)  # a_i = s * x_i
    qc = QuantumCircuit(n)
    if kind == "angle":
        for i in range(n):
            qc.ry(a[i], i)
        return qc, a
    for _ in range(2):
        qc.h(range(n))
        for i in range(n):
            qc.rz(2 * a[i] if kind == "zz" else a[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if kind == "zz":
                    qc.rzz(2 * (np.pi - a[i]) * (np.pi - a[j]), i, j)
                else:
                    qc.rzz(a[i] * a[j], i, j)
    return qc, a


def check_templates(n=4):
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(1)
    for kind in KINDS:
        qc, a = template(kind, n, "a")
        x, s = rng.uniform(0, np.pi, n), 0.7
        got = Statevector(qc.assign_parameters(dict(zip(a, s * x)))).data
        ref = Statevector(embed_circuit(x, s, kind)).data
        assert abs(abs(np.vdot(got, ref)) - 1) < 1e-9, f"template mismatch for {kind}"


BACKENDS = ["FakeFez", "FakeSherbrooke", "FakeCusco"]  # low-error Heron / median Eagle / high-error Eagle (2025 snapshots)


def device_setup(backend="FakeSherbrooke"):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit_aer import AerSimulator

    be = getattr(fp, backend)()
    return be, AerSimulator.from_backend(be, method="density_matrix")


def backend_table(names):
    """Calibration summary of each stored snapshot (what the replay uses)."""
    rows = []
    for nm in names:
        be, _ = device_setup(nm)
        p = be.properties()
        q = range(be.num_qubits)
        two = [g.parameters[0].value for g in p.gates if len(g.qubits) == 2 and g.parameters and g.parameters[0].value < 0.5]
        one = [g.parameters[0].value for g in p.gates if len(g.qubits) == 1 and g.gate in ("sx", "x") and g.parameters]
        rows.append(dict(backend=nm, n_qubits=be.num_qubits, snapshot=str(p.last_update_date)[:19],
                         median_T1_us=float(np.median([p.t1(i) for i in q])) * 1e6,
                         median_T2_us=float(np.median([p.t2(i) for i in q])) * 1e6,
                         median_readout_err=float(np.median([p.readout_error(i) for i in q])),
                         median_1q_err=float(np.median(one)), median_2q_err=float(np.median(two)),
                         two_qubit_gate=sorted({g.gate for g in p.gates if len(g.qubits) == 2})[0]))
    return pd.DataFrame(rows)


def build_device_templates(be, kind, n, seed=0):
    from qiskit import QuantumCircuit, transpile

    st, a1 = template(kind, n, "p")
    st2, a2 = template(kind, n, "q")
    fid = st.compose(st2.inverse())
    fid.measure_all()
    tf = transpile(fid, backend=be, optimization_level=3, seed_transpiler=seed, initial_layout=list(range(n)))
    basis = {}
    for b in ("X", "Y", "Z"):
        c = st.copy()
        if b == "X":
            c.h(range(n))
        elif b == "Y":
            c.sdg(range(n))
            c.h(range(n))
        c.measure_all()
        basis[b] = transpile(c, backend=be, optimization_level=3, seed_transpiler=seed, initial_layout=list(range(n)))
    return dict(fid=tf, basis=basis, a1=a1, a2=a2, st=st)


def twoq(c):
    ops = c.count_ops()
    return int(sum(v for k, v in ops.items() if k in ("ecr", "cx", "cz", "rzz")))


def run_counts(sim, circs, shots, seed):
    res = sim.run(circs, shots=shots, seed_simulator=seed).result()
    return [res.get_counts(i) for i in range(len(circs))]


def p_zero(cnt, shots, n):
    return cnt.get("0" * n, 0) / shots


def exp_z(cnt, shots, n):
    """<Z_i> for i = 0..n-1 from counts (bit string is little-endian: rightmost char is qubit 0)."""
    z = np.zeros(n)
    for bits, c in cnt.items():
        for i in range(n):
            z[i] += c * (1 if bits[::-1][i] == "0" else -1)
    return z / shots


def _subset(y, k, seed):
    """Stratified subset of k indices; all indices if k >= len(y) (small test folds, e.g. wine has 18)."""
    if k >= len(y):
        return np.arange(len(y))
    return train_test_split(np.arange(len(y)), train_size=k, stratify=y, random_state=seed)[0]


def device_job(name, ri, Xtr_raw, Xte_raw, ytr, yte, n_train, n_test, shots, backend):
    be, sim = device_setup(backend)
    Xtr, Xte = preprocess(Xtr_raw, Xte_raw)
    # stratified subsets, fixed by seed
    seed = BASE_SEED + ri
    itr = _subset(ytr, n_train, seed)
    ite = _subset(yte, n_test, seed)
    Xtr, ytr, Xte, yte = Xtr[itr], ytr[itr], Xte[ite], yte[ite]
    ntr, nte = len(ytr), len(yte)
    n = Xtr.shape[1]
    models = ideal_models(Xtr, Xte, ytr, seed, s_values=S_WIDE)
    X = np.vstack([Xtr, Xte])
    rows, gates = [], []
    rng_c = np.random.default_rng(seed + 4242)
    for kind in KINDS:
        T = build_device_templates(be, kind, n)
        gates.append(dict(backend=backend, dataset=name, resample=ri, embedding=kind, circuit="fidelity", twoq=twoq(T["fid"]),
                          depth=T["fid"].depth()))
        for b, c in T["basis"].items():
            gates.append(dict(backend=backend, dataset=name, resample=ri, embedding=kind, circuit=f"state_{b}", twoq=twoq(c),
                              depth=c.depth()))
        # ---------------- fidelity kernel
        m = models[f"{kind}/fidelity"]
        s = m["s"]
        A = s * X
        pairs = [(i, j) for i in range(ntr) for j in range(i + 1, ntr)] + [(i, j) for i in range(ntr, ntr + nte)
                                                                           for j in range(ntr)]
        circs = []
        for i, j in pairs:
            circs.append(T["fid"].assign_parameters({**dict(zip(T["a1"], A[i])), **dict(zip(T["a2"], A[j]))}))
        cnts = []
        for k in range(0, len(circs), 400):
            cnts += run_counts(sim, circs[k:k + 400], shots, seed + k)
        Kd = np.eye(ntr + nte)
        for (i, j), c in zip(pairs, cnts):
            Kd[i, j] = Kd[j, i] = p_zero(c, shots, n)
        Ktr = psd_clip(Kd[:ntr, :ntr])
        Kte = Kd[ntr:, :ntr]
        K = m["K"]
        rms, rel = gram_err(Kd[:ntr, :ntr], K[:ntr, :ntr])
        rows.append(dict(backend=backend, dataset=name, resample=ri, kernel=f"{kind}/fidelity", s_star=s, C_star=m["C"],
                         acc_ideal=acc_of(K[:ntr, :ntr], K[ntr:, :ntr], m["C"], ytr, yte),
                         acc_device=acc_of(Ktr, Kte, m["C"], ytr, yte), acc_shotonly=shot_only_acc(m, ntr, shots, rng_c, ytr, yte),
                         gram_bias=float((Kd[:ntr, :ntr] - K[:ntr, :ntr])[~np.eye(ntr, dtype=bool)].mean()),
                         gram_rms_err=rms, gram_rel_frob=rel,
                         twoq_per_circuit=twoq(T["fid"]), depth=T["fid"].depth(), circuits=len(circs)))
        # ---------------- projected kernel
        m = models[f"{kind}/projected"]
        s = m["s"]
        A = s * X
        Fh = np.zeros((ntr + nte, n, 3))
        for bi, b in enumerate(("X", "Y", "Z")):
            circs = [T["basis"][b].assign_parameters(dict(zip(T["a1"], A[i]))) for i in range(len(A))]
            cnts = run_counts(sim, circs, shots, seed + 7 * bi)
            for i, c in enumerate(cnts):
                Fh[i, :, bi] = exp_z(c, shots, n)
        Fh = Fh.reshape(len(A), -1)
        Ktr = projected_gram(Fh[:ntr], Fh[:ntr], m["gamma"])
        Kte = projected_gram(Fh[ntr:], Fh[:ntr], m["gamma"])
        K = m["K"]
        rms, rel = gram_err(Ktr, K[:ntr, :ntr])
        rows.append(dict(backend=backend, dataset=name, resample=ri, kernel=f"{kind}/projected", s_star=s, C_star=m["C"],
                         acc_ideal=acc_of(K[:ntr, :ntr], K[ntr:, :ntr], m["C"], ytr, yte),
                         acc_device=acc_of(Ktr, Kte, m["C"], ytr, yte), acc_shotonly=shot_only_acc(m, ntr, shots, rng_c, ytr, yte),
                         gram_bias=float((Ktr - K[:ntr, :ntr])[~np.eye(ntr, dtype=bool)].mean()),
                         gram_rms_err=rms, gram_rel_frob=rel,
                         twoq_per_circuit=twoq(T["basis"]["Z"]), depth=T["basis"]["Z"].depth(), circuits=3 * len(A)))
    return rows, gates


# --------------------------------------------------------------------------- main
def get_splits(name, R):
    X, y, _ = load_full(name)
    sp = resamples(y, FOLDS, 2)[:R]
    return [(name, ri, X[tr], X[te], y[tr], y[te]) for ri, (tr, te) in enumerate(sp)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--part", default="both", choices=["shots", "device", "both"])
    ap.add_argument("--R", type=int, default=10, help="resamples per dataset for the shot study (max 20)")
    ap.add_argument("--R-device", type=int, default=10, help="resamples per dataset for the device replay (max 20)")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--device-shots", type=int, default=1000)
    ap.add_argument("--shot-datasets", nargs="+", default=SHOT_DATASETS)
    ap.add_argument("--backends", nargs="+", default=BACKENDS)
    ap.add_argument("--device-datasets", nargs="+", default=DEVICE_DATASETS)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.quick:
        a.R, a.R_device, a.n_train, a.n_test = 2, 1, 24, 12
        a.shot_datasets, a.device_datasets = a.shot_datasets[:2], a.device_datasets[:1]
        a.backends = a.backends[:2]
    ratio = 1.0 / (FOLDS - 1)
    ratio_dev = a.n_test / a.n_train   # device replay subsamples 40 train / 20 test points: use that ratio in the correction
    lines = []

    if a.part in ("shots", "both"):
        jobs = [j for d in a.shot_datasets for j in get_splits(d, a.R)]
        t0 = time.time()
        res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(shots_job)(*j) for j in jobs)
        cells = pd.DataFrame([r for rs in res for r in rs])
        cells.to_csv(out / "shots_cells.csv", index=False)
        cells["loss"] = cells.acc_ideal - cells.acc_shots
        summ = cells.groupby(["kernel", "shots"]).agg(acc_ideal=("acc_ideal", "mean"), acc_shots=("acc_shots", "mean"),
                                                      loss=("loss", "mean"), gram_rms_err=("gram_rms_err", "mean"),
                                                      gram_rel_frob=("gram_rel_frob", "mean")).reset_index()
        summ.to_csv(out / "shots_summary.csv", index=False)
        pt = paired_table(cells, "acc_shots", ["shots"], ratio, "shots")
        pt.to_csv(out / "shots_paired.csv", index=False)
        lines += [f"PART 1 SHOTS: {len(set(cells.dataset))} datasets x {a.R} resamples, N in {SHOTS}; time {time.time() - t0:.0f}s",
                  "(loss = ideal accuracy - accuracy with shot-noisy Gram, hyper-parameters fixed from ideal tuning)", "",
                  summ.round(4).to_string(index=False), "",
                  "PAIRED LOSS DIFFERENCE vs ANGLE (pooled; positive = embedding loses more accuracy than angle)",
                  pt[pt.dataset == "POOLED"][["shots", "cons", "contrast", "mean", "lo", "hi", "p"]].round(4).to_string(index=False), ""]

    if a.part in ("device", "both"):
        try:
            check_templates()
            device_setup()
        except Exception as e:  # missing qiskit-aer / fake provider
            raise SystemExit(f"device replay unavailable: {type(e).__name__}: {e}")
        bt = backend_table(a.backends)
        bt.to_csv(out / "device_backends.csv", index=False)
        jobs = [j + (a.n_train, a.n_test, a.device_shots, b) for b in a.backends for d in a.device_datasets
                for j in get_splits(d, a.R_device)]
        t0 = time.time()
        res = Parallel(n_jobs=a.n_jobs, verbose=5 if a.n_jobs != 1 else 0)(delayed(device_job)(*j) for j in jobs)
        cells = pd.DataFrame([r for rs, _ in res for r in rs])
        gates = pd.DataFrame([g for _, gs in res for g in gs])
        cells["change"] = cells.acc_device - cells.acc_ideal          # total change = shot noise + device noise
        cells["change_shot"] = cells.acc_shotonly - cells.acc_ideal   # control: same shots, no device noise
        cells["change_noise"] = cells.acc_device - cells.acc_shotonly  # effect of device noise beyond shot noise
        cells.to_csv(out / "device_cells.csv", index=False)
        gates.to_csv(out / "device_gates.csv", index=False)
        summ = cells.groupby(["backend", "kernel"]).agg(acc_ideal=("acc_ideal", "mean"), acc_shotonly=("acc_shotonly", "mean"),
                                                        acc_device=("acc_device", "mean"),
                                                        gram_bias=("gram_bias", "mean"), gram_rms_err=("gram_rms_err", "mean"),
                                                        gram_rel_frob=("gram_rel_frob", "mean"),
                                                        twoq_per_circuit=("twoq_per_circuit", "median"),
                                                        depth=("depth", "median")).reset_index()
        for col in ("change", "change_shot", "change_noise"):
            rows = []
            for (b_, k), g in cells.groupby(["backend", "kernel"]):
                piv = g.pivot_table(index="resample", columns="dataset", values=col)
                rows.append(dict(backend=b_, kernel=k, **corrected(piv.mean(axis=1).values, 1.0, ratio_dev)))
            ci = pd.DataFrame(rows).rename(columns={"mean": f"{col}_mean", "lo": f"{col}_lo", "hi": f"{col}_hi"})
            summ = summ.merge(ci[["backend", "kernel", f"{col}_mean", f"{col}_lo", f"{col}_hi"]], on=["backend", "kernel"])
        summ.to_csv(out / "device_summary.csv", index=False)
        pt = paired_table(cells.assign(acc_dev=cells.acc_device), "acc_dev", ["backend"], ratio_dev, "device")
        pt.to_csv(out / "device_paired.csv", index=False)
        pt2 = paired_table(cells.assign(acc_dev=cells.acc_shotonly), "acc_dev", ["backend"], ratio_dev, "shot-only control")
        pt2.to_csv(out / "device_paired_shotonly.csv", index=False)
        lines += [f"PART 2 DEVICE REPLAY (stored 2025 calibration snapshots, Aer density matrix, {a.device_shots} shots; "
                  f"NOT hardware): {len(set(cells.dataset))} datasets x {a.R_device} resamples x {len(a.backends)} backends, "
                  f"{a.n_train} train / {a.n_test} test; time {time.time() - t0:.0f}s", "",
                  "BACKEND CALIBRATION SUMMARY", bt.round(5).to_string(index=False), "",
                  summ.round(4).to_string(index=False), "",
                  "PAIRED LOSS DIFFERENCE vs ANGLE (pooled; loss = ideal - device accuracy)",
                  pt[pt.dataset == "POOLED"][["backend", "cons", "contrast", "mean", "lo", "hi", "p"]].round(4).to_string(index=False), ""]

    (out / "shots_device_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
