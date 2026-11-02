"""Two-qubit gate counts of EXACT amplitude loading (hypothesis H1) from independent toolchains.

  qiskit_opt1 / qiskit_opt3 : Qiskit StatePreparation transpiled to {cx, rz, sx, x}, no connectivity constraint
  qiskit_routed             : the same circuit routed on the heavy-hex coupling map of the stored FakeSherbrooke snapshot
  pennylane                 : PennyLane MottonenStatePreparation decomposition (CNOT count), optional (needs pennylane)

Vector families: dense real (|Gaussian|), dense complex, Gaussian bump, four non-zero entries, product state.
Also writes the cost ratio Q_entry / C_cl = 2 G_load S(eps) / (2 N), S(eps) = ceil(1 / (4 eps^2)), and the log-linear
slope fit of log2 G_load against n.

Outputs (--out): loading_counts.csv, loading_fit.csv, loading_ratio.csv, loading_report.txt
Usage:
  python loading_counts.py --out results
  python loading_counts.py --out results --quick           # n <= 8
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
BASIS = ["cx", "rz", "sx", "x"]
EPS = [0.1, 0.05, 0.01]


def family_vector(fam, n, rng):
    N = 2 ** n
    if fam == "dense_real":
        v = np.abs(rng.normal(size=N))
    elif fam == "dense_complex":
        v = rng.normal(size=N) + 1j * rng.normal(size=N)
    elif fam == "gaussian_bump":
        x = np.arange(N)
        v = np.exp(-((x - N / 2) ** 2) / (2 * (N / 8) ** 2)) + 1e-6   # tiny floor keeps every amplitude non-zero
    elif fam == "sparse4":
        v = np.zeros(N)
        v[rng.choice(N, size=min(4, N), replace=False)] = np.abs(rng.normal(size=min(4, N))) + 0.1
    elif fam == "product":
        v = np.array([1.0])
        for _ in range(n):
            a = rng.uniform(0.2, 1.3)
            v = np.kron(v, np.array([np.cos(a), np.sin(a)]))
    else:
        raise ValueError(fam)
    return v / np.linalg.norm(v)


def qiskit_count(v, n, opt, coupling=None, seed=0):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation

    qc = QuantumCircuit(n)
    qc.append(StatePreparation(v), range(n))
    if coupling is None:
        tc = transpile(qc, basis_gates=BASIS, optimization_level=opt, seed_transpiler=seed)
    else:
        tc = transpile(qc, basis_gates=BASIS, coupling_map=coupling, optimization_level=opt, seed_transpiler=seed)
    return int(tc.num_nonlocal_gates()), int(tc.depth())


def pennylane_count(v, n):
    import pennylane as qml

    ops = qml.MottonenStatePreparation(v, wires=range(n)).decomposition()
    return int(sum(1 for op in ops if op.name == "CNOT")), len(ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--nmax", type=int, default=12)
    ap.add_argument("--nmax-qiskit", type=int, default=12)
    ap.add_argument("--nmax-routed", type=int, default=8)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.nmax = a.nmax_qiskit = 8
        a.nmax_routed = 6
        a.draws = 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import pennylane  # noqa: F401
        have_pl = True
    except Exception:
        have_pl = False
        print("pennylane not installed: PennyLane counts skipped")
    coupling = None
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        from qiskit.transpiler import CouplingMap
        coupling = CouplingMap(FakeSherbrooke().coupling_map.get_edges())
    except Exception as e:
        print("routing skipped:", e)

    rows = []
    t0 = time.time()
    fams = ["dense_real", "dense_complex", "gaussian_bump", "sparse4", "product"]
    for n in range(1, a.nmax + 1):
        for fam in fams:
            nd = a.draws if fam in ("dense_real", "dense_complex", "sparse4", "product") else 1
            if n >= 11:
                nd = 1
            for d in range(nd):
                rng = np.random.default_rng(1000 * n + d)
                v = family_vector(fam, n, rng)
                if have_pl and n >= 1 and fam in ("dense_real", "dense_complex", "gaussian_bump", "sparse4", "product"):
                    c, g = pennylane_count(v, n)
                    rows.append(dict(n=n, family=fam, draw=d, toolchain="pennylane", twoq=c, depth=np.nan))
                if n <= a.nmax_qiskit and n >= 2:
                    for opt in (1, 3):
                        if fam == "dense_complex" and opt == 3 and n > 10:
                            continue
                        try:
                            c, dep = qiskit_count(v, n, opt)
                        except Exception as e:
                            print(f"  skipped qiskit {fam} n={n} opt{opt}: {str(e)[:60]}")
                            continue
                        rows.append(dict(n=n, family=fam, draw=d, toolchain=f"qiskit_opt{opt}", twoq=c, depth=dep))
                    if coupling is not None and fam == "dense_real" and n <= a.nmax_routed and n >= 2:
                        for opt in (1, 3):
                            c, dep = qiskit_count(v, n, opt, coupling=coupling)
                            rows.append(dict(n=n, family=fam, draw=d, toolchain=f"qiskit_routed_opt{opt}", twoq=c, depth=dep))
        print(f"n={n} done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "loading_counts.csv", index=False)

    dense = df[df.family == "dense_real"].groupby(["toolchain", "n"]).twoq.median().unstack(0)
    fits = []
    for tc in dense.columns:
        s = dense[tc].dropna()
        s = s[s.index >= (2 if tc == "pennylane" else 5)]
        s = s[s > 0]
        if len(s) < 3:
            continue
        r = stats.linregress(s.index.values.astype(float), np.log2(s.values))
        tcrit = stats.t.ppf(0.975, len(s) - 2)
        fits.append(dict(toolchain=tc, n_from=int(s.index.min()), n_to=int(s.index.max()), slope=r.slope,
                         lo=r.slope - tcrit * r.stderr, hi=r.slope + tcrit * r.stderr))
    pd.DataFrame(fits).to_csv(out / "loading_fit.csv", index=False)

    ratio = []
    for tc in ("pennylane", "qiskit_opt1"):
        if tc not in dense.columns:
            continue
        for n, g in dense[tc].dropna().items():
            if n < 2:
                continue
            for e in EPS:
                S = math.ceil(1 / (4 * e * e))
                ratio.append(dict(toolchain=tc, n=int(n), N=2 ** int(n), G_load=int(g), eps=e, S=S,
                                  Q_entry=2 * g * S, C_cl=2 * 2 ** int(n), ratio=(2 * g * S) / (2 * 2 ** int(n))))
    pd.DataFrame(ratio).to_csv(out / "loading_ratio.csv", index=False)

    lines = ["EXACT AMPLITUDE LOADING, dense real vector: median two-qubit gates", dense.to_string(), ""]
    chk = []
    for n, row in dense.iterrows():
        if "pennylane" in row and not np.isnan(row.get("pennylane", np.nan)):
            chk.append((n, "pennylane", row["pennylane"], 2 ** n - 2))
        if "qiskit_opt1" in row and not np.isnan(row.get("qiskit_opt1", np.nan)):
            chk.append((n, "qiskit_opt1", row["qiskit_opt1"], 2 ** n - n - 1))
    bad = [c for c in chk if c[2] != c[3] and c[0] >= 2]
    lines += [f"FORMULA CHECK: PennyLane = 2^n - 2, Qiskit opt1 = 2^n - n - 1 ; mismatches: {bad if bad else 'none'}", ""]
    lines += ["LOG-LINEAR SLOPE of log2(G_load) vs n", pd.DataFrame(fits).round(4).to_string(index=False), ""]
    lines += ["FAMILIES at n = 8 and the largest n (median two-qubit gates; qiskit_opt1, pennylane)"]
    fam = df[df.toolchain.isin(["qiskit_opt1", "pennylane"]) & df.n.isin([8, a.nmax_qiskit])]
    lines += [fam.groupby(["n", "family", "toolchain"]).twoq.median().unstack().to_string(), ""]
    lines += ["COST RATIO Q_entry / C_cl = 2 G S / (2 N)", pd.DataFrame(ratio).pivot_table(
        index=["toolchain", "n"], columns="eps", values="ratio").round(2).to_string()]
    (out / "loading_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
