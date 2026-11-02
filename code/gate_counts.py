"""Two-qubit gate and circuit counts for every tested embedding (angle / IQP / ZZ),
fidelity (compute-uncompute) and projected constructions.

Writes (in --out):
  gate_counts.csv        n, embedding, circuit, mode, opt_level, 2q gates (min/median/max over seeds), depth
  circuits_per_gram.csv  circuits needed for one train+test Gram matrix, per construction
  gate_counts_check.txt  self-check against the analytic two-qubit count

mode = all_to_all : transpiled to {cx, rz, sx, x} with no connectivity constraint
mode = routed     : transpiled to the coupling map / native basis of the hardware snapshot
                    (FakeSherbrooke unless a snapshot is given)

Usage:
  python gate_counts.py --out results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from qkcommon import KINDS, L_REPS, embed_circuit, fidelity_circuit

ALLOWED_NATIVE = {"cx", "ecr", "cz", "rz", "sx", "x", "rx", "ry", "id"}


def hardware_target(snapshot_path):
    if snapshot_path:
        snap = json.loads(Path(snapshot_path).read_text())
        edges = [tuple(e) for e in snap["coupling_map"]]
        basis = snap["native_basis"]
        label = f'{snap["backend_name"]} ({snap["source"]}, {snap["dumped_at_utc"]})'
        return CouplingMap(edges), basis, label
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    b = FakeSherbrooke()
    basis = [g for g in b.operation_names if g in ALLOWED_NATIVE]
    return CouplingMap(list(b.coupling_map.get_edges())), basis, "FakeSherbrooke (stored 2025 replay, NOT live hardware)"


def count_2q(qc):
    return qc.num_nonlocal_gates()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--nmin", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5, help="transpiler seeds for routed counts")
    ap.add_argument("--opt-levels", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--M", type=int, default=100, help="training points")
    ap.add_argument("--nte", type=int, default=50, help="test points")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cmap, native, label = hardware_target(a.snapshot)
    print("Routing target:", label, "| native basis:", native)
    rng = np.random.default_rng(2026)
    rows = []
    for n in range(a.nmin, a.nmax + 1):
        x1 = rng.uniform(0.2, np.pi - 0.2, n)
        x2 = rng.uniform(0.2, np.pi - 0.2, n)
        s = 1.0
        for kind in KINDS:
            circs = {
                "state": embed_circuit(x1, s, kind),
                "fidelity": fidelity_circuit(x1, x2, s, kind, measure=False),
            }
            for cname, qc in circs.items():
                for lv in a.opt_levels:
                    t = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=lv, seed_transpiler=0)
                    rows.append(dict(n=n, embedding=kind, circuit=cname, mode="all_to_all", opt_level=lv,
                                     twoq_min=count_2q(t), twoq_median=count_2q(t), twoq_max=count_2q(t),
                                     depth_median=t.depth()))
                    if n <= 12:
                        cnt, dep = [], []
                        for sd in range(a.seeds):
                            tr = transpile(qc, basis_gates=native, coupling_map=cmap, optimization_level=lv,
                                           seed_transpiler=sd)
                            cnt.append(count_2q(tr))
                            dep.append(tr.depth())
                        rows.append(dict(n=n, embedding=kind, circuit=cname, mode="routed", opt_level=lv,
                                         twoq_min=min(cnt), twoq_median=float(np.median(cnt)), twoq_max=max(cnt),
                                         depth_median=float(np.median(dep))))
        print(f"n={n} done")
    df = pd.DataFrame(rows)
    df["hardware_target"] = label
    df.to_csv(out / "gate_counts.csv", index=False)

    # ---- circuits per Gram matrix (train Gram + test-vs-train block) ----
    M, nte = a.M, a.nte
    gram = pd.DataFrame([
        dict(construction="fidelity", circuits_train=M * (M - 1) // 2, circuits_test=nte * M,
             circuits_total=M * (M - 1) // 2 + nte * M,
             note="one compute-uncompute circuit per kernel entry"),
        dict(construction="projected", circuits_train=3 * M, circuits_test=3 * nte,
             circuits_total=3 * (M + nte),
             note="one state-preparation circuit per point, measured in X, Y and Z bases (3 settings)"),
    ])
    gram["M_train"], gram["n_test"] = M, nte
    gram.to_csv(out / "circuits_per_gram.csv", index=False)

    # ---- checks ----
    lines = []
    bad = False
    for n in range(a.nmin, a.nmax + 1):
        for kind in ("iqp", "zz"):
            r = df[(df.n == n) & (df.embedding == kind) & (df.circuit == "state") & (df["mode"] == "all_to_all")
                   & (df.opt_level == 1)]
            exp = L_REPS * n * (n - 1)
            if len(r) and int(r.twoq_median.iloc[0]) != exp:
                bad = True
                lines.append(f"ANALYTIC MISMATCH n={n} {kind}: got {int(r.twoq_median.iloc[0])}, expected {exp}")
        r = df[(df.n == n) & (df.embedding == "angle") & (df.circuit == "state") & (df["mode"] == "all_to_all")]
        if len(r) and (r.twoq_median != 0).any():
            bad = True
            lines.append(f"ANALYTIC MISMATCH n={n} angle: expected 0 two-qubit gates")
    lines.append("Analytic check (state circuit, all-to-all, opt 1: IQP/ZZ = L*n*(n-1), angle = 0): "
                 + ("FAILED" if bad else "passed"))

    (out / "gate_counts_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
