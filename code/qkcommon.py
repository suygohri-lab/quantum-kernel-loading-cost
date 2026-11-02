"""Shared definitions for the quantum-kernel loading-cost scripts.

Definitions as in Sect. 2 (Background) of the paper:
  * angle, IQP and ZZ embeddings, L = 2 layer repetitions, features scaled to [0, pi]
  * fidelity kernel and projected kernel with gamma = 1/(3 n Var(F))
  * default grids for the pilot study; the studies of the paper use the 12-value bandwidth grid set by the
    environment variable QK_S_GRID (see extended_study.py) and C in {0.1, 1, 10, 100}

States are simulated with plain numpy (fast).  The same embeddings are also built
as Qiskit circuits (embed_circuit) for gate counting and device-model replay, and
`python qkcommon.py --selftest` checks that both agree.
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache

import numpy as np

S_GRID = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
C_GRID = [0.1, 1.0, 10.0, 100.0]
L_REPS = 2
N_QUBITS = 4
BASE_SEED = 2026
DATASETS = ["breast_cancer", "circles", "digits38", "iris", "moons", "wine"]
KINDS = ["angle", "iqp", "zz"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _lift(X2):
    x1, x2 = X2[:, 0], X2[:, 1]
    return np.column_stack([x1, x2, x1 * x2, x1 ** 2 - x2 ** 2])


def load_split(name: str, rep: int, n_qubits: int = N_QUBITS):
    """Return Xtr, Xte, ytr, yte already mapped to [0, pi] (fit on train only)."""
    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    seed = BASE_SEED + rep
    if name == "moons":
        X, y = datasets.make_moons(n_samples=150, noise=0.15, random_state=seed)
        X = _lift(X)
    elif name == "circles":
        X, y = datasets.make_circles(n_samples=150, noise=0.1, factor=0.5, random_state=seed)
        X = _lift(X)
    else:
        if name == "breast_cancer":
            X, y = datasets.load_breast_cancer(return_X_y=True)
        elif name == "iris":
            X, y = datasets.load_iris(return_X_y=True)
        elif name == "wine":
            X, y = datasets.load_wine(return_X_y=True)
        elif name == "digits38":
            X, y = datasets.load_digits(return_X_y=True)
            m = (y == 3) | (y == 8)
            X, y = X[m], (y[m] == 8).astype(int)
        else:
            raise ValueError(name)
        if len(y) > 150:
            X, _, y, _ = train_test_split(X, y, train_size=150, stratify=y, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=100, test_size=50, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if Xtr.shape[1] > n_qubits:
        pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    mm = MinMaxScaler(feature_range=(0.0, np.pi)).fit(Xtr)
    return mm.transform(Xtr), mm.transform(Xte), ytr, yte


# --------------------------------------------------------------------------
# Numpy statevector simulation (little-endian: qubit i is bit i of the index)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _hadamard_n(n):
    H = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)
    out = np.array([[1.0]])
    for _ in range(n):
        out = np.kron(H, out)
    return out


@lru_cache(maxsize=None)
def _zmat(n):
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> np.arange(n)[None, :]) & 1
    return (1 - 2 * bits).astype(float)  # (2^n, n): z_i = +1 for bit 0


def states(X, s, kind, n=None, L=L_REPS):
    """Embedded states, shape (B, 2^n)."""
    X = np.atleast_2d(np.asarray(X, float))
    B, n_feat = X.shape
    n = n_feat if n is None else n
    assert n_feat == n
    if kind == "angle":
        psi = np.ones((B, 1), complex)
        for q in range(n):
            th = s * X[:, q]
            amp = np.stack([np.cos(th / 2), np.sin(th / 2)], axis=1)  # RY(th)|0>
            psi = np.einsum("bk,bj->bkj", amp, psi).reshape(B, -1)
        return psi
    Z = _zmat(n)
    Hn = _hadamard_n(n)
    psi = np.zeros((B, 2 ** n), complex)
    psi[:, 0] = 1.0
    if kind == "zz":
        u = np.pi - s * X
        w = u @ Z.T
        pair = (w ** 2 - (u ** 2).sum(1, keepdims=True)) / 2.0  # sum_{i<j} u_i u_j z_i z_j
        phase = -(s * X @ Z.T) - pair
    elif kind == "iqp":
        v = s * X
        w = v @ Z.T
        pair = (w ** 2 - (v ** 2).sum(1, keepdims=True)) / 2.0  # sum_{i<j} v_i v_j z_i z_j
        phase = -(v @ Z.T) / 2.0 - pair / 2.0
    else:
        raise ValueError(kind)
    ph = np.exp(1j * phase)
    for _ in range(L):
        psi = psi @ Hn.T
        psi = psi * ph
    return psi


def fidelity_gram(A, B):
    return np.abs(A @ B.conj().T) ** 2


def bloch(psi, n):
    """Bloch vectors, shape (B, n, 3) as (<X>,<Y>,<Z>)."""
    B = psi.shape[0]
    out = np.zeros((B, n, 3))
    t = psi.reshape((B,) + (2,) * n)
    for q in range(n):
        arr = np.moveaxis(t, 1 + (n - 1 - q), 1).reshape(B, 2, -1)
        rho = np.einsum("bik,bjk->bij", arr, arr.conj())
        out[:, q, 0] = 2 * rho[:, 0, 1].real
        out[:, q, 1] = -2 * rho[:, 0, 1].imag
        out[:, q, 2] = (rho[:, 0, 0] - rho[:, 1, 1]).real
    return out


def projected_gamma(Ftrain):
    """gamma = 1/(3 n Var(F)); F has shape (M, 3n) -> 3n features."""
    v = Ftrain.var()
    return 1.0 / (Ftrain.shape[1] * v) if v > 0 else 1.0


def projected_gram(Fa, Fb, gamma):
    d2 = ((Fa[:, None, :] - Fb[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


# --------------------------------------------------------------------------
# Qiskit circuits (for gate counting / hardware); same conventions as above
# --------------------------------------------------------------------------
def embed_circuit(x, s, kind, L=L_REPS):
    from qiskit import QuantumCircuit

    x = np.asarray(x, float)
    n = len(x)
    qc = QuantumCircuit(n)
    if kind == "angle":
        for i in range(n):
            qc.ry(float(s * x[i]), i)
        return qc
    for _ in range(L):
        qc.h(range(n))
        for i in range(n):
            # exp(-i s x Z) = RZ(2 s x) (zz) ; exp(-i s x Z / 2) = RZ(s x) (iqp)
            qc.rz(float(2 * s * x[i]) if kind == "zz" else float(s * x[i]), i)
        for i in range(n):
            for j in range(i + 1, n):
                if kind == "zz":
                    phi = (np.pi - s * x[i]) * (np.pi - s * x[j])
                    qc.rzz(float(2 * phi), i, j)  # exp(-i phi ZZ) = RZZ(2 phi)
                else:
                    qc.rzz(float(s * s * x[i] * x[j]), i, j)  # exp(-i s^2 xi xj ZZ / 2) = RZZ(s^2 xi xj)
    return qc


def fidelity_circuit(x1, x2, s, kind, measure=True):
    """Compute-uncompute circuit; P(all zeros) = |<phi(x1)|phi(x2)>|^2."""
    qc = embed_circuit(x1, s, kind).compose(embed_circuit(x2, s, kind).inverse())
    if measure:
        qc.measure_all()
    return qc


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------
def selftest():
    from qiskit.quantum_info import Pauli, Statevector

    rng = np.random.default_rng(0)
    ok = True
    for n in (3, 4):
        for kind in KINDS:
            for s in (0.05, 0.7, 2.0):
                x = rng.uniform(0, np.pi, size=(2, n))
                ps = states(x, s, kind, n)
                for b in range(2):
                    sv = Statevector(embed_circuit(x[b], s, kind)).data
                    ov = abs(np.vdot(sv, ps[b]))
                    if abs(ov - 1) > 1e-9:
                        ok = False
                        print(f"FAIL state n={n} {kind} s={s} overlap={ov}")
                    br = bloch(ps[b:b + 1], n)[0]
                    for q in range(n):
                        for k, P in enumerate("XYZ"):
                            lab = ["I"] * n
                            lab[n - 1 - q] = P
                            ev = Statevector(sv).expectation_value(Pauli("".join(lab))).real
                            if abs(ev - br[q, k]) > 1e-9:
                                ok = False
                                print(f"FAIL bloch n={n} {kind} s={s} q={q} {P}: {ev} vs {br[q, k]}")
    x = rng.uniform(0, np.pi, size=(5, 4))
    s = 0.8
    K = fidelity_gram(states(x, s, "angle"), states(x, s, "angle"))
    Kf = np.prod(np.cos(s * (x[:, None, :] - x[None, :, :]) / 2) ** 2, axis=2)
    if np.abs(K - Kf).max() > 1e-12:
        ok = False
        print("FAIL angle factorisation")
    F = bloch(states(x, s, "angle"), 4).reshape(5, -1)
    g = 0.37
    Kp = projected_gram(F, F, g)
    Kpf = np.exp(-4 * g * (np.sin(s * (x[:, None, :] - x[None, :, :]) / 2) ** 2).sum(2))
    if np.abs(Kp - Kpf).max() > 1e-12:
        ok = False
        print("FAIL angle projected closed form")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    ap.print_help()
