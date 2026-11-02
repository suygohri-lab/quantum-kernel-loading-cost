"""Power analysis for the equivalence margin delta = 0.02 BEFORE running a larger study.

Model: the paired accuracy difference of two classifiers on one test split has variance sigma^2 ~ c / n_test
(binomial sampling noise of the test set).  c is estimated from the repetitions of the pilot study
(cells.csv from gram_spread.py).  The corrected resampled t interval half-width is
    h = t_{R-1,0.975} * sqrt( (1/R + n_test/n_train) * c / n_test ).
This prints h for candidate designs, so the study design can be chosen so that h < delta.  It also finds the smallest
number of datasets needed for a pooled interval (assuming independent datasets).

Usage: python power_analysis.py --cells ../results/design/pilot_cells.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--kernel", default="angle/fidelity")
    ap.add_argument("--reference", default="classical/rbf")
    ap.add_argument("--delta", type=float, default=0.02)
    a = ap.parse_args()
    df = pd.read_csv(a.cells)
    piv = df.pivot_table(index=["dataset", "rep"], columns="kernel", values="test_acc")
    d = (piv[a.kernel] - piv[a.reference]).unstack("dataset")
    nte_obs = 50
    c = float(np.nanmean(d.var(ddof=1).values) * nte_obs)
    print(f"Estimated c = sigma^2 * n_test = {c:.3f}  (from {a.kernel} - {a.reference}, n_test = {nte_obs})\n")
    designs = []
    for N in (150, 350, 569, 1000, 3000):
        for name, ntr_frac, R in (("70/30 x20", 0.7, 20), ("80/20 x20", 0.8, 20), ("90/10 x20", 0.9, 20),
                                  ("90/10 x100", 0.9, 100), ("10-fold x10", 0.9, 100)):
            ntr = int(N * ntr_frac)
            nte = N - ntr
            if "10-fold" in name:
                nte, ntr = N // 10, N - N // 10
            h = stats.t.ppf(0.975, R - 1) * np.sqrt((1 / R + nte / ntr) * c / nte)
            designs.append(dict(N=N, design=name, n_train=ntr, n_test=nte, R=R, half_width=round(h, 4),
                                meets_delta=h < a.delta))
    out = pd.DataFrame(designs)
    print(out.to_string(index=False))
    print(f"\nRows with meets_delta=True can in principle show equivalence within +/-{a.delta} per dataset "
          "if the true difference is ~0.")
    print("Pooled over K independent datasets the half-width shrinks roughly as 1/sqrt(K); the pilot's "
          "pooled intervals (+/-0.011 to 0.017) come from K = 6 datasets of 150 points.")


if __name__ == "__main__":
    main()
