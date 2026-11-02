"""Re-analyse the stored device-replay cells with the correct variance-correction ratio (no simulation is run).

The device replay subsamples n_train = 40 training and n_test = 20 test points from every resample, so the corrected
resampled t of Nadeau and Bengio must use n_test / n_train = 0.5 (not the 1/9 of the full 10-fold splits).  This script
reads device_cells.csv and rewrites device_summary.csv, device_paired.csv, device_paired_shotonly.csv and
device_reanalysis_report.txt.  (shots_device_study.py now uses the right ratio itself; this is for result files that
were produced before that change.)

Usage: python device_reanalysis.py --out results --n-train 40 --n-test 20
"""
import argparse
from pathlib import Path

import pandas as pd

from extended_study import corrected
from shots_device_study import paired_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    ratio = a.n_test / a.n_train
    cells = pd.read_csv(out / "device_cells.csv")
    summ = cells.groupby(["backend", "kernel"]).agg(
        acc_ideal=("acc_ideal", "mean"), acc_shotonly=("acc_shotonly", "mean"), acc_device=("acc_device", "mean"),
        gram_bias=("gram_bias", "mean"), gram_rms_err=("gram_rms_err", "mean"), gram_rel_frob=("gram_rel_frob", "mean"),
        twoq_per_circuit=("twoq_per_circuit", "median"), depth=("depth", "median")).reset_index()
    for col in ("change", "change_shot", "change_noise"):
        rows = []
        for (b, k), g in cells.groupby(["backend", "kernel"]):
            piv = g.pivot_table(index="resample", columns="dataset", values=col)
            rows.append(dict(backend=b, kernel=k, **corrected(piv.mean(axis=1).values, 1.0, ratio)))
        ci = pd.DataFrame(rows).rename(columns={"mean": f"{col}_mean", "lo": f"{col}_lo", "hi": f"{col}_hi"})
        summ = summ.merge(ci[["backend", "kernel", f"{col}_mean", f"{col}_lo", f"{col}_hi"]], on=["backend", "kernel"])
    summ.to_csv(out / "device_summary.csv", index=False)
    pt = paired_table(cells.assign(acc_dev=cells.acc_device), "acc_dev", ["backend"], ratio, "device")
    pt.to_csv(out / "device_paired.csv", index=False)
    pt2 = paired_table(cells.assign(acc_dev=cells.acc_shotonly), "acc_dev", ["backend"], ratio, "shot-only control")
    pt2.to_csv(out / "device_paired_shotonly.csv", index=False)
    lines = [f"DEVICE RE-ANALYSIS with n_test/n_train = {ratio:.3f}",
             "", summ[["backend", "kernel", "change_mean", "change_lo", "change_hi", "change_shot_mean", "change_shot_lo", "change_shot_hi",
                      "change_noise_mean", "change_noise_lo", "change_noise_hi"]].round(4).to_string(index=False), "",
             "PAIRED LOSS DIFFERENCE vs ANGLE (total loss = ideal - device)",
             pt[pt.dataset == "POOLED"][["backend", "cons", "contrast", "mean", "lo", "hi", "p"]].round(4).to_string(index=False), "",
             "PAIRED LOSS DIFFERENCE vs ANGLE (shot-noise-only control)",
             pt2[pt2.dataset == "POOLED"][["backend", "cons", "contrast", "mean", "lo", "hi", "p"]].round(4).to_string(index=False)]
    (out / "device_reanalysis_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
