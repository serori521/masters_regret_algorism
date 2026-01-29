"""
plot_scale_bias_logs.py

Quick plotting helper for scale_bias_logs/*.csv
- No external dependencies beyond pandas + matplotlib (standard in many setups).
- If pandas isn't installed in your env, you can adapt to CSV.jl in Julia or install pandas.

Usage (from repo root):
  python plot_scale_bias_logs.py --csv data/metrics_julia/scale_bias_logs/case_summary__rule=minimax_regret__N=6__tw=A__utility=u2__method=eAMRw.csv

You can also point to true_to_pred__...csv or pred_to_true__...csv.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to a scale-bias log csv")
    ap.add_argument("--out", default=None, help="output png path (default: same dir)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    base = os.path.splitext(os.path.basename(args.csv))[0]
    out = args.out or os.path.join(os.path.dirname(args.csv), base + ".png")

    # choose plot based on columns
    if "mean_best_score_01" in df.columns:
        # case summary
        plt.figure()
        plt.scatter(df["mean_alpha_pred"], df["mean_best_score_01"])
        plt.xlabel("mean_alpha_pred (0..1 in predicted t-range)")
        plt.ylabel("mean_best_score_01 (avg best pair-match / denom_pairs)")
        plt.title(base)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        return

    if "alpha_pred" in df.columns and "best_score_01" in df.columns:
        plt.figure()
        plt.scatter(df["alpha_pred"], df["best_score_01"])
        plt.xlabel("alpha_pred (0..1 in predicted t-range)")
        plt.ylabel("best_score_01 (best pair-match / denom_pairs)")
        plt.title(base)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        return

    # fallback: histogram if possible
    if "alpha_pred" in df.columns:
        plt.figure()
        plt.hist(df["alpha_pred"].dropna().values, bins=30)
        plt.xlabel("alpha_pred")
        plt.ylabel("count")
        plt.title(base)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        return

    raise SystemExit("Unknown CSV schema: " + args.csv)

if __name__ == "__main__":
    main()
