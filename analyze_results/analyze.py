#!/usr/bin/env python3
"""
summarize_grid_results.py

Reads:
  - grid_summary_maximinmaximax_v3.csv   (rule in {maximin,maximax})
  - grid_summary_minimax_regret_v2.csv   (rule == minimax_regret)

Computes:
  1) Method-wise average scores across ALL settings (N, tw, utility), per rule.
  2) Same, but restricted to N=6.
  3) Stability (variance/std across N×tw) per (rule, utility, method).

Outputs:
  - By default: prints Markdown tables to stdout (Top-K).
  - Optionally writes ONE tidy CSV (long format) that contains all summaries:
      columns: section, rule, utility, method, metric, value

Usage:
  python summarize_grid_results.py \
    --maximinmaximax /path/to/grid_summary_maximinmaximax_v3.csv \
    --minimax        /path/to/grid_summary_minimax_regret_v2.csv \
    --out            /path/to/summary_tidy.csv \
    --topk 10

If you omit --out, no file is written.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd


REQUIRED_COLS = [
    "rule", "N", "tw", "utility", "method",
    "sum_precision", "sum_recall", "sum_f1",
    "sum_diag_mean", "sum_full_mean",
    "sum_top1", "sum_top2_comp", "sum_top2_include",
    "cases",
]


def load_grid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # per-case averages
    for col, out in [
        ("sum_precision", "precision"),
        ("sum_recall", "recall"),
        ("sum_f1", "f1"),
        ("sum_diag_mean", "diag_mean"),
        ("sum_full_mean", "full_mean"),
        ("sum_top1", "top1"),
        ("sum_top2_comp", "top2_comp"),
        ("sum_top2_include", "top2_inc"),
    ]:
        df[out] = df[col] / df["cases"]

    df["diag_minus_full"] = df["diag_mean"] - df["full_mean"]

    # normalize types
    df["rule"] = df["rule"].astype(str)
    df["tw"] = df["tw"].astype(str)
    df["utility"] = df["utility"].astype(str)
    df["method"] = df["method"].astype(str)
    df["N"] = df["N"].astype(int)
    return df


def weighted_avgs(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    # weighted by cases (usually constant, but safe)
    agg = df.groupby(group_cols, as_index=False).agg(
        sum_precision=("sum_precision", "sum"),
        sum_recall=("sum_recall", "sum"),
        sum_f1=("sum_f1", "sum"),
        sum_diag_mean=("sum_diag_mean", "sum"),
        sum_full_mean=("sum_full_mean", "sum"),
        sum_top1=("sum_top1", "sum"),
        sum_top2_comp=("sum_top2_comp", "sum"),
        sum_top2_include=("sum_top2_include", "sum"),
        cases=("cases", "sum"),
    )
    out = agg.copy()
    out["precision"] = out["sum_precision"] / out["cases"]
    out["recall"] = out["sum_recall"] / out["cases"]
    out["f1"] = out["sum_f1"] / out["cases"]
    out["diag_mean"] = out["sum_diag_mean"] / out["cases"]
    out["full_mean"] = out["sum_full_mean"] / out["cases"]
    out["top1"] = out["sum_top1"] / out["cases"]
    out["top2_comp"] = out["sum_top2_comp"] / out["cases"]
    out["top2_inc"] = out["sum_top2_include"] / out["cases"]
    out["diag_minus_full"] = out["diag_mean"] - out["full_mean"]
    keep = group_cols + ["precision", "recall", "f1", "diag_mean", "full_mean", "diag_minus_full", "top1", "top2_comp", "top2_inc"]
    return out[keep]


def stability_stats(df: pd.DataFrame) -> pd.DataFrame:
    # variance/std across N×tw cells within each (rule, utility, method)
    g = df.groupby(["rule", "utility", "method"], as_index=False).agg(
        var_precision=("precision", "var"),
        sd_precision=("precision", "std"),
        var_recall=("recall", "var"),
        sd_recall=("recall", "std"),
        var_f1=("f1", "var"),
        sd_f1=("f1", "std"),
        mean_diag_minus_full=("diag_minus_full", "mean"),
    )

    # decomposition: how much varies across tw within each N, then averaged
    v_tw = df.groupby(["rule", "utility", "method", "N"], as_index=False).agg(
        var_f1_over_tw=("f1", "var")
    )
    v_tw2 = v_tw.groupby(["rule", "utility", "method"], as_index=False).agg(
        mean_var_f1_over_tw=("var_f1_over_tw", "mean"),
        max_var_f1_over_tw=("var_f1_over_tw", "max"),
    )

    # how much varies across N within each tw, then averaged
    v_N = df.groupby(["rule", "utility", "method", "tw"], as_index=False).agg(
        var_f1_over_N=("f1", "var")
    )
    v_N2 = v_N.groupby(["rule", "utility", "method"], as_index=False).agg(
        mean_var_f1_over_N=("var_f1_over_N", "mean"),
        max_var_f1_over_N=("var_f1_over_N", "max"),
    )

    out = g.merge(v_tw2, on=["rule", "utility", "method"], how="left").merge(
        v_N2, on=["rule", "utility", "method"], how="left"
    )
    return out


def to_tidy(dfwide: pd.DataFrame, section: str, id_cols: List[str]) -> pd.DataFrame:
    metric_cols = [c for c in dfwide.columns if c not in id_cols]
    rows = []
    for _, r in dfwide.iterrows():
        base = {c: r[c] for c in id_cols}
        for mc in metric_cols:
            v = r[mc]
            if pd.isna(v):
                continue
            rows.append(
                {
                    "section": section,
                    "rule": base.get("rule", ""),
                    "utility": base.get("utility", pd.NA),
                    "method": base["method"],
                    "metric": mc,
                    "value": float(v),
                }
            )
    return pd.DataFrame(rows)


def print_markdown_table(df: pd.DataFrame, title: str, cols: List[str], float_cols: Optional[List[str]] = None):
    float_cols = float_cols or []
    print(f"\n## {title}")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in float_cols:
                cells.append(f"{float(v):.4f}")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maximinmaximax", default="/workspaces/inulab_julia_devcontainer/data/metrics_julia/grid_summary_maximinmaximax_v3.csv")
    ap.add_argument("--minimax", default="/workspaces/inulab_julia_devcontainer/data/metrics_julia/grid_summary_minimax_regret_v2.csv")
    ap.add_argument("--out", default="", help="If set, write ONE tidy CSV to this path.")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    df = pd.concat([load_grid(args.maximinmaximax), load_grid(args.minimax)], ignore_index=True)

    # (1) overall all settings
    overall = weighted_avgs(df, ["rule", "method"]).sort_values(["rule", "f1"], ascending=[True, False])
    # (2) N=6
    overall_N6 = weighted_avgs(df[df["N"] == 6], ["rule", "method"]).sort_values(["rule", "f1"], ascending=[True, False])
    # (3) stability
    stab = stability_stats(df).sort_values(["rule", "utility", "sd_f1"], ascending=[True, True, True])

    print(f"Loaded rows: {len(df)}")
    print(f"Rules: {sorted(df['rule'].unique().tolist())}")
    print(f"Utilities: {sorted(df['utility'].unique().tolist())}")
    print(f"N values: {sorted(df['N'].unique().tolist())}")

    show_cols = ["method", "precision", "recall", "f1", "diag_mean", "full_mean", "diag_minus_full"]
    float_cols = [c for c in show_cols if c != "method"]

    # Top-K per rule overall
    for rule in sorted(df["rule"].unique()):
        sub = overall[overall["rule"] == rule].head(args.topk).copy()
        print_markdown_table(sub, f"Top {args.topk} methods by F1 (overall) — rule={rule}", show_cols, float_cols)

    # Top-K per rule N=6
    for rule in sorted(df["rule"].unique()):
        sub = overall_N6[overall_N6["rule"] == rule].head(args.topk).copy()
        print_markdown_table(sub, f"Top {args.topk} methods by F1 (N=6 only) — rule={rule}", show_cols, float_cols)

    # Stability Top-K (lowest sd_f1) per (rule, utility)
    stab_cols = ["method", "sd_f1", "var_f1", "mean_var_f1_over_tw", "mean_var_f1_over_N", "mean_diag_minus_full"]
    stab_float = [c for c in stab_cols if c != "method"]
    for rule in sorted(df["rule"].unique()):
        for util in sorted(df["utility"].unique()):
            sub = stab[(stab["rule"] == rule) & (stab["utility"] == util)].head(args.topk).copy()
            print_markdown_table(sub, f"Stability Top {args.topk} (lowest sd_f1) — rule={rule}, utility={util}", stab_cols, stab_float)

    # Optional: write ONE tidy CSV
    if args.out:
        tidy_overall = to_tidy(
            overall[["rule","method","precision","recall","f1","diag_mean","full_mean","diag_minus_full","top1","top2_comp","top2_inc"]],
            "overall_all",
            ["rule","method"],
        )
        tidy_overall_N6 = to_tidy(
            overall_N6[["rule","method","precision","recall","f1","diag_mean","full_mean","diag_minus_full","top1","top2_comp","top2_inc"]],
            "overall_N6",
            ["rule","method"],
        )
        tidy_stab = to_tidy(stab, "stability_N_tw", ["rule","utility","method"])

        tidy = pd.concat([tidy_overall, tidy_overall_N6, tidy_stab], ignore_index=True)
        tidy.to_csv(args.out, index=False)
        print(f"\nSaved tidy summary CSV -> {args.out}")
    else:
        print("\n(--out not specified) Not writing any file. Use --out to save a single tidy CSV.")

if __name__ == "__main__":
    main()