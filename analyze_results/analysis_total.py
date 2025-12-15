#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分割 tidy（例: tidy_raw_v2_regret_u1_A_R1000_N6.csv）を全部集めて結合し、
analysis_typeごとに

| Method | Recall::u1,A ... Recall::u2,E | Precision::... | F1::... |

のワイド表を results/csv/ に出力する。
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np


# =========================
# 設定（ここだけ変えればOK）
# =========================
PROJECT_ROOT = Path("/workspaces/inulab_julia_devcontainer")
IN_DIR = PROJECT_ROOT / "results" / "metrics_python"
OUT_DIR = PROJECT_ROOT / "results" / "csv"

DECISION_RULE = "regret"
R = 1000
N = 6

CASE_ORDER = ["u1", "u2"]
TW_ORDER = ["A", "B", "C", "D", "E"]

# split tidy の想定ファイル名
# tidy_raw_v2_regret_u1_A_R1000_N6.csv
SPLIT_PATTERN = re.compile(
    r"^tidy_raw_v2_(?P<rule>[^_]+)_(?P<case>u[12])_(?P<tw>[A-E])_R(?P<R>\d+)_N(?P<N>\d+)\.csv$"
)


def ensure_f1(df: pd.DataFrame) -> pd.DataFrame:
    if "F1_Score" not in df.columns:
        df["F1_Score"] = 2 * (df["Precision"] * df["Recall"]) / (df["Precision"] + df["Recall"])
        df["F1_Score"] = df["F1_Score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def ordered_case_tw_cols() -> list:
    return [f"{c},{w}" for c in CASE_ORDER for w in TW_ORDER]


def collect_split_tidies() -> pd.DataFrame:
    """
    分割 tidy を全部読み、1つの DataFrame に結合して返す。
    ファイル名由来の case/tw を優先し、CSV中の空白も strip する。
    """
    files = []
    for p in IN_DIR.glob("tidy_raw_v2_*.csv"):
        m = SPLIT_PATTERN.match(p.name)
        if not m:
            continue
        if m.group("rule") != DECISION_RULE:
            continue
        if int(m.group("R")) != R or int(m.group("N")) != N:
            continue
        files.append((p, m.group("case"), m.group("tw")))

    if not files:
        raise FileNotFoundError(f"split tidy not found in {IN_DIR}")

    dfs = []
    for p, case, tw in sorted(files, key=lambda x: (x[1], x[2], x[0].name)):
        df = pd.read_csv(p, encoding="utf-8-sig")

        # 必須列チェック
        need = {"Method_Name", "Analysis_Type", "Precision", "Recall"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{p.name} missing columns: {miss}")

        # 正規化：空白除去
        df["Method_Name"] = df["Method_Name"].astype(str).str.strip()
        df["Analysis_Type"] = df["Analysis_Type"].astype(str).str.strip()
        if "Missing_File" in df.columns:
            # 念のため
            df["Missing_File"] = pd.to_numeric(df["Missing_File"], errors="coerce").fillna(0).astype(int)

        # Case/Width_Pattern は「ファイル名」を信用（中身がズレてても強制でそろえる）
        df["Case"] = case
        df["Width_Pattern"] = tw

        dfs.append(df)

    big = pd.concat(dfs, ignore_index=True)
    big = ensure_f1(big)
    return big


def build_summary_for_analysis(df: pd.DataFrame, analysis_type: str) -> pd.DataFrame:
    sub = df[df["Analysis_Type"] == analysis_type].copy()
    if sub.empty:
        return pd.DataFrame()

    # Missing_File=1 は除外（NaN行大量で邪魔）
    if "Missing_File" in sub.columns:
        sub = sub[sub["Missing_File"] == 0].copy()
        if sub.empty:
            return pd.DataFrame()

    sub["CaseTW"] = sub["Case"].astype(str) + "," + sub["Width_Pattern"].astype(str)

    # 重複は平均で潰す（通常は起きないはず）
    g = sub.groupby(["Method_Name", "CaseTW"], as_index=False)[["Recall", "Precision", "F1_Score"]].mean()

    recall = g.pivot(index="Method_Name", columns="CaseTW", values="Recall")
    prec   = g.pivot(index="Method_Name", columns="CaseTW", values="Precision")
    f1     = g.pivot(index="Method_Name", columns="CaseTW", values="F1_Score")

    col_order = ordered_case_tw_cols()
    recall = recall.reindex(columns=col_order)
    prec   = prec.reindex(columns=col_order)
    f1     = f1.reindex(columns=col_order)

    recall.columns = [f"Recall::{c}" for c in recall.columns]
    prec.columns   = [f"Precision::{c}" for c in prec.columns]
    f1.columns     = [f"F1::{c}" for c in f1.columns]

    out = pd.concat([recall, prec, f1], axis=1).reset_index()
    out = out.rename(columns={"Method_Name": "Method"})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] collecting split tidies from: {IN_DIR}")
    df = collect_split_tidies()
    print(f"[INFO] rows={len(df)}  analysis_types={sorted(df['Analysis_Type'].unique().tolist())}")

    analysis_types = sorted(df["Analysis_Type"].unique().tolist())
    for at in analysis_types:
        wide = build_summary_for_analysis(df, at)
        if wide.empty:
            print(f"[SKIP] {at} (no data after filtering)")
            continue

        outpath = OUT_DIR / f"summary_{DECISION_RULE}_{at}_R{R}_N{N}.csv"
        wide.to_csv(outpath, index=False, encoding="utf-8-sig")
        print(f"[WRITE] {outpath}  rows={len(wide)}")


if __name__ == "__main__":
    main()
