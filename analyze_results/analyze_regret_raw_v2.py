
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================
# 定数（必要ならここだけ編集）
# =========================
R = 1000          # *_minimax_regret_{R}.csv
N = 6
ALT_N = 5
MAX_PAIRS = ALT_N * (ALT_N - 1) / 2.0   # concord の最大値（ペア数）
DECISION_RULE = "regret"

which_utility = ["u1", "u2"]
true_weights = ["A", "B", "C", "D", "E"]

# methods = ["/EV","/GM","/WMIN","/DMIN",
# "/MMRW","/MMRD","/E-MMRW","/G-MMRW","/E-MMRD","/G-MMRD",
# "/MMRwc","/eMMRw","/eMMRwc","/gMMRw","/gMMRwc",
# "/eMMRd", "/eMMRdc", "/gMMRd", "/gMMRdc",
# "/AMRW","/AMRD","/E-AMRW","/G-AMRW","/E-AMRD","/G-AMRD",
# "/AMRwc","/eAMRw","/eAMRwc","/gAMRw","/gAMRwc",
# "/eAMRd", "/eAMRdc", "/gAMRd", "/gAMRdc"]
# method_names = ["EV","GM","MSW","DMIN",
# "MMRW","MMRD","E-MMRW","G-MMRW","E-MMRD","G-MMRD",
# "MMRwc","eMMRw","eMMRwc","gMMRw","gMMRwc",
# "eMMRd", "eMMRdc", "gMMRd", "gMMRdc",
# "AMRW","AMRD","E-AMRW","G-AMRW","E-AMRD","G-AMRD",
# "AMRwc","eAMRw","eAMRwc","gAMRw","gAMRwc",
# "eAMRd", "eAMRdc", "gAMRd", "gAMRdc"]
methods = ["/EV","/GM","/WMIN",
"/MMRW",
"/MMRwc","/eMMRw","/eMMRwc","/gMMRw","/gMMRwc",
"/eMMRd", "/eMMRdc", "/gMMRd", "/gMMRdc",
"/AMRW",
"/AMRwc","/eAMRw","/eAMRwc","/gAMRw","/gAMRwc",
"/eAMRd", "/eAMRdc", "/gAMRd", "/gAMRdc"]
method_names = ["EV","GM","MSW",
"MMRW",
"MMRwc","eMMRw","eMMRwc","gMMRw","gMMRwc",
"eMMRd", "eMMRdc", "gMMRd", "gMMRdc",
"AMRW",
"AMRwc","eAMRw","eAMRwc","gAMRw","gAMRwc",
"eAMRd", "eAMRdc", "gAMRd", "gAMRdc"]
traditional_methods = {"EV", "GM"}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
raw_v2 (minimax_regret_{R}.csv) を読み、
concord 行列の「中身の数値」から以下を計算して tidy CSV を出力する。

- Recall  : 真(列)ごとに max を取り、その平均  = mean(max(m, axis=0))
- Precision: 推定(行)ごとに max を取り、その平均 = mean(max(m, axis=1))
- F1      : 調和平均

さらに、min_average / absolute_min も従来スクリプト互換で出す。

加えて：
- 並列化（ProcessPoolExecutor）
- Case×Width_Pattern（例 u1×A）ごとに分割CSVも同時出力

VSCode 右上 ▶ 実行だけで動く想定。
"""



# 並列数（I/OもあるのでCPU全部より少なめが安定しやすい）
CPU = os.cpu_count() or 2
N_WORKERS = max(1, min(16, CPU - 2))  # 上限8くらいに抑える（環境により調整）


# =========================
# パス
# data/a3/regret/{utility}/N=6/{tw}/{method}/{utility}_minimax_regret_{R}.csv
# =========================
def build_input_path(base_data: Path, utility: str, tw: str, method_clean: str) -> Path:
    return (base_data / "a3" / DECISION_RULE / utility / f"N={N}" / tw / method_clean
            / f"{utility}_minimax_regret_{R}.csv")


# =========================
# 指標計算
# =========================
def _parse_float_row_fast(line: str, cnt_true: int) -> np.ndarray:
    """
    CSVの行:  t_method, v1, v2, ..., v_cnt_true, ...
    から v1..v_cnt_true を numpy.fromstring で高速に読む
    """
    # 先頭の t_method を捨てて、最初のカンマ以降だけ
    _, _, rest = line.partition(",")
    arr = np.fromstring(rest, sep=",", count=cnt_true)
    if arr.size < cnt_true:
        out = np.full(cnt_true, np.nan, dtype=float)
        out[:arr.size] = arr
        return out
    return arr


def scan_raw_v2_metrics_concord(path: Path, normalize: bool = True) -> Dict[str, float]:
    """
    raw_v2 を走査して、ブロック（=1行列）ごとに
    - max_based:  recall=mean(col_max), precision=mean(row_max)
    - min_average_analysis: recall=mean(col_min), precision=mean(row_min)
    - absolute_min_analysis: 行列の最小値（precision=recall=absmin）
    を計算し、最後にブロック平均を返す。
    """
    blocks = 0

    recall_max_sum = 0.0
    prec_max_sum = 0.0

    recall_min_sum = 0.0
    prec_min_sum = 0.0

    absmin_sum = 0.0

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        while True:
            header = f.readline()
            if not header:
                break
            h = header.strip()
            if not h:
                continue

            parts = [p.strip() for p in h.split(",")]
            if len(parts) < 4:
                continue

            try:
                _utl_num = int(parts[0])
                _repeat = int(parts[1])
                cnt_true = int(parts[2])
                cnt_method = int(parts[3])
            except ValueError:
                continue

            # true line（時刻列）を読み飛ばす
            true_line = f.readline()
            if not true_line:
                break

            # --- このブロックの集計器 ---
            col_max = np.full(cnt_true, -np.inf, dtype=float)
            row_max_sum = 0.0
            row_max_cnt = 0

            col_min = np.full(cnt_true, +np.inf, dtype=float)
            row_min_sum = 0.0
            row_min_cnt = 0

            block_min = +np.inf

            # method 行を読む
            for _ in range(cnt_method):
                line = f.readline()
                if not line:
                    break

                vals = _parse_float_row_fast(line, cnt_true)

                if normalize:
                    vals = vals / MAX_PAIRS

                if np.all(np.isnan(vals)):
                    continue

                # max-based
                v_for_max = np.where(np.isnan(vals), -np.inf, vals)
                col_max = np.maximum(col_max, v_for_max)
                row_max = np.max(v_for_max)
                if row_max != -np.inf:
                    row_max_sum += float(row_max)
                    row_max_cnt += 1

                # min-average
                v_for_min = np.where(np.isnan(vals), +np.inf, vals)
                col_min = np.minimum(col_min, v_for_min)
                row_min = np.min(v_for_min)
                if row_min != +np.inf:
                    row_min_sum += float(row_min)
                    row_min_cnt += 1

                # absolute min
                m = float(np.min(v_for_min))
                if m != +np.inf:
                    block_min = min(block_min, m)

            # ブロックの recall/precision を確定
            col_max2 = np.where(np.isneginf(col_max), np.nan, col_max)
            col_min2 = np.where(np.isposinf(col_min), np.nan, col_min)

            recall_max = float(np.nanmean(col_max2)) if np.any(~np.isnan(col_max2)) else np.nan
            prec_max = (row_max_sum / row_max_cnt) if row_max_cnt > 0 else np.nan

            recall_min = float(np.nanmean(col_min2)) if np.any(~np.isnan(col_min2)) else np.nan
            prec_min = (row_min_sum / row_min_cnt) if row_min_cnt > 0 else np.nan

            absmin = block_min if block_min != +np.inf else np.nan

            blocks += 1
            recall_max_sum += 0.0 if np.isnan(recall_max) else recall_max
            prec_max_sum += 0.0 if np.isnan(prec_max) else prec_max

            recall_min_sum += 0.0 if np.isnan(recall_min) else recall_min
            prec_min_sum += 0.0 if np.isnan(prec_min) else prec_min

            absmin_sum += 0.0 if np.isnan(absmin) else absmin

    if blocks == 0:
        return {"blocks": 0}

    return {
        "blocks": blocks,
        "recall_max": recall_max_sum / blocks,
        "prec_max": prec_max_sum / blocks,
        "recall_min": recall_min_sum / blocks,
        "prec_min": prec_min_sum / blocks,
        "absmin": absmin_sum / blocks,
    }


# =========================
# tidy 保存
# =========================
def save_results_tidy(results: List[dict], output_file: Path):
    if not results:
        print("結果が空です。")
        return
    df = pd.DataFrame(results)

    # F1 を統一計算
    df["F1_Score"] = 2 * (df["Precision"] * df["Recall"]) / (df["Precision"] + df["Recall"])
    df["F1_Score"] = df["F1_Score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    col_order = [
        "Method_Type", "Method_Name",
        "Case", "Decision_Rule", "Width_Pattern",
        "Analysis_Type",
        "Blocks",
        "Precision", "Recall", "F1_Score",
        "Missing_File",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"【Tidy Data】保存完了: {output_file}")


def save_split_by_case_weight(all_results: List[dict], out_dir: Path):
    """
    Case×Width_Pattern（例：u1×A）ごとに tidy CSV を分割保存
    """
    if not all_results:
        return
    df = pd.DataFrame(all_results)

    # split 側でも F1 を付与（単体でも見れるように）
    df["F1_Score"] = 2 * (df["Precision"] * df["Recall"]) / (df["Precision"] + df["Recall"])
    df["F1_Score"] = df["F1_Score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out_dir.mkdir(parents=True, exist_ok=True)

    for (case, tw), sub in df.groupby(["Case", "Width_Pattern"], dropna=False):
        outpath = out_dir / f"tidy_raw_v2_{DECISION_RULE}_{case}_{tw}_R{R}_N{N}.csv"
        sub.to_csv(outpath, index=False, encoding="utf-8-sig")
        print(f"[SPLIT] wrote {outpath}")


# =========================
# 並列 worker
# =========================
def worker_one_file(args: Tuple[str, str, str, str, str]) -> List[dict]:
    """
    1ファイル分を処理して、3つの analysis_type の結果を返す。
    args = (inpath_str, utility, tw, method_name, method_type)
    """
    inpath_str, utility, tw, mname, method_type = args
    inpath = Path(inpath_str)

    if not inpath.exists():
        out = []
        for analysis_type in ["max_based", "min_average_analysis", "absolute_min_analysis"]:
            out.append({
                "Method_Name": mname,
                "Method_Type": method_type,
                "Case": utility,
                "Decision_Rule": DECISION_RULE,
                "Width_Pattern": tw,
                "Analysis_Type": analysis_type,
                "Blocks": 0,
                "Precision": np.nan,
                "Recall": np.nan,
                "Missing_File": 1,
            })
        return out

    met = scan_raw_v2_metrics_concord(inpath, normalize=True)
    if met.get("blocks", 0) == 0:
        out = []
        for analysis_type in ["max_based", "min_average_analysis", "absolute_min_analysis"]:
            out.append({
                "Method_Name": mname,
                "Method_Type": method_type,
                "Case": utility,
                "Decision_Rule": DECISION_RULE,
                "Width_Pattern": tw,
                "Analysis_Type": analysis_type,
                "Blocks": 0,
                "Precision": np.nan,
                "Recall": np.nan,
                "Missing_File": 0,
            })
        return out

    blocks = int(met["blocks"])
    out = []

    # 1) max_based
    out.append({
        "Method_Name": mname,
        "Method_Type": method_type,
        "Case": utility,
        "Decision_Rule": DECISION_RULE,
        "Width_Pattern": tw,
        "Analysis_Type": "max_based",
        "Blocks": blocks,
        "Precision": float(met["prec_max"]),
        "Recall": float(met["recall_max"]),
        "Missing_File": 0,
    })
    # 2) min_average_analysis
    out.append({
        "Method_Name": mname,
        "Method_Type": method_type,
        "Case": utility,
        "Decision_Rule": DECISION_RULE,
        "Width_Pattern": tw,
        "Analysis_Type": "min_average_analysis",
        "Blocks": blocks,
        "Precision": float(met["prec_min"]),
        "Recall": float(met["recall_min"]),
        "Missing_File": 0,
    })
    # 3) absolute_min_analysis（Precision/Recall同値）
    absmin = float(met["absmin"])
    out.append({
        "Method_Name": mname,
        "Method_Type": method_type,
        "Case": utility,
        "Decision_Rule": DECISION_RULE,
        "Width_Pattern": tw,
        "Analysis_Type": "absolute_min_analysis",
        "Blocks": blocks,
        "Precision": absmin,
        "Recall": absmin,
        "Missing_File": 0,
    })
    return out


# =========================
# メイン（並列）
# =========================
def main_analysis_tidy_parallel(base_data_path: Path, base_output_path: Path):
    tasks: List[Tuple[str, str, str, str, str]] = []

    for utility in which_utility:
        for tw in true_weights:
            for md, mname in enumerate(method_names):
                method_clean = methods[md].lstrip("/")
                inpath = build_input_path(base_data_path, utility, tw, method_clean)
                method_type = "従来法" if mname in traditional_methods else "区間推定法"
                tasks.append((str(inpath), utility, tw, mname, method_type))

    print(f"[INFO] workers={N_WORKERS} tasks={len(tasks)}")

    all_results: List[dict] = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(worker_one_file, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            rows = fut.result()
            all_results.extend(rows)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"[DONE] {done}/{len(tasks)}")

    # 全体 tidy
    outpath = base_output_path / f"tidy_raw_v2_{DECISION_RULE}_R{R}_N{N}.csv"
    save_results_tidy(all_results, outpath)

    # u1,A など Case×Width_Pattern 分割 tidy
    save_split_by_case_weight(all_results, base_output_path)

    return all_results


if __name__ == "__main__":
    base_data_path = Path("/workspaces/inulab_julia_devcontainer/data")
    base_output_path = Path("/workspaces/inulab_julia_devcontainer/results/metrics_python")
    main_analysis_tidy_parallel(base_data_path, base_output_path)
