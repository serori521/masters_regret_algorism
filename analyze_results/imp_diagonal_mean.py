#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Extract diagonal values (main diagonal) from the averaged 10x10 heatmap-ready
matrices for each method, split by true weight pattern and utility case.

Update:
- Diagonal の計算は従来どおり 10 個取る。
- ただし保存は「縦持ち tidy 形式」で 1 つの CSV にまとめる。
  1 行 = (Rule, Utility, TrueWeight, Method, DiagMean)

• Only N=6 is processed.
• Decision rules handled separately: regret (minimax regret), maximin, maximax.
• Output is saved under:
    ./results/diagonal_summaries/diagonal_tidy_all_rules_u1u2_N6.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ===================== Config =====================
BASE_INPUT_PATH = "/workspaces/inulab_julia_devcontainer/data"  # keep as-is
OUTPUT_ROOT = "./results"

R_VALUES = {
    "regret": 100,
    "maximin": 1000,
    "Maximax": 1000,
}

N = 6
N_STR = "/N=6"
UTILITIES = [1, 2]
TRUE_WEIGHTS = ["A", "B", "C", "D", "E"]

METHOD_DIRS = [
    "/EV","/GM","/WMIN","/DMIN",
    "/MMRW","/MMRD","/E-MMRW","/G-MMRW","/E-MMRD","/G-MMRD",
    "/MMRwc","/eMMRw","/eMMRwc","/gMMRw","/gMMRwc",
    "/eMMRd", "/eMMRdc", "/gMMRd", "/gMMRdc",
    "/AMRW","/AMRD","/E-AMRW","/G-AMRW","/E-AMRD","/G-AMRD",
    "/AMRwc","/eAMRw","/eAMRwc","/gAMRw","/gAMRwc",
    "/eAMRd", "eAMRdc", "gAMRd", "gAMRdc",
]
METHOD_NAMES = [
    "EV","GM","MSW","DMIN",
    "MMRW","MMRD","E-MMRW","G-MMRW","E-MMRD","G-MMRD",
    "MMRwc","eMMRw","eMMRwc","gMMRw","gMMRwc",
    "eMMRd", "eMMRdc", "gMMRd", "gMMRdc",
    "AMRW","AMRD","E-AMRW","G-AMRW","E-AMRD","G-AMRD",
    "AMRwc","eAMRw","eAMRwc","gAMRw","gAMRwc",
    "eAMRd", "eAMRdc", "gAMRd", "gAMRdc",
]

# for並び順を安定させたいとき用（最後のソートに使う）
METHOD_ORDER = {name: idx for idx, name in enumerate(METHOD_NAMES)}

TARGET_ROWS = 10
TARGET_COLS = 10

# ===================== Helpers =====================
def _resize_to_target(matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((rows, cols))
    r0, c0 = matrix.shape
    if r0 == rows and c0 == cols:
        return matrix
    r_zoom = rows / r0
    c_zoom = cols / c0
    resized = zoom(matrix, (r_zoom, c_zoom), order=0)
    resized = resized[:rows, :cols]
    if resized.shape[0] < rows:
        resized = np.pad(resized, ((0, rows - resized.shape[0]), (0, 0)), mode='edge')
    if resized.shape[1] < cols:
        resized = np.pad(resized, ((0, 0), (0, cols - resized.shape[1])), mode='edge')
    return resized


def read_regret_matrices(filepath: str) -> list:
    matrices = []
    if not os.path.exists(filepath):
        return matrices
    try:
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')
        i = 0
        while i < len(lines):
            parts = lines[i].split(',')
            if len(parts) < 4:
                i += 1
                continue
            try:
                k, l, cnt1, cnt2 = map(int, parts[:4])
            except ValueError:
                i += 1
                continue
            i += 2
            block = []
            for _ in range(cnt2):
                if i >= len(lines):
                    break
                row = lines[i].split(',')
                if len(row) >= cnt1 + 1:
                    vals = [float(x) for x in row[1:cnt1+1]]
                    block.append(vals)
                i += 1
            if block:
                matrices.append(np.array(block))
    except Exception:
        pass
    return matrices


def read_c_count_pairs(filepath: str) -> list:
    matrices = []
    if not os.path.exists(filepath):
        return matrices
    try:
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')
        i = 0
        while i < len(lines):
            parts = lines[i].split(',')
            if len(parts) < 4:
                i += 1
                continue
            try:
                _, _, cnt1, cnt2 = map(int, parts[:4])
            except ValueError:
                i += 1
                continue
            i += 2
            block = []
            for _ in range(cnt2):
                if i >= len(lines):
                    break
                row = lines[i].split(',')
                if len(row) >= cnt1 + 3:
                    try:
                        vals = [int(x) for x in row[3:cnt1+3]]
                    except ValueError:
                        vals = []
                    if vals:
                        block.append(vals)
                i += 1
            if block:
                matrices.append(np.array(block, dtype=float))
    except Exception:
        pass
    return matrices


def average_resized_matrix(mats: list) -> np.ndarray:
    if not mats:
        return np.zeros((TARGET_ROWS, TARGET_COLS))
    acc = np.zeros((TARGET_ROWS, TARGET_COLS), dtype=float)
    n = 0
    for m in mats:
        if m.size == 0:
            continue
        acc += _resize_to_target(m, TARGET_ROWS, TARGET_COLS)
        n += 1
    if n == 0:
        return np.zeros((TARGET_ROWS, TARGET_COLS))
    return acc / n


def diagonal10(avg: np.ndarray) -> np.ndarray:
    # 10個分とる（元コードと同じ振る舞い）
    return np.diag(avg)[:10]

# ===================== Parallel unit =====================

def build_input_path(rule: str, u: int, tw: str, method_dir: str, R: int) -> str:
    which_u = {1: "/u1", 2: "/u2"}
    if rule == 'regret':
        return (
            f"{BASE_INPUT_PATH}/a3/{rule}{which_u[u]}{N_STR}/"
            f"{tw}{method_dir}{which_u[u]}_minimax_regret_{R}.csv"
        )
    else:
        return (
            f"{BASE_INPUT_PATH}/a3/{rule}{which_u[u]}{N_STR}/"
            f"{tw}{method_dir}/{rule}_count_pairs_in_squares_{R}_u{u}.csv"
        )


def load_and_compute(rule: str, u: int, tw: str,
                     method_dir: str, method_name: str, R: int):
    """1 (rule, u, tw, method) について平均対角値を計算して返す"""
    thread_id = threading.get_ident()
    print(f"[Thread {thread_id}] Start: rule={rule}, u{u}, TW={tw}, method={method_name}")
    path = build_input_path(rule, u, tw, method_dir, R)
    if rule == 'regret':
        mats = read_regret_matrices(path)
    else:
        mats = read_c_count_pairs(path)
    if not mats:
        # データなし
        return False, (rule, u, tw, method_name), None
    avg = average_resized_matrix(mats)
    d = diagonal10(avg)
    diag_mean = float(np.mean(d))  # ★ 10 個の平均をとる
    return True, (rule, u, tw, method_name), diag_mean

# ===================== Core routine =====================

def process_rule(rule: str, tidy_records: list):
    """
    1つの rule について全 (u, TW, method) を並列処理し、
    結果を tidy_records (外から渡された list) に append していく。
    """
    R = R_VALUES[rule]

    futures = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        for u in UTILITIES:
            for tw in TRUE_WEIGHTS:
                for mdir, mname in zip(METHOD_DIRS, METHOD_NAMES):
                    futures.append(
                        ex.submit(load_and_compute, rule, u, tw, mdir, mname, R)
                    )

        for f in as_completed(futures):
            ok, key, diag_mean = f.result()
            if not ok:
                continue
            rule_k, u_k, tw_k, mname_k = key
            tidy_records.append(
                {
                    "Rule": rule_k,
                    "Utility": u_k,
                    "TrueWeight": tw_k,
                    "Method": mname_k,
                    "DiagMean": diag_mean,
                }
            )


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ここに全 rule 分の (rule, u, TW, method, DiagMean) を貯める
    tidy_records = []

    for rule in ["regret", "maximin", "Maximax"]:
        print(f"===== Processing rule: {rule} =====")
        process_rule(rule, tidy_records)

    # DataFrame 化して並び順を整えつつ保存
    if not tidy_records:
        print("No diagonal data found. No CSV will be written.")
        return

    df = pd.DataFrame(tidy_records)

    # 並び替え: Rule, Utility, TrueWeight, Method (METHOD_ORDER に従って)
    df["MethodOrder"] = df["Method"].map(lambda m: METHOD_ORDER.get(m, 9999))
    df = df.sort_values(
        by=["Rule", "Utility", "TrueWeight", "MethodOrder"]
    ).drop(columns=["MethodOrder"])

    out_file = os.path.join(OUTPUT_ROOT, "diagonal_tidy_all_rules_u1u2_N6.csv")
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Tidy diagonal CSV saved: {out_file} (rows={len(df)})")


if __name__ == "__main__":
    main()
