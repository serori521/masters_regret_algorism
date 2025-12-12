#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== Config =====================

BASE_INPUT_PATH = "/workspaces/inulab_julia_devcontainer/data"
OUTPUT_ROOT = "./analyze_the_results/heatmaps"
SUBSAMPLE_STEP = 100

R_VALUES = {
    "regret": 100,
    "maximin": 1000,
    "Maximax": 1000,
}

N = 6
N_STR = "/N=6"

UTILITIES = [1, 2]
TRUE_WEIGHTS = ["A", "B", "C", "D", "E"]
# METHOD_DIRS = [
#     "/EV","/GM","/WMIN","/DMIN",
#     "/MMRW","/MMRD","/E-MMRW","/G-MMRW","/E-MMRD","/G-MMRD",
#     "/MMRwc","/eMMRw","/eMMRwc","/gMMRw","/gMMRwc",
#     "/eMMRd", "/eMMRdc", "/gMMRd", "/gMMRdc",
#     "/AMRW","/AMRD","/E-AMRW","/G-AMRW","/E-AMRD","/G-AMRD",
#     "/AMRwc","/eAMRw","/eAMRwc","/gAMRw","/gMMRwc",
#     "/eAMRd", "eAMRdc", "gAMRd", "gAMRdc",
# ]
METHOD_DIRS = [
    "/EV","/GM","/WMIN",
    "/MMRW","/E-MMRW","/G-MMRW",
    "/MMRwc","/eMMRw","/eMMRwc","/gMMRw","/gMMRwc",
    "/eMMRd", "/eMMRdc", "/gMMRd", "/gMMRdc",
    "/AMRW","/E-AMRW",
    "/AMRwc","/eAMRw","/eAMRwc","/gAMRw","/gAMRwc",
    "/eAMRd", "/eAMRdc", "/gAMRd", "/gAMRdc",
]
METHOD_NAMES = [
    "EV","GM","MSW",
    "MMRW","E-MMRW","G-MMRW",
    "MMRwc","eMMRw","eMMRwc","gMMRw","gMMRwc",
    "eMMRd", "eMMRdc", "gMMRd", "gMMRdc",
    "AMRW","E-AMRW",
    "AMRwc","eAMRw","eAMRwc","gAMRw","gMMRwc",
    "eAMRd", "eAMRdc", "gAMRd", "gAMRdc",
]

TARGET_ROWS = 10
TARGET_COLS = 10

which_u = {1: "/u1", 2: "/u2"}

# ===================== Helper functions =====================

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
    except Exception as e:
        print(f"[read_regret_matrices] error for {filepath}: {e}")
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
    except Exception as e:
        print(f"[read_c_count_pairs] error for {filepath}: {e}")
    return matrices


def average_resized_matrix(mats: list, step: int = SUBSAMPLE_STEP) -> np.ndarray:
    if not mats:
        return np.zeros((TARGET_ROWS, TARGET_COLS))

    if step > 1 and len(mats) > step:
        mats = mats[::step]
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


def build_input_path(rule: str, u: int, tw: str, method_dir: str, R: int) -> str:
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


# ===================== Heatmap core =====================

def get_color_settings(utility: int):
    """utilityごとの vmin / vmax / cmap を返す（必須条件）"""
    if utility == 1:
        return 7.3, 9.2, 'YlOrRd'
    elif utility == 2:
        return 4.8, 7.3, 'viridis'
    else:
        return None
def load_and_average(rule: str, u: int, tw: str,
                     mdir: str, mname: str, R: int):
    """
    ファイル読み込み＋10x10平均行列の計算だけを行う。
    描画（matplotlib）は一切ここでやらない。
    """
    path = build_input_path(rule, u, tw, mdir, R)
    if not os.path.exists(path):
        return False, (rule, u, tw, mname), None, "no-file"

    if rule == "regret":
        mats = read_regret_matrices(path)
    else:
        mats = read_c_count_pairs(path)

    if not mats:
        return False, (rule, u, tw, mname), None, "no-mats"

    avg = average_resized_matrix(mats)
    return True, (rule, u, tw, mname), avg, None

# ===================== 描画含むメイン処理（描画はシングルスレッド） =====================

def create_heatmaps_for_rule(rule: str):
    R = R_VALUES[rule]
    print(f"\n===== Heatmaps for rule = {rule}, R={R} =====")

    # 1. 並列で avg を計算して results に貯める
    results = []  # (rule, u, tw, mname, avg, vmin, vmax, cmap)

    # future -> 付帯情報 (u, tw, mname, vmin_val, vmax_val, cmap_val)
    future_info = {}

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        # ----- ジョブ投入 -----
        for u in UTILITIES:
            vmin_val, vmax_val, cmap_val = get_color_settings(u)
            for tw in TRUE_WEIGHTS:
                for mdir, mname in zip(METHOD_DIRS, METHOD_NAMES):
                    fut = ex.submit(load_and_average, rule, u, tw, mdir, mname, R)
                    future_info[fut] = (u, tw, mname, vmin_val, vmax_val, cmap_val)

        total = len(future_info)
        done = 0

        # ----- 結果回収 ＋ 進捗表示 -----
        for fut in as_completed(future_info):
            done += 1
            u, tw, mname, vmin_val, vmax_val, cmap_val = future_info[fut]
            ok, key, avg, err = fut.result()

            if not ok:
                # データは確実にある前提なら、ここはおかしい場合だけログになる
                print(f"  [skip] {err}: rule={rule}, u{u}, TW={tw}, method={mname}")
            else:
                results.append((rule, u, tw, mname, avg, vmin_val, vmax_val, cmap_val))

            # ★ここで簡易進捗ログ★
            if done == 1 or done % 10 == 0 or done == total:
                pct = done / total * 100
                print(f"  [progress] {done}/{total} ({pct:.1f}%) jobs finished for rule={rule}")

    # 2. シングルスレッドで描画
    for rule_k, u_k, tw_k, mname_k, avg, vmin_val, vmax_val, cmap_val in results:
        flipped = np.flipud(avg)

        out_dir = os.path.join(OUTPUT_ROOT, rule_k, f"u{u_k}", "N6")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"heatmap_{rule_k}_{mname_k}_u{u_k}_N6_{tw_k}.png"
        )

        plt.figure(figsize=(8, 7))
        sns.heatmap(
            flipped,
            annot=True,
            fmt=".2f",
            cmap=cmap_val,
            vmin=vmin_val,
            vmax=vmax_val,
            cbar_kws={"label": "value"},
            square=True,
            xticklabels=range(TARGET_COLS),
            yticklabels=range(TARGET_ROWS-1, -1, -1)
        )
        plt.title(
            f"{rule_k.upper()} / {mname_k}\n"
            f"u={u_k}, TW={tw_k}, resized {TARGET_ROWS}x{TARGET_COLS}"
        )
        plt.xlabel("True rank index")
        plt.ylabel("Estimated rank index")

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  [ok] {out_path}  [min={avg.min():.3f}, max={avg.max():.3f}]")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for rule in [ "Maximax"]:
        create_heatmaps_for_rule(rule)
    print("\nAll heatmaps done.")


if __name__ == "__main__":
    main()