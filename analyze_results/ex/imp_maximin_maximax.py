import os
import csv
import math
from pathlib import Path

# ==== 元の C++ に合わせた定数たち ====
R = 1000          # ひとつの真の重要度の組に対する繰り返し回数
Generated_method = 3
N = 6             # 評価基準数
Alt = 5           # 代替案数
M = 100           # 効用行列の個数
Rt = 5            # 真の重要度 A〜E
epsi = 1e-6

# N=6 固定の想定
evaluation_num = N - 4
eval_num_suffix = ["N=4", "N=5", "N=6", "N=7+", "N=8+"]
util_num_suffix = ["N=4", "N=5", "N=6_M=5", "N=7+", "N=8+"]

true_weight_list = ["A", "B", "C", "D", "E"]

# ---- 元コードの method 群（コメントアウトも含めて統合） ----
# 使われていた 16 手法
active_method_dirs = [
    "EV", "GM", "WMIN", "DMIN",
    "MMRW", "MMRD", "E-MMRW", "G-MMRW", "E-MMRD", "G-MMRD",
    "AMRW", "AMRD", "E-AMRW", "G-AMRW", "E-AMRD", "G-AMRD",
]

# コメントアウトされていた旧 31 手法（ファイル名側）
legacy_method_dirs = [
    "MSW","MSWW","MMRW","AMRW","E-MMRW","G-MMRW","MSD","MMRD","AMRD",
    "E-MMRD","G-MMRD","MMRLD","AMRLD",
    "EV","GM","E-MSW","G-MSW","E-MSD","G-MSD",
    "E-AMRW","G-AMRW","E-AMRD","G-AMRD",
    "MMRWW","AMRWW","E-MMRWW","G-MMRWW","E-AMRWW","G-AMRWW",
    "E-MSWW","G-MSWW"
]

# コメントアウトされていた「w 付き」16 手法
weighted_method_dirs = [
    "E-AMRw", "E-MMRw", "G-AMRw", "G-MMRw",
    "eAMRw", "eAMRwc", "eMMRw", "eMMRwc",
    "gAMRw", "gAMRwc", "gMMRw", "gMMRwc",
    "lAMRw", "lAMRwc", "lMMRw", "lMMRwc"
]

# 全部まとめて重複を削除
ALL_METHOD_DIRS = sorted(set(active_method_dirs + legacy_method_dirs + weighted_method_dirs))

max_or_min_list = ["maximin", "Maximax"]
which_utility_list = ["u1", "u2"]


# ===== ユーティリティ関数 =====

def load_true_interval_weight(base_dir, eval_idx, rt_idx):
    """真の区間重要度 wL, wR を読み込む"""
    eval_suffix = eval_num_suffix[eval_idx]
    tw_label = true_weight_list[rt_idx]

    path = Path(base_dir) / "true_interval_weight_set" / eval_suffix / tw_label / "Given_interval_weight.csv"
    trueW = []
    with path.open("r", encoding="cp932") as f:
        row = next(csv.reader(f))
        # 2N 個の値
        trueW = [float(x) for x in row[:2 * N]]

    true_wL = [trueW[2 * i] for i in range(N)]
    true_wR = [trueW[2 * i + 1] for i in range(N)]
    return true_wL, true_wR


def load_utility_matrix(base_dir, eval_idx, utility_idx):
    """
    u[Alt][N] を M 個分読み込む。
    C++ では Alt*M 行, 各行 8 列(u1,u2,...u8) の CSV を読んでいた。
    Python では (M, Alt, N) の3次元リストで返す。
    """
    eval_suffix = util_num_suffix[eval_idx]
    which_u = which_utility_list[utility_idx]

    path = Path(base_dir) / f"効用値行列"/which_u / eval_suffix / "u.csv"
    u_all = []
    with path.open("r", encoding="cp932") as f:
        reader = csv.reader(f)
        rows = [ [float(x) for x in row] for row in reader ]

    if len(rows) != Alt * M:
        raise ValueError(f"効用値行数が想定と違います: got {len(rows)}, expected {Alt*M}")

    for m in range(M):
        block = rows[m * Alt:(m + 1) * Alt]
        # 各行の先頭 N 列だけ使う
        u_block = [row[:N] for row in block]
        u_all.append(u_block)

    # 形は [M][Alt][N]
    return u_all


def load_estimated_interval_weights(base_dir, eval_idx, rt_idx, method_dir):
    eval_suffix = eval_num_suffix[eval_idx]
    tw_label = true_weight_list[rt_idx]

    path = Path(base_dir) / "Simp" / eval_suffix / f"a{Generated_method}" / tw_label / method_dir / "Simp.csv"
    if not path.exists():
        return None, None

    # ---- どのエンコードでも読めるように試行 ----
    rows = None
    for enc in ("utf-8-sig", "cp932", "shift_jis", "latin1"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                rows = list(csv.reader(f))
            break
        except Exception:
            pass

    if rows is None:
        raise RuntimeError(f"Could not decode {path}")

    # ---- ヘッダー行を自動スキップ ----
    # 「最左列が '1' で始まる＝1行目データと判定」
    start_idx = None
    for i, row in enumerate(rows):
        if len(row) > 0:
            try:
                # 数値1ならOK（文字列"1"でもfloat変換で通る）
                if float(row[0]) == 1:
                    start_idx = i
                    break
            except Exception:
                pass

    if start_idx is None:
        raise RuntimeError(f"Could not find data start (col0==1) in {path}")

    data_rows = rows[start_idx : start_idx + R]
    if len(data_rows) < R:
        raise RuntimeError(f"Not enough rows in Simp.csv: need {R}, got {len(data_rows)}")

    # ---- 区間重要度に変換 ----
    wL_list = []
    wR_list = []

    for row in data_rows:
        vals = [float(x) for x in row[1 : 1 + 2 * N]]  # idの次から2N値
        wL = [vals[2*j]     for j in range(N)]
        wR = [vals[2*j + 1] for j in range(N)]
        wL_list.append(wL)
        wR_list.append(wR)

    return wL_list, wR_list

def sort_indices_by_value(values, descending=True):
    """値リストのソート順 index 配列を返す（大きい順 / 小さい順）"""
    return sorted(range(len(values)), key=lambda i: values[i], reverse=descending)


def maximin_total_utility(u, wL, wR, perm):
    """
    C++ の maximin 関数を Python に移植。
    u: [Alt][N]
    wL, wR: [N]
    perm: [Alt][N]  (各代替案ごとの基準ソート順)
    戻り値: totalU[Alt], z[Alt][N], star[Alt]
    """
    totalU = [0.0] * Alt
    z = [[0.0] * N for _ in range(Alt)]
    star = [0] * Alt

    for k in range(Alt):
        cap = sum(wL)
        it = 0

        # Process2
        while it < N - 1 and cap + wR[perm[k][it]] - wL[perm[k][it]] <= 1.0 + 1e-12:
            j = perm[k][it]
            z[k][j] = wR[j]
            cap += wR[j] - wL[j]
            it += 1

        # Process3
        j = perm[k][it]
        z[k][j] = 1.0 - cap + wL[j]
        star[k] = j
        it += 1

        # Process4
        while it < N:
            j = perm[k][it]
            z[k][j] = wL[j]
            it += 1

        # total utility
        tu = 0.0
        for i in range(N):
            tu += u[k][i] * z[k][i]
        totalU[k] = tu

    return totalU, z, star


def compute_rank_from_maximin(u, wL, wR, morm):
    """
    ある 1 ケース (ある m, rp, 方法など) について、
    maximin / Maximax の総合効用と 1〜Alt 位の順位を返す。

    morm = 0 → maximin (C++ の BubSort_Ascend と同じ)
    morm = 1 → Maximax (C++ の BubSort_Descend と同じ)
    """
    # 各代替案ごとに u を昇順/降順ソートした perm を作る
    perm = [[0] * N for _ in range(Alt)]
    for j in range(Alt):
        # (値, index) のペアを並べ替え
        order = sort_indices_by_value(u[j], descending=(morm == 1))
        for i in range(N):
            perm[j][i] = order[i]

    totalU, _, _ = maximin_total_utility(u, wL, wR, perm)
    # 大きい順に並べた index が順位
    rank = sort_indices_by_value(totalU, descending=True)
    return totalU, rank


def top1_and_top2_match(true_rank, est_rank):
    """1位一致, Top2一致(順不同) を 0/1 で返す"""
    top1_true = true_rank[0]
    top1_est = est_rank[0]
    top1_ok = 1 if top1_true == top1_est else 0

    top2_true = set(true_rank[:2])
    top2_est = set(est_rank[:2])
    top2_ok = 1 if top2_true == top2_est else 0
    return top1_ok, top2_ok


def loss_amount(true_totalU, true_rank, est_choice):
    """
    損失量: 真の重みの下で「本当に最適な代替案」と
    「推定に基づいて選んだ代替案」の総合効用差。
    """
    best_true_idx = true_rank[0]
    return true_totalU[best_true_idx] - true_totalU[est_choice]


# ===== メイン評価ループ =====

def evaluate_all_methods(base_data_dir="../data", output_dir="../data/metrics_python"):
    base_data_dir = str(base_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []  # 後で CSV 用にまとめる

    for morm, max_or_min in enumerate(max_or_min_list):
        for utility_idx, which_u in enumerate(which_utility_list):
            # u[m][Alt][N]
            u_all = load_utility_matrix(base_data_dir, evaluation_num, utility_idx)

            for rt_idx in range(Rt):
                true_wL, true_wR = load_true_interval_weight(base_data_dir, evaluation_num, rt_idx)

                # 「真の」順位は s=t=1 の maximin で固定して計算（定義は必要に応じて調整可）
                # ここでは M 個の u それぞれについて true ranking を計算しておく
                true_totalU_list = []
                true_rank_list = []
                for m_idx in range(M):
                    u = u_all[m_idx]
                    totalU_true, rank_true = compute_rank_from_maximin(u, true_wL, true_wR, morm)
                    true_totalU_list.append(totalU_true)
                    true_rank_list.append(rank_true)

                for method_dir in ALL_METHOD_DIRS:
                    wL_list, wR_list = load_estimated_interval_weights(base_data_dir, evaluation_num, rt_idx, method_dir)
                    if wL_list is None:
                        # 該当ファイルがない手法はスキップ
                        continue

                    # 集計用
                    sum_top1 = 0
                    sum_top2 = 0
                    sum_loss = 0.0
                    count = 0

                    # C++ と同じイメージで m (効用) × rp (標本) を走査
                    for m_idx in range(M):
                        u = u_all[m_idx]
                        true_totalU = true_totalU_list[m_idx]
                        true_rank = true_rank_list[m_idx]

                        for rp_idx in range(R):
                            wL = wL_list[rp_idx]
                            wR = wR_list[rp_idx]

                            est_totalU, est_rank = compute_rank_from_maximin(u, wL, wR, morm)
                            top1_ok, top2_ok = top1_and_top2_match(true_rank, est_rank)
                            loss = loss_amount(true_totalU, true_rank, est_rank[0])

                            sum_top1 += top1_ok
                            sum_top2 += top2_ok
                            sum_loss += loss
                            count += 1

                    if count == 0:
                        continue

                    top1_rate = sum_top1 / count
                    top2_rate = sum_top2 / count
                    avg_loss = sum_loss / count

                    results.append({
                        "max_or_min": max_or_min,
                        "utility": which_u,
                        "eval_suffix": eval_num_suffix[evaluation_num],
                        "true_weight": true_weight_list[rt_idx],
                        "method": method_dir,
                        "top1_rate": top1_rate,
                        "top2_rate": top2_rate,
                        "avg_loss": avg_loss,
                        "samples": count,
                    })
                    print(f"[{max_or_min}][{which_u}][{true_weight_list[rt_idx]}][{method_dir}] "
                          f"Top1={top1_rate:.3f}, Top2={top2_rate:.3f}, Loss={avg_loss:.4f} (n={count})")

    # 全体まとめを1つの CSV に出力
    out_path = output_dir / "rank_metrics_summary.csv"
    with out_path.open("w", newline="", encoding="cp932") as f:
        writer = csv.writer(f)
        writer.writerow([
            "max_or_min", "utility", "eval_suffix", "true_weight", "method",
            "top1_rate", "top2_rate", "avg_loss", "samples"
        ])
        for r in results:
            writer.writerow([
                r["max_or_min"],
                r["utility"],
                r["eval_suffix"],
                r["true_weight"],
                r["method"],
                f"{r['top1_rate']:.6f}",
                f"{r['top2_rate']:.6f}",
                f"{r['avg_loss']:.10f}",
                r["samples"],
            ])

    print(f"\n==> 集計結果を書き出しました: {out_path}")


if __name__ == "__main__":
    # C++ と同じ ../data 配下を前提にして集計
    evaluate_all_methods(base_data_dir="/workspaces/inulab_julia_devcontainer/data", output_dir="/workspaces/inulab_julia_devcontainer/data/metrics_python")
