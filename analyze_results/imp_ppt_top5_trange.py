import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# Paths
# ======================
BASE = "/workspaces/inulab_julia_devcontainer"
INPUT = f"{BASE}/results/trange_tidy/trange_tidy_u1u2_all_methods.csv"

OUT_DIR = f"{BASE}/results/trange_tidy"
OUT_CSV = f"{OUT_DIR}/trange_stats_for_ppt.csv"
OUT_FIG = f"{OUT_DIR}/trange_mean_abs_mid_error.png"

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# Settings
# ======================

# Slide に載せたい5手法（Method列は "/EV" みたいに "/" 付きなので正規化する）
SELECTED_METHODS = ["EV", "eAMRw", "MMRW", "gAMRw"]
UTILITY_CASES = ["u1", "u2"]  # UtilityCase 列の値

def normalize_method(name: str) -> str:
    """Tidy CSV の Method 列 (例: '/EV') を 'EV' のように正規化する。"""
    s = name.strip()
    if s.startswith("/"):
        s = s[1:]
    return s

def main():
    # -------------
    # 1. 読み込み
    # -------------
    df = pd.read_csv(INPUT)

    # Method名を正規化（/EV → EV など）
    df["MethodShort"] = df["Method"].apply(normalize_method)

    # 対象 5 手法に絞る
    df = df[df["MethodShort"].isin(SELECTED_METHODS)]

    # 念のため UtilityCase も絞る（u1/u2のみ）
    df = df[df["UtilityCase"].isin(UTILITY_CASES)]

    # -------------
    # 2. 誤差列の追加
    # -------------
    # t* の位置の誤差
    df["mid_error"] = df["t_mid_est"] - df["s_mid_true"]
    df["abs_mid_error"] = df["mid_error"].abs()

    # t 区間幅の誤差
    df["width_error"] = df["t_width_est"] - df["s_width_true"]
    df["abs_width_error"] = df["width_error"].abs()

    # -------------
    # 3. UtilityCase × MethodShort ごとに統計量を集計
    # -------------
    group_cols = ["UtilityCase", "MethodShort"]

    agg = (
        df.groupby(group_cols)
          .agg(
              mean_t_mid_est=("t_mid_est", "mean"),
              std_t_mid_est=("t_mid_est", "std"),
              mean_t_width_est=("t_width_est", "mean"),
              std_t_width_est=("t_width_est", "std"),
              mean_abs_mid_error=("abs_mid_error", "mean"),
              mean_abs_width_error=("abs_width_error", "mean"),
              n_samples=("t_mid_est", "size"),
          )
          .reset_index()
    )

    # 保存
    agg.to_csv(OUT_CSV, index=False)
    print("Saved summary CSV:", OUT_CSV)

    # -------------
    # 4. スライド用の図 (mean_abs_mid_error)
    #    x軸: Method, 棒: u1 / u2 の mean_abs_mid_error
    # -------------
    methods = SELECTED_METHODS
    x = np.arange(len(methods))
    width = 0.35

    # u1 / u2 の値を method の順に並べる
    u1 = agg[agg["UtilityCase"] == "u1"].set_index("MethodShort")
    u2 = agg[agg["UtilityCase"] == "u2"].set_index("MethodShort")

    y1 = [u1.loc[m, "mean_abs_mid_error"] for m in methods]
    y2 = [u2.loc[m, "mean_abs_mid_error"] for m in methods]

    plt.figure()

    plt.bar(x - width/2, y1, width, label="u1", color="#E69F00")  # orange系
    plt.bar(x + width/2, y2, width, label="u2", color="#0072B2")  # blue系

    plt.xticks(x, methods, rotation=20)
    plt.ylabel("Mean abs error of t*")
    plt.xlabel("Method")
    plt.title("Location error of estimated t* (EV vs AMR variants)")
    plt.legend(title="Utility case")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(OUT_FIG, dpi=400)
    plt.close()
    print("Saved figure:", OUT_FIG)


if __name__ == "__main__":
    main()
