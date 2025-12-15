import os
import pandas as pd


# 入力と出力フォルダ
INPUT_CSV = "/workspaces/inulab_julia_devcontainer/data/metrics_python/rank_metrics_summary.csv"
OUTPUT_DIR = "/workspaces/inulab_julia_devcontainer/data/metrics_python/summary_for_ppt"


def summarize_for_powerpoint(
    input_csv: str = INPUT_CSV,s
    output_dir: str = OUTPUT_DIR,
    sort_metric: str = "top1_rate",  # "top1_rate" / "top2_rate" / "avg_loss" から選ぶ
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # 必要な列が揃っているかチェック
    required_columns = [
        "max_or_min",
        "utility",
        "eval_suffix",
        "true_weight",
        "method",
        "top1_rate",
        "top2_rate",
        "avg_loss",
        "samples",
    ]
    for c in required_columns:
        if c not in df.columns:
            raise ValueError(f"列 {c} が見つかりません。rank_metrics_summary.csv を確認してください。")

    # ---- ① true_weight(A〜E) で平均をとった「手法ごとの成績」 ----
    # グループ単位: max_or_min × utility × method
    grouped = (
        df.groupby(["max_or_min", "utility", "method"], as_index=False)[
            ["top1_rate", "top2_rate", "avg_loss"]
        ]
        .mean()
    )

    # ---- ② maximin/u1, maximin/u2, Maximax/u1, ... ごとにランキングを出力 ----
    for (morm, util), sub in grouped.groupby(["max_or_min", "utility"]):
        # sort_metric が avg_loss のときだけ昇順（小さい方が良い）
        ascending = True if sort_metric == "avg_loss" else False

        sub_sorted = sub.sort_values(by=sort_metric, ascending=ascending)

        # utility は "u1" / "u2" 形式なので、末尾の数字だけ使ってファイル名を作る
        util_id = util.replace("u", "")
        outpath = os.path.join(
            output_dir,
            f"{morm}_u{util_id}_method_ranking_by_{sort_metric}.csv",
        )
        sub_sorted.to_csv(outpath, index=False)
        print(f"保存しました: {outpath}")

    # ---- ③ 追加：ヒートマップ・表用のピボットも作っておく（任意） ----
    # 例：maximin/u1 だけ抜き出して「行: true_weight(A〜E), 列: method, 値: top1_rate」
    for metric in ["top1_rate", "top2_rate", "avg_loss"]:
        pivot = df.pivot_table(
            index=["max_or_min", "utility", "true_weight"],
            columns="method",
            values=metric,
            aggfunc="mean",
        )
        pivot_out = os.path.join(output_dir, f"pivot_{metric}.csv")
        pivot.to_csv(pivot_out)
        print(f"保存しました: {pivot_out}")


def main():
    summarize_for_powerpoint(
        input_csv=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        sort_metric="top1_rate",  # ここを変えれば他の指標順のランキングも作れる
    )


if __name__ == "__main__":
    main()
