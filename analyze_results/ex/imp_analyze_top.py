import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# Paths and settings
# ======================
INPUT = "/workspaces/inulab_julia_devcontainer/results/metrics_python/rank_metrics_summary.csv"

OUT_TABLE_DIR = "/workspaces/inulab_julia_devcontainer/results/metrics_python/ppt_summary_tables"
OUT_FIG_DIR   = "/workspaces/inulab_julia_devcontainer/results/metrics_python/ppt_summary_figs"

os.makedirs(OUT_TABLE_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

EV_NAME = "EV"   # method name for the baseline

# We will create one figure per metric
METRICS = ["top1_rate", "top2_rate", "avg_loss"]
METRIC_LABEL = {
    "top1_rate": "Top1 accuracy",
    "top2_rate": "Top2 accuracy",
    "avg_loss":  "Average loss",
}

# We compare only these evaluation rules on the plots
MORMS = ["maximin", "Maximax"]
UTILS = ["u1", "u2"]


def select_top_methods(df: pd.DataFrame) -> list[str]:
    """
    Select EV + 4 best methods based on a combined score of:
    - top1_rate (higher is better)
    - top2_rate (higher is better)
    - avg_loss  (lower is better)

    The ranking is computed using data from:
    max_or_min in MORMS and utility in UTILS.
    """
    df_mm = df[df["max_or_min"].isin(MORMS)].copy()

    # Aggregate over utility and max_or_min
    agg = (
        df_mm.groupby("method", as_index=False)[["top1_rate", "top2_rate", "avg_loss"]]
             .mean()
    )

    if EV_NAME not in agg["method"].values:
        raise ValueError(f"EV method '{EV_NAME}' not found in 'method' column.")

    # Ranking: higher is better for top1/top2, lower is better for avg_loss
    agg["rank_top1"] = agg["top1_rate"].rank(ascending=False, method="min")
    agg["rank_top2"] = agg["top2_rate"].rank(ascending=False, method="min")
    agg["rank_loss"] = agg["avg_loss"].rank(ascending=True,  method="min")

    # Combined score (smaller is better)
    agg["score_total"] = agg["rank_top1"] + agg["rank_top2"] + agg["rank_loss"]

    # EV row (always included)
    ev_row = agg[agg["method"] == EV_NAME]

    # Top 4 non-EV methods by total score
    non_ev = (
        agg[agg["method"] != EV_NAME]
        .sort_values("score_total")
        .head(4)
    )

    top_methods = [EV_NAME] + list(non_ev["method"])
    print("Selected methods (EV + top 4):", top_methods)

    # Save the selection table (for checking)
    selection_table_path = os.path.join(OUT_TABLE_DIR, "selected_methods_score.csv")
    pd.concat([ev_row, non_ev], ignore_index=True).to_csv(selection_table_path, index=False)
    print("Saved selection table:", selection_table_path)

    return top_methods


def main():
    df = pd.read_csv(INPUT)

    # -----------------------------
    # 1. Decide EV + 4 best methods
    # -----------------------------
    top_methods = select_top_methods(df)

    # -----------------------------
    # 2. For each metric, create one figure:
    #    x-axis: method (EV + 4 methods)
    #    bars:  maximin/u1, Maximax/u1, maximin/u2, Maximax/u2
    # -----------------------------
    for metric in METRICS:
        label = METRIC_LABEL[metric]

        # Filter to relevant rows
        sub = df[
            df["method"].isin(top_methods)
            & df["max_or_min"].isin(MORMS)
            & df["utility"].isin(UTILS)
        ].copy()

        # Aggregate: method x max_or_min x utility
        g = (
            sub.groupby(["method", "max_or_min", "utility"], as_index=False)[metric]
               .mean()
        )

        # Pivot to a 4-column table:
        # columns = (maximin,u1), (Maximax,u1), (maximin,u2), (Maximax,u2)
        # We will build it explicitly to control the order.
        rows = []
        for m in top_methods:
            row = {"method": m}
            for util in UTILS:
                for morm in MORMS:
                    value = g[
                        (g["method"] == m)
                        & (g["max_or_min"] == morm)
                        & (g["utility"] == util)
                    ][metric]
                    if len(value) == 0:
                        row[f"{morm}_{util}"] = 0.0
                    else:
                        row[f"{morm}_{util}"] = float(value.iloc[0])
            rows.append(row)

        table = pd.DataFrame(rows)
        table_path = os.path.join(OUT_TABLE_DIR, f"summary_{metric}.csv")
        table.to_csv(table_path, index=False)
        print("Saved metric table:", table_path)

        # -------------------------
        # Plot
        # -------------------------
        x = np.arange(len(top_methods))
        width = 0.2  # width for each bar

        # Order of bars and colors:
        # blue  : maximin, u1
        # orange: Maximax, u1
        # green : maximin, u2
        # red   : Maximax, u2
        y_maximin_u1 = table["maximin_u1"].values
        y_maximax_u1 = table["Maximax_u1"].values
        y_maximin_u2 = table["maximin_u2"].values
        y_maximax_u2 = table["Maximax_u2"].values

        plt.figure()

        # BAR colors (color-blind friendly & grouped by utility)
        COLOR_u1_maximin  = "#D55E00"  # dark orange
        COLOR_u1_maximax  = "#E69F00"  # light orange
        COLOR_u2_maximin  = "#0072B2"  # dark blue
        COLOR_u2_maximax  = "#56B4E9"  # light blue

        plt.bar(x - 1.5 * width, y_maximin_u1, width,
                label="maximin, u1", color=COLOR_u1_maximin)
        plt.bar(x - 0.5 * width, y_maximax_u1, width,
                label="Maximax, u1", color=COLOR_u1_maximax)
        plt.bar(x + 0.5 * width, y_maximin_u2, width,
                label="maximin, u2", color=COLOR_u2_maximin)
        plt.bar(x + 1.5 * width, y_maximax_u2, width,
                label="Maximax, u2", color=COLOR_u2_maximax)


        plt.xticks(x, top_methods, rotation=20)
        plt.xlabel("Method")
        plt.ylabel(label)
        plt.title(f"{label} (EV + top 4 methods)")
        plt.legend(title="Rule / utility")
        plt.tight_layout()

        fig_path = os.path.join(OUT_FIG_DIR, f"{metric}_summary.png")
        plt.savefig(fig_path, dpi=400)
        plt.close()
        print("Saved figure:", fig_path)


if __name__ == "__main__":
    main()
