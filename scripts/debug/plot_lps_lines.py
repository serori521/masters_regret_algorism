import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========= paths =========
BASE = Path("results/tmp")
LINE_CSV = BASE / "lps_debug_lines.csv"
OUTDIR = BASE / "fig_lps"
OUTDIR.mkdir(exist_ok=True)

# ========= load =========
df = pd.read_csv(LINE_CSV)

# t ごとに図を作る
for tchg, g in df.groupby("t"):
    fig, ax = plt.subplots(figsize=(7, 5))

    # t の近傍だけ描く（見やすさ）
    ts = np.linspace(tchg - 0.05, tchg + 0.05, 200)

    for _, row in g.iterrows():
        y = row["slope"] * ts + row["intercept"]

        if row["role"] == "qstar":
            ax.plot(ts, y, lw=2, label=f"p{int(row.p)}-q{int(row.q)} (q*)")
        else:
            ax.plot(ts, y, lw=1, ls="--", alpha=0.7,
                    label=f"p{int(row.p)}-q{int(row.q)} (hat)")

    # 縦線：LPS change
    ax.axvline(tchg, color="red", lw=2, label="LPS change")

    ax.set_title(f"LPS local lines around t = {tchg:.6f}")
    ax.set_xlabel("t")
    ax.set_ylabel("regret")
    ax.legend(fontsize=8)
    ax.grid(True)

    fname = OUTDIR / f"lps_lines_t{tchg:.6f}.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

    print(f"saved: {fname}")
