import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("results/tmp")
LINE_CSV = BASE / "lps_debug_lines.csv"
OUTDIR = BASE / "fig_lps_full"
OUTDIR.mkdir(exist_ok=True)

# ========= load csv without pandas =========
rows = []
with open(LINE_CSV, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        # columns: t,p,q,slope,intercept,role
        rows.append({
            "t": float(r["t"]),
            "p": int(r["p"]),
            "q": int(r["q"]),
            "slope": float(r["slope"]),
            "intercept": float(r["intercept"]),
            "role": r["role"],
        })

if not rows:
    raise RuntimeError("lps_debug_lines.csv is empty")

# t values (change points)
t_list = sorted(set(r["t"] for r in rows), reverse=True)

# infer tL, tR from observed changes (you can hardcode if you prefer)
tR = max(t_list)
tL = min(t_list)

# make a slightly wider range for nicer edges
pad = 0.01 * (tR - tL)
x_min = tL - pad
x_max = tR + pad

# group by p
Ps = sorted(set(r["p"] for r in rows))

# For each p: plot all qstar/hat lines that appeared at any change t
# Option 1: plot each unique (p,q,role) as a single line over full range
for p in Ps:
    fig, ax = plt.subplots(figsize=(10, 5))

    # collect unique lines for this p
    uniq = {}
    for r in rows:
        if r["p"] != p:
            continue
        key = (r["q"], r["role"])
        # keep the first occurrence (same (p,q,role) should be identical most of the time)
        if key not in uniq:
            uniq[key] = (r["slope"], r["intercept"])

    xs = np.linspace(x_min, x_max, 400)

    for (q, role), (A, B) in uniq.items():
        ys = A * xs + B
        if role == "qstar":
            ax.plot(xs, ys, lw=2, label=f"q* line q={q}")
        else:
            ax.plot(xs, ys, lw=1, ls="--", alpha=0.7, label=f"hat line q={q}")

    # vertical lines at each LPS change
    for tchg in t_list:
        ax.axvline(tchg, lw=1, alpha=0.3)

    ax.set_title(f"Full-range LPS lines for p={p}  (t in [{x_min:.3f},{x_max:.3f}])")
    ax.set_xlabel("t")
    ax.set_ylabel("regret")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)

    out = OUTDIR / f"full_p{p}.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("saved:", out)

# Option 2 (bonus): a single figure overlaying all p (can get crowded)
fig, ax = plt.subplots(figsize=(12, 6))
xs = np.linspace(x_min, x_max, 400)

uniq_all = {}
for r in rows:
    key = (r["p"], r["q"], r["role"])
    if key not in uniq_all:
        uniq_all[key] = (r["slope"], r["intercept"])

for (p, q, role), (A, B) in uniq_all.items():
    ys = A * xs + B
    if role == "qstar":
        ax.plot(xs, ys, lw=1.5, label=f"p{p}-q{q} q*")
    else:
        ax.plot(xs, ys, lw=1, ls="--", alpha=0.6, label=f"p{p}-q{q} hat")

for tchg in t_list:
    ax.axvline(tchg, lw=1, alpha=0.2)

ax.set_title("Full-range LPS lines (all p) + vertical LPS change markers")
ax.set_xlabel("t")
ax.set_ylabel("regret")
ax.grid(True)

# legend can be huge; comment out if too cluttered
# ax.legend(fontsize=6, ncol=4)

out = OUTDIR / "full_all_p.png"
plt.tight_layout()
plt.savefig(out)
plt.close()
print("saved:", out)
