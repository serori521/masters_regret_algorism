# batch_make_best_ts_plots_fixed.py

from pathlib import Path
import csv
import matplotlib.pyplot as plt

# ★ここを固定値にする
INDIR  = Path("/workspaces/inulab_julia_devcontainer/data/metrics_julia/tsresults")
OUTDIR = Path("/workspaces/inulab_julia_devcontainer/data/metrics_julia/tsresults/plots")
GLOB_PATTERN = "*_summary.csv"
SKIP_EXISTING = True


def _get(row, keys):
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    raise KeyError(f"Missing any of columns: {keys}")


def read_summary(path: Path):
    t=[]; r_mean=[]; pr0=[]; pr1=[]; ts_med=[]; q25=[]; q75=[]
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path.name}: empty or invalid CSV")

        for row in reader:
            t.append(float(_get(row, ["t"])))
            r_mean.append(float(_get(row, ["r_mean"])))
            pr0.append(float(_get(row, ["Pr0","r0"])))
            pr1.append(float(_get(row, ["Pr1","r1"])))
            ts_med.append(float(_get(row, ["ts_q50","ts_median"])))
            q25.append(float(_get(row, ["ts_q25","q25"])))
            q75.append(float(_get(row, ["ts_q75","q75"])))

    order = sorted(range(len(t)), key=lambda i: t[i])
    def reord(a): return [a[i] for i in order]
    return dict(t=reord(t), r_mean=reord(r_mean), pr0=reord(pr0), pr1=reord(pr1),
                ts_med=reord(ts_med), q25=reord(q25), q75=reord(q75))


def plot_r_boundary(data, outpath: Path, title_suffix: str):
    plt.figure(figsize=(8, 6))
    plt.plot(data["t"], data["r_mean"], label="mean(r)")
    plt.plot(data["t"], data["pr0"], label="P(r=0)")
    plt.plot(data["t"], data["pr1"], label="P(r=1)")
    plt.title(f"Where the optimizer lands in ts-range (r)\n{title_suffix}")
    plt.xlabel("t"); plt.ylabel("r statistics")
    plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()


def plot_tsstar_iqr(data, outpath: Path, title_suffix: str):
    plt.figure(figsize=(8, 6))
    plt.plot(data["t"], data["ts_med"], label="median(ts_star)")
    plt.fill_between(data["t"], data["q25"], data["q75"], alpha=0.2, label="IQR (25-75%)")
    plt.title(f"Best ts* vs t (median and IQR across pcm_id)\n{title_suffix}")
    plt.xlabel("t (true scale grid)"); plt.ylabel("ts_star (best estimated scale)")
    plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(INDIR.glob(GLOB_PATTERN))
    if not csv_paths:
        raise SystemExit(f"No CSV matched: {INDIR}/{GLOB_PATTERN}")

    ok, ng = 0, 0
    for csv_path in csv_paths:
        stem = csv_path.stem
        out_r  = OUTDIR / f"{stem}_r_boundary_vs_t.png"
        out_ts = OUTDIR / f"{stem}_tsstar_vs_t.png"

        if SKIP_EXISTING and out_r.exists() and out_ts.exists():
            print(f"[skip] {csv_path.name}")
            continue

        try:
            data = read_summary(csv_path)
            plot_r_boundary(data, out_r, stem)
            plot_tsstar_iqr(data, out_ts, stem)
            print(f"[ok] {csv_path.name}")
            ok += 1
        except Exception as e:
            print(f"[ng] {csv_path.name}: {e}")
            ng += 1

    print(f"done: ok={ok}, ng={ng}, outdir={OUTDIR}")


if __name__ == "__main__":
    main()
