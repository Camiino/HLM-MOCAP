#!/usr/bin/env python3
"""
Step 10: Pattern Reference Analysis
Compare averaged trajectories against predefined pattern references (circle, zigzag),
aggregate nRMSE metrics, and produce summary tables and visuals.
"""

import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
EXPORTS_ROOT = "Exports"
SEARCH_ROOTS = [os.path.join(EXPORTS_ROOT, "Clustered")]

PATTERN_CIRCLE = os.path.join("PatternReferences", "Pattern_Circle.csv")
PATTERN_ZIGZAG = os.path.join("PatternReferences", "Pattern_Zigzag.csv")

N_RESAMPLE = 200  # samples per path for comparison
DELIM = ";"

# Output files
SUMMARY_OUT = os.path.join(EXPORTS_ROOT, "PatternDeviation_Summary.csv")
OUT_SIMPLE = os.path.join(EXPORTS_ROOT, "PatternDeviation_PercentTable.csv")
OUT_DETAILED = os.path.join(EXPORTS_ROOT, "PatternDeviation_PercentTable_detailed.csv")
OUT_TOTAL_TXT = os.path.join(EXPORTS_ROOT, "PatternDeviation_Totals.txt")
OUT_HEATMAP = os.path.join(EXPORTS_ROOT, "PatternDeviation_Heatmap_nRMSE.png")
OUT_BOXPLOT = os.path.join(EXPORTS_ROOT, "PatternDeviation_Boxplot_byTask.png")
OUT_BAR_COMBINED = os.path.join(EXPORTS_ROOT, "PatternDeviation_Bar_Combined.png")


def load_path_xy(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CSV with columns: Frame;1_X;1_Y;1_Z (semicolon separated)
    Returns (t, x, y) as float arrays. If columns missing, raises.
    """
    df = pd.read_csv(csv_path, sep=DELIM)
    if not {"Frame", "1_X", "1_Y"}.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} missing required columns Frame;1_X;1_Y")
    t = df["Frame"].astype(float).to_numpy()
    x = df["1_X"].astype(float).to_numpy()
    y = df["1_Y"].astype(float).to_numpy()
    return t, x, y


def resample_arclength(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Resample 2D path by arc length to n points. Returns shape (n, 2)."""
    xy = np.column_stack([x, y]).astype(float)
    if len(xy) == 0:
        return np.zeros((n, 2))
    mask = np.all(np.isfinite(xy), axis=1)
    xy = xy[mask]
    if len(xy) == 0:
        return np.zeros((n, 2))

    diff = np.diff(xy, axis=0)
    seg = np.sqrt((diff**2).sum(axis=1))
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = seg > 0
    xy = xy[keep]
    if len(xy) == 1:
        return np.repeat(xy, n, axis=0)

    d = np.sqrt(((np.diff(xy, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] <= 0:
        return np.repeat(xy[:1], n, axis=0)
    u = s / s[-1]

    ut = np.linspace(0.0, 1.0, n)
    xt = np.interp(ut, u, xy[:, 0])
    yt = np.interp(ut, u, xy[:, 1])
    return np.column_stack([xt, yt])


def orthogonal_procrustes(target: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Align source to target via similarity transform (rotation + uniform scale + translation).
    Returns (aligned_source, scale, rotation_matrix, translation).
    Enforces proper rotation (determinant +1), i.e., no reflection.
    """
    X = target.astype(float)
    Y = source.astype(float)
    if X.shape != Y.shape:
        raise ValueError("Shapes must match for Procrustes alignment")

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0 = X - muX
    Y0 = Y - muY

    normY = np.linalg.norm(Y0)
    if normY == 0:
        R = np.eye(2)
        s = 1.0
        t = muX - muY
        Y_aligned = (Y0 @ R.T) * s + muX
        return Y_aligned, s, R, t

    C = Y0.T @ X0
    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    s = (np.trace((Y0 @ R).T @ X0)) / (np.linalg.norm(Y0) ** 2 + 1e-12)
    t = muX - s * (muY @ R)
    Y_aligned = (s * (Y @ R)) + t
    return Y_aligned, s, R, t


def bbox_diag(xy: np.ndarray) -> float:
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    return float(np.hypot(xmax - xmin, ymax - ymin))


def compute_metrics(ref: np.ndarray, obs: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAX, Hausdorff and normalized percentages."""
    if ref.shape != obs.shape:
        raise ValueError("ref and obs must have same shape for metrics")
    diff = ref - obs
    dist = np.sqrt((diff**2).sum(axis=1))
    rmse = float(np.sqrt(np.mean(dist**2)))
    mae = float(np.mean(np.abs(dist)))
    dmax = float(np.max(dist))

    def _hausdorff(A, B):
        from scipy.spatial.distance import cdist

        D = cdist(A, B)
        return max(float(D.min(axis=1).max()), float(D.min(axis=0).max()))

    h = _hausdorff(ref, obs)
    dref = bbox_diag(ref)
    dref = dref if dref > 0 else 1.0
    return {
        "rmse": rmse,
        "mae": mae,
        "max": dmax,
        "hausdorff": h,
        "nrmse_percent": 100.0 * rmse / dref,
        "nmae_percent": 100.0 * mae / dref,
        "hausdorff_percent": 100.0 * h / dref,
        "ref_bbox_diag": dref,
    }


def plot_overlay(ref: np.ndarray, obs: np.ndarray, out_path: str, title: str = ""):
    plt.figure(figsize=(5.5, 5.5), dpi=120)
    plt.plot(ref[:, 0], ref[:, 1], label="Pattern reference", color="#1f77b4", linewidth=2)
    plt.plot(obs[:, 0], obs[:, 1], label="Mean (aligned)", color="#d62728", linewidth=2, alpha=0.8)
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def analyze_one(mean_csv: str, pattern_csv: str) -> dict:
    _, mx, my = load_path_xy(mean_csv)
    _, rx, ry = load_path_xy(pattern_csv)
    M = resample_arclength(mx, my, N_RESAMPLE)
    R = resample_arclength(rx, ry, N_RESAMPLE)
    M_aligned, s, Rmat, t = orthogonal_procrustes(R, M)
    metrics = compute_metrics(R, M_aligned)
    metrics.update(
        {
            "scale": float(s),
            "rot_deg": float(np.degrees(np.arctan2(Rmat[1, 0], Rmat[0, 0]))),
            "tx": float(t[0]),
            "ty": float(t[1]),
            "n_points": int(N_RESAMPLE),
        }
    )
    try:
        head = os.path.dirname(mean_csv)
        task = infer_task_from_path(mean_csv)
        out_plot = os.path.join(head, "pattern_compare.png")
        title = f"{task} | deviation: {metrics['nrmse_percent']:.1f}% (nRMSE)"
        plot_overlay(R, M_aligned, out_plot, title)
    except Exception as e:
        print(f"[warn] plotting failed for {mean_csv}: {e}")
    return metrics


def infer_task_from_path(path: str) -> str:
    p = str(path).lower()
    if re.search(r"[\\/]circle[\\/]", p):
        return "circle"
    if re.search(r"[\\/]zigzag[\\/]", p):
        return "zigzag"
    return "unknown"


def find_mean_files(search_roots: list[str]) -> list[str]:
    paths = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for fp in glob(os.path.join(root, "**", "mean.csv"), recursive=True):
            if re.search(r"[\\/](circle|zigzag)[\\/]", fp, flags=re.IGNORECASE):
                paths.append(fp)
    paths.sort()
    return paths


def _norm_path(p: str) -> list[str]:
    return re.split(r"[\\/]+", p.strip())


def cluster_from_relfile(relpath: str, task: str) -> str:
    parts = _norm_path(relpath)
    if len(parts) < 4:
        return relpath.replace(os.sep, "/")
    try:
        i = parts.index("Clustered")
    except ValueError:
        i = -1
    if i >= 0:
        parts = parts[i + 1 :]
    if parts and parts[-1].lower() == "mean.csv":
        parts = parts[:-1]
    if parts and parts[-1].lower() in ("circle", "zigzag"):
        parts = parts[:-1]
    cluster = "/".join(parts)
    return cluster


def run_analysis() -> pd.DataFrame:
    """Run the trajectory analysis and return the results DataFrame."""
    print("[Step10] Comparing averaged trajectories to pattern references")
    files = find_mean_files(SEARCH_ROOTS)
    if not files:
        print("No mean.csv files found under search roots.")
        return pd.DataFrame()
    print(f"Found {len(files)} mean.csv files")

    if not os.path.isfile(PATTERN_CIRCLE) or not os.path.isfile(PATTERN_ZIGZAG):
        print("Pattern reference CSVs not found. Expected at:"
              f"\n  {PATTERN_CIRCLE}\n  {PATTERN_ZIGZAG}")
        return pd.DataFrame()

    rows = []
    for fp in files:
        task = infer_task_from_path(fp)
        pattern = PATTERN_CIRCLE if task == "circle" else PATTERN_ZIGZAG if task == "zigzag" else None
        if pattern is None:
            print(f"[skip] Unknown task for {fp}")
            continue
        try:
            m = analyze_one(fp, pattern)
            rel = os.path.relpath(fp, EXPORTS_ROOT)
            row = {"file": rel, "task": task, **m}
            rows.append(row)
            print(f"  -> {rel}: nRMSE = {m['nrmse_percent']:.2f}% | Hausdorff = {m['hausdorff_percent']:.2f}%")
        except Exception as e:
            print(f"[error] Failed {fp}: {e}")

    if not rows:
        print("No results to save.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(SUMMARY_OUT)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(SUMMARY_OUT, index=False)
    print(f"Saved summary to {SUMMARY_OUT}")
    return df


def aggregate_results(df: pd.DataFrame):
    """Create aggregated tables from the analysis results."""
    if df.empty:
        print("No data to aggregate.")
        return

    print("[Step10] Creating aggregated tables")

    if "file" not in df.columns or "task" not in df.columns or "nrmse_percent" not in df.columns:
        print("DataFrame missing required columns: file, task, nrmse_percent")
        return

    df["cluster"] = [cluster_from_relfile(rel, task) for rel, task in zip(df["file"], df["task"])]

    pivot = df.pivot_table(index="cluster", columns="task", values="nrmse_percent", aggfunc="mean")
    if "circle" not in pivot.columns:
        pivot["circle"] = np.nan
    if "zigzag" not in pivot.columns:
        pivot["zigzag"] = np.nan
    pivot = pivot[[c for c in ["circle", "zigzag"] if c in pivot.columns]]
    pivot = pivot.rename(columns={"circle": "circle_nrmse_percent", "zigzag": "zigzag_nrmse_percent"})
    pivot["combined_mean_nrmse_percent"] = pivot.mean(axis=1, skipna=True)

    out_dir = os.path.dirname(OUT_DETAILED)
    os.makedirs(out_dir, exist_ok=True)
    pivot.reset_index().to_csv(OUT_DETAILED, index=False)

    simple = pivot[["combined_mean_nrmse_percent"]].rename(columns={"combined_mean_nrmse_percent": "deviation_percent"})
    simple = simple.reset_index()

    total_over_entries = float(df["nrmse_percent"].mean())
    total_over_clusters = float(simple["deviation_percent"].mean())

    simple_total = pd.DataFrame(
        [
            {"cluster": "TOTAL (over clusters)", "deviation_percent": total_over_clusters},
            {"cluster": "TOTAL (over all entries)", "deviation_percent": total_over_entries},
        ]
    )
    simple_with_total = pd.concat([simple, simple_total], ignore_index=True)
    simple_with_total.to_csv(OUT_SIMPLE, index=False)

    with open(OUT_TOTAL_TXT, "w", encoding="utf-8") as f:
        f.write(f"TOTAL over clusters (mean of cluster means): {total_over_clusters:.3f}%\n")
        f.write(f"TOTAL over all entries (mean across circle+zigzag): {total_over_entries:.3f}%\n")

    print(f"Saved simple table to {OUT_SIMPLE}")
    print(f"Saved detailed table to {OUT_DETAILED}")
    print(f"Saved totals to {OUT_TOTAL_TXT}")

    try:
        _plot_heatmap(pivot, OUT_HEATMAP)
        print(f"Saved heatmap to {OUT_HEATMAP}")
    except Exception as e:
        print(f"[warn] Heatmap failed: {e}")
    try:
        _plot_boxplot_by_task(df, OUT_BOXPLOT)
        print(f"Saved boxplot to {OUT_BOXPLOT}")
    except Exception as e:
        print(f"[warn] Boxplot failed: {e}")
    try:
        _plot_bar_combined(simple, OUT_BAR_COMBINED)
        print(f"Saved combined bar chart to {OUT_BAR_COMBINED}")
    except Exception as e:
        print(f"[warn] Bar chart failed: {e}")

    print("\n=== SUMMARY ===")
    print(f"Total deviation over clusters: {total_over_clusters:.3f}%")
    print(f"Total deviation over all entries: {total_over_entries:.3f}%")


def _shorten_labels(labels: list[str], maxlen: int = 30) -> list[str]:
    out = []
    for s in labels:
        s2 = str(s)
        if len(s2) > maxlen:
            s2 = s2[: maxlen - 1] + "…"
        out.append(s2)
    return out


def _plot_heatmap(pivot: pd.DataFrame, out_path: str):
    if pivot.empty:
        return
    cols = [c for c in ["circle_nrmse_percent", "zigzag_nrmse_percent"] if c in pivot.columns]
    if not cols:
        return
    M = pivot[cols].to_numpy()
    clusters = list(pivot.index)
    clusters_s = _shorten_labels(clusters, maxlen=32)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(clusters))))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_yticklabels(clusters_s)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("_nrmse_percent", "") for c in cols])
    ax.set_xlabel("Task")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("nRMSE [%]")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_boxplot_by_task(df: pd.DataFrame, out_path: str):
    d = df.dropna(subset=["task", "nrmse_percent"]).copy()
    if d.empty:
        return
    groups = ["circle", "zigzag"]
    data = [d.loc[d["task"].str.lower() == g, "nrmse_percent"].values for g in groups]
    labels = [g.capitalize() for g in groups]
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("nRMSE [%]")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_bar_combined(simple_df: pd.DataFrame, out_path: str):
    d = simple_df.copy()
    d = d[~d["cluster"].str.startswith("TOTAL")] if "cluster" in d.columns else d
    if d.empty:
        return
    d = d.sort_values("deviation_percent", ascending=True)
    labels = _shorten_labels(d["cluster"].tolist(), maxlen=38)
    values = d["deviation_percent"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#1f77b4", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Combined deviation (nRMSE %) for circle & zigzag")
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("=== Step 10: Pattern Reference Analysis ===")
    print("Part 1: Trajectory Analysis")
    results_df = run_analysis()

    print("\nPart 2: Aggregation")
    aggregate_results(results_df)

    print("\n=== Step 10 Complete ===")


if __name__ == "__main__":
    main()
