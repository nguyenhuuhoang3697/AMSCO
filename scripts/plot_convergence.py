import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_convergence(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {
        "dataset",
        "model",
        "optimizer",
        "raw_f1_history",
        "cumulative_time"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu các cột trong convergence_history.csv: {missing}")
    return df


def decode_json_array(value) -> list:
    if pd.isna(value):
        return []
    try:
        arr = json.loads(value)
        if isinstance(arr, list):
            cleaned = []
            for x in arr:
                try:
                    if x is None:
                        cleaned.append(None)
                    else:
                        cleaned.append(float(x))
                except (TypeError, ValueError):
                    cleaned.append(None)
            return cleaned
    except Exception:
        pass
    return []


def resample_curve(times, values, grid):
    if not times or not values:
        return None
    try:
        times_arr = np.array([float(t) for t in times], dtype=float)
    except Exception:
        return None
    values_arr = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            values_arr.append(np.nan)
        else:
            try:
                values_arr.append(float(v))
            except Exception:
                values_arr.append(np.nan)
    values_arr = np.array(values_arr, dtype=float)
    valid = np.isfinite(times_arr) & np.isfinite(values_arr)
    if not np.any(valid):
        return None
    times_arr = times_arr[valid]
    values_arr = values_arr[valid]
    if len(times_arr) == 0:
        return None
    if times_arr[0] > 0:
        times_arr = np.insert(times_arr, 0, 0.0)
        values_arr = np.insert(values_arr, 0, values_arr[0])
    dedup_times = [times_arr[0]]
    dedup_values = [values_arr[0]]
    for t, v in zip(times_arr[1:], values_arr[1:]):
        if abs(t - dedup_times[-1]) < 1e-9:
            dedup_values[-1] = v
        else:
            dedup_times.append(t)
            dedup_values.append(v)
    dedup_times = np.array(dedup_times, dtype=float)
    dedup_values = np.array(dedup_values, dtype=float)
    if len(dedup_times) < 2:
        dedup_times = np.append(dedup_times, dedup_times[-1] + 1e-6)
        dedup_values = np.append(dedup_values, dedup_values[-1])
    return np.interp(grid, dedup_times, dedup_values, left=dedup_values[0], right=dedup_values[-1])


def plot_convergence(df: pd.DataFrame, out_path: str = "figure_4_1_convergence.png"):
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    if n == 0:
        raise ValueError("Không có dataset nào trong convergence_history.csv")

    # Bố trí subplot 2x2 nếu có 4 dataset, nếu ít hơn vẫn dùng lưới này
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 7), sharex=False, sharey=False)
    axes = axes.flatten()

    color_map = {
        "Optuna (TPE)": "#1f77b4",  # xanh dương
        "AMSCO": "#2ca02c",        # xanh lá
    }

    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break
        ax = axes[idx]
        sub = df[df["dataset"] == dataset]

        max_time = 0.0
        curves_by_opt = {"Optuna (TPE)": [], "AMSCO": []}
        for opt in ["Optuna (TPE)", "AMSCO"]:
            opt_rows = sub[sub["optimizer"] == opt]
            for _, row in opt_rows.iterrows():
                times = decode_json_array(row.get("cumulative_time"))
                f1_hist = decode_json_array(row.get("raw_f1_history"))
                if not times or not f1_hist:
                    continue
                try:
                    last_time = float(times[-1])
                except Exception:
                    last_time = 0.0
                max_time = max(max_time, last_time)
                curves_by_opt[opt].append((times, f1_hist))

        if not curves_by_opt["Optuna (TPE)"] and not curves_by_opt["AMSCO"]:
            ax.set_title(dataset.title())
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        max_time = max(max_time, 1e-6)
        grid = np.linspace(0, max_time, num=200)
        mean_curves = {}
        for opt, runs in curves_by_opt.items():
            resampled = []
            for times, values in runs:
                curve_vals = resample_curve(times, values, grid)
                if curve_vals is not None:
                    resampled.append(curve_vals)
            if resampled:
                mean_curves[opt] = np.nanmean(np.vstack(resampled), axis=0)

        if not mean_curves:
            ax.set_title(dataset.title())
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        for opt, curve in mean_curves.items():
            label = "Optuna" if opt.startswith("Optuna") else opt
            ax.plot(grid, curve, label=label, color=color_map.get(opt, "gray"), linewidth=2)

        best_final = max(curve[-1] for curve in mean_curves.values())
        threshold = 0.95 * best_final
        ax.axhline(threshold, linestyle="--", color="red", alpha=0.4, label="95% of best F1")

        times_at_threshold = {}
        for opt, curve in mean_curves.items():
            idx = np.where(curve >= threshold)[0]
            if idx.size:
                times_at_threshold[opt] = grid[idx[0]]

        if all(opt in times_at_threshold for opt in ("Optuna (TPE)", "AMSCO")):
            t_optuna = times_at_threshold["Optuna (TPE)"]
            t_amsco = times_at_threshold["AMSCO"]
            if t_amsco > 0 and math.isfinite(t_optuna) and math.isfinite(t_amsco):
                speedup = t_optuna / t_amsco
                text = f"≈{speedup:.1f}× faster to 95% F1"
                x_pos = grid[-1] * 0.55 if grid[-1] > 0 else 0.0
                y_pos = threshold
                ax.annotate(
                    text,
                    xy=(x_pos, y_pos),
                    xytext=(x_pos, y_pos + 0.01),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                    fontsize=8,
                    color="gray"
                )

                optuna_note = f"Optuna: {t_optuna:.1f}s"
                amsco_note = f"AMSCO: {t_amsco:.1f}s"
                ax.text(0.02, 0.95, optuna_note, transform=ax.transAxes, fontsize=8, color=color_map["Optuna (TPE)"])
                ax.text(0.02, 0.88, amsco_note, transform=ax.transAxes, fontsize=8, color=color_map["AMSCO"])

        ax.set_title(dataset.replace("_", " ").title())
        ax.set_xlabel("Optimization Time (s)")
        ax.set_ylabel("Best F1 (CV)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Ẩn subplot thừa nếu số dataset < số ô
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Hình 4.1: Raw F1 vs. Optimization Time", fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Đã lưu {out_path}")


if __name__ == "__main__":
    csv_path = "convergence_history.csv"
    df_conv = load_convergence(csv_path)
    plot_convergence(df_conv)
