import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import defaultdict
import re

# --- CONFIG ---
AVERAGE_ROOT = "Exports/Averaged_Data_1M"
CLUSTER_OUTPUT = "Exports/Clustered"
DELIMITER = ";"
MARKERS = [1, 2, 3, 4, 5]
AXES = ["X", "Y", "Z"]

# --- Participant metadata ---
participant_meta = {
    6: {"Age": "18-20", "Gender": "Male"},
    7: {"Age": "21-25", "Gender": "Male"},
    8: {"Age": "21-25", "Gender": "Female"},
    9: {"Age": "21-25", "Gender": "Male"},
    10: {"Age": "21-25", "Gender": "Male"},
    11: {"Age": "25-35", "Gender": "Male"},
    12: {"Age": "18-20", "Gender": "Female"},
    13: {"Age": "21-25", "Gender": "Male"},
    14: {"Age": "21-25", "Gender": "Male"},
    15: {"Age": "21-25", "Gender": "Male"},
    16: {"Age": "21-25", "Gender": "Male"},
    17: {"Age": "25-35", "Gender": "Male"},
    18: {"Age": "25-35", "Gender": "Male"},
    19: {"Age": "25-35", "Gender": "Male"},
    20: {"Age": "21-25", "Gender": "Male"},
    21: {"Age": "25-35", "Gender": "Male"},
    22: {"Age": "35+", "Gender": "Female"},
    23: {"Age": "21-25", "Gender": "Male"},
    24: {"Age": "21-25", "Gender": "Male"},
    25: {"Age": "18-20", "Gender": "Male"},
    26: {"Age": "25-35", "Gender": "Female"},
    27: {"Age": "25-35", "Gender": "Female"},
    28: {"Age": "25-35", "Gender": "Male"},
    29: {"Age": "25-35", "Gender": "Male"},
    30: {"Age": "25-35", "Gender": "Female"},
    31: {"Age": "25-35", "Gender": "Male"},
    32: {"Age": "35+", "Gender": "Male"},
    33: {"Age": "35+", "Gender": "Female"},
    34: {"Age": "21-25", "Gender": "Female"},
    35: {"Age": "21-25", "Gender": "Male"},
    36: {"Age": "21-25", "Gender": "Female"},
    37: {"Age": "21-25", "Gender": "Male"},
}


def get_meta(participant_str: str):
    try:
        num_match = re.search(r"\d+", participant_str)
        if not num_match:
            return "Male", "21-25"
        num = int(num_match.group(0))
        meta = participant_meta.get(num, {"Age": "21-25", "Gender": "Male"})
        return meta["Gender"], meta["Age"]
    except Exception:
        return "Male", "21-25"


def resample_df(df, target_len):
    df_interp = {}
    for col in df.columns:
        if col == "Frame":
            continue
        y = df[col].astype(float).values
        x_old = np.linspace(0, 1, len(y))
        x_new = np.linspace(0, 1, target_len)
        if np.isnan(y).all():
            y_new = np.full(target_len, np.nan)
        else:
            y_new = np.interp(x_new, x_old, y)
        df_interp[col] = y_new
    return pd.DataFrame(df_interp)


def plot_derivatives_scalar_style(df, out_dir, markers=MARKERS, axes=AXES):
    frame = df["Frame"].values

    from scipy.ndimage import median_filter

    def compute_scalar(df, marker_id):
        vs, accs = [], []
        for axis in AXES:
            col = f"{marker_id}_{axis}"
            if col in df.columns:
                pos = df[col].astype(float).values
                v = np.gradient(pos)
                a = np.gradient(v)
                vs.append(v)
                accs.append(a)

        if len(vs) == 3:
            vel = np.sqrt(vs[0] ** 2 + vs[1] ** 2 + vs[2] ** 2)
            acc = np.sqrt(accs[0] ** 2 + accs[1] ** 2 + accs[2] ** 2)

            win_v = 11 if len(vel) >= 11 else len(vel) // 2 * 2 + 1
            vel_smooth = savgol_filter(vel, window_length=win_v, polyorder=3)

            z_v = (vel_smooth - np.nanmean(vel_smooth)) / np.nanstd(vel_smooth)
            vel_smooth[np.abs(z_v) > 2.5] = np.nan
            vel_smooth = pd.Series(vel_smooth).interpolate(limit_direction="both").values
            vel_smooth = median_filter(vel_smooth, size=7)

            win_a = 21 if len(acc) >= 21 else len(acc) // 2 * 2 + 1
            acc_smooth = savgol_filter(acc, window_length=win_a, polyorder=3)

            z_a = (acc_smooth - np.nanmean(acc_smooth)) / np.nanstd(acc_smooth)
            acc_smooth[np.abs(z_a) > 2.5] = np.nan
            acc_smooth = pd.Series(acc_smooth).interpolate(limit_direction="both").values
            acc_smooth = median_filter(acc_smooth, size=11)

            return vel_smooth, acc_smooth
        return None, None

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Magnitude of speed and acceleration", fontsize=14)

    for marker in markers:
        v, a = compute_scalar(df, marker)
        if v is not None and a is not None:
            ax_v.plot(frame, v, label=f"Marker {marker}")
            ax_a.plot(frame, a, label=f"Marker {marker}")

    ax_v.set_title("Speed (||v||)")
    ax_a.set_title("Acceleration (||a||)")
    ax_a.set_xlabel("Movement progression (%)")
    ax_v.set_ylabel("||v||")
    ax_a.set_ylabel("||a||")
    ax_v.legend()
    ax_a.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(os.path.join(out_dir, "velocity_acc_scalar.png"))
    plt.close()


def save_cluster_average(mode, cluster_key, experiment, dfs):
    if len(dfs) < 2:
        print(f"[warn] Not enough data for {mode}/{cluster_key} - {experiment}")
        return
    try:
        avg_len = int(round(np.mean([len(df) for df in dfs])))

        resampled = [resample_df(df.drop(columns="Frame"), avg_len) for df in dfs]
        stack = np.stack([df.values for df in resampled])
        mean = np.nanmean(stack, axis=0)

        frame = np.arange(avg_len)
        columns = dfs[0].columns[1:]
        out_df = pd.DataFrame(mean, columns=columns)
        out_df.insert(0, "Frame", frame)

        out_dir = os.path.join(CLUSTER_OUTPUT, mode, cluster_key, experiment)
        os.makedirs(out_dir, exist_ok=True)

        out_df.to_csv(os.path.join(out_dir, "mean.csv"), sep=DELIMITER, index=False)
        plot_derivatives_scalar_style(out_df, out_dir)

        print(f"[ok] Saved: {mode}/{cluster_key}/{experiment}")
    except Exception as e:
        print(f"[error] Error saving {mode}/{cluster_key} - {experiment}: {e}")


grouped_data = {
    "Gender": defaultdict(lambda: defaultdict(list)),
    "AgeWithinGender": defaultdict(lambda: defaultdict(list)),
    "AgeOnly": defaultdict(lambda: defaultdict(list)),
}

if not os.path.exists(AVERAGE_ROOT):
    print(f"[error] Directory not found: {AVERAGE_ROOT}")
    exit()

files = [f for f in os.listdir(AVERAGE_ROOT) if f.endswith("_mean.csv")]

for file in files:
    match = re.search(r"(participant|probant)\d+_([^_]+)_mean\.csv", file.lower())
    if not match:
        continue
    participant_str, experiment = match.groups()
    gender, age_group = get_meta(participant_str)

    path = os.path.join(AVERAGE_ROOT, file)
    df = pd.read_csv(path, delimiter=DELIMITER)

    grouped_data["Gender"][gender][experiment].append(df)
    grouped_data["AgeWithinGender"][f"{gender}_{age_group}"][experiment].append(df)
    grouped_data["AgeOnly"][age_group][experiment].append(df)

for mode, cluster_dict in grouped_data.items():
    for cluster_key, experiments in cluster_dict.items():
        for experiment, dfs in experiments.items():
            save_cluster_average(mode, cluster_key, experiment, dfs)
