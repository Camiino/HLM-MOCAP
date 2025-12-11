import os
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict

# --- CONFIG ---
INPUT_DIR = "Exports/Averaged_Data_1M"
OUTPUT_DIR = "Exports/Final_Averages_1M"
DELIMITER = ";"
EXPERIMENTS = [
    "circle",
    "ptp",
    "ptp2",
    "ptp3",
    "zigzag",
    "sequential",
    "precision",
    "grasp",
    "weight",
]


def resample_df(df, target_len):
    def safe_interp(col):
        col = col.astype(float)
        if col.isna().all():
            return np.full(target_len, np.nan)
        return np.interp(np.linspace(0, len(col) - 1, target_len), np.arange(len(col)), col)

    return df.apply(safe_interp)


files_by_experiment = defaultdict(list)
for path in glob(f"{INPUT_DIR}/*_mean.csv"):
    basename = os.path.basename(path).lower().replace("_mean.csv", "")
    for exp in sorted(EXPERIMENTS, key=lambda x: -len(x)):
        if basename.endswith(f"_{exp}"):
            files_by_experiment[exp].append(path)
            break

os.makedirs(OUTPUT_DIR, exist_ok=True)

for experiment, file_list in files_by_experiment.items():
    print(f"\n[info] Averaging {len(file_list)} mean files for experiment: {experiment}")

    dfs = []
    all_columns = set()
    lengths = []

    for path in file_list:
        try:
            df = pd.read_csv(path, delimiter=DELIMITER)
            df = df.drop(columns=["Frame"], errors="ignore")
            df = df.astype(float)
            lengths.append(len(df))
            all_columns.update(df.columns)
            dfs.append(df)
        except Exception as e:
            print(f"[warn] Skipped {path}: {e}")

    if len(dfs) < 2:
        print(f"[error] Not enough valid mean files for: {experiment}")
        continue

    avg_frame_count = int(round(np.mean(lengths)))
    print(f"[info] Using average frame count: {avg_frame_count}")

    all_columns = sorted(all_columns)
    resampled_dfs = []

    for df in dfs:
        aligned = df.reindex(columns=all_columns, fill_value=np.nan)
        resampled = resample_df(aligned, avg_frame_count)
        resampled_dfs.append(resampled)

    try:
        data_stack = np.stack([df.values for df in resampled_dfs])
    except Exception as e:
        print(f"[error] Error stacking data: {e}")
        continue

    avg = np.nanmean(data_stack, axis=0)
    frame_column = np.linspace(0, 100, avg_frame_count)  # percent scale

    final_df = pd.DataFrame(avg, columns=all_columns)
    final_df.insert(0, "Frame", frame_column)

    out_path = os.path.join(OUTPUT_DIR, f"{experiment}_final_mean.csv")
    final_df.to_csv(out_path, sep=DELIMITER, index=False)
    print(f"[ok] Saved: {out_path}")
