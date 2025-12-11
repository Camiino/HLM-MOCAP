import os
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict
import re

# --- CONFIG ---
INPUT_ROOT = "Exports/Trimmed_Data"
OUTPUT_ROOT = "Exports/Averaged_Data_1M"
DELIMITER = ";"
AXES = ("X", "Y", "Z")
MARKERS = [1, 2, 3, 4, 5]
NORMALIZE_START = True
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

# For these experiments we only use marker #3
SINGLE_MARKER_EXPERIMENTS = {"precision", "grasp", "weight"}
SINGLE_MARKER_ID = 3


def normalize_start(df: pd.DataFrame) -> pd.DataFrame:
    """Shift all marker coordinates to start at zero."""
    for marker in MARKERS:
        for axis in AXES:
            col = f"{marker}_{axis}"
            if col in df.columns:
                df[col] -= df[col].iloc[0]
    return df


def resample_df(df: pd.DataFrame, target_len: int) -> pd.DataFrame:
    def safe_interp(col):
        col = col.astype(float)
        if col.isna().all():
            return np.full(target_len, np.nan)
        return np.interp(np.linspace(0, len(col) - 1, target_len), np.arange(len(col)), col)

    return df.apply(safe_interp)


def get_participant_and_experiment(filename: str):
    name = os.path.basename(filename).lower()
    match = re.search(r"(participant|probant)\d+", name)
    participant = match.group(0) if match else None
    for exp in sorted(EXPERIMENTS, key=lambda x: -len(x)):
        if f"_{exp}" in name or name.endswith(exp):
            return participant, exp
    return participant, None


files_by_group = defaultdict(list)
for path in glob(f"{INPUT_ROOT}/**/*.csv", recursive=True):
    participant, experiment = get_participant_and_experiment(path)
    if participant and experiment:
        files_by_group[(participant, experiment)].append(path)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for (participant, experiment), file_list in sorted(files_by_group.items()):
    print(f"\n[info] Processing {len(file_list)} files for {participant} - {experiment}")
    raw_dfs = []
    all_columns = set()
    original_lengths = []

    for file in file_list:
        try:
            df = pd.read_csv(file, delimiter=DELIMITER)
            frame_col = df.columns[0]
            df = df.drop(columns=frame_col)
            df = df.astype(float)

            if NORMALIZE_START:
                df = normalize_start(df)

            if experiment in SINGLE_MARKER_EXPERIMENTS:
                keep_cols = [f"{SINGLE_MARKER_ID}_{axis}" for axis in AXES]
                df = df[[c for c in df.columns if c in keep_cols]]

            all_columns.update(df.columns)
            raw_dfs.append(df)
            original_lengths.append(len(df))
        except Exception as e:
            print(f"[warn] Skipped file {file} due to error: {e}")

    if len(raw_dfs) < 2:
        print(f"[error] Not enough valid files for averaging: {participant} - {experiment}")
        continue

    avg_frame_count = int(round(np.mean(original_lengths)))
    print(f"[info] Resampling all to average frame count: {avg_frame_count}")

    all_columns = sorted(all_columns)
    aligned_resampled = []

    for df in raw_dfs:
        aligned = df.reindex(columns=all_columns, fill_value=np.nan)
        resampled = resample_df(aligned, avg_frame_count)
        aligned_resampled.append(resampled)

    try:
        data_stack = np.stack([df.values for df in aligned_resampled])
    except Exception as e:
        print(f"[error] Error stacking data for {participant}-{experiment}: {e}")
        continue

    avg_data = np.nanmean(data_stack, axis=0)
    std_data = np.nanstd(data_stack, axis=0)

    frame_index = np.arange(avg_frame_count)
    avg_df = pd.DataFrame(avg_data, columns=all_columns)
    std_df = pd.DataFrame(std_data, columns=all_columns)
    avg_df.insert(0, "Frame", frame_index)
    std_df.insert(0, "Frame", frame_index)

    base = f"{participant}_{experiment}"
    avg_path = os.path.join(OUTPUT_ROOT, f"{base}_mean.csv")
    std_path = os.path.join(OUTPUT_ROOT, f"{base}_std.csv")

    avg_df.to_csv(avg_path, sep=DELIMITER, index=False)
    std_df.to_csv(std_path, sep=DELIMITER, index=False)
    print(f"[ok] Saved: {avg_path}, {std_path}")
