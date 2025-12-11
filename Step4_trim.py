import os
import pandas as pd
import numpy as np
from glob import glob

# --- CONFIG ---
INPUT_ROOT = "Exports/Raw_Data_Interpolated"
OUTPUT_ROOT = "Exports/Trimmed_Data"
DELIMITER = ";"
FPS = 200
WINDOW_SIZE = 60  # frames for rolling window
MOVEMENT_THRESHOLD = 3  # mm/frame


def compute_trimmed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Trim after the last active movement based on rolling displacement."""
    marker_cols = [col for col in df.columns if any(axis in col for axis in ["_X", "_Y", "_Z"])]
    marker_df = df[marker_cols]

    def frame_distance(row1, row2):
        return np.linalg.norm(row1 - row2)

    distances = [0.0]
    for i in range(1, len(marker_df)):
        distances.append(frame_distance(marker_df.iloc[i], marker_df.iloc[i - 1]))

    df["frame_distance"] = distances
    df["rolling_mean_movement"] = df["frame_distance"].rolling(WINDOW_SIZE).mean()

    threshold_crossed = df[df["rolling_mean_movement"] > MOVEMENT_THRESHOLD].index
    if len(threshold_crossed) > 0:
        last_active_idx = threshold_crossed[-1]
        df_trimmed = df.iloc[: last_active_idx + 1].copy()
        print(f"[ok] Truncated at frame {last_active_idx} (of {len(df)}).")
    else:
        df_trimmed = df.copy()
        print("[info] No truncation; kept full recording.")

    return df_trimmed.drop(columns=["frame_distance", "rolling_mean_movement"])


os.makedirs(OUTPUT_ROOT, exist_ok=True)
csv_files = glob(f"{INPUT_ROOT}/**/*.csv", recursive=True)

for path in csv_files:
    try:
        print(f"\n[info] Processing: {path}")
        df = pd.read_csv(path, delimiter=DELIMITER)
        trimmed_df = compute_trimmed_df(df)

        base_name = os.path.basename(path).replace(".csv", "_trimmed.csv")
        save_path = os.path.join(OUTPUT_ROOT, base_name)
        trimmed_df.to_csv(save_path, sep=DELIMITER, index=False)
        print(f"[ok] Saved: {save_path}")
    except Exception as e:
        print(f"[error] Error processing {path}: {e}")
