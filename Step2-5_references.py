import os
import pandas as pd

INPUT_ROOT = "Exports/Raw_Data_Clean"
REFERENCE_DIR = "Exports/Reference"
DELIMITER = ";"

# Experiment categories
EXPERIMENT_TYPES = ["weight", "grasp", "precision"]

MARKER4_COLS = [f"4_{a}" for a in ("X", "Y", "Z")]
MARKER5_COLS = [f"5_{a}" for a in ("X", "Y", "Z")]
REQUIRED_COLS = [f"{m}_{a}" for m in range(1, 6) for a in ("X", "Y", "Z")]

os.makedirs(REFERENCE_DIR, exist_ok=True)

best_refs = {
    exp: {
        "marker4": {"file": None, "valid_rows": 0, "df": None},
        "marker5": {"file": None, "valid_rows": 0, "df": None},
    }
    for exp in EXPERIMENT_TYPES
}

for root, _, files in os.walk(INPUT_ROOT):
    for file in files:
        if not file.lower().endswith(".csv"):
            continue

        path = os.path.join(root, file)

        for exp in EXPERIMENT_TYPES:
            if exp in file.lower() or exp in root.lower():
                try:
                    df = pd.read_csv(path, delimiter=DELIMITER, encoding="utf-8")

                    if not all(col in df.columns for col in REQUIRED_COLS):
                        continue

                    if all(col in df.columns for col in MARKER4_COLS):
                        valid_4 = df[MARKER4_COLS].dropna().shape[0]
                        if valid_4 > best_refs[exp]["marker4"]["valid_rows"]:
                            best_refs[exp]["marker4"] = {"file": file, "valid_rows": valid_4, "df": df}

                    if all(col in df.columns for col in MARKER5_COLS):
                        valid_5 = df[MARKER5_COLS].dropna().shape[0]
                        if valid_5 > best_refs[exp]["marker5"]["valid_rows"]:
                            best_refs[exp]["marker5"] = {"file": file, "valid_rows": valid_5, "df": df}

                except Exception as e:
                    print(f"[error] Failed to process {file}: {e}")

for exp in EXPERIMENT_TYPES:
    for marker in ["marker4", "marker5"]:
        info = best_refs[exp][marker]
        if info["df"] is not None:
            out_path = os.path.join(REFERENCE_DIR, f"{marker}_{exp}.csv")
            info["df"].to_csv(out_path, sep=DELIMITER, index=False, encoding="utf-8")
            print(f"[ok] Saved {marker}_{exp}.csv from {info['file']} ({info['valid_rows']} valid rows)")
        else:
            print(f"[warn] No suitable file found for {marker}_{exp}")
