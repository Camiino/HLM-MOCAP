import os
import pandas as pd
import numpy as np
import re

# --- CONFIG ---
INPUT_ROOT = "Exports/Raw_Data_Formatted"
OUTPUT_ROOT = "Exports/Raw_Data_Clean"


def determine_base_markers(path: str) -> list[int]:
    """Choose which marker IDs must be present based on filename."""
    lower = path.lower()
    if any(k in lower for k in ["weight", "grasp", "precision"]):
        return [1, 2, 3, 4, 5]
    if any(k in lower for k in ["ptp", "ptp2", "ptp3", "sequential", "zickzack", "circle", "zigzag"]):
        return [1]
    print(f"[warn] Unknown experiment group for '{path}', defaulting to [1]")
    return [1]


def parse_val(val):
    """Parse numbers that may use German thousand separators."""
    try:
        val = str(val).replace(" ", "")
        parts = val.split(".")
        if len(parts) > 2:
            val = "".join(parts[:-1]) + "." + parts[-1]
        return float(val)
    except Exception:
        return np.nan


def clean_csv(input_path: str, output_path: str):
    base_markers = determine_base_markers(input_path)

    try:
        df_raw = pd.read_csv(input_path, sep=";", header=None, dtype=str, encoding="cp1252")
    except Exception as e:
        print(f"[error] Failed to read {input_path}: {e}")
        return

    try:
        marker_hdr_idx = df_raw.apply(lambda r: r.str.contains(r"\*1", regex=True, na=False).any(), axis=1).idxmax()
        coord_hdr_idx = marker_hdr_idx + 1
    except Exception as e:
        print(f"[error] Could not find headers in {input_path}: {e}")
        return

    marker_row = df_raw.iloc[marker_hdr_idx].fillna("")
    coord_row = df_raw.iloc[coord_hdr_idx].fillna("")

    new_cols = []
    current_marker = None
    for col_val, coord in zip(marker_row, coord_row):
        col_val = col_val.strip()
        coord = coord.strip()

        if coord.lower().startswith("field"):
            new_cols.append("Frame")
            continue

        m = re.match(r"\*(\d+)", col_val)
        if m:
            current_marker = m.group(1)

        if current_marker and coord in ("X", "Y", "Z"):
            new_cols.append(f"{current_marker}_{coord}")
        else:
            new_cols.append(f"unk_{len(new_cols)}")

    df = df_raw.iloc[coord_hdr_idx + 1 :].copy()
    df.columns = new_cols

    for col in df.columns:
        if col != "Frame":
            df[col] = df[col].apply(parse_val)

    all_markers = sorted({int(c.split("_")[0]) for c in df.columns if c != "Frame"})
    extra_markers = [m for m in all_markers if m not in base_markers]

    for row_idx, row in df.iterrows():
        for extra_id in extra_markers:
            ex, ey, ez = f"{extra_id}_X", f"{extra_id}_Y", f"{extra_id}_Z"
            if pd.notna(row.get(ex)) and pd.notna(row.get(ey)) and pd.notna(row.get(ez)):
                for base_id in base_markers:
                    bx, by, bz = f"{base_id}_X", f"{base_id}_Y", f"{base_id}_Z"
                    if (
                        bx in df.columns
                        and by in df.columns
                        and bz in df.columns
                        and pd.isna(row.get(bx))
                        and pd.isna(row.get(by))
                        and pd.isna(row.get(bz))
                    ):
                        df.loc[row_idx, [bx, by, bz]] = [row[ex], row[ey], row[ez]]
                        break

    cols_to_drop = [f"{e}_{axis}" for e in extra_markers for axis in ("X", "Y", "Z") if f"{e}_{axis}" in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        df.to_csv(output_path, sep=";", index=False, encoding="cp1252")
        print(f"[ok] Cleaned: {input_path} -> {output_path}")
    except Exception as e:
        print(f"[error] Failed to save {output_path}: {e}")


for root, _, files in os.walk(INPUT_ROOT):
    for file in files:
        if file.lower().endswith(".csv"):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, INPUT_ROOT)
            output_path = os.path.join(OUTPUT_ROOT, rel_path)
            clean_csv(input_path, output_path)
