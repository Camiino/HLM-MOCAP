import os
import csv

# --- CONFIG ---
input_root = "Exports/Raw_Data"
output_root = "Exports/Raw_Data_Formatted"

# --- Ensure output root exists ---
os.makedirs(output_root, exist_ok=True)


def clean_and_convert_file(input_path, output_path):
    """Normalize raw CSVs to semicolon-delimited rectangular files."""
    try:
        with open(input_path, "r", encoding="cp1252") as f:
            lines = [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError as e:
        print(f"[error] Encoding error in {input_path}: {e}")
        return

    try:
        traj_start = lines.index("TRAJECTORIES")
    except ValueError:
        print(f"[error] 'TRAJECTORIES' not found in: {input_path}")
        return

    field_lines = lines[traj_start + 1 :]
    split_lines = [line.split(",") for line in field_lines]
    max_fields = max(len(line) for line in split_lines)

    padded_lines = [line + [""] * (max_fields - len(line)) for line in split_lines]

    header_lines = lines[:traj_start]
    converted_lines = []

    for line in header_lines:
        if "," in line:
            parts = line.split(",")
            converted_lines.append(";".join(parts + [""] * (max_fields - len(parts))))
        else:
            converted_lines.append(line + ";" * (max_fields - 1))

    converted_lines.append("TRAJECTORIES" + ";" * (max_fields - 1))

    for line in padded_lines:
        converted_lines.append(";".join(line))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="cp1252", newline="") as f:
        for line in converted_lines:
            f.write(line + "\n")

    print(f"[ok] Converted: {input_path} -> {output_path}")


# --- STEP THROUGH ALL CSV FILES ---
for root, _, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".csv"):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)
            clean_and_convert_file(input_path, output_path)
