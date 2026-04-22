import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

INPUT_FILE   = r"CLEANED_MERGED_FILE\vital_signs_merged.xlsx"
INPUT_SHEET  = 'Vital_Signs_Merged'
OUTPUT_DIR   = r"TRAIN_TEST_VAL\splits"

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15

RANDOM_STATE = 42
TARGET_COL   = "risk_level"       # 0=Normal, 1=Abnormal, 2=Critical
FEATURE_COLS = ["heart_rate", "temperature", "spo2"]

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print("Loading merged dataset...")
df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# STEP 1 — VERIFY REQUIRED COLUMNS
# ─────────────────────────────────────────────

print("Step 1 — Verifying required columns...")
missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
print(f"  ✅ All required columns present: {FEATURE_COLS + [TARGET_COL]}\n")

# ─────────────────────────────────────────────
# STEP 2 — DROP ROWS WITH NaN IN REQUIRED COLS
# ─────────────────────────────────────────────

before = len(df)
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
removed = before - len(df)
print(f"Step 2 — Dropped {removed} row(s) with NaN in required columns.")
print(f"  Shape after: {df.shape[0]} rows\n")

# ─────────────────────────────────────────────
# STEP 3 — CLASS DISTRIBUTION BEFORE SPLIT
# ─────────────────────────────────────────────

print("── Class Distribution (full dataset) ────────")
label_map = {0: "Normal", 1: "Abnormal", 2: "Critical"}
counts = df[TARGET_COL].value_counts().sort_index()
for val, count in counts.items():
    label = label_map.get(val, str(val))
    print(f"  {val} ({label:8s}): {count:>7}  ({count/len(df)*100:.2f}%)")
print()

# ─────────────────────────────────────────────
# STEP 4 — SPLIT
# stratify=TARGET_COL ensures each split has the
# same class proportions as the full dataset
# ─────────────────────────────────────────────

print("Step 4 — Splitting dataset...")

# First split: train vs (val + test)
df_train, df_temp = train_test_split(
    df,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_STATE,
    stratify=df[TARGET_COL]
)

# Second split: val vs test (equal halves of temp)
df_val, df_test = train_test_split(
    df_temp,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_STATE,
    stratify=df_temp[TARGET_COL]
)

total = len(df)
print(f"  Train : {len(df_train):>7} rows  ({len(df_train)/total*100:.1f}%)")
print(f"  Val   : {len(df_val):>7} rows  ({len(df_val)/total*100:.1f}%)")
print(f"  Test  : {len(df_test):>7} rows  ({len(df_test)/total*100:.1f}%)\n")

# ─────────────────────────────────────────────
# STEP 5 — VERIFY CLASS BALANCE ACROSS SPLITS
# ─────────────────────────────────────────────

print("── Class Distribution per Split ─────────────")
for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    counts = split_df[TARGET_COL].value_counts().sort_index()
    parts  = []
    for val, count in counts.items():
        label = label_map.get(val, str(val))
        parts.append(f"{val}({label})={count} ({count/len(split_df)*100:.1f}%)")
    print(f"  {name:5s}: {' | '.join(parts)}")
print()

# ─────────────────────────────────────────────
# STEP 6 — SAVE SPLITS
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = {
    "train": df_train,
    "val":   df_val,
    "test":  df_test,
}

print("Step 6 — Saving splits...")
for split_name, split_df in splits.items():
    # Save as Excel
    xlsx_path = os.path.join(OUTPUT_DIR, f"vital_signs_{split_name}.xlsx")
    split_df.to_excel(xlsx_path, index=False, sheet_name=f"Vital_Signs_{split_name.capitalize()}", engine='openpyxl')
    print(f"  ✅ Saved: vital_signs_{split_name}.xlsx  ({len(split_df)} rows)")

print(f"\n✅ All splits saved to '{OUTPUT_DIR}'")
print(f"\n── Summary ──────────────────────────────────")
print(f"  Total rows : {total}")
print(f"  Train      : {len(df_train)} rows  (70%)")
print(f"  Validate   : {len(df_val)} rows  (15%)")
print(f"  Test       : {len(df_test)} rows  (15%)")
print(f"  Features   : {FEATURE_COLS}")
print(f"  Target     : '{TARGET_COL}'  (0=Normal, 1=Abnormal, 2=Critical)")
