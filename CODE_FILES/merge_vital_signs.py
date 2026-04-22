import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────

# Paths to the cleaned datasets
FILE_A = r"CLEANED_FILES\human_vital_signs_cleaned_datasetv03.xlsx"   # original dataset
FILE_B = r"CLEANED_FILES\icu with timestamp.xlsx"                    # ICU dataset
FILE_C = r"ORIGINAL_FILES\critical_dataset\critical_vital_signs.csv"  # critical cases dataset

SHEET_A = 'Human_Vital_Signs'
SHEET_B = 'Vital_Signs'
# FILE_C is a CSV — no sheet name needed

OUTPUT_FILE  = r"CLEANED_MERGED_FILE\vital_signs_merged.xlsx"
OUTPUT_SHEET = 'Vital_Signs_Merged'
PLOTS_DIR    = r"CLEANED_MERGED_FILE\visuals_merged"

# ─────────────────────────────────────────────
# COLUMN RENAME MAP — standardize both datasets
# to lowercase_with_underscores before merging
# ─────────────────────────────────────────────

RENAME_A = {
    "patient_ID":  "patient_id",
    "Timestamp":   "timestamp",       # ← add this line
    "Heart_Rate":  "heart_rate",
    "Temperature": "temperature",
    "SpO2":        "spo2",
    "Risk_Level":  "risk_level",
}
RENAME_B = {
    "Risk_Level": "risk_level",
}
RENAME_C = {
    "Risk_Level": "risk_level",
}

REQUIRED_COLS = ["patient_id", "heart_rate", "temperature", "spo2", "risk_level"]

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print("Loading cleaned datasets...")

df_a = pd.read_excel(FILE_A, sheet_name=SHEET_A)
df_b = pd.read_excel(FILE_B, sheet_name=SHEET_B)
df_c = pd.read_csv(FILE_C)
# Parse timestamp immediately so it's a proper datetime object,
# not a plain string — this ensures Excel renders it correctly
df_c["timestamp"] = pd.to_datetime(df_c["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

print(f"  Dataset A — shape: {df_a.shape[0]} rows × {df_a.shape[1]} columns")
print(f"             columns: {list(df_a.columns)}")
print(f"  Dataset B — shape: {df_b.shape[0]} rows × {df_b.shape[1]} columns")
print(f"             columns: {list(df_b.columns)}")
print(f"  Dataset C — shape: {df_c.shape[0]} rows × {df_c.shape[1]} columns")
print(f"             columns: {list(df_c.columns)}\n")



# ─────────────────────────────────────────────
# STEP 1 — STANDARDIZE COLUMN NAMES
# ─────────────────────────────────────────────

print("Step 1 — Standardizing column names to lowercase_with_underscores...")

df_a.rename(columns=RENAME_A, inplace=True)
df_b.rename(columns=RENAME_B, inplace=True)
df_c.rename(columns=RENAME_C, inplace=True)

df_a.columns = [c.lower() for c in df_a.columns]
df_b.columns = [c.lower() for c in df_b.columns]
df_c.columns = [c.lower() for c in df_c.columns]

print(f"  Dataset A columns after rename: {list(df_a.columns)}")
print(f"  Dataset B columns after rename: {list(df_b.columns)}")
print(f"  Dataset C columns after rename: {list(df_c.columns)}\n")

# ─────────────────────────────────────────────
# STEP 2 — TAG SOURCE
# ─────────────────────────────────────────────

df_a["source"] = "dataset_A"
df_b["source"] = "dataset_B"
df_c["source"] = "dataset_C"

# ─────────────────────────────────────────────
# STEP 3 — STACK
# ─────────────────────────────────────────────

print("Step 2 — Stacking datasets (A + B + C)...")
df = pd.concat([df_a, df_b, df_c], ignore_index=True, sort=False)
print(f"  Shape after stack: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 4 — DEDUPLICATE
# ─────────────────────────────────────────────

print("Step 3 — Normalizing timestamps & Deduplicating...")
before = len(df)

ts_col = "timestamp" if "timestamp" in df.columns else None

# Robust multi-format timestamp parser — handles all 3 formats in the data:
#   "2026-03-25T14:19:49Z"       ISO 8601 UTC (Z suffix)
#   "2026-03-25 21:35:28+05:00"  timezone-aware with offset
#   "2026-03-23 02:25:52"        naive datetime (assumed UTC)
# utc=True alone silently coerces naive datetimes to NaT — this avoids that.
def parse_mixed_timestamps(series):
    result = pd.Series([pd.NaT] * len(series), index=series.index, dtype="datetime64[ns]")
    for i, val in series.items():
        if pd.isna(val):
            continue
        try:
            ts = pd.to_datetime(str(val).strip())
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            result[i] = ts
        except Exception:
            result[i] = pd.NaT
    return result

if ts_col:
    print(f"  Parsing mixed-format timestamps in '{ts_col}'...")
    df[ts_col] = parse_mixed_timestamps(df[ts_col])
    nulls_after = df[ts_col].isna().sum()
    print(f"  Timestamp NaT count after normalization: {nulls_after}")
    has_ts  = df[df[ts_col].notna()].copy()
    no_ts   = df[df[ts_col].isna()].copy()
    has_ts_deduped = has_ts.drop_duplicates(subset=["patient_id", ts_col], keep="first")
    proxy_cols = [c for c in ["patient_id", "heart_rate", "temperature", "spo2"] if c in df.columns]
    no_ts_deduped = no_ts.drop_duplicates(subset=proxy_cols, keep="first")
    df = pd.concat([has_ts_deduped, no_ts_deduped], ignore_index=True)
else:
    proxy_cols = [c for c in ["patient_id", "heart_rate", "temperature", "spo2"] if c in df.columns]
    df = df.drop_duplicates(subset=proxy_cols, keep="first")

removed = before - len(df)
print(f"  Removed {removed} duplicate row(s).")
print(f"  Shape after dedup: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 5 — COLUMN ALIGNMENT REPORT
# ─────────────────────────────────────────────

print("── Column Alignment Report ──────────────────")
all_cols  = set(df_a.columns) | set(df_b.columns) | set(df_c.columns) - {"source"}
in_all    = (set(df_a.columns) & set(df_b.columns) & set(df_c.columns)) - {"source"}
only_in_a = set(df_a.columns) - set(df_b.columns) - set(df_c.columns) - {"source"}
only_in_b = set(df_b.columns) - set(df_a.columns) - set(df_c.columns) - {"source"}
only_in_c = set(df_c.columns) - set(df_a.columns) - set(df_b.columns) - {"source"}

print(f"  Shared by all three : {sorted(in_all)}")
print(f"  Only in Dataset A   : {sorted(only_in_a)}  → NaN for B & C rows")
print(f"  Only in Dataset B   : {sorted(only_in_b)}  → NaN for A & C rows")
print(f"  Only in Dataset C   : {sorted(only_in_c)}  → NaN for A & B rows\n")

# ─────────────────────────────────────────────
# STEP 6 — MISSING VALUE CHECK
# ─────────────────────────────────────────────

print("── Post-Merge Missing Value Check ──────────")
missing = df[REQUIRED_COLS].isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("  ✅ No missing values in required columns.\n")
else:
    print("  ⚠  Missing values found in required columns:")
    for col, n in missing.items():
        print(f"    '{col}': {n} missing ({n/len(df)*100:.2f}%)")
    print()

# ─────────────────────────────────────────────
# STEP 7 — RISK LEVEL DISTRIBUTION
# ─────────────────────────────────────────────

print("── Risk Level Distribution (merged) ────────")
if "risk_level" in df.columns:
    counts      = df["risk_level"].value_counts()
    percentages = df["risk_level"].value_counts(normalize=True) * 100
    for label in ["Normal", "Abnormal", "Critical"]:
        if label in counts:
            print(f"  {label:10s}: {counts[label]:>6}  ({percentages[label]:.2f}%)")
    ratio = percentages.max() / percentages.min()
    print(f"\n  Imbalance ratio: {ratio:.2f}x")
    if ratio < 1.5:
        verdict = "✅ Balanced."
    elif ratio < 3.0:
        verdict = "⚠️  Mildly imbalanced — consider SMOTE or class weights."
    else:
        verdict = "❌ Highly imbalanced — recommend SMOTE, undersampling, or class_weight='balanced'."
    print(f"  Verdict: {verdict}\n")

# ─────────────────────────────────────────────
# STEP 8 — SOURCE BREAKDOWN
# ─────────────────────────────────────────────

print("── Source Breakdown ─────────────────────────")
src_counts = df["source"].value_counts()
for src, n in src_counts.items():
    print(f"  {src}: {n} rows ({n/len(df)*100:.1f}%)")
print()

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="Set2")

vital_signs     = ["heart_rate", "temperature", "spo2"]
existing_vitals = [c for c in vital_signs if c in df.columns]
risk_order      = [r for r in ["Normal", "Abnormal", "Critical"] if r in df["risk_level"].unique()]

print("\n── Generating Visualizations ────────────────")

# ── Plot 01: Risk Level Distribution (bar + pie) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Merged Dataset — Risk Level Distribution", fontsize=16, fontweight='bold')

rl_counts = df["risk_level"].value_counts()
rl_counts.plot(kind='bar', ax=axes[0],
               color=sns.color_palette("Set2", len(rl_counts)), edgecolor='black')
axes[0].set_title("Count per Risk Level")
axes[0].set_xlabel("Risk Level")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)
for bar in axes[0].patches:
    axes[0].annotate(f'{int(bar.get_height())}',
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10)

axes[1].pie(rl_counts, labels=rl_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("Set2", len(rl_counts)),
            startangle=140, wedgeprops={'edgecolor': 'black'})
axes[1].set_title("Proportion per Risk Level")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_risk_level_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 01_risk_level_distribution.png")

# ── Plot 02: Source Breakdown (bar + pie) ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Row Contribution by Source Dataset", fontsize=16, fontweight='bold')

src_counts.plot(kind='bar', ax=axes[0],
                color=sns.color_palette("Set2", len(src_counts)), edgecolor='black')
axes[0].set_title("Row Count per Source")
axes[0].set_xlabel("Source")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=0)
for bar in axes[0].patches:
    axes[0].annotate(f'{int(bar.get_height())}',
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10)

axes[1].pie(src_counts, labels=src_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("Set2", len(src_counts)),
            startangle=140, wedgeprops={'edgecolor': 'black'})
axes[1].set_title("Proportion per Source")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_source_breakdown.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 02_source_breakdown.png")

# ── Plot 03: Risk Level by Source (grouped + stacked bar) ─────────────────────
cross     = pd.crosstab(df["source"], df["risk_level"])
cross     = cross[[c for c in risk_order if c in cross.columns]]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Risk Level by Source Dataset", fontsize=16, fontweight='bold')

cross.plot(kind='bar', ax=axes[0],
           color=sns.color_palette("Set2", 3), edgecolor='black')
axes[0].set_title("Absolute Count")
axes[0].set_xlabel("Source")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title="Risk Level")

cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
cross_pct.plot(kind='bar', stacked=True, ax=axes[1],
               color=sns.color_palette("Set2", 3), edgecolor='black')
axes[1].set_title("Percentage (stacked)")
axes[1].set_xlabel("Source")
axes[1].set_ylabel("Percentage (%)")
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title="Risk Level")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_risk_by_source.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 03_risk_by_source.png")

# ── Plot 04: Numerical Feature Distributions ──────────────────────────────────
numeric_cols = df[existing_vitals].select_dtypes(include='number').columns.tolist()

fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(14, 5 * len(numeric_cols)))
fig.suptitle("Distribution of Numerical Features (Merged)", fontsize=16, fontweight='bold')
if len(numeric_cols) == 1:
    axes = [axes]

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i][0],
                 color=sns.color_palette("Set2")[i % 8], edgecolor='black')
    axes[i][0].set_title(f"{col} — Histogram & KDE")
    axes[i][0].set_xlabel(col)
    axes[i][0].set_ylabel("Count")

    sns.boxplot(x=df[col], ax=axes[i][1], color=sns.color_palette("Set2")[i % 8])
    axes[i][1].set_title(f"{col} — Boxplot")
    axes[i][1].set_xlabel(col)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_numerical_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 04_numerical_distributions.png")

# ── Plot 05: Vital Signs KDE — Dataset A vs B ────────────────────────────────
fig, axes = plt.subplots(1, len(existing_vitals), figsize=(6 * len(existing_vitals), 5))
fig.suptitle("Vital Signs Distribution — Dataset A vs B vs C (KDE)", fontsize=16, fontweight='bold')
if len(existing_vitals) == 1:
    axes = [axes]

palette = {"dataset_A": "steelblue", "dataset_B": "darkorange", "dataset_C": "crimson"}
for ax, col in zip(axes, existing_vitals):
    for src, color in palette.items():
        subset = df[df["source"] == src][col].dropna()
        sns.kdeplot(subset, ax=ax, label=src, fill=True, color=color, alpha=0.4)
    ax.set_title(f"{col} — A vs B vs C")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_vitals_kde_by_source.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 05_vitals_kde_by_source.png")

# ── Plot 06: Vital Signs Boxplot by Risk Level ────────────────────────────────
fig, axes = plt.subplots(1, len(existing_vitals), figsize=(6 * len(existing_vitals), 6))
fig.suptitle("Vital Signs Distribution by Risk Level (Merged)", fontsize=16, fontweight='bold')
if len(existing_vitals) == 1:
    axes = [axes]

for ax, col in zip(axes, existing_vitals):
    sns.boxplot(data=df, x="risk_level", y=col, ax=ax,
                palette="Set2", order=risk_order)
    ax.set_title(f"{col} by Risk Level")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel(col)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_vitals_boxplot_by_risk.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 06_vitals_boxplot_by_risk.png")

# ── Plot 07: Vital Signs Violin by Risk Level ─────────────────────────────────
fig, axes = plt.subplots(1, len(existing_vitals), figsize=(6 * len(existing_vitals), 6))
fig.suptitle("Vital Signs Spread by Risk Level (Merged)", fontsize=16, fontweight='bold')
if len(existing_vitals) == 1:
    axes = [axes]

for ax, col in zip(axes, existing_vitals):
    sns.violinplot(data=df, x="risk_level", y=col, ax=ax,
                   palette="Set2", order=risk_order, inner="quartile")
    ax.set_title(f"{col} — Risk Level")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel(col)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_vitals_violin_by_risk.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 07_vitals_violin_by_risk.png")

# ── Plot 08: Correlation Matrix ───────────────────────────────────────────────
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"label": "Correlation Coefficient"})
    ax.set_title("Correlation Matrix — Merged Dataset", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_correlation_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 08_correlation_matrix.png")

# ── Plot 09: Mean Vital Signs Heatmap per Risk Level ─────────────────────────
mean_vitals = df.groupby("risk_level")[existing_vitals].mean().round(2)
mean_vitals = mean_vitals.reindex([r for r in risk_order if r in mean_vitals.index])

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(mean_vitals, annot=True, fmt=".2f", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Mean Value"})
ax.set_title("Mean Vital Signs per Risk Level (Merged)", fontsize=14, fontweight='bold')
ax.set_xlabel("Vital Sign")
ax.set_ylabel("Risk Level")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_vitals_heatmap_mean_by_risk.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 09_vitals_heatmap_mean_by_risk.png")

# ── Plot 10: Vital Signs Boxplot — Dataset A vs B ────────────────────────────
fig, axes = plt.subplots(1, len(existing_vitals), figsize=(6 * len(existing_vitals), 6))
fig.suptitle("Vital Signs Comparison — Dataset A vs B vs C", fontsize=16, fontweight='bold')
if len(existing_vitals) == 1:
    axes = [axes]

for ax, col in zip(axes, existing_vitals):
    sns.boxplot(data=df, x="source", y=col, ax=ax,
                palette=["steelblue", "darkorange", "crimson"],
                order=["dataset_A", "dataset_B", "dataset_C"])
    ax.set_title(f"{col} — A vs B vs C")
    ax.set_xlabel("Source")
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "10_vitals_boxplot_by_source.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 10_vitals_boxplot_by_source.png")

print(f"\n✅ All plots saved to '{PLOTS_DIR}'")

# ─────────────────────────────────────────────
# STEP — ENCODE RISK LEVEL TO NUMERIC
# ─────────────────────────────────────────────

print("── Encoding Risk Level to Numeric ───────────")
RISK_ENCODING = {"Normal": 0, "Abnormal": 1, "Critical": 2}
df["risk_level"] = df["risk_level"].map(RISK_ENCODING)

unmapped = df["risk_level"].isna().sum()
if unmapped == 0:
    print("  ✅ All values encoded successfully.")
    print("     Normal → 0  |  Abnormal → 1  |  Critical → 2")
else:
    print(f"  ⚠  {unmapped} row(s) could not be mapped — check for unexpected risk_level values.")

print(f"  Encoded value counts:")
for val, label in sorted({0: "Normal", 1: "Abnormal", 2: "Critical"}.items()):
    count = (df["risk_level"] == val).sum()
    print(f"    {val} ({label:8s}): {count:>6}")
print()

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_excel(OUTPUT_FILE, index=False, sheet_name=OUTPUT_SHEET, engine='openpyxl')
print(f"\n✅ Merged dataset saved to '{OUTPUT_FILE}'")
print(f"   Final shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Final columns: {list(df.columns)}")