import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────

INPUT_FILE  = r"ORIGINAL_FILES\icu with timestamp\icu_vitals_timeseries_messy.csv"          # ← update path
OUTPUT_FILE = r"CLEANED_FILES\icu with timestamp.xlsx"   # ← update path
SHEET_NAME  = 'Vital_Signs'
PLOTS_DIR   = r"CLEANED_FILES\visuals02"

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"  Original shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}\n")

print("Sample timestamp values:")
print(df["timestamp"].head(20).to_string())
print(f"\nUnique timestamp formats (sample):")
print(df["timestamp"].astype(str).str.extract(r'(\d{4}[-/]\d{2}|\d{2}[-/]\d{4}|[A-Za-z]+\s\d)')[0].value_counts())

# ─────────────────────────────────────────────
# STEP 0 — DROP ROWS WITH ANY MISSING VALUES
# ─────────────────────────────────────────────

before = len(df)
df.dropna(inplace=True)
removed = before - len(df)
print(f"Step 0 — Removed {removed} row(s) with missing values.")
print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 1 — DROP device_id COLUMN
# ─────────────────────────────────────────────

col_to_drop = "device_id"
if col_to_drop in df.columns:
    df.drop(columns=[col_to_drop], inplace=True)
    print(f"Step 1 — Dropped column: '{col_to_drop}'")
else:
    print(f"Step 1 — ⚠  Column '{col_to_drop}' not found, skipped.")
print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 2 — CONVERT TEMPERATURE TO CELSIUS
# ─────────────────────────────────────────────

print("Step 2 — Converting temperature to Celsius...")

if "temp_unit" not in df.columns:
    print("  ⚠  'temp_unit' column not found. Skipping conversion.")
elif "temperature" not in df.columns:
    print("  ⚠  'temperature' column not found. Skipping conversion.")
else:
    # Normalize temp_unit values (strip whitespace, uppercase)
    df["temp_unit"] = df["temp_unit"].astype(str).str.strip().str.upper()

    fahrenheit_mask = df["temp_unit"].isin(["F", "FAHRENHEIT"])
    n_converted = fahrenheit_mask.sum()

    # Convert: °C = (°F − 32) × 5/9
    df.loc[fahrenheit_mask, "temperature"] = (
        (df.loc[fahrenheit_mask, "temperature"] - 32) * 5 / 9
    ).round(2)

    print(f"  Converted {n_converted} Fahrenheit value(s) → Celsius.")

    # Drop the now-redundant temp_unit column
    df.drop(columns=["temp_unit"], inplace=True)
    print("  Dropped 'temp_unit' column (no longer needed).")

print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 3 — DROP DUPLICATE ROWS
# ─────────────────────────────────────────────

before = len(df)
df.drop_duplicates(inplace=True)
removed = before - len(df)
print(f"Step 3 — Removed {removed} duplicate row(s).")
print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print("Step 3b — Normalizing timestamp column...")

def parse_mixed_timestamps(series):
    result = pd.Series([pd.NaT] * len(series), index=series.index)
    
    for i, val in series.items():
        if pd.isna(val):
            continue
        val_str = str(val).strip()
        try:
            ts = pd.to_datetime(val_str)        # parse whatever format
            if ts.tzinfo is not None:
                ts = ts.tz_convert('UTC').tz_localize(None)   # aware → strip tz
            # naive timestamps assumed UTC, left as-is
            result[i] = ts
        except Exception:
            result[i] = pd.NaT                  # truly unparseable → NaT
    
    return result

df["timestamp"] = parse_mixed_timestamps(df["timestamp"])

nulls = df["timestamp"].isna().sum()
print(f"  Timestamp nulls after normalization: {nulls}")
print(f"  Sample parsed values:")
print(df["timestamp"].dropna().head(5).to_string())

# ─────────────────────────────────────────────
# STEP 4 — REMOVE OUTLIERS (medical ranges)
# ─────────────────────────────────────────────

# Medical reference ranges (uses current column names after rename above)
MEDICAL_RANGES = {
    "heart_rate":   (40,   200),    # bpm  — bradycardia floor / tachy ceiling
    "temperature":  (35.0, 41.5),   # °C   — hypothermia / hyperpyrexia
    "spo2":         (70,   100),    # %    — severe hypoxia floor
}

print("Step 4 — Removing outliers outside medical reference ranges...")
before = len(df)

for col, (lower, upper) in MEDICAL_RANGES.items():
    if col not in df.columns:
        print(f"  ⚠  Column '{col}' not found — skipped.")
        continue
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"  '{col}': removed {outliers} outlier(s)  [range: {lower} – {upper}]")

removed = before - len(df)
print(f"  Total outlier rows removed: {removed}")
print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 5 — ASSIGN RISK LEVEL
# ─────────────────────────────────────────────

print("Step 5 — Assigning Risk Level from vital sign thresholds...")

def classify_heart_rate(hr):
    if 60 <= hr <= 100:
        return 0   # Normal
    elif (50 <= hr < 60) or (100 < hr <= 120):
        return 1   # Abnormal
    else:
        return 2   # Critical  (< 50 or > 120)

def classify_spo2(spo2):
    if spo2 >= 95:
        return 0   # Normal
    elif 90 <= spo2 < 95:
        return 1   # Abnormal
    else:
        return 2   # Critical  (< 90)

def classify_temperature(temp):
    if 36.1 <= temp <= 37.2:
        return 0   # Normal
    elif (35.0 <= temp < 36.1) or (37.2 < temp <= 38.5):
        return 1   # Abnormal
    else:
        return 2   # Critical  (< 35 or > 38.5)

def assign_risk(row):
    scores = [
        classify_heart_rate(row["heart_rate"]),
        classify_spo2(row["spo2"]),
        classify_temperature(row["temperature"]),
    ]
    return max(scores)   # worst reading drives the final label

RISK_MAPPING = {0: "Normal", 1: "Abnormal", 2: "Critical"}
df["Risk_Level"] = df.apply(assign_risk, axis=1).map(RISK_MAPPING)

counts      = df["Risk_Level"].value_counts()
percentages = df["Risk_Level"].value_counts(normalize=True) * 100
print("  Risk Level distribution:")
for label in ["Normal", "Abnormal", "Critical"]:
    if label in counts:
        print(f"    {label:10s}: {counts[label]:>6}  ({percentages[label]:.2f}%)")
print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 6 — CLASS BALANCE CHECK
# ─────────────────────────────────────────────

print("── Class Balance Check ──────────────────────")
balance_df = pd.DataFrame({
    "Count":      counts,
    "Percentage": percentages.round(2)
})
print(balance_df.to_string())

max_pct = percentages.max()
min_pct = percentages.min()
ratio   = max_pct / min_pct

print(f"\n  Majority class : {percentages.idxmax()}  ({max_pct:.2f}%)")
print(f"  Minority class : {percentages.idxmin()}  ({min_pct:.2f}%)")
print(f"  Imbalance ratio: {ratio:.2f}x")

if ratio < 1.5:
    verdict = "✅ Balanced — no action needed."
elif ratio < 3.0:
    verdict = "⚠️  Mildly imbalanced — consider oversampling (SMOTE) or class weights."
else:
    verdict = "❌ Highly imbalanced — recommend SMOTE, undersampling, or class_weight='balanced'."
print(f"\n  Verdict: {verdict}\n")

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="Set2")

vital_signs     = ["heart_rate", "temperature", "spo2"]
existing_vitals = [c for c in vital_signs if c in df.columns]

# -- Plot 1: Risk Level Distribution
print("── Generating Visualizations ────────────────")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Risk Level Distribution", fontsize=16, fontweight='bold')

rl_counts = df["Risk_Level"].value_counts()

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

# -- Plot 2: Numerical Feature Distributions
numeric_cols = df.select_dtypes(include='number').columns.tolist()
fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(14, 5 * len(numeric_cols)))
fig.suptitle("Distribution of Numerical Features", fontsize=16, fontweight='bold')
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
plt.savefig(os.path.join(PLOTS_DIR, "02_numerical_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 02_numerical_distributions.png")

# -- Plot 3: Correlation Matrix
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"label": "Correlation Coefficient"})
    ax.set_title("Correlation Matrix — Numerical Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_correlation_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 03_correlation_matrix.png")

# -- Plot 4: Vital Signs Boxplot by Risk Level
fig, axes = plt.subplots(1, len(existing_vitals), figsize=(6 * len(existing_vitals), 6))
fig.suptitle("Vital Signs Distribution by Risk Level", fontsize=16, fontweight='bold')
if len(existing_vitals) == 1:
    axes = [axes]
order = sorted(df["Risk_Level"].unique())
for ax, col in zip(axes, existing_vitals):
    sns.boxplot(data=df, x="Risk_Level", y=col, ax=ax, palette="Set2", order=order)
    ax.set_title(f"{col} by Risk Level")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel(col)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_vitals_boxplot_by_risk.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 04_vitals_boxplot_by_risk.png")

# -- Plot 5: Vital Signs Violin by Risk Level
fig, axes = plt.subplots(1, len(existing_vitals), figsize=(6 * len(existing_vitals), 6))
fig.suptitle("Vital Signs Spread by Risk Level", fontsize=16, fontweight='bold')
if len(existing_vitals) == 1:
    axes = [axes]
for ax, col in zip(axes, existing_vitals):
    sns.violinplot(data=df, x="Risk_Level", y=col, ax=ax,
                   palette="Set2", order=order, inner="quartile")
    ax.set_title(f"{col} Distribution — Risk Level")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel(col)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_vitals_violin_by_risk.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 05_vitals_violin_by_risk.png")

# -- Plot 6: Mean Vital Signs Heatmap per Risk Level
mean_vitals = df.groupby("Risk_Level")[existing_vitals].mean().round(2)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(mean_vitals, annot=True, fmt=".2f", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Mean Value"})
ax.set_title("Mean Vital Signs per Risk Level", fontsize=14, fontweight='bold')
ax.set_xlabel("Vital Sign")
ax.set_ylabel("Risk Level")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_vitals_heatmap_mean_by_risk.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 06_vitals_heatmap_mean_by_risk.png")

print(f"\n✅ All plots saved to '{PLOTS_DIR}'")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_excel(OUTPUT_FILE, index=False, sheet_name=SHEET_NAME, engine='openpyxl')
print(f"\n✅ Cleaned dataset saved to '{OUTPUT_FILE}'")
print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Final columns: {list(df.columns)}")