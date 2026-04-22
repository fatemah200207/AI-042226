import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────

INPUT_FILE  = r"ORIGINAL_FILES\human vital signs\human_vital_signs_dataset_2024.csv"
OUTPUT_FILE = r"CLEANED_FILES\human_vital_signs_cleaned_datasetv03.xlsx"
SHEET_NAME  = 'Human_Vital_Signs'

COLUMNS_TO_DROP = [
    "Respiratory Rate", "Systolic Blood Pressure",
    "Diastolic Blood Pressure", "Age", "Gender", "Weight (kg)", "Height (m)",
    "Derived_HRV", "Derived_Pulse_Pressure", "Derived_BMI", "Derived_MAP"
]

COLUMNS_TO_RENAME = {
    "Patient ID":        "patient_ID",
    "Heart Rate":        "Heart_Rate",
    "Body Temperature":  "Temperature",
    "Oxygen Saturation": "SpO2",
    "Risk Category":     "Risk_Level",
}

MEDICAL_RANGES = {
    "Heart Rate":        (40,  200),
    "Body Temperature":  (35.0, 41.5),
    "Oxygen Saturation": (70,  100),
}

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"  Original shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
print(f"  Columns: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# STEP 0 — REASSIGN RISK LEVEL FROM THRESHOLDS
# ─────────────────────────────────────────────

print("Step 0 — Reassigning Risk Level from vital sign thresholds...")

def classify_heart_rate(hr):
    if 60 <= hr <= 100:
        return 0
    elif (50 <= hr < 60) or (100 < hr <= 120):
        return 1
    else:
        return 2

def classify_spo2(spo2):
    if spo2 >= 95:
        return 0
    elif 90 <= spo2 < 95:
        return 1
    else:
        return 2

def classify_temperature(temp):
    if 36.1 <= temp <= 37.2:
        return 0
    elif (35.0 <= temp < 36.1) or (37.2 < temp <= 38.5):
        return 1
    else:
        return 2

def assign_risk(row):
    scores = [
        classify_heart_rate(row["Heart Rate"]),
        classify_spo2(row["Oxygen Saturation"]),
        classify_temperature(row["Body Temperature"]),
    ]
    return max(scores)

df["Risk Category"] = df.apply(assign_risk, axis=1)
mapping = {0: "Normal", 1: "Abnormal", 2: "Critical"}
df["Risk Category"] = df["Risk Category"].map(mapping)

counts      = df["Risk Category"].value_counts()
percentages = df["Risk Category"].value_counts(normalize=True) * 100
print("  Risk Level distribution after reassignment:")
for label in ["Normal", "Abnormal", "Critical"]:
    if label in counts:
        print(f"    {label:10s}: {counts[label]:>6}  ({percentages[label]:.2f}%)")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 1 — DROP SPECIFIC COLUMNS
# ─────────────────────────────────────────────

existing_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
missing_cols  = [c for c in COLUMNS_TO_DROP if c not in df.columns]

if missing_cols:
    print(f"  ⚠  Columns not found (skipped): {missing_cols}")

df.drop(columns=existing_cols, inplace=True)
print(f"Step 1 — Dropped columns: {existing_cols}")
print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 2 — DROP DUPLICATE ROWS
# ─────────────────────────────────────────────

before = len(df)
df.drop_duplicates(inplace=True)
removed = before - len(df)
print(f"Step 2 — Removed {removed} duplicate row(s).")
print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 3 — REMOVE OUTLIERS (medical ranges)
# ─────────────────────────────────────────────

before = len(df)

for col, (lower, upper) in MEDICAL_RANGES.items():
    if col not in df.columns:
        print(f"  ⚠  Column '{col}' not found — skipped.")
        continue
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"  '{col}': removed {outliers} outlier(s)  [medical range: {lower} – {upper}]")

removed = before - len(df)
print(f"Step 3 — Removed {removed} outlier row(s) total.")
print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 4 — RENAME COLUMNS
# ─────────────────────────────────────────────

existing_renames = {k: v for k, v in COLUMNS_TO_RENAME.items() if k in df.columns}
missing_renames  = [k for k in COLUMNS_TO_RENAME if k not in df.columns]

if missing_renames:
    print(f"  ⚠  Rename columns not found (skipped): {missing_renames}")

df.rename(columns=existing_renames, inplace=True)
print(f"Step 4 — Renamed columns: {existing_renames}")
print(f"  Current columns: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# STEP 5 — UNDERSAMPLE NORMAL TO MATCH ABNORMAL
# ─────────────────────────────────────────────

print("Step 5 — Undersampling 'Normal' to match 'Abnormal' count...")

counts_before = df["Risk_Level"].value_counts()
print("  Class counts BEFORE undersampling:")
for label in ["Normal", "Abnormal", "Critical"]:
    if label in counts_before:
        print(f"    {label:10s}: {counts_before[label]:>6}")

abnormal_count = counts_before.get("Abnormal", 0)

if abnormal_count == 0:
    print("  ⚠  'Abnormal' class not found — skipping undersampling.")
elif counts_before.get("Normal", 0) <= abnormal_count:
    print(f"  ℹ  Normal ({counts_before.get('Normal', 0)}) is already ≤ Abnormal ({abnormal_count}) — no action needed.")
else:
    # Randomly sample Normal rows down to Abnormal count (reproducible)
    normal_rows    = df[df["Risk_Level"] == "Normal"].sample(n=abnormal_count, random_state=42)
    non_normal     = df[df["Risk_Level"] != "Normal"]
    df             = pd.concat([normal_rows, non_normal], ignore_index=True)

    # Shuffle so Normal rows aren't all at the top
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    removed = counts_before.get("Normal", 0) - abnormal_count
    print(f"  Removed {removed} Normal row(s) — kept {abnormal_count} (= Abnormal count).")

counts_after = df["Risk_Level"].value_counts()
percentages  = df["Risk_Level"].value_counts(normalize=True) * 100
print("\n  Class counts AFTER undersampling:")
for label in ["Normal", "Abnormal", "Critical"]:
    if label in counts_after:
        print(f"    {label:10s}: {counts_after[label]:>6}  ({percentages[label]:.2f}%)")
print(f"  Shape after: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 6 — CHECK CLASS BALANCE
# ─────────────────────────────────────────────

print("── Class Balance Check ──────────────────────")

target_col = "Risk_Level"
counts      = df[target_col].value_counts()
percentages = df[target_col].value_counts(normalize=True) * 100

balance_df = pd.DataFrame({
    "Count":      counts,
    "Percentage": percentages.round(2)
})
print(balance_df.to_string())
print(f"\n  Total rows     : {len(df)}")
print(f"  Unique classes : {df[target_col].nunique()}")

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
print(f"\n  Verdict: {verdict}")

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

PLOTS_DIR = r"CLEANED_FILES\visuals01"
os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="Set2")

vital_signs     = ["Heart_Rate", "Temperature", "SpO2"]
existing_vitals = [c for c in vital_signs if c in df.columns]

print("\n── Risk Level Visualizations ────────────────")

# ── Plot 01: Risk Level Distribution ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Risk Level Distribution", fontsize=16, fontweight='bold')

counts = df["Risk_Level"].value_counts()
counts.plot(kind='bar', ax=axes[0],
            color=sns.color_palette("Set2", len(counts)), edgecolor='black')
axes[0].set_title("Count per Risk Level")
axes[0].set_xlabel("Risk Level")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)
for bar in axes[0].patches:
    axes[0].annotate(f'{int(bar.get_height())}',
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10)

axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("Set2", len(counts)),
            startangle=140, wedgeprops={'edgecolor': 'black'})
axes[1].set_title("Proportion per Risk Level")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_risk_level_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 01_risk_level_distribution.png")

# ── Plot 02: Numerical Feature Distributions ──────────────────────────────────
print("\n── Numerical Feature Distributions ─────────")
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

# ── Plot 03: Correlation Matrix ───────────────────────────────────────────────
print("\n── Correlation Matrix ───────────────────────")
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

# ── Plot 04: Vital Signs Boxplot by Risk Level ────────────────────────────────
print("\n── Vital Signs vs Risk Level ────────────────")
risk_order = [r for r in ["Normal", "Abnormal", "Critical"] if r in df["Risk_Level"].unique()]

fig, axes = plt.subplots(1, len(existing_vitals), figsize=(6 * len(existing_vitals), 6))
fig.suptitle("Vital Signs Distribution by Risk Level", fontsize=16, fontweight='bold')
if len(existing_vitals) == 1:
    axes = [axes]

for ax, col in zip(axes, existing_vitals):
    sns.boxplot(data=df, x="Risk_Level", y=col, ax=ax,
                palette="Set2", order=risk_order)
    ax.set_title(f"{col} by Risk Level")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel(col)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_vitals_boxplot_by_risk.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 04_vitals_boxplot_by_risk.png")

# ── Plot 05: Vital Signs Violin by Risk Level ─────────────────────────────────
fig, axes = plt.subplots(1, len(existing_vitals), figsize=(6 * len(existing_vitals), 6))
fig.suptitle("Vital Signs Spread by Risk Level", fontsize=16, fontweight='bold')
if len(existing_vitals) == 1:
    axes = [axes]

for ax, col in zip(axes, existing_vitals):
    sns.violinplot(data=df, x="Risk_Level", y=col, ax=ax,
                   palette="Set2", order=risk_order, inner="quartile")
    ax.set_title(f"{col} Distribution — Risk Level")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel(col)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_vitals_violin_by_risk.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 05_vitals_violin_by_risk.png")

# ── Plot 06: KDE — Critical vs Others ────────────────────────────────────────
fig, axes = plt.subplots(1, len(existing_vitals), figsize=(6 * len(existing_vitals), 5))
fig.suptitle("Vital Signs: Critical vs Others (KDE)", fontsize=16, fontweight='bold')
if len(existing_vitals) == 1:
    axes = [axes]

for ax, col in zip(axes, existing_vitals):
    critical_df = df[df["Risk_Level"] == "Critical"][col].dropna()
    other_df    = df[df["Risk_Level"] != "Critical"][col].dropna()
    sns.kdeplot(critical_df, ax=ax, label="Critical", fill=True, color="crimson",   alpha=0.4)
    sns.kdeplot(other_df,    ax=ax, label="Other",    fill=True, color="steelblue", alpha=0.4)
    ax.set_title(f"{col} — Critical vs Others")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_vitals_kde_critical_vs_others.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 06_vitals_kde_critical_vs_others.png")

# ── Plot 07: Mean Vital Signs Heatmap per Risk Level ─────────────────────────
mean_vitals = df.groupby("Risk_Level")[existing_vitals].mean().round(2)
mean_vitals = mean_vitals.reindex([r for r in risk_order if r in mean_vitals.index])

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(mean_vitals, annot=True, fmt=".2f", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Mean Value"})
ax.set_title("Mean Vital Signs per Risk Level", fontsize=14, fontweight='bold')
ax.set_xlabel("Vital Sign")
ax.set_ylabel("Risk Level")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_vitals_heatmap_mean_by_risk.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 07_vitals_heatmap_mean_by_risk.png")

print(f"\n✅ All plots saved to '{PLOTS_DIR}'")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_excel(OUTPUT_FILE, index=False, sheet_name=SHEET_NAME, engine='openpyxl')
print(f"\n✅ Cleaned dataset saved to '{OUTPUT_FILE}'")
print(f"   Final shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Final columns: {list(df.columns)}")