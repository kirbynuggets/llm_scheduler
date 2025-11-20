import os
import pandas as pd

DATA_DIR = "data"
ACCURACY_FILE = os.path.join(DATA_DIR, "model_accuracy_scores.csv")
REGRESSION_FILE = os.path.join(DATA_DIR, "model_regression_summary.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "model_performance_profile.csv")

# -------------------------------------------------------------
# Load
# -------------------------------------------------------------
print(" Loading regression and accuracy data...")
acc_df = pd.read_csv(ACCURACY_FILE)
reg_df = pd.read_csv(REGRESSION_FILE)

# -------------------------------------------------------------
# Normalize model names to line up
# -------------------------------------------------------------
def simplify(name):
    name = name.lower()
    name = name.replace("metrics_", "")
    name = name.replace("microsoft/", "")
    name = name.replace("mistralai/", "")
    name = name.replace("tiiuae/", "")
    name = name.replace("-v0.1", "")
    name = name.replace("-v0.2", "")
    name = name.replace("instruct", "instruct")
    name = name.replace("-", "").replace("_", "")
    return name

acc_df["key"] = acc_df["model"].apply(simplify)
reg_df["key"] = reg_df["model"].apply(simplify)

# -------------------------------------------------------------
# Merge
# -------------------------------------------------------------
merged = pd.merge(reg_df, acc_df[["key", "accuracy"]], on="key", how="inner")

if merged.empty:
    print(" Merge produced no matches — check model names!")
else:
    cols = [
        "model",
        "accuracy",
        "energy_const", "energy_in_tokens", "energy_out_tokens", "energy_R2",
        "runtime_const", "runtime_in_tokens", "runtime_out_tokens", "runtime_R2"
    ]
    final = merged[cols].copy()
    os.makedirs(DATA_DIR, exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Model performance profile saved to {OUTPUT_FILE}\n")
    print(final)