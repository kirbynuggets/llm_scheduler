import os
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
INPUT_DIR = "models"     # change to your actual folder if needed
OUTPUT_DIR = "normalized_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columns to normalize
COLUMNS = ["input_tokens", "output_tokens", "runtime_seconds", "energy_joules"]

# -------------------------------------------------------------
# NORMALIZATION HELPERS
# -------------------------------------------------------------
def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)

def normalize_file(path, output_path):
    df = pd.read_csv(path)
    print(f"🔹 Normalizing {os.path.basename(path)}  ({len(df)} rows)")
    
    # Drop NAs and filter out nonsensical rows
    df = df.dropna()
    df = df[(df["runtime_seconds"] > 0) & (df["input_tokens"] > 0)]

    # Apply z-score normalization
    stats = {}
    for col in COLUMNS:
        mean, std = df[col].mean(), df[col].std(ddof=0)
        df[col + "_norm"] = (df[col] - mean) / std
        stats[col] = {"mean": mean, "std": std}
    
    # Save normalized data
    df.to_csv(output_path, index=False)
    print(f"✅ Saved normalized file → {output_path}")
    
    # Return stats for reuse later (for scheduler normalization)
    return stats

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
all_stats = {}
for file in os.listdir(INPUT_DIR):
    if file.endswith(".csv") and file.startswith("metrics_"):
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, file.replace("metrics_", "norm_"))
        stats = normalize_file(input_path, output_path)
        all_stats[file] = stats

# Save per-model normalization stats
stats_path = os.path.join(OUTPUT_DIR, "normalization_stats.json")
pd.DataFrame(all_stats).to_json(stats_path, indent=4)
print(f"\n📊 Normalization parameters saved → {stats_path}")
