import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Compare multiple scheduler runs visually.")
parser.add_argument(
    "--logfiles",
    nargs="+",
    required=True,
    help="List of CSV logs to compare (space separated)."
)
args = parser.parse_args()

# ------------------------------
# Load all logs
# ------------------------------
runs = []
for log in args.logfiles:
    try:
        df = pd.read_csv(log)
        df["run_name"] = Path(log).stem
        runs.append(df)
    except Exception as e:
        print(f"⚠️ Could not load {log}: {e}")

if not runs:
    raise ValueError("No valid logs found!")

all_runs = pd.concat(runs, ignore_index=True)

# ------------------------------
# Normalize and compute relative ranks
# ------------------------------
def z_norm(series):
    return (series - series.mean()) / (series.std() + 1e-8)

all_runs["runtime_norm"] = z_norm(all_runs["runtime_pred"].abs())
all_runs["energy_norm"] = z_norm(all_runs["energy_pred"].abs())

if "accuracy" in all_runs.columns:
    all_runs["accuracy_norm"] = (all_runs["accuracy"] - all_runs["accuracy"].min()) / (
        all_runs["accuracy"].max() - all_runs["accuracy"].min()
    )
else:
    all_runs["accuracy_norm"] = np.nan

# ------------------------------
# Build summary table
# ------------------------------
summary = (
    all_runs.groupby(["run_name", "model"])
    .agg(
        {
            "score": "mean",
            "runtime_norm": "mean",
            "energy_norm": "mean",
            "accuracy_norm": "mean",
        }
    )
    .reset_index()
)

print("\n📊 Comparison Summary:\n")
print(summary.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

# ------------------------------
# Plot comparative bar chart
# ------------------------------
Path("plots").mkdir(exist_ok=True)
plt.figure(figsize=(9, 5))

for run_name in summary["run_name"].unique():
    subset = summary[summary["run_name"] == run_name]
    plt.plot(subset["model"], subset["score"], marker="o", label=run_name)

plt.title("Scheduler Score Comparison Across Runs (lower = better)")
plt.ylabel("Composite Score")
plt.xlabel("Model")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plot_path = f"plots/comparison_{'_'.join(Path(f).stem for f in args.logfiles)}.png"
plt.savefig(plot_path)
print(f"\n📈 Saved comparison plot → {plot_path}")
plt.close()
