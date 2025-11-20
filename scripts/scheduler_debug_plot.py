import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="LLM Scheduler Debug Visualizer")
parser.add_argument("--logfile", type=str, required=True, help="Path to the logged CSV from llm_fair_ols")
args = parser.parse_args()

# ===========================
# Load data
# ===========================
df = pd.read_csv(args.logfile)
df = df.dropna(subset=["score"]).reset_index(drop=True)

print("\n🧠 Debugging Scheduler Decision")
print(f"Loaded log: {args.logfile}\n")

# ===========================
# Normalize values for visualization
# ===========================
def z_norm(series):
    return (series - series.mean()) / (series.std() + 1e-8)

df["runtime_norm"] = z_norm(df["runtime_pred"].abs())
df["energy_norm"] = z_norm(df["energy_pred"].abs())
df["accuracy_norm"] = (df["accuracy"] - df["accuracy"].min()) / (df["accuracy"].max() - df["accuracy"].min())

# ===========================
# Compute weighted contributions (reverse-engineered)
# ===========================
weights = {
    "accuracy": 0.5,
    "energy": 0.3,
    "time": 0.2
}  # manually set or infer from log naming

df["accuracy_contrib"] = (1 - df["accuracy_norm"]) * weights["accuracy"]
df["energy_contrib"] = df["energy_norm"] * weights["energy"]
df["runtime_contrib"] = df["runtime_norm"] * weights["time"]
df["total_score_est"] = df["accuracy_contrib"] + df["energy_contrib"] + df["runtime_contrib"]

# ===========================
# Show tabular view
# ===========================
print("📊 Model Breakdown (normalized & weighted):\n")
print(
    df[
        [
            "model",
            "accuracy",
            "runtime_pred",
            "energy_pred",
            "accuracy_contrib",
            "energy_contrib",
            "runtime_contrib",
            "total_score_est",
        ]
    ].sort_values("total_score_est")
    .to_string(index=False, float_format=lambda x: f"{x:0.3f}")
)

winner = df.iloc[df["total_score_est"].idxmin()]["model"]
print(f"\n✅ Winner (lowest total score): {winner}")

# ===========================
# Optional visualization
# ===========================
Path("plots").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.25
x = np.arange(len(df))

ax.bar(x - bar_width, df["accuracy_contrib"], bar_width, label="Accuracy (lower=better)")
ax.bar(x, df["energy_contrib"], bar_width, label="Energy")
ax.bar(x + bar_width, df["runtime_contrib"], bar_width, label="Runtime")

ax.set_xticks(x)
ax.set_xticklabels(df["model"], rotation=15)
ax.set_ylabel("Normalized Contribution to Total Score")
ax.set_title("Model Trade-offs Visualization")
ax.legend()
plt.tight_layout()
plot_path = f"plots/scheduler_debug_{Path(args.logfile).stem}.png"
plt.savefig(plot_path)
print(f"\n📈 Saved visualization → {plot_path}")
plt.close()