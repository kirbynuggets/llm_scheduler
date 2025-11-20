import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path


# Load regression model for output tokens

def estimate_output_tokens(input_tokens):
    try:
        params = json.load(open("data/output_token_regression.json"))
        alpha = params["alpha"]
        beta = params["beta"]
        out = alpha * input_tokens + beta
        return max(5, int(out))   # avoid zero-token generation
    except:
        return 100  # fallback



# Main scheduler script

parser = argparse.ArgumentParser(description="Fair OLS Scheduler with Regression Output Tokens")
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--weight_accuracy", type=float, default=0.5)
parser.add_argument("--weight_energy", type=float, default=0.3)
parser.add_argument("--weight_time", type=float, default=0.2)
parser.add_argument("--input_tokens", type=int, default=200)
args = parser.parse_args()

# predict output tokens dynamically 👇
args.output_tokens = estimate_output_tokens(args.input_tokens)


# Load regression + accuracy profiles

reg = pd.read_csv("data/model_regression_summary.csv")
acc = pd.read_csv("data/model_accuracy_scores.csv")

# normalize names
def clean_name(name):
    return (
        name.lower()
        .replace("microsoft/", "")
        .replace("tiiuae/", "")
        .replace("mistralai/", "")
        .replace("instruct-v0.1", "instruct-v0.2")
        .replace(" ", "")
        .strip()
    )

reg["model"] = reg["model"].apply(clean_name)
acc["model"] = acc["model"].apply(clean_name)

reg = reg.merge(acc, on="model", how="left")

if reg["accuracy"].isnull().any():
    print("⚠️ Missing accuracy for:", reg[reg["accuracy"].isnull()]["model"].tolist())


# Predict runtime & energy using OLS

reg["runtime_pred"] = (
    abs(reg["runtime_in_tokens"] * args.input_tokens +
        reg["runtime_out_tokens"] * args.output_tokens)
)

reg["energy_pred"] = (
    abs(reg["energy_in_tokens"] * args.input_tokens +
        reg["energy_out_tokens"] * args.output_tokens)
)


# Hybrid normalisation

def hybrid_norm(series):
    z = (series - series.mean()) / (series.std() + 1e-8)
    mm = (series - series.min()) / (series.max() - series.min() + 1e-8)
    return 0.5*z + 0.5*mm

reg["runtime_norm"] = hybrid_norm(reg["runtime_pred"])
reg["energy_norm"] = hybrid_norm(reg["energy_pred"])
reg["accuracy_norm"] = (reg["accuracy"].max() - reg["accuracy"]) / (
    reg["accuracy"].max() - reg["accuracy"].min() + 1e-8
)


# Softmax scaling

def softmax(x):
    e = np.exp(-x)
    return e / np.sum(e)

reg["runtime_soft"] = softmax(reg["runtime_norm"])
reg["energy_soft"] = softmax(reg["energy_norm"])
reg["accuracy_soft"] = softmax(reg["accuracy_norm"])


# Weighted score
reg["score"] = (
    args.weight_accuracy * reg["accuracy_soft"] +
    args.weight_energy * reg["energy_soft"] +
    args.weight_time * reg["runtime_soft"]
)

reg = reg.sort_values("score")
best = reg.iloc[0]

# Output results
print(f"\n📊 Fair OLS Scheduler for Prompt: “{args.prompt}”")
print(f" Predicted Output Tokens: {args.output_tokens}")
print("\n🏁 Model Ranking (lower score = better):")
print(reg[["model", "score", "accuracy", "runtime_pred", "energy_pred"]].to_string(index=False))

print(f"\n✅ Selected Model: {best['model']}")


# Log the result
Path("logs/scheduler_runs").mkdir(exist_ok=True)
log_path = f"logs/scheduler_runs/fair_scheduler_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
reg.to_csv(log_path, index=False)
print(f"\n🗂️ Logged results → {log_path}")
