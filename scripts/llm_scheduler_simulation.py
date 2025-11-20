import os
import argparse
import pandas as pd
import torch
from datetime import datetime
from transformers import AutoTokenizer

PROFILE_PATH = "data/model_performance_profile.csv"
NORMALIZATION_STATS = "normalized_models/normalization_stats.json"
LOG_DIR = "logs/scheduler_runs"
os.makedirs(LOG_DIR, exist_ok=True)

def estimate_tokens(prompt, model_id):
    """Estimate number of input tokens for a prompt using tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(prompt, return_tensors="pt")
        return len(tokens["input_ids"][0])
    except Exception:
        return len(prompt.split()) * 1.5

def normalize_value(value, mean, std):
    return (value - mean) / std if std != 0 else 0.0

def score_model(row, input_tokens, output_tokens, weights, norms):
    """Predict runtime, energy, and compute weighted score for one model."""
    model = row["model"]
    acc = row["accuracy"]

    model_stats = norms.get(f"metrics_{model}.csv", {})
    in_mean = model_stats.get("input_tokens", {}).get("mean", 0)
    in_std = model_stats.get("input_tokens", {}).get("std", 1)
    out_mean = model_stats.get("output_tokens", {}).get("mean", 0)
    out_std = model_stats.get("output_tokens", {}).get("std", 1)

    in_norm = normalize_value(input_tokens, in_mean, in_std)
    out_norm = normalize_value(output_tokens, out_mean, out_std)

    runtime = (
        row["runtime_const"]
        + row["runtime_in_tokens"] * in_norm
        + row["runtime_out_tokens"] * out_norm
    )
    energy = (
        row["energy_const"]
        + row["energy_in_tokens"] * in_norm
        + row["energy_out_tokens"] * out_norm
    )

    return {"model": model, "acc": acc, "runtime": abs(runtime), "energy": abs(energy)}

def main():
    parser = argparse.ArgumentParser(description="LLM Scheduler - Normalized Simulation v2")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--weight_accuracy", type=float, default=0.6)
    parser.add_argument("--weight_energy", type=float, default=0.2)
    parser.add_argument("--weight_time", type=float, default=0.2)
    parser.add_argument("--batch_file", type=str, default=None)
    args = parser.parse_args()

    prof = pd.read_csv(PROFILE_PATH)
    norms = pd.read_json(NORMALIZATION_STATS)

    weights = {
        "accuracy": args.weight_accuracy,
        "energy": args.weight_energy,
        "time": args.weight_time,
    }

    print("\n📊 Loaded models:", prof["model"].tolist())

    prompts = []
    if args.batch_file:
        with open(args.batch_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [args.prompt]

    log_entries = []

    for idx, prompt in enumerate(prompts, 1):
        est_out_tokens = min(150, len(prompt.split()) * 3)
        results = []

        # Step 1: Predict runtime/energy for all models
        for _, row in prof.iterrows():
            if "phi-3" in row["model"]:
                model_id = "microsoft/Phi-3-mini-4k-instruct"
            elif "falcon" in row["model"]:
                model_id = "tiiuae/Falcon-7B-Instruct"
            else:
                model_id = "mistralai/Mistral-7B-Instruct-v0.1"

            input_tokens = estimate_tokens(prompt, model_id)
            pred = score_model(row, input_tokens, est_out_tokens, weights, norms)
            results.append(pred)

        df = pd.DataFrame(results)

        # Step 2: Min–max normalize predicted metrics across models
        for col in ["runtime", "energy"]:
            df[f"{col}_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)

        # Step 3: Compute final weighted score (lower = better)
        df["score"] = (
            weights["accuracy"] * (1 - df["acc"])
            + weights["energy"] * df["energy_norm"]
            + weights["time"] * df["runtime_norm"]
        )

        best = df.loc[df["score"].idxmin()]
        print(f"\n🧩 Prompt {idx}/{len(prompts)}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(df[["model", "score", "runtime", "energy", "acc"]].round(4))
        print(f"✅ Selected model: {best['model']} (score={best['score']:.4f})")

        log_entries.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "best_model": best["model"],
            "best_score": round(best["score"], 4),
            "est_runtime": round(best["runtime"], 4),
            "est_energy": round(best["energy"], 4),
            "accuracy": round(best["acc"], 3),
            "weights_accuracy": args.weight_accuracy,
            "weights_energy": args.weight_energy,
            "weights_time": args.weight_time
        })

    # Step 4: Log everything
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"scheduler_run_{timestamp}.csv")
    pd.DataFrame(log_entries).to_csv(log_file, index=False)

    print(f"\n🗂️  Saved run log: {log_file}")
    print(f"🧠  Logged {len(log_entries)} prompts.")


if __name__ == "__main__":
    main()
