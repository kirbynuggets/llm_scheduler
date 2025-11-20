import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
PROFILE_PATH = "data/model_performance_profile.csv"
NORMALIZATION_STATS = "normalized_models/normalization_stats.json"

# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------
def estimate_tokens(prompt, model_id):
    """Estimate number of input tokens for a prompt using the model tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokens = tokenizer(prompt, return_tensors="pt")
    return len(tokens["input_ids"][0])

def normalize_value(value, mean, std):
    """Normalize value using mean/std to match regression scaling."""
    return (value - mean) / std if std != 0 else 0.0

# -------------------------------------------------------------
# MODEL SCORING FUNCTION
# -------------------------------------------------------------
def score_model(row, input_tokens, output_tokens, weights, norms):
    """
    Predict runtime, energy and compute final weighted score for one model.
    """
    model = row["model"]
    acc = row["accuracy"]

    # Fetch normalization stats
    model_stats = norms.get(f"metrics_{model}.csv", {})
    in_mean = model_stats.get("input_tokens", {}).get("mean", 0)
    in_std = model_stats.get("input_tokens", {}).get("std", 1)
    out_mean = model_stats.get("output_tokens", {}).get("mean", 0)
    out_std = model_stats.get("output_tokens", {}).get("std", 1)

    # Normalize inputs
    in_norm = normalize_value(input_tokens, in_mean, in_std)
    out_norm = normalize_value(output_tokens, out_mean, out_std)

    # Predict normalized runtime and energy
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

    # Weighted composite score
    score = (
        weights["accuracy"] * (1 - acc)
        + weights["energy"] * abs(energy)
        + weights["time"] * abs(runtime)
    )
    return score, runtime, energy

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dynamic LLM Scheduler (runtime selector)")
    parser.add_argument("--prompt", required=True, help="User input text")
    parser.add_argument("--weight_accuracy", type=float, default=0.6)
    parser.add_argument("--weight_energy", type=float, default=0.2)
    parser.add_argument("--weight_time", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=100, help="Output token cap")
    parser.add_argument("--run", action="store_true", help="Actually execute best model and show response")
    args = parser.parse_args()

    # Load regression + accuracy data
    prof = pd.read_csv(PROFILE_PATH)
    norms = pd.read_json(NORMALIZATION_STATS)

    weights = {
        "accuracy": args.weight_accuracy,
        "energy": args.weight_energy,
        "time": args.weight_time,
    }

    print("\n📊 Loaded performance profile for models:")
    print(prof["model"].tolist())

    # Rough guess of output tokens based on input length
    est_out_tokens = min(150, len(args.prompt.split()) * 3)

    scores = []
    for _, row in prof.iterrows():
        if "phi-3" in row["model"]:
            model_id = "microsoft/Phi-3-mini-4k-instruct"
        elif "falcon" in row["model"]:
            model_id = "tiiuae/Falcon-7B-Instruct"
        else:
            model_id = "mistralai/Mistral-7B-Instruct-v0.1"

        inp_tokens = estimate_tokens(args.prompt, model_id)
        score, rt, en = score_model(row, inp_tokens, est_out_tokens, weights, norms)
        scores.append({
            "model": row["model"],
            "score": score,
            "runtime_est": rt,
            "energy_est": en,
            "accuracy": row["accuracy"]
        })

    df = pd.DataFrame(scores).sort_values("score")
    print("\n🏁 Scheduler results (lower score = better):\n")
    print(df.round(4))

    best = df.iloc[0]
    print(f"\n✅ Selected model: {best['model']} (score={best['score']:.4f})")

    # -------------------------------------------------------------
    # SAFE MODEL EXECUTION
    # -------------------------------------------------------------
    if args.run:
        model_name = best["model"]
        if "phi-3" in model_name:
            hf_id = "microsoft/Phi-3-mini-4k-instruct"
        elif "falcon" in model_name:
            hf_id = "tiiuae/Falcon-7B-Instruct"
        else:
            hf_id = "mistralai/Mistral-7B-Instruct-v0.1"

        print(f"\n🚀 Running {hf_id} (safe memory mode)...")

        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(hf_id)

        # Load safely — try FP16 sequential, fallback to 8-bit
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                device_map="sequential",
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="eager"
            )
        except RuntimeError as e:
            print(f"⚠️ FP16 load failed: {e}")
            print("Retrying in 8-bit mode to save VRAM...")
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                device_map="auto",
                load_in_8bit=True
            )

        # Run generation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_tokens)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("\n🧠 Model output:\n")
        print(text)


if __name__ == "__main__":
    main()
