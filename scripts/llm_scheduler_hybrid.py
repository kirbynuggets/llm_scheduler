import argparse
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ============================================================
# FINAL VARIANCE-SCALED HYBRID SCHEDULER
# ============================================================

def minmax_scale(series):
    """Safely scale a pandas series between 0 and 1."""
    return (series - series.min()) / (series.max() - series.min() + 1e-8)


def load_regression_profiles():
    reg_df = pd.read_csv("data/model_regression_summary.csv")
    acc_df = None
    perf_path = "data/model_performance_profile.csv"
    if os.path.exists(perf_path):
        acc_df = pd.read_csv(perf_path)
    return reg_df, acc_df


def main():
    parser = argparse.ArgumentParser(description="Hybrid-normalized LLM scheduler with variance weighting")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text for inference simulation")
    parser.add_argument("--weight_accuracy", type=float, default=0.6)
    parser.add_argument("--weight_energy", type=float, default=0.2)
    parser.add_argument("--weight_time", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=50)
    args = parser.parse_args()

    weights = {
        "accuracy": args.weight_accuracy,
        "energy": args.weight_energy,
        "time": args.weight_time,
    }

    # ------------------------------
    # Load regression + accuracy data
    # ------------------------------
    reg_df, acc_df = load_regression_profiles()
    print("\n📊 Loaded regression + accuracy profiles for:")
    print(list(reg_df["model"].values))

    if acc_df is not None and "accuracy" in acc_df.columns:
        df = reg_df.merge(acc_df[["model", "accuracy"]], on="model", how="left")
    else:
        df = reg_df.copy()
        df["accuracy"] = 0.7  # default fallback
    df.fillna(0, inplace=True)

    # ------------------------------
    # Hybrid normalization
    # ------------------------------
    df["runtime_est"] = abs(df["runtime_in_tokens"]) + abs(df["runtime_out_tokens"])
    df["energy_est"] = abs(df["energy_in_tokens"]) + abs(df["energy_out_tokens"])

    # Inverted min-max (so higher runtime/energy → worse)
    df["runtime_norm"] = 1 - minmax_scale(df["runtime_est"])
    df["energy_norm"] = 1 - minmax_scale(df["energy_est"])
    df["accuracy_norm"] = minmax_scale(df["accuracy"])

    print("\n📐 Normalization ranges:")
    print({
        "runtime": [float(df['runtime_est'].min()), float(df['runtime_est'].max())],
        "energy": [float(df['energy_est'].min()), float(df['energy_est'].max())],
        "accuracy": [float(df['accuracy'].min()), float(df['accuracy'].max())],
    })

    # ------------------------------
    # Variance-based scaling
    # ------------------------------
    var_runtime = df["runtime_norm"].var()
    var_energy = df["energy_norm"].var()
    var_acc = df["accuracy_norm"].var()

    scale_runtime = 1 / (var_runtime + 1e-6)
    scale_energy = 1 / (var_energy + 1e-6)
    scale_acc = 1 / (var_acc + 1e-6)

    print("\n⚖️ Variance scales:")
    print({
        "runtime_scale": scale_runtime,
        "energy_scale": scale_energy,
        "accuracy_scale": scale_acc,
    })

    # ------------------------------
    # Scoring
    # ------------------------------
    scores, details = [], []
    for _, row in df.iterrows():
        acc_term = weights["accuracy"] * (1 - row["accuracy_norm"]) * scale_acc
        energy_term = weights["energy"] * (1 - row["energy_norm"]) * scale_energy
        time_term = weights["time"] * (1 - row["runtime_norm"]) * scale_runtime
        score = acc_term + energy_term + time_term

        scores.append(score)
        details.append({
            "model": row["model"],
            "acc_term": acc_term,
            "energy_term": energy_term,
            "time_term": time_term,
            "score": score
        })

    df["score"] = scores
    df = df.sort_values("score", ascending=True).reset_index(drop=True)

    # ------------------------------
    # Print results
    # ------------------------------
    print("\n🏁 Scheduler results (lower score = better):")
    print(df[["model", "score", "runtime_norm", "energy_norm", "accuracy"]])

    winner = df.iloc[0]
    print(f"\n✅ Selected model: {winner['model']} (score={winner['score']:.4f})")

    print("\n📊 Feature contributions (for debugging):")
    for d in details:
        print(f"  {d['model']}: "
              f"acc={d['acc_term']:.4f}, "
              f"energy={d['energy_term']:.4f}, "
              f"time={d['time_term']:.4f}, "
              f"total={d['score']:.4f}")

    # ------------------------------
    # Log decision
    # ------------------------------
    os.makedirs("logs/scheduler_runs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/scheduler_runs/scheduler_run_{ts}.csv"
    df.to_csv(log_path, index=False)
    print(f"\n🗂️  Logged decision to: {log_path}")


if __name__ == "__main__":
    main()
