import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. EMBEDDED MODEL REGISTRY (From Table 5.1)
# ==========================================
# This ensures the code matches your report's data perfectly.
# No more "999" errors due to missing JSON files.

REGISTRY = {
    "falcon-7b": {
        "accuracy": 0.705,  # From Leaderboard
        # Energy Coeffs (Alpha): [Const, In, Out, Interaction]
        "energy_coeffs": [112.60, -17.92, 13.59, 0.089],
        # Time Coeffs (Beta): [Const, In, Out, Interaction]
        "time_coeffs":   [0.551, -0.161, 0.132, 0.0008]
    },
    "mistral-7b": {
        "accuracy": 0.723,
        "energy_coeffs": [30.11, -4.48, 6.14, -0.016],
        "time_coeffs":   [-0.417, -0.017, 0.058, -0.0003]
    },
    "phi-3-mini": {
        "accuracy": 0.685,
        "energy_coeffs": [44.99, -5.76, 5.72, 0.004],
        "time_coeffs":   [-0.166, -0.028, 0.048, -0.0001]
    }
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def estimate_tokens(prompt):
    """Heuristic from Eq 3.5 in the report"""
    t_in = max(1, len(prompt.split())) * 1.3 # Crude word-to-token ratio
    t_out = (1.028 * t_in) + 44.12
    return int(t_in), int(t_out)

def predict_metrics(model_data, tin, tout):
    """Implements Eq 3.3 and 3.4 (The Regression Models)"""
    interaction = tin * tout
    
    # Energy Calculation
    ce = model_data["energy_coeffs"]
    energy = ce[0] + (ce[1]*tin) + (ce[2]*tout) + (ce[3]*interaction)
    
    # Runtime Calculation
    ct = model_data["time_coeffs"]
    runtime = ct[0] + (ct[1]*tin) + (ct[2]*tout) + (ct[3]*interaction)
    
    # --- SAFETY CLAMP ---
    # Fixes the "Negative Energy" bug for Mistral on short prompts.
    # We clamp to the base constant or a minimum floor.
    energy = max(energy, 5.0) 
    runtime = max(runtime, 0.1)
    
    return energy, runtime

# ==========================================
# 3. MAIN SCHEDULER LOGIC
# ==========================================

def main():
    print("\n" + "="*40)
    print(" Energy, Time, Correctness-Aware Scheduler")
    print("="*40)

    print("\nEnter prompt:")
    prompt = input("> ").strip()
    if not prompt: prompt = "Test prompt"

    print("\nEnter weights as: Accuracy Energy Time (e.g., 0.5 0.5 0.0)")
    try:
        w_input = input("> ").split()
        if len(w_input) != 3: raise ValueError
        w_acc, w_eng, w_time = map(float, w_input)
        if abs((w_acc + w_eng + w_time) - 1.0) > 0.01:
            print("Warning: Weights do not sum to 1.0. Normalizing...")
            total = w_acc + w_eng + w_time
            w_acc /= total; w_eng /= total; w_time /= total
    except:
        print("Invalid input. Using default Balanced weights (0.33 0.33 0.33)")
        w_acc = w_eng = w_time = 0.33

    # 1. Estimate Tokens
    tin, tout = estimate_tokens(prompt)

    # 2. Generate Predictions
    data = []
    for name, meta in REGISTRY.items():
        e_pred, t_pred = predict_metrics(meta, tin, tout)
        data.append({
            "model": name,
            "accuracy": meta["accuracy"],
            "energy_pred": e_pred,
            "runtime_pred": t_pred
        })
    
    df = pd.DataFrame(data)

    # 3. Normalization (Equation 3.6 - Min-Max)
    # This maps values to 0.0 (Best) - 1.0 (Worst)
    def min_max_norm(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    df["n_energy"] = min_max_norm(df["energy_pred"])
    df["n_time"]   = min_max_norm(df["runtime_pred"])
    
    # 4. Accuracy Cost (Equation 3.7)
    # We invert accuracy because we want to MINIMIZE the score.
    # Higher accuracy (0.8) becomes lower cost (0.2).
    df["n_acc"]    = 1.0 - df["accuracy"]
    # Normalize the accuracy cost so it competes fairly with energy/time
    df["n_acc"]    = min_max_norm(df["n_acc"])

    # 5. Final Scoring (Equation 3.8)
    df["score"] = (w_acc * df["n_acc"]) + \
                  (w_eng * df["n_energy"]) + \
                  (w_time * df["n_time"])

    # Sort: Lowest Score wins
    df = df.sort_values("score", ascending=True)
    best_model = df.iloc[0]["model"]

    # ==========================================
    # 4. OUTPUT & LOGGING
    # ==========================================
    
    print("\n" + "-"*60)
    print(" Fair-OLS Scheduling Results")
    print("-"*60)
    print(f"Prompt: \"{prompt[:50]}...\"")
    print(f"Est. Tokens: In={tin}, Out={tout}")
    print(f"User Weights: Acc={w_acc:.1f}, Eng={w_eng:.1f}, Time={w_time:.1f}")
    print("\n🏁 Candidate Ranking (Lower Score = Better):")
    
    # formatting for clean output
    out_df = df[["model", "score", "accuracy", "energy_pred", "runtime_pred"]].copy()
    out_df.columns = ["Model", "Score", "Acc", "Energy(J)", "Time(s)"]
    print(out_df.to_string(index=False, float_format="%.4f"))

    print(f"\n🎯 SELECTED MODEL: {best_model.upper()}")
    print("-"*60)

    # Save Log
    Path("logs/scheduler_runs").mkdir(parents=True, exist_ok=True)
    log_file = f"logs/scheduler_runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(log_file, index=False)
    print(f"Log saved to: {log_file}\n")

if __name__ == "__main__":
    main()