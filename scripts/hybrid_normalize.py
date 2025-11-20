#!/usr/bin/env python3
import os
import json
import math
import argparse
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
INPUT_DIR = "models"               # where metrics_*.csv are
ACC_FILE = "data/model_accuracy_scores.csv"
OUT_DIR = "normalized_models"
REGRESSION_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REGRESSION_DIR, exist_ok=True)

METRIC_PREFIX = "metrics_"
COLUMNS = ["input_tokens", "output_tokens", "runtime_seconds", "energy_joules"]

# ---------- HELPERS ----------
def safe_div(a, b):
    return a / b if (b is not None and b != 0) else 0.0

def zscore_series(s):
    mu = s.mean()
    sigma = s.std(ddof=0)
    return (s - mu) / sigma, mu, sigma

def minmax_series(s):
    mn = s.min()
    mx = s.max()
    denom = mx - mn if mx != mn else 1.0
    return (s - mn) / denom, mn, mx

def log1p_series(s):
    return np.log1p(s)

def normalize_accuracy_value(acc):
    # convert percentages >1 to fraction
    if acc is None:
        return None
    if acc > 1.0:
        return acc / 100.0
    return acc

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description="Hybrid normalization: tokens (z-score), time/energy (log+minmax), accuracy (0-1)")
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--acc_file", default=ACC_FILE)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--regression_out_dir", default=REGRESSION_DIR)
    parser.add_argument("--save_regression_csv", action="store_true", help="also save regression-ready CSVs")
    args = parser.parse_args()

    # load accuracy table
    if not os.path.exists(args.acc_file):
        raise FileNotFoundError(f"Accuracy file not found: {args.acc_file}")
    acc_df = pd.read_csv(args.acc_file)
    # normalise model id strings to map files
    def simplify(name):
        n = str(name).lower()
        n = n.replace("microsoft/","").replace("mistralai/","").replace("tiiuae/","")
        n = n.replace("-v0.1","").replace("-v0.2","")
        n = n.replace("-","").replace("_","")
        return n
    acc_df["key"] = acc_df["model"].apply(simplify)
    acc_map = {}
    for _, r in acc_df.iterrows():
        key = r["key"]
        acc_map[key] = normalize_accuracy_value(float(r["accuracy"]))

    normalization_stats = {}  # per-model stats

    # Process each metrics_*.csv
    files = sorted([f for f in os.listdir(args.input_dir) if f.startswith(METRIC_PREFIX) and f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(f"No files {METRIC_PREFIX}*.csv found in {args.input_dir}")

    for fname in files:
        path = os.path.join(args.input_dir, fname)
        print(f"Processing {fname} ...")
        df = pd.read_csv(path)
        # basic cleaning
        df = df.dropna(subset=["input_tokens", "runtime_seconds"])
        df = df[(df["runtime_seconds"] > 0) & (df["input_tokens"] > 0)]

        model_short = fname.replace(METRIC_PREFIX, "").replace(".csv","")
        key = simplify(model_short)

        # --- tokens: z-score (for regression) ---
        inp_z, inp_mu, inp_sigma = zscore_series(df["input_tokens"])
        out_z, out_mu, out_sigma = zscore_series(df["output_tokens"])

        df["input_tokens_z"] = inp_z
        df["output_tokens_z"] = out_z

        # --- runtime & energy: log1p then min-max (0-1) for scheduler use ---
        df["runtime_log"] = log1p_series(df["runtime_seconds"])
        df["energy_log"] = log1p_series(df["energy_joules"].fillna(0.0))

        runtime_norm, runtime_log_min, runtime_log_max = minmax_series(df["runtime_log"])
        energy_norm, energy_log_min, energy_log_max = minmax_series(df["energy_log"])

        df["runtime_log_norm"] = runtime_norm
        df["energy_log_norm"] = energy_norm

        # --- For regression targets: z-score runtime and energy (we keep separate) ---
        runtime_z, runtime_mu, runtime_sigma = zscore_series(df["runtime_seconds"])
        energy_z, energy_mu, energy_sigma = zscore_series(df["energy_joules"].fillna(0.0))

        df["runtime_z"] = runtime_z
        df["energy_z"] = energy_z

        # --- accuracy mapping ---
        acc_val = acc_map.get(key, None)
        if acc_val is None:
            # fallback: if absent, try to match by original model_short (rare)
            acc_val = None
            for k,v in acc_map.items():
                if k in model_short.lower():
                    acc_val = v
                    break
        if acc_val is None:
            print(f"⚠️ No accuracy entry found for {model_short}. Setting accuracy=0.0")
            acc_val = 0.0
        df["accuracy"] = acc_val
        # also min-max normalize accuracy across models later; for now keep raw 0-1
        # We'll compute global acc min/max across models after loop if desired

        # --- Save normalized CSV for this model ---
        out_csv = os.path.join(args.out_dir, f"norm_{model_short}.csv")
        df.to_csv(out_csv, index=False)

        # --- save regression ready csv if requested (z-scored features + z-scored targets) ---
        if args.save_regression_csv:
            reg_df = df[["input_tokens_z", "output_tokens_z", "runtime_z", "energy_z", "accuracy"]].copy()
            reg_csv = os.path.join(args.regression_out_dir, f"regression_{model_short}.csv")
            reg_df.to_csv(reg_csv, index=False)

        # --- record normalization stats ---
        normalization_stats[model_short] = {
            "input_tokens": {"mean": float(inp_mu), "std": float(inp_sigma)},
            "output_tokens": {"mean": float(out_mu), "std": float(out_sigma)},
            "runtime": {
                "log_min": float(runtime_log_min),
                "log_max": float(runtime_log_max),
                "raw_mean": float(runtime_mu),
                "raw_std": float(runtime_sigma)
            },
            "energy": {
                "log_min": float(energy_log_min),
                "log_max": float(energy_log_max),
                "raw_mean": float(energy_mu),
                "raw_std": float(energy_sigma)
            },
            "accuracy": {"value": float(acc_val)}
        }

        print(f" → wrote: {out_csv}")
        if args.save_regression_csv:
            print(f" → regression csv: {reg_csv}")

    # Optionally, compute accuracy min/max across models and add normalized acc
    acc_values = [v["accuracy"]["value"] for v in normalization_stats.values() if v["accuracy"]["value"] is not None]
    if acc_values:
        acc_min = min(acc_values)
        acc_max = max(acc_values)
        for model_short, stats in normalization_stats.items():
            accv = stats["accuracy"]["value"]
            stats["accuracy"]["min"] = acc_min
            stats["accuracy"]["max"] = acc_max
            stats["accuracy"]["norm"] = safe_div((accv - acc_min), (acc_max - acc_min)) if acc_max != acc_min else 1.0

    # Save normalization stats
    stats_path = os.path.join(args.out_dir, "normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(normalization_stats, f, indent=2)
    print(f"\nSaved normalization stats -> {stats_path}")
    print("Done.")

if __name__ == "__main__":
    main()