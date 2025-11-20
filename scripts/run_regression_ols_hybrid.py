#!/usr/bin/env python3
import os
import pandas as pd
import statsmodels.api as sm

REG_DIR = "data"
OUT_FILE = "data/model_regression_summary.csv"

def run_ols(df, target_col):
    X = df[["input_tokens_z", "output_tokens_z"]]
    X = sm.add_constant(X)
    y = df[target_col]
    model = sm.OLS(y, X).fit()
    return model

def main():
    files = sorted([f for f in os.listdir(REG_DIR) if f.startswith("regression_") and f.endswith(".csv")])
    if not files:
        print("No regression_*.csv files found in data/. Run hybrid_normalize.py first.")
        return

    rows = []

    for fname in files:
        model_name = fname.replace("regression_", "").replace(".csv", "")
        print(f"📈 Running OLS for {model_name}")
        df = pd.read_csv(os.path.join(REG_DIR, fname))

        # Runtime regression
        rt_model = run_ols(df, "runtime_z")
        # Energy regression
        en_model = run_ols(df, "energy_z")

        rows.append({
            "model": model_name,
            "runtime_const": rt_model.params["const"],
            "runtime_in_tokens": rt_model.params["input_tokens_z"],
            "runtime_out_tokens": rt_model.params["output_tokens_z"],
            "runtime_R2": rt_model.rsquared,
            "energy_const": en_model.params["const"],
            "energy_in_tokens": en_model.params["input_tokens_z"],
            "energy_out_tokens": en_model.params["output_tokens_z"],
            "energy_R2": en_model.rsquared,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_FILE, index=False)
    print(f"\n✅ Regression summary saved → {OUT_FILE}\n")
    print(summary.round(4))

if __name__ == "__main__":
    main()