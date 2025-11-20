import os
import pandas as pd
import statsmodels.api as sm

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DATA_DIR = "normalized_models"
OUTPUT_FILE = "data/model_regression_summary.csv"
os.makedirs("data", exist_ok=True)

# -------------------------------------------------------------
# REGRESSION FUNCTION
# -------------------------------------------------------------
def fit_regression(df, y_col):
    X = df[["input_tokens_norm", "output_tokens_norm"]]
    X = sm.add_constant(X)
    y = df[y_col]
    model = sm.OLS(y, X).fit()
    return model.params, model.rsquared

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
rows = []
for file in os.listdir(DATA_DIR):
    if file.startswith("norm_") and file.endswith(".csv"):
        path = os.path.join(DATA_DIR, file)
        model_name = file.replace("norm_", "").replace(".csv", "")
        print(f"📈 Running OLS for {model_name}")

        df = pd.read_csv(path)
        df = df.dropna(subset=[
            "input_tokens_norm", "output_tokens_norm",
            "runtime_seconds_norm", "energy_joules_norm"
        ])

        rt_params, rt_r2 = fit_regression(df, "runtime_seconds_norm")
        en_params, en_r2 = fit_regression(df, "energy_joules_norm")

        rows.append({
            "model": model_name,
            "runtime_const": rt_params["const"],
            "runtime_in_tokens": rt_params["input_tokens_norm"],
            "runtime_out_tokens": rt_params["output_tokens_norm"],
            "runtime_R2": rt_r2,
            "energy_const": en_params["const"],
            "energy_in_tokens": en_params["input_tokens_norm"],
            "energy_out_tokens": en_params["output_tokens_norm"],
            "energy_R2": en_r2
        })

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Regression summary saved → {OUTPUT_FILE}\n")
print(summary)
