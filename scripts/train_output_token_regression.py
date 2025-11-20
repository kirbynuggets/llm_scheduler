import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from pathlib import Path

# =====================================
# Load profiling data from all models
# =====================================

csv_paths = [
    "metrics_falcon-7b-instruct.csv",
    "metrics_mistral-7B-instruct-v0.2.csv",
    "metrics_phi-3-mini-4k-instruct.csv"
]

df_list = []
for p in csv_paths:
    df = pd.read_csv(p)

    # ensure required columns exist
    if {"input_tokens", "output_tokens"} <= set(df.columns):
        df_list.append(df)
    else:
        print(f"⚠️ Skipping {p}: missing required columns")

data = pd.concat(df_list, axis=0)
print(f"Loaded {len(data)} rows for regression training.")

# =====================================
# Prepare data for regression
# =====================================

X = data["input_tokens"].values.reshape(-1, 1)
y = data["output_tokens"].values.reshape(-1, 1)

# =====================================
# Train Simple Linear Regression
# =====================================

model = LinearRegression()
model.fit(X, y)

alpha = float(model.coef_[0][0])
beta = float(model.intercept_[0])

print("\n📌 Trained Regression Model:")
print(f"α (slope)     = {alpha}")
print(f"β (intercept) = {beta}")

# =====================================
# Save parameters
# =====================================

out = {"alpha": alpha, "beta": beta}
Path("data").mkdir(exist_ok=True)

with open("data/output_token_regression.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n✅ Saved -> data/output_token_regression.json")