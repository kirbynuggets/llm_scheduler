import pandas as pd
import glob
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression

def clean(name: str):
    return (
        name.lower()
        .replace("metrics_", "")
        .replace(".csv", "")
        .strip()
    )

def main():
    files = glob.glob("models/metrics_*.csv")
    if not files:
        raise FileNotFoundError("No metrics_*.csv files found in models/ folder.")

    output = {}

    for f in files:
        model_name = clean(Path(f).name)
        df = pd.read_csv(f)

        # Ensure required columns exist
        required = {"input_tokens", "output_tokens", "runtime_seconds", "energy_joules", "tin_tout"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing required columns in {f}")

        X = df[["input_tokens", "output_tokens", "tin_tout"]]
        X.insert(0, "const", 1.0)

        reg_runtime = LinearRegression().fit(X, df["runtime_seconds"])
        reg_energy = LinearRegression().fit(X, df["energy_joules"])

        output[model_name] = {
            "runtime": dict(zip(X.columns, reg_runtime.coef_,)),
            "energy": dict(zip(X.columns, reg_energy.coef_,))
        }

        # Fix intercept placement (coef_ does not include intercept)
        output[model_name]["runtime"]["const"] = float(reg_runtime.intercept_)
        output[model_name]["energy"]["const"] = float(reg_energy.intercept_)

        print(f"✔ Trained model regression for {model_name}")

    Path("data").mkdir(exist_ok=True)
    with open("data/model_runtime_energy_regression.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\nSaved → data/model_runtime_energy_regression.json")

if __name__ == "__main__":
    main()
