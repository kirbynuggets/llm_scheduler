import pandas as pd
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
import glob


def main():
    # ----------------------------------------------------
    # 1. Find all metrics CSVs
    # ----------------------------------------------------
    # Adjust the pattern if your CSVs live in another folder
    csv_paths = list(Path("models").glob("metrics_*.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            "No metrics_*.csv files found in the project root. "
            "Make sure your profiling files are named like 'metrics_falcon-7b-instruct.csv'."
        )

    all_rows = []

    for path in csv_paths:
        df = pd.read_csv(path)

        # We only need these two columns
        if not {"input_tokens", "output_tokens"}.issubset(df.columns):
            raise ValueError(
                f"{path} is missing 'input_tokens' or 'output_tokens' columns. "
                f"Found columns: {list(df.columns)}"
            )

        # Drop rows with NaNs just to be safe
        sub = df[["input_tokens", "output_tokens"]].dropna()
        all_rows.append(sub)

    data = pd.concat(all_rows, ignore_index=True)

    if data.empty:
        raise ValueError("No valid rows found across metrics_*.csv files.")

    print(f"Loaded {len(data)} samples from {len(csv_paths)} metrics files.")

    # ----------------------------------------------------
    # 2. Prepare X, y for regression
    # ----------------------------------------------------
    X = data["input_tokens"].values.reshape(-1, 1)
    y = data["output_tokens"].values.reshape(-1, 1)

    # ----------------------------------------------------
    # 3. Fit linear regression
    # ----------------------------------------------------
    model = LinearRegression()
    model.fit(X, y)

    alpha = float(model.coef_[0][0])      # slope
    beta = float(model.intercept_[0])     # intercept

    print("\nTrained output-token regression:")
    print(f"alpha (slope)     = {alpha:.6f}")
    print(f"beta  (intercept) = {beta:.6f}")

    # ----------------------------------------------------
    # 4. Save to JSON
    # ----------------------------------------------------
    Path("data").mkdir(exist_ok=True)

    out_path = Path("data/output_token_regression.json")
    with out_path.open("w") as f:
        json.dump({"alpha": alpha, "beta": beta}, f, indent=2)

    print(f"\nSaved regression parameters to {out_path.resolve()}")


if __name__ == "__main__":
    main()