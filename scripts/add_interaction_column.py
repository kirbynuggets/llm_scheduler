import pandas as pd
from pathlib import Path

def main():
    root = Path(".")
    metric_files = list(root.glob("./models/metrics_*.csv"))

    if not metric_files:
        print("No metrics_*.csv files found in the project root.")
        return

    print(f"Found {len(metric_files)} metrics files:")
    for f in metric_files:
        print("  -", f.name)

    for path in metric_files:
        print(f"\nProcessing {path.name} ...")
        df = pd.read_csv(path)

        # Basic sanity check
        required_cols = {"input_tokens", "output_tokens"}
        if not required_cols.issubset(df.columns):
            print(f"  ⚠️ Skipping {path.name}: missing required columns {required_cols}")
            continue

        # Compute interaction term tin * tout
        df["tin_tout"] = df["input_tokens"] * df["output_tokens"]

        # Save back to the same file
        df.to_csv(path, index=False)
        print(f"  ✅ Added 'tin_tout' column and saved {path.name}")

    print("\nDone adding interaction column to metrics files.")

if __name__ == "__main__":
    main()