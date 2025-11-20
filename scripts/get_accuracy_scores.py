import os
import pandas as pd

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    print(" Hugging Face Hub not available — using manual fallback scores.")

# -------------------------------------------------------------
# MANUAL FALLBACK ACCURACIES
# (values roughly reflect average leaderboard correctness)
# -------------------------------------------------------------
fallback_scores = {
    "microsoft/Phi-3-mini-4k-instruct": 0.685,   
    "mistralai/Mistral-7B-Instruct-v0.1": 0.723, 
    "tiiuae/Falcon-7B-Instruct": 0.705          
}

# -------------------------------------------------------------
# TRY FETCHING FROM HUGGING FACE LEADERBOARD
# -------------------------------------------------------------
def get_accuracy_from_hf(model_id: str) -> float:
    api = HfApi()
    info = api.model_info(model_id)
    card = info.cardData or {}
    for key in ["mmlu", "eval_accuracy", "average_score"]:
        if key in card:
            val = card[key]
            if isinstance(val, (float, int)):
                return val / 100 if val > 1 else val
    raise ValueError("No accuracy field found.")

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    models = list(fallback_scores.keys())
    results = []

    for model_id in models:
        print(f" Getting accuracy for {model_id} ...")
        if HF_AVAILABLE:
            try:
                acc = get_accuracy_from_hf(model_id)
                print(f" Found on HF: {acc*100:.2f}%")
            except Exception as e:
                acc = fallback_scores[model_id]
                print(f" Using fallback {acc*100:.2f}% ({e})")
        else:
            acc = fallback_scores[model_id]

        results.append({"model": model_id, "accuracy": acc})

    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    out_path = "data/model_accuracy_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"\n Accuracy table saved to {out_path}\n")
    print(df)

if __name__ == "__main__":
    main()