import os

block_keys = [
    "FLASH_ATTENTION_FORCE_DISABLED",
    "FLASH_ATTENTION_SKIP_CHECKS",
    "USE_FLASH_ATTENTION",
    "USE_FLASH_ATTENTION_2",
    "ATTN_IMPL",
]
for k in block_keys:
    os.environ[k] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

os.environ["DISABLE_TRITON"] = "1"

# ----- IMPORTS -----
import time
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ----- ENERGY LOGGER -----
try:
    from gpu_profiler import GPUPowerLogger
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False
    print("⚠️ Power logging disabled (NVML not available).")

# ----- CLI ARGS -----
parser = argparse.ArgumentParser(description="LLM Energy/Time Profiler (no-FlashAttention build)")
parser.add_argument("--model", required=True, help="Model name or HuggingFace ID")
parser.add_argument("--gpu", type=int, default=0, help="GPU index")
parser.add_argument("--tokens", type=int, default=40, help="Max new tokens")
parser.add_argument("--interval", type=float, default=0.5, help="Energy-sampling interval")
args = parser.parse_args()

MODEL_ID = args.model
GPU_INDEX = args.gpu
TOKENS = args.tokens
INTERVAL = args.interval

# ----- CHECK CUDA -----
print("\n📦 Environment ready.")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🧩 GPU in use: {torch.cuda.get_device_name(GPU_INDEX)}")

# ----- LOAD DATASET -----
print("\n📚 Loading Alpaca dataset...")
ds = load_dataset("tatsu-lab/alpaca", split="train")
prompts = [ex["instruction"] for ex in ds]
print(f"Loaded {len(prompts)} prompts.\n")

# ----- LOAD MODEL SAFELY -----
print(f"🚀 Loading model with sequential offload: {MODEL_ID}")
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="sequential",       # one block at a time
    offload_folder="offload_cache",
    low_cpu_mem_usage=True,
    attn_implementation="eager"    # absolutely no flash
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()
torch.cuda.empty_cache()
print("✅ Model loaded successfully.\n")

# ----- CSV SETUP -----
os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", f"results_{MODEL_ID.split('/')[-1]}.csv")
header = "prompt_id,input_tokens,output_tokens,runtime_seconds,energy_joules\n"
if not os.path.exists(csv_path):
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header)

# ----- MAIN LOOP -----
print(f"⚙️ Profiling {len(prompts)} prompts...\n")

for i, prompt in enumerate(prompts, start=1):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        if torch.cuda.is_available():
            inputs = inputs.to(f"cuda:{GPU_INDEX}")
        input_tokens = len(inputs["input_ids"][0])

        # measure
        if NVML_AVAILABLE:
            logger = GPUPowerLogger(gpu_index=GPU_INDEX, interval=INTERVAL, log_to_csv=False)
            with logger:
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=TOKENS)
                t1 = time.time()
                logger.sample(t1 - t0)
            energy = logger.energy_joules
        else:
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=TOKENS)
            t1 = time.time()
            energy = 0.0

        runtime = t1 - t0
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        output_tokens = len(tokenizer(text)["input_ids"])

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{i},{input_tokens},{output_tokens},{runtime:.4f},{energy:.4f}\n")

        if i % 10 == 0:
            print(f"✅ {i} / {len(prompts)} done  |  last {runtime:.2f}s")

        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error on prompt {i}: {e}")
        torch.cuda.empty_cache()
        continue

print(f"\n🎯 Profiling finished for {MODEL_ID}")
print(f"📄 Results saved to {csv_path}\n")