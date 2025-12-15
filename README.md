# Energy, Time, and Correctness-Aware LLM Scheduler

A regression-driven, performance-aware system for dynamically routing LLM queries across heterogeneous models.

## Overview

Large Language Models (LLMs) vary drastically in accuracy, latency, and energy consumption.
Yet most real-world deployments still use a single large model for every query, leading to unnecessary cost, latency, and power waste.

This project implements an online, performance-aware LLM scheduler that dynamically selects the best model for each prompt by explicitly trading off:

- **Correctness** (accuracy)
- **Inference latency**
- **Energy consumption**

Rather than guessing or load-balancing blindly, the scheduler predicts runtime and energy before execution using regression models trained from real GPU profiling data, then selects the optimal model using a weighted cost function.

This work was developed as part of a B.Tech dissertation and builds on workload-based energy modeling methodologies from recent systems research.

## Why This Matters

> Using a large model for every query is like using a cargo ship to deliver pizza.

- Simple prompts don't need massive models
- Complex reasoning deserves better accuracy
- Mobile, edge, and sustainable deployments demand efficiency

This scheduler enables intelligent, user-controlled routing across multiple LLMs—without sacrificing predictability or transparency.

## Key Contributions

- Real-time LLM scheduling using regression-based cost prediction
- Joint optimization of accuracy, energy, and latency
- Explicit modeling of attention complexity via token interaction terms
- User-driven trade-offs using weighted objectives
- Reproducible and deterministic routing decisions

## System Architecture

The system is divided into two clean phases:

### 1. Offline Profiling & Modeling

A one-time process that characterizes model behavior.

- Run each model on thousands of prompts
- Measure:
  - Runtime (seconds)
  - Energy consumption (Joules)
  - Input & output token counts
- Train OLS regression models to predict cost

### 2. Online Scheduling

A lightweight, real-time decision engine.

- Estimate token counts for incoming prompt
- Predict runtime & energy for each model
- Combine predictions with accuracy scores
- Select the model with the lowest weighted cost

Scheduler overhead is under 1 ms, making it safe for real-time systems.

## Models Evaluated

| Model | Parameters | Strength |
|-------|-----------|----------|
| Phi-3 Mini | 3.8B | Energy & latency efficiency |
| Mistral-7B | 7B | High accuracy per watt |
| Falcon-7B | 7B | Legacy dense architecture (baseline) |

## Mathematical Modeling

### Regression-Based Cost Prediction

Each model learns two linear predictors:

**Runtime**
```
T = β₀ + β₁·τ_in + β₂·τ_out + β₃·(τ_in × τ_out)
```

**Energy**
```
E = α₀ + α₁·τ_in + α₂·τ_out + α₃·(τ_in × τ_out)
```

The interaction term captures the quadratic scaling of attention in Transformers.

### Output Token Estimation

Since output length is unknown before execution, it is estimated as:
```
τ_out ≈ 1.028 · τ_in + 44.12
```

This heuristic is derived empirically and is sufficient for relative ranking.

### Fair-OLS Scheduling Algorithm

Because accuracy, energy, and time exist on wildly different scales, the scheduler first normalizes all predicted costs:
```
x_norm = (x − min(x)) / (max(x) − min(x) + ε)
```

Accuracy is inverted to form a minimization objective:
```
Accuracy Cost = 1 − Accuracy
```

**Final Score**
```
Score =
  w_acc   · AccuracyCost +
  w_energy· EnergyNorm +
  w_time  · TimeNorm
```

The model with the lowest score is selected.

## What Makes This Different

❌ No black-box heuristics  
❌ No hard-coded thresholds  
❌ No static routing rules

Every decision is:
- ✅ Predictive
- ✅ Explainable
- ✅ Reproducible
- ✅ Tunable by the user

## Project Structure
```
.
├── data/
│   ├── model_regression_summary.csv
│   ├── model_accuracy_scores.csv
│   └── raw_profiles/
├── scripts/
│   ├── build_model_profile.py
│   ├── get_accuracy_scores.py
│   ├── run_regression_ols_hybrid.py
│   ├── llm_fair_ols.py
│   ├── llm_scheduler_hybrid.py
│   └── hybrid_normalize.py
├── plots/
├── requirements.txt
└── README.md
```

## Quick Start

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Profile Models
```bash
python scripts/build_model_profile.py
```

### Step 2: Train Regression Models
```bash
python scripts/run_regression_ols_hybrid.py
```

### Step 3: Run the Scheduler
```bash
python scripts/llm_fair_ols.py \
  --prompt "Explain reinforcement learning" \
  --weight_accuracy 0.7 \
  --weight_energy 0.2 \
  --weight_time 0.1
```

## Example Scheduling Behavior

- **Energy-priority user** → Phi-3 Mini
- **Accuracy-priority user** → Mistral-7B
- **Latency-priority user** → Phi-3 Mini
- **Balanced user** → Dynamic selection based on prompt

Falcon-7B is often excluded due to Pareto inefficiency—higher energy with lower accuracy than Mistral.

## Visualization & Analysis

The project includes scripts to generate:

- Accuracy vs Energy trade-off plots
- Scheduler sensitivity curves
- Model dominance regions
- Console-level routing diagnostics

These are especially useful for:
- Debugging
- Research validation
- Interview explanations

## Limitations

- Output token estimation is heuristic
- Energy measurement relies on NVML polling
- Accuracy scores are static (leaderboard-based)

All of these are explicit design trade-offs, not hidden assumptions.

## Future Enhancements

- Quantized & GGUF models
- Reinforcement-learning-based scheduler
- GPU-aware and multi-node routing
- Prompt-type classification
- REST API for production deployment
- Real hardware power meters

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

---

**Built with rigor. Optimized for reality.**
