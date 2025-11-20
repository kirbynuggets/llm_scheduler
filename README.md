# **Dynamic LLM Inference Scheduler**

*A data-driven system that automatically selects the best Large Language Model for a given prompt using hybrid normalization, regression-based cost prediction, and multi-objective scoring.*

This project builds an adaptive model-selection engine that chooses between multiple LLMs (Phi-3, Falcon-7B, Mistral-7B) based on predicted runtime, energy usage, and output quality.
Instead of routing every user prompt to a single large model, the system predicts cost–accuracy trade-offs and selects the optimal model automatically.

The goal is to make LLM inference cheaper, faster, and more intelligent—especially in multi-model deployment environments.

---

# **Project Overview**

Modern LLM applications often rely on a single large model for all queries.
This creates unnecessary latency, cost and energy waste.

This project provides a full, end-to-end **Dynamic LLM Scheduler** that:

1. Profiles each model’s behavior
2. Learns runtime and energy patterns using OLS regression
3. Normalizes cost and accuracy metrics through a multi-stage hybrid pipeline
4. Scores every model using user-defined priorities
5. Selects the best model with a deterministic, reproducible algorithm

Everything here is built from scratch and modular, so new models can be plugged in with zero friction.

---

# **Key Innovations**

### **1. Multi-Metric Profiling Layer**

Each LLM is benchmarked for:

• Input & output token counts
• Runtime
• Measured/estimated energy usage

This creates a reproducible numeric profile per model.

---

### **2. Hybrid Normalization Pipeline**

Because model metrics differ dramatically in scale, the system applies a 5-stage normalization method:

1. Z-score
2. Log transformation
3. Min-max scaling
4. Softmax amplification
5. Variance rebasing

This ensures stable comparisons across different LLMs and hardware environments.

---

### **3. Regression-Based Runtime & Energy Prediction**

The system trains Ordinary Least Squares (OLS) models of the form:

```
runtime = α0 + α1 * input_tokens + α2 * output_tokens  
energy  = β0 + β1 * input_tokens + β2 * output_tokens
```

This allows the scheduler to *predict cost before running the model*.

---

### **4. Weighted Multi-Objective Scoring**

Users control the trade-off:

```
--weight_accuracy
--weight_energy
--weight_time
```

This means the scheduler adapts to:

• High-accuracy tasks
• Low-energy deployments
• Latency-sensitive environments
• Balanced inference pipelines

The final score is:

```
score =
  w_acc * accuracy_score +
  w_energy * energy_score +
  w_time * runtime_score
```

Lower score = better model.

---

### **5. Visualization & Diagnostics**

The project includes scripts to visualize:

• Trade-offs between accuracy, energy, runtime
• Score changes across multiple runs
• Model dominance regions

These visuals are helpful for debugging and for explaining model routes in interviews.

---

# **System Architecture**

### **1. Profiling**

Benchmark each LLM on controlled inputs to build a profile dataset.

### **2. Normalization**

Apply the hybrid pipeline to convert raw numbers into comparable metrics.

### **3. Regression**

Fit OLS models to estimate per-token cost.

### **4. Scheduler**

Combine predictions + accuracy scores + user weights to pick the optimal model.

---

# **Tech Stack**

### **Languages**

Python 3.10+

### **Libraries**

NumPy, Pandas, Statsmodels, PyTorch, Transformers, Matplotlib

### **System Requirements**

Linux recommended (deterministic inference), virtualenv for isolation.

---

# **Project Structure**

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
└── README.md
```

---

# **Quick Start**

### Install

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Generate profiling data

```
python scripts/build_model_profile.py
```

### Train OLS regression

```
python scripts/run_regression_ols_hybrid.py
```

### Run scheduler

```
python scripts/llm_fair_ols.py \
  --prompt "Explain reinforcement learning" \
  --weight_accuracy 0.7 \
  --weight_energy 0.2 \
  --weight_time 0.1
```

---

# **Future Enhancements**

• Add quantized models (INT4, GGUF)
• Real hardware power-draw logging
• Reinforcement-learning-based scheduler
• GPU-aware routing
• Prompt-type classification & custom routing rules
• REST API server for production use

---

# **License**

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

Under this license:

• Anyone can use, study, modify, and share the project
• Any redistributed version must remain open-source
• Any modified version must also be open-sourced under GPL-3.0
• Commercial use is allowed, but only if derivatives stay open-source

The LICENSE file contains the full text of GPL-3.0.
