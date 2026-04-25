# Production SLM Fine-Tuning and Evaluation Framework

This project implements a production-style pipeline for fine-tuning and evaluating a compact language model (SLM) for SDG-driven community problem understanding, structured reasoning, and actionable recommendation generation.

---

## 🔍 Overview

This framework focuses on adapting a base model (Phi-3 Mini) using LoRA to improve:

- SDG mapping accuracy
- structured response generation
- community-level reasoning
- safe and actionable recommendations

---

## ⚙️ What it includes

- LoRA-based fine-tuning (MLX + Hugging Face compatible)
- SDG instruction dataset
- validation dataset split
- evaluation pipeline (custom SDG relevance scoring)
- base vs fine-tuned model comparison
- config-driven training setup

---

## 🎯 Project Goal

Fine-tune a compact SLM to produce structured outputs for:

- community problem understanding  
- SDG identification  
- actionable solution generation  
- structured response formatting  
- reliability-focused outputs  

---

## 📊 Evaluation Results

The base Phi-3 model and fine-tuned Phi-3 LoRA adapter were evaluated on community SDG problem scenarios.

| Model | SDG Hit Rate | Structure Score | Safety Score | Actionability Score | Latency |
|---|---:|---:|---:|---:|---:|
| Base Phi-3 | 0.75 | 0.60 | 0.00 | 0.50 | 15.02s |
| Fine-tuned Phi-3 LoRA | 0.92 | 1.00 | 0.50 | 0.84 | 14.58s |

The fine-tuned model improved SDG mapping, structured response compliance, safety-warning behavior, and actionability while maintaining similar latency.

## 🧪 Evaluation Method

We evaluated the base and fine-tuned models using:

- **SDG Hit Rate:** whether expected SDGs were correctly identified
- **Structure Score:** whether the response followed the required format
- **Safety Score:** whether health/safety warnings were included when relevant
- **Actionability Score:** whether the response included practical next steps
- **Latency:** end-to-end generation time per test case

---

## 🧠 Training Approach

### Option 1: MLX (Recommended for Mac)

```bash
python -m mlx_lm lora \
  --model microsoft/Phi-3-mini-4k-instruct \
  --train \
  --data data \
  --iters 100 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --adapter-path outputs/sdg_slm_lora
  ```