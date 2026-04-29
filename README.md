# SDG Community Assistant + LLM Systems

An end-to-end GenAI system combining SLM fine-tuning, LLMOps model serving, and a real-time SDG-focused application for community problem solving.

---

## 🔍 Overview

This repository demonstrates a complete pipeline:

- Fine-tuning a compact language model (SLM) for domain-specific reasoning  
- Serving models via a production-style LLMOps platform  
- Consuming the model in a user-facing SDG assistant application  

---

## 📦 Projects

| Project | Description | Link |
|---|---|---|
| Production SLM Fine-Tuning and Evaluation Framework | LoRA-based fine-tuning and evaluation of SLMs for structured SDG reasoning | [Open Project](./03_slm_finetuning_evaluation) |
| LLMOps Platform for Local and Hybrid Inference | FastAPI-based model serving with routing, fallback, observability, and latency tracking | [Open Project](./02_llmops_local_hybrid_inference) |
| SDG Case-Based Community Assistant | Interactive app mapping community problems to SDGs with actionable recommendations | [Open Project](./01_sdg_case_based_assistant) |

---

## ⚙️ System Architecture

```text
User Input (Community Problem)
        ↓
SDG Assistant (Project 1 - Streamlit UI)
        ↓
LLMOps API (Project 2 - FastAPI)
        ↓
Fine-Tuned SLM (Project 3 - LoRA Adapter)
        ↓
Fallback → Local Model → Remote Model
        ↓
Structured SDG Response

## System Flow
03_slm_finetuning_evaluation
        ↓
Fine-tuned SLM (LoRA adapter)

02_llmops_local_hybrid_inference
        ↓
Model routing + serving + monitoring

01_sdg_case_based_assistant
        ↓
Real-time SDG problem solving UI