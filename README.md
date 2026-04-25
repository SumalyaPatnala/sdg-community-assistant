# SDG Community Assistant + LLM Systems

This repository contains three connected GenAI projects focused on SLM fine-tuning, LLMOps deployment, and SDG case-based AI assistance.

## Projects

| Project | Description | Link |
|---|---|---|
| Production SLM Fine-Tuning and Evaluation Framework | LoRA/QLoRA fine-tuning and evaluation for structured SDG reasoning and extraction | [Open Project](./03_slm_finetuning_evaluation) |
| LLMOps Platform for Local and Hybrid Inference | FastAPI-based local/remote model serving with routing, observability, latency logging, and monitoring | [Open Project](./02_llmops_local_hybrid_inference) |
| SDG Case-Based Community Assistant | Public assistant mapping community problems to SDGs and trusted global solution cases | [Open Project](./01_sdg_case_based_assistant) |

## System Flow

```text
03_slm_finetuning_evaluation
        ↓
Fine-tuned / evaluated SLM behavior

02_llmops_local_hybrid_inference
        ↓
Serves models locally or through hosted endpoints

01_sdg_case_based_assistant
        ↓
Uses model inference + trusted case retrieval to generate SDG action guidance