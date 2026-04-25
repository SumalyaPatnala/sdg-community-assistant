# Production SLM Fine-Tuning and Evaluation Framework

This project demonstrates a production-style framework for fine-tuning and evaluating a compact language model for SDG/community reasoning and structured extraction.

## What it includes

- LoRA/QLoRA fine-tuning script
- SDG instruction dataset format
- Validation split
- Evaluation script for latency and SDG hit-rate
- Config-driven training
- Hugging Face compatible workflow

## Project goal

Fine-tune a compact SLM to produce structured outputs for:

- community problem understanding
- SDG mapping
- safe action recommendations
- structured extraction
- reliability-oriented responses

## Folder structure

```text
03_slm_finetuning_evaluation/
├── train_lora.py
├── evaluate.py
├── requirements.txt
├── configs/
│   └── lora_config.yaml
├── data/
│   ├── train.jsonl
│   └── valid.jsonl
├── src/
│   ├── data_utils.py
│   └── prompts.py
└── outputs/
```

## Install

```bash
pip install -r requirements.txt
```

## Train LoRA adapter

```bash
python train_lora.py --config configs/lora_config.yaml
```

## Evaluate base model

```bash
python evaluate.py --model microsoft/Phi-3-mini-4k-instruct
```

## Evaluate adapted model

```bash
python evaluate.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --adapter outputs/sdg_slm_lora
```

## Metrics demonstrated

- SDG hit-rate
- latency
- output reliability checks
- base vs adapted model comparison

## Production relevance

This framework supports the resume claim:

> Fine-tuned a compact language model for structured reasoning and extraction across SDG and enterprise datasets. Evaluated base and adapted models across accuracy, latency, memory, inference efficiency, and reliability metrics.

## Notes

For Mac local training, MLX may be easier than bitsandbytes.  
For free GPU experiments, use Kaggle or Colab.
