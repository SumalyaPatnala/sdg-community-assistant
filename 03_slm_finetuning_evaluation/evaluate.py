import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import build_eval_prompt


TEST_CASES = [
    {
        "problem": "There is stagnant water near homes, mosquitoes everywhere, and children are getting fever.",
        "expected_sdgs": ["SDG 3", "SDG 6", "SDG 11"]
    },
    {
        "problem": "People drink untreated water and many households report diarrhea.",
        "expected_sdgs": ["SDG 3", "SDG 6"]
    },
    {
        "problem": "Girls are missing school because there are no usable toilets.",
        "expected_sdgs": ["SDG 4", "SDG 5", "SDG 6"]
    }
]


def load_model(model_name: str, adapter_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    latency = round((time.time() - start) * 1000, 2)

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text, latency


def sdg_hit_rate(output: str, expected_sdgs):
    output_lower = output.lower()
    hits = sum(1 for sdg in expected_sdgs if sdg.lower() in output_lower)
    return hits / len(expected_sdgs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default="outputs/eval_results.csv")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapter)

    rows = []
    for case in TEST_CASES:
        prompt = build_eval_prompt(case["problem"])
        output, latency_ms = generate(model, tokenizer, prompt)

        rows.append({
            "problem": case["problem"],
            "expected_sdgs": ", ".join(case["expected_sdgs"]),
            "sdg_hit_rate": sdg_hit_rate(output, case["expected_sdgs"]),
            "latency_ms": latency_ms,
            "output": output
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(df[["problem", "expected_sdgs", "sdg_hit_rate", "latency_ms"]])
    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
