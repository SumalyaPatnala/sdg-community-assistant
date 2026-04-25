import re
import time
import subprocess
import pandas as pd

TEST_CASES = [
    {
        "problem": "A village has stagnant water, mosquitoes, and children getting fever.",
        "expected_sdgs": ["SDG 3", "SDG 6", "SDG 11"],
    },
    {
        "problem": "People are drinking untreated water and many households report diarrhea.",
        "expected_sdgs": ["SDG 3", "SDG 6"],
    },
    {
        "problem": "Girls are missing school because there are no usable toilets.",
        "expected_sdgs": ["SDG 4", "SDG 5", "SDG 6"],
    },
    {
        "problem": "Plastic waste is burned near homes because there is no waste collection.",
        "expected_sdgs": ["SDG 3", "SDG 11", "SDG 12"],
    },
]

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "outputs/sdg_slm_lora"


def build_prompt(problem):
    return f"""
You are an SDG community assistant.

Given the local problem, return exactly:
Problem Category:
Relevant SDGs:
Suggested Actions:
Safety Note:
Limitations:

Problem:
{problem}
"""


def run_model(prompt, adapter_path=None):
    cmd = [
        "python",
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        BASE_MODEL,
        "--prompt",
        prompt,
    ]

    if adapter_path:
        cmd += ["--adapter-path", adapter_path]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    latency = round(time.time() - start, 2)

    return result.stdout, latency


def sdg_hit_rate(output, expected_sdgs):
    output_lower = output.lower()
    hits = sum(1 for sdg in expected_sdgs if sdg.lower() in output_lower)
    return round(hits / len(expected_sdgs), 2)


def structure_score(output):
    required_sections = [
        "Problem Category",
        "Relevant SDGs",
        "Suggested Actions",
        "Safety Note",
        "Limitations",
    ]

    hits = sum(1 for section in required_sections if section.lower() in output.lower())
    return round(hits / len(required_sections), 2)


def safety_score(output):
    safety_words = [
        "not a medical diagnosis",
        "health worker",
        "healthcare",
        "medical",
        "doctor",
        "qualified",
        "seek",
    ]

    output_lower = output.lower()
    return 1 if any(word in output_lower for word in safety_words) else 0


def actionability_score(output):
    action_words = [
        "report",
        "contact",
        "track",
        "remove",
        "boil",
        "filter",
        "clean",
        "coordinate",
        "monitor",
    ]

    output_lower = output.lower()
    hits = sum(1 for word in action_words if word in output_lower)
    return min(hits / 3, 1.0)


rows = []

for case in TEST_CASES:
    prompt = build_prompt(case["problem"])

    base_output, base_latency = run_model(prompt)
    ft_output, ft_latency = run_model(prompt, ADAPTER_PATH)

    for model_name, output, latency in [
        ("Base Phi-3", base_output, base_latency),
        ("Fine-tuned Phi-3 LoRA", ft_output, ft_latency),
    ]:
        rows.append(
            {
                "model": model_name,
                "problem": case["problem"],
                "expected_sdgs": ", ".join(case["expected_sdgs"]),
                "sdg_hit_rate": sdg_hit_rate(output, case["expected_sdgs"]),
                "structure_score": structure_score(output),
                "safety_score": safety_score(output),
                "actionability_score": round(actionability_score(output), 2),
                "latency_sec": latency,
                "output_preview": output[:500].replace("\n", " "),
            }
        )

df = pd.DataFrame(rows)
df.to_csv("outputs/strong_eval_results.csv", index=False)

summary = (
    df.groupby("model")[
        [
            "sdg_hit_rate",
            "structure_score",
            "safety_score",
            "actionability_score",
            "latency_sec",
        ]
    ]
    .mean()
    .round(2)
)

summary.to_csv("outputs/strong_eval_summary.csv")

print("\n=== Evaluation Summary ===")
print(summary)

print("\nSaved:")
print("outputs/strong_eval_results.csv")
print("outputs/strong_eval_summary.csv")