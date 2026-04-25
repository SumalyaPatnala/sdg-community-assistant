import subprocess

test_cases = [
    "A village has stagnant water and mosquitoes, children getting fever",
    "People drinking unsafe water, diarrhea cases increasing",
    "No toilets in school, girls missing classes",
]

def run_model(prompt, adapter=None):
    cmd = [
        "python", "-m", "mlx_lm.generate",
        "--model", "microsoft/Phi-3-mini-4k-instruct",
        "--prompt", prompt
    ]

    if adapter:
        cmd += ["--adapter-path", adapter]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def score(output):
    # simple SDG keyword check
    score = 0
    if "SDG 3" in output: score += 1
    if "SDG 6" in output: score += 1
    if "SDG" in output: score += 1
    return score


print("\n=== Evaluation ===\n")

for case in test_cases:
    print(f"\nProblem: {case}")

    base_out = run_model(case)
    ft_out = run_model(case, "outputs/sdg_slm_lora")

    base_score = score(base_out)
    ft_score = score(ft_out)

    print(f"Base Score: {base_score}")
    print(f"Fine-tuned Score: {ft_score}")