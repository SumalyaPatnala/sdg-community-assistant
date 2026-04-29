import argparse
import json
from pathlib import Path

def load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    parser = argparse.ArgumentParser(description="Promote verified candidate to trusted case library.")
    parser.add_argument("--candidate-file", default="data/candidate_cases.jsonl")
    parser.add_argument("--trusted-file", default="../01_sdg_case_based_assistant/data/trusted_case_library.json")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()

    candidates = load_jsonl(args.candidate_file)
    selected = next((c for c in candidates if c.get("case_id") == args.case_id), None)

    if selected is None:
        raise ValueError(f"Case ID not found: {args.case_id}")

    if selected.get("verification_status") != "verified":
        raise ValueError("Only verified cases can be promoted. Re-ingest with --cross-verified after manual validation.")

    trusted_path = Path(args.trusted_file)
    if trusted_path.exists():
        with open(trusted_path, "r", encoding="utf-8") as f:
            trusted_cases = json.load(f)
    else:
        trusted_cases = []

    trusted_case = {
        "case_id": selected["case_id"].replace("candidate_", "trusted_"),
        "source_name": selected.get("source_title", ""),
        "source_type": selected.get("source_type", ""),
        "source_url": selected.get("source_url", ""),
        "trust_level": selected.get("trust_level", "Medium"),
        "country": selected.get("country", ""),
        "region": selected.get("region", ""),
        "community_problem": selected.get("community_problem", ""),
        "problem_category": [],
        "sdgs": selected.get("sdgs", []),
        "intervention": [selected.get("solution_summary", "")],
        "recommended_transfer": selected.get("recommended_transfer", []),
        "limitations": selected.get("limitations", ""),
        "last_verified": selected.get("created_at", ""),
    }

    trusted_cases.append(trusted_case)
    trusted_path.parent.mkdir(parents=True, exist_ok=True)

    with open(trusted_path, "w", encoding="utf-8") as f:
        json.dump(trusted_cases, f, indent=2, ensure_ascii=False)

    print(f"Promoted {args.case_id} to trusted library: {trusted_path}")

if __name__ == "__main__":
    main()
