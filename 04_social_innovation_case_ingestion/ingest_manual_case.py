import argparse
import json
from pathlib import Path

from src.case_schema import build_candidate_case
from src.heuristic_extractor import infer_sdgs, extract_problem_and_solution, infer_recommended_transfer

def main():
    parser = argparse.ArgumentParser(description="Create a candidate social innovation case from URL + notes.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--country", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--cross-verified", action="store_true")
    parser.add_argument("--out", default="data/candidate_cases.jsonl")
    args = parser.parse_args()

    sdgs = infer_sdgs(args.notes)
    problem, solution = extract_problem_and_solution(args.notes)
    actions = infer_recommended_transfer(args.notes)

    case = build_candidate_case(
        source_url=args.url,
        title=args.title,
        raw_description=args.notes,
        community_problem=problem,
        solution_summary=solution,
        sdgs=sdgs,
        country=args.country,
        region=args.region,
        recommended_transfer=actions,
        cross_verified=args.cross_verified,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(json.dumps(case, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
