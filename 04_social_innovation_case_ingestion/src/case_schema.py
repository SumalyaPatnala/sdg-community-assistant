from datetime import datetime
from urllib.parse import urlparse

TRUSTED_DOMAINS = [
    "who.int", "unicef.org", "worldbank.org", "undp.org",
    "sdgs.un.org", "wateraid.org", "gov.in", "niti.gov.in", "mospi.gov.in"
]

def detect_source_type(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if "youtube.com" in domain or "youtu.be" in domain:
        return "social_video"
    if any(d in domain for d in TRUSTED_DOMAINS):
        return "trusted_institutional_source"
    return "web_source"

def assign_trust_level(url: str, cross_verified: bool = False) -> str:
    domain = urlparse(url).netloc.lower()
    if any(d in domain for d in TRUSTED_DOMAINS):
        return "High"
    if cross_verified:
        return "Medium"
    return "Candidate"

def build_candidate_case(
    source_url: str,
    title: str,
    raw_description: str,
    community_problem: str,
    solution_summary: str,
    sdgs: list,
    country: str = "",
    region: str = "",
    beneficiaries: list | None = None,
    recommended_transfer: list | None = None,
    cross_verified: bool = False,
):
    return {
        "case_id": f"candidate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "source_type": detect_source_type(source_url),
        "source_url": source_url,
        "source_title": title,
        "raw_description": raw_description,
        "verification_status": "verified" if cross_verified else "needs_review",
        "trust_level": assign_trust_level(source_url, cross_verified),
        "country": country,
        "region": region,
        "community_problem": community_problem,
        "solution_summary": solution_summary,
        "sdgs": sdgs,
        "beneficiaries": beneficiaries or [],
        "recommended_transfer": recommended_transfer or [],
        "limitations": (
            "Candidate social innovation case. Validate with additional trusted sources, "
            "local experts, or institutional documentation before adding to trusted library."
        ),
        "created_at": datetime.utcnow().isoformat(),
    }
