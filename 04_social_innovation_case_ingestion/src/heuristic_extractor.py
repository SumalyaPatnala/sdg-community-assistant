SDG_KEYWORDS = {
    "SDG 3": ["health", "fever", "disease", "clinic", "doctor", "diarrhea", "nutrition", "hygiene"],
    "SDG 4": ["school", "education", "students", "girls", "learning", "teacher"],
    "SDG 5": ["women", "girls", "gender", "menstrual", "safety"],
    "SDG 6": ["water", "sanitation", "toilet", "drainage", "hygiene", "wash", "sewage"],
    "SDG 7": ["energy", "solar", "electricity", "clean cooking"],
    "SDG 8": ["jobs", "workers", "livelihood", "income", "employment"],
    "SDG 11": ["community", "village", "city", "housing", "waste", "transport", "settlement"],
    "SDG 12": ["waste", "recycling", "plastic", "reuse", "compost"],
    "SDG 13": ["climate", "flood", "drought", "rainfall", "disaster"],
}

def infer_sdgs(text: str) -> list[str]:
    text_l = text.lower()
    matched = []
    for sdg, keywords in SDG_KEYWORDS.items():
        if any(k in text_l for k in keywords):
            matched.append(sdg)
    return sorted(set(matched))

def extract_problem_and_solution(text: str) -> tuple[str, str]:
    text = " ".join(text.split())
    if not text:
        return "Not enough information provided.", "Solution details require manual review."
    problem = text[:350]
    solution = text[350:850] if len(text) > 350 else "Solution details require manual review."
    return problem, solution

def infer_recommended_transfer(text: str) -> list[str]:
    text_l = text.lower()
    actions = []
    if "water" in text_l or "diarrhea" in text_l:
        actions.extend(["Check water source safety", "Promote boiling/filtering", "Use covered storage"])
    if "waste" in text_l or "plastic" in text_l:
        actions.extend(["Create local collection points", "Promote segregation", "Coordinate with local authorities"])
    if "school" in text_l or "girls" in text_l:
        actions.extend(["Assess school sanitation access", "Engage teachers and parents", "Create safe reporting channel"])
    if "health" in text_l or "fever" in text_l:
        actions.extend(["Track affected households", "Contact qualified health workers", "Escalate severe symptoms"])
    if not actions:
        actions = ["Validate the project locally", "Identify responsible stakeholders", "Run a small pilot"]
    return list(dict.fromkeys(actions))
