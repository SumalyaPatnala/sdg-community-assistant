# Social Innovation Case Ingestion Pipeline

This module extends the SDG assistant with a human-in-the-loop pipeline for adding new social innovation examples from videos, NGO pages, public articles, or institutional reports.

## Workflow

```text
Source URL / Video / Article
        ↓
Candidate case extraction
        ↓
Trust scoring
        ↓
Human review / cross-verification
        ↓
Promote to trusted case library
        ↓
FAISS retrieval in SDG assistant
```

## Create a candidate case

```bash
python ingest_manual_case.py \
  --url "https://m.youtube.com/shorts/OvZAhfCDeUk" \
  --title "Community social innovation example" \
  --notes "Paste transcript or your description here. Example: villagers created a low-cost drainage and waste solution to reduce stagnant water and improve hygiene." \
  --country "India" \
  --region "Local community"
```

## Create a verified candidate

Use this only when the example is cross-verified with an NGO, government report, article, or institutional source.

```bash
python ingest_manual_case.py \
  --url "https://example.org/verified-project" \
  --title "Verified WASH social innovation project" \
  --notes "Project summary..." \
  --country "India" \
  --region "Rural community" \
  --cross-verified
```

## Promote a verified case

```bash
python promote_to_trusted.py --case-id candidate_YYYYMMDD_HHMMSS
```

## Paper framing

The system includes a human-in-the-loop case-ingestion pipeline that converts newly discovered social innovation examples into structured candidate cases, assigns verification status and trust level, and promotes only reviewed cases into the trusted retrieval library.
