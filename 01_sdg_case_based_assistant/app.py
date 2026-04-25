
import json
import time
from pathlib import Path

import faiss
import fitz
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="SDG Community Assistant", layout="wide")

LLMOPS_API_URL = "http://localhost:8001/infer"

def ask_llmops(prompt: str, task: str = "summarization") -> str:
    """
    Project 1 → Project 2 → Project 3 integration

    Flow:
    UI → LLMOps API → (Fine-tuned → Local → Remote fallback)
    """

    try:
        response = requests.post(
            LLMOPS_API_URL,
            json={
                "task": task,
                "text": prompt,
                "model_preference": "auto",  # switch to "finetuned" after training
                "max_tokens": 700
            },
            timeout=180,
        )
        response.raise_for_status()

        data = response.json()

        model_used = data.get("model_used", "unknown")
        output = data.get("output", "")

        # ✅ Clean formatting for Streamlit
        return f"""
### 🔍 Model Used: `{model_used}`

{output}
"""

    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Try again."

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to LLMOps API. Is FastAPI running?"

    except Exception as e:
        return f"❌ Unexpected error: {e}"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CASE_FILE = DATA_DIR / "trusted_case_library.json"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


EMBED_MODEL = load_embedding_model()


def ask_phi(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error calling local model: {e}"


def build_faiss_index_from_texts(texts):
    if not texts:
        return None
    embeddings = EMBED_MODEL.encode(texts)
    embeddings = np.array(embeddings).astype("float32")
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def load_cases():
    if not CASE_FILE.exists():
        st.error(
            "trusted_case_library.json not found. Put it inside the data/ folder."
        )
        st.stop()
    with open(CASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def case_to_text(case):
    return f"""
Community Problem: {case.get("community_problem", "")}
Problem Category: {", ".join(case.get("problem_category", []))}
Country/Region: {case.get("country", "")} / {case.get("region", "")}
SDGs: {", ".join(case.get("sdgs", []))}
Intervention: {", ".join(case.get("intervention", []))}
Evidence Summary: {case.get("evidence_summary", "")}
Recommended Transfer: {", ".join(case.get("recommended_transfer", []))}
Limitations: {case.get("limitations", "")}
Source: {case.get("source_name", "")}
Trust Level: {case.get("trust_level", "")}
"""


@st.cache_resource
def build_case_retriever(cases_json_text):
    cases = json.loads(cases_json_text)
    case_texts = [case_to_text(c) for c in cases]
    index = build_faiss_index_from_texts(case_texts)
    return index, case_texts


def retrieve_similar_cases(user_problem, cases, index, top_k=4):
    query_embedding = EMBED_MODEL.encode([user_problem])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, ids = index.search(query_embedding, top_k)

    matched = []
    for rank, idx in enumerate(ids[0]):
        if 0 <= idx < len(cases):
            item = cases[idx].copy()
            item["_rank"] = rank + 1
            item["_distance"] = float(distances[0][rank])
            matched.append(item)
    return matched


def extract_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        pages.append({"page": i + 1, "text": page.get_text()})
    return pages


def chunk_text(pages, chunk_size=500, overlap=100):
    chunks = []
    for page in pages:
        text = page["text"].replace("\n", " ").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append({"page": page["page"], "text": chunk})
            start += chunk_size - overlap
    return chunks


def retrieve_pdf_chunks(query, chunks, index, top_k=5):
    query_embedding = EMBED_MODEL.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, ids = index.search(query_embedding, top_k)

    results = []
    for idx in ids[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results


st.title("Offline SDG Case-Based Assistant")
st.caption("Offline community problem solving + trusted global case transfer")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Community Problem Solver", "PDF SDG Assistant", "Case Library Browser"]
)

st.sidebar.markdown("---")
st.sidebar.write(f"Local model: `{OLLAMA_MODEL}`")
st.sidebar.write("Case library: trusted offline JSON")
st.sidebar.write("Internet needed only for manual updates, not for use.")


if mode == "Community Problem Solver":
    st.header("Community Problem Solver")
    st.write(
        "Describe a local community issue. The app matches it to trusted global cases "
        "and generates safe, practical, source-aware next steps."
    )

    cases = load_cases()
    cases_json_text = json.dumps(cases, sort_keys=True)
    case_index, _ = build_case_retriever(cases_json_text)

    user_problem = st.text_area(
        "Describe the community problem",
        height=170,
        placeholder=(
            "Example: Our community has stagnant water near homes, mosquitoes, "
            "and children are frequently getting fever and stomach infections."
        )
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        solve_button = st.button("Find Trusted Similar Cases")
    with col2:
        top_k = st.slider("Number of cases", 2, 6, 4)

    if solve_button and user_problem.strip():
        matched_cases = retrieve_similar_cases(user_problem, cases, case_index, top_k=top_k)

        case_context = "\n\n".join(
            [
                f"""
Case Rank: {case["_rank"]}
Source: {case.get("source_name")}
Source Type: {case.get("source_type")}
Source URL: {case.get("source_url")}
Published Year: {case.get("published_year")}
Last Verified: {case.get("last_verified")}
Trust Level: {case.get("trust_level")}
Country/Region: {case.get("country")} / {case.get("region")}
Problem: {case.get("community_problem")}
Problem Categories: {", ".join(case.get("problem_category", []))}
SDGs: {", ".join(case.get("sdgs", []))}
Interventions Used Elsewhere: {", ".join(case.get("intervention", []))}
Evidence Summary: {case.get("evidence_summary")}
Recommended Transfer: {", ".join(case.get("recommended_transfer", []))}
Urgent Warning: {case.get("urgent_warning")}
Limitations: {case.get("limitations")}
"""
                for case in matched_cases
            ]
        )

        prompt = f"""
You are an offline community SDG case-based assistant.

Your role:
- Understand the user's local problem
- Map it to relevant SDGs
- Use ONLY the trusted similar cases below
- Suggest practical, safe, community-level actions
- Do NOT diagnose disease or prescribe treatment
- Always include source awareness and local validation needs
- If the problem mentions serious health symptoms, advise contacting qualified health workers

User Local Problem:
{user_problem}

Trusted Similar Cases:
{case_context}

Return EXACTLY in this format:

Problem Category:
- 

Relevant SDGs:
- SDG number and name:
  Why relevant:

Trusted Similar Cases Used:
1.
Source:
Country/Region:
Why it matches:
Intervention used elsewhere:
What can transfer locally:
Trust Level:
Last Verified:

Suggested Community Action Plan:
Immediate actions:
- 
Short-term actions:
- 
Who to contact:
- 
Data to collect:
- 

Safety / Healthcare Warning:
- 

Local Validation Needed:
- 

Confidence:
High / Medium / Low

Limitations:
- 
"""

        start = time.time()
        answer = ask_llmops(prompt)
        end = time.time()

        st.subheader("Source-Aware Solution Plan")
        st.write(answer)
        st.info(f"Response time: {round(end - start, 2)} seconds")

        st.subheader("Retrieved Trusted Cases")
        for case in matched_cases:
            with st.expander(
                f"Rank {case['_rank']} | {case.get('source_name')} | {case.get('country')}"
            ):
                st.write(f"**Trust Level:** {case.get('trust_level')}")
                st.write(f"**Source Type:** {case.get('source_type')}")
                st.write(f"**Published Year:** {case.get('published_year')}")
                st.write(f"**Last Verified:** {case.get('last_verified')}")
                st.write(f"**Source URL:** {case.get('source_url')}")
                st.write(f"**Problem:** {case.get('community_problem')}")
                st.write(f"**SDGs:** {', '.join(case.get('sdgs', []))}")
                st.write("**Interventions:**")
                for item in case.get("intervention", []):
                    st.write(f"- {item}")
                st.write("**Recommended Transfer:**")
                for item in case.get("recommended_transfer", []):
                    st.write(f"- {item}")
                st.write(f"**Limitations:** {case.get('limitations')}")
    elif solve_button:
        st.warning("Please describe a community problem first.")


elif mode == "Case Library Browser":
    st.header("Trusted Case Library Browser")
    cases = load_cases()

    query = st.text_input("Search cases", placeholder="water, sanitation, healthcare, girls education...")
    filtered = cases
    if query:
        q = query.lower()
        filtered = [
            c for c in cases
            if q in json.dumps(c).lower()
        ]

    st.write(f"Showing {len(filtered)} of {len(cases)} cases")

    for case in filtered:
        with st.expander(f"{case.get('case_id')} | {case.get('source_name')} | {case.get('country')}"):
            st.json(case)


elif mode == "PDF SDG Assistant":
    st.header("PDF SDG Assistant")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        pages = extract_pdf_text(uploaded_file)
        chunks = chunk_text(pages)

        if not chunks:
            st.error("No text chunks found. This may be a scanned PDF.")
            st.stop()

        texts = [c["text"] for c in chunks]
        pdf_index = build_faiss_index_from_texts(texts)
        st.success(f"PDF processed: {len(pages)} pages, {len(chunks)} chunks")

        tab1, tab2, tab3 = st.tabs(["Ask Questions", "SDG Classification", "Summary"])

        with tab1:
            question = st.text_input("Ask a question from the PDF")
            if question:
                evidence = retrieve_pdf_chunks(question, chunks, pdf_index, top_k=3)
                evidence_text = "\n\n".join([f"Page {e['page']}: {e['text']}" for e in evidence])

                prompt = f"""
You are an offline SDG evidence assistant.

Use ONLY the retrieved evidence.
Do NOT infer beyond what is written.
If evidence is insufficient, say so.

Question:
{question}

Retrieved Evidence:
{evidence_text}

Return:
Direct Answer:
- 

Evidence Table:
1.
Page:
Evidence:
Supports:
Confidence:

Limitations:
- 
"""
                start = time.time()
                answer = ask_llmops(prompt)
                end = time.time()

                st.subheader("Answer")
                st.write(answer)
                st.info(f"Response time: {round(end - start, 2)} seconds")

                st.subheader("Retrieved Evidence")
                for e in evidence:
                    with st.expander(f"Page {e['page']}"):
                        st.write(e["text"])

        with tab2:
            if st.button("Classify SDGs"):
                context = "\n\n".join([c["text"] for c in chunks[:8]])
                prompt = f"""
Classify this document into relevant UN SDGs using only the text below.

Document:
{context}

Return:
Relevant SDGs:
- SDG number and name:
  Evidence:
  Confidence:

Limitations:
- 
"""
                st.write(ask_llmops(prompt))

        with tab3:
            if st.button("Generate Summary"):
                context = "\n\n".join([c["text"] for c in chunks[:10]])
                prompt = f"""
Summarize this SDG-related document using only the text below.

Document:
{context}

Return:
Short Summary:
- 
Key Problems:
- 
Beneficiaries:
- 
Relevant SDGs:
- 
Limitations:
- 
"""
                st.write(ask_llmops(prompt))
