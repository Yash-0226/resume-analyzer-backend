# analyzer.py
# ─────────────────────────────────────────────────────────────────────────────
# All AI/NLP logic lives here. Flask (app.py) just calls these functions.
# This is your reader.py — cleaned up and turned into reusable functions.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
from pypdf import PdfReader
import spacy
from spacy.matcher import Matcher
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model once at startup (not on every request — saves time)
nlp = spacy.load("en_core_web_sm")

SKILLS = [
    "python", "java", "c++", "machine learning", "deep learning",
    "data analysis", "sql", "flask", "django", "html", "css",
    "javascript", "numpy", "pandas", "tensorflow", "pytorch",
    "git", "docker", "fastapi", "mongodb", "postgresql", "react",
    "node", "typescript", "aws", "azure", "linux", "rest api"
]


# ── 1. Extract text from uploaded PDF file object ────────────────────────────
def extract_text_from_file(file_obj) -> str:
    """
    Accepts a file-like object (from Flask's request.files).
    Returns extracted plain text string.
    """
    reader = PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        raise ValueError(
            "No text found in PDF. "
            "It may be a scanned/image-based resume — try a text-based PDF."
        )
    return text


# ── 2. Extract skills using spaCy pattern matching ───────────────────────────
def extract_skills(text: str) -> list:
    """
    Uses spaCy Matcher to find known skills in the resume text.
    Returns a sorted list of found skill strings.
    """
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    patterns = [[{"LOWER": word} for word in skill.split()] for skill in SKILLS]
    matcher.add("SKILLS", patterns)

    found = set()
    for _, start, end in matcher(doc):
        found.add(doc[start:end].text.lower())

    return sorted(list(found))


# ── 3. ATS Score via TF-IDF + Cosine Similarity ──────────────────────────────
def calculate_ats_score(resume_text: str, job_description: str) -> dict:
    """
    Vectorizes both resume and JD using TF-IDF, then measures
    cosine similarity between them (0–100 scale).
    Also returns matched and missing keywords.
    """
    resume_clean = resume_text.lower().strip()
    jd_clean     = job_description.lower().strip()

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
    similarity   = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    ats_score    = round(float(similarity) * 100, 1)

    # Find matched and missing keywords
    jd_words     = set(jd_clean.split())
    resume_words = set(resume_clean.split())

    stopwords = {
        'with', 'have', 'that', 'this', 'will', 'from', 'they',
        'been', 'were', 'their', 'about', 'which', 'when', 'your',
        'also', 'into', 'what', 'more', 'some', 'such', 'than'
    }
    jd_keywords = {w for w in jd_words if len(w) > 4 and w not in stopwords}

    matched = sorted(list(jd_keywords & resume_words))[:20]
    missing = sorted(list(jd_keywords - resume_words))[:15]

    return {
        "ats_score":        ats_score,
        "matched_keywords": matched,
        "missing_keywords": missing,
    }


# ── 4. AI Resume Analysis via Groq LLM ───────────────────────────────────────
def analyze_with_llm(resume_text: str, found_skills: list) -> dict:
    """
    Sends resume text + detected skills to Groq (Llama 3.3).
    Returns structured JSON with score, summary, strengths,
    weaknesses, and suggestions.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    prompt = f"""You are an expert resume reviewer and career coach.

Analyze the resume below and return ONLY a valid JSON object.
No explanation, no markdown, no extra text — just the JSON.

Detected skills (from NLP): {found_skills}

Resume text:
\"\"\"
{resume_text[:4000]}
\"\"\"

Return exactly this JSON structure:
{{
  "score": <integer 0-100>,
  "summary": "<2 sentence overall impression>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],
  "suggestions": ["<actionable tip 1>", "<actionable tip 2>", "<actionable tip 3>"]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a professional resume analyst. Always respond with valid JSON only. No markdown."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())
