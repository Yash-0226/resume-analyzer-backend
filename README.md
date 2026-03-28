# ResumeAI — Flask Backend

## Setup

### 1. Create and activate virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Add your API key
Open `.env` and replace `gsk_...` with your real Groq API key.

### 4. Run the server
```bash
python app.py
```

Server runs at: http://localhost:5000

### 5. Test it's working
Open browser and visit: http://localhost:5000
You should see: `{"status": "ResumeAI backend is running ✓"}`

## API

### POST /analyze
**Body (multipart/form-data):**
- `resume` — PDF file
- `job_description` — string

**Response:**
```json
{
  "ats_score": 72.4,
  "matched_keywords": ["python", "machine learning", ...],
  "missing_keywords": ["kubernetes", "aws", ...],
  "score": 78,
  "summary": "Strong technical background...",
  "strengths": ["...", "...", "..."],
  "weaknesses": ["...", "...", "..."],
  "suggestions": ["...", "...", "..."],
  "detected_skills": ["python", "sql", ...]
}
```
