# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Flask API — the web server that connects React frontend to analyzer.py
#
# ONE endpoint:
#   POST /analyze
#     accepts: multipart form with 'resume' (PDF file) + 'job_description' (text)
#     returns: JSON with all scores and analysis
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

from analyzer import (
    extract_text_from_file,
    extract_skills,
    calculate_ats_score,
    analyze_with_llm,
)

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Allow React (localhost:3000) to talk to Flask (localhost:5000)
# Without this, browser blocks all requests — very common beginner issue
CORS(app)


# ── Health check — visit http://localhost:5000/ to confirm server is running ──
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ResumeAI backend is running ✓"})


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    # ── Step 1: Validate incoming request ────────────────────────────────────
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded."}), 400

    resume_file     = request.files["resume"]
    job_description = request.form.get("job_description", "").strip()

    if resume_file.filename == "":
        return jsonify({"error": "Resume file is empty."}), 400

    if not resume_file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    if len(job_description) < 20:
        return jsonify({"error": "Job description is too short."}), 400

    # ── Step 2: Extract text from PDF ────────────────────────────────────────
    try:
        resume_text = extract_text_from_file(resume_file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    # ── Step 3: Extract skills with spaCy ────────────────────────────────────
    found_skills = extract_skills(resume_text)

    # ── Step 4: Calculate ATS score ───────────────────────────────────────────
    ats_result = calculate_ats_score(resume_text, job_description)

    # ── Step 5: LLM analysis via Groq ────────────────────────────────────────
    try:
        llm_result = analyze_with_llm(resume_text, found_skills)
    except Exception as e:
        return jsonify({"error": f"LLM analysis failed: {str(e)}"}), 500

    # ── Step 6: Merge everything and return ───────────────────────────────────
    response = {
        # ATS scores
        "ats_score":        ats_result["ats_score"],
        "matched_keywords": ats_result["matched_keywords"],
        "missing_keywords": ats_result["missing_keywords"],

        # LLM analysis
        "score":       llm_result["score"],
        "summary":     llm_result["summary"],
        "strengths":   llm_result["strengths"],
        "weaknesses":  llm_result["weaknesses"],
        "suggestions": llm_result["suggestions"],

        # spaCy skills
        "detected_skills": found_skills,
    }

    return jsonify(response), 200


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ── Run server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=True → auto-reloads when you save a file (great for development)
    # Remove debug=True when deploying to production
    app.run(debug=True, port=5000)
