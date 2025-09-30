from flask import Flask, request, jsonify, render_template
import re, json, sqlite3, os
import numpy as np
from numpy.linalg import norm

app = Flask(__name__)

# --- Config ---
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "0").lower() in ("1", "true", "yes")
model = None

def load_model():
    """Lazy load model only if USE_EMBEDDINGS=1."""
    global model
    if model is not None:
        return model
    if not USE_EMBEDDINGS:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
        return model
    except Exception as e:
        print("⚠️ Could not load embeddings:", e)
        return None

# --- Load offers ---
with open("offers.json") as f:
    offers = json.load(f)

# --- Skill normalization ---
SKILL_MAP = {
    "py": "python", "python": "python",
    "js": "javascript", "ts": "typescript",
    "flask": "flask", "angular": "angular",
    "sql": "sql", "pandas": "pandas", "ml": "ml", "css": "css"
}

# --- SQLite ---
DB_FILE = "students.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    skills TEXT
)
""")
conn.commit()

# --- Helpers ---
def normalize_skills(skills):
    return list({SKILL_MAP.get(s.lower(), s.lower()) for s in skills})

def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))

def extract_cv(text):
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    name = text.split("\n")[0].strip() if text else "Unknown"
    skills = [s for s in SKILL_MAP if s.lower() in text.lower()]
    normalized = normalize_skills(skills)

    emb = None
    m = load_model()
    if m:
        emb = m.encode(text)

    return {
        "name": name,
        "email": email.group(0) if email else None,
        "skills": normalized,
        "embedding": emb,
        "cv_text": text
    }

def highlight_matched_skills(cv_text, skills):
    return {s: bool(re.search(re.escape(s), cv_text, re.IGNORECASE)) for s in skills}

def recommend_offers_keyword(student):
    student_skills = set(student.get("skills", []))
    results = []
    for offer in offers:
        core = set(offer.get("core_skills", []))
        other = set(offer.get("skills", [])) - core
        score = len(core & student_skills) * 2 + len(other & student_skills)
        if score > 0:
            results.append({"offer": offer, "score": score})
    return sorted(results, key=lambda x: -x["score"])[:3]

def recommend_offers_semantic(student):
    if not student.get("embedding"):
        return []
    student_vec = np.array(student["embedding"])
    results = []
    for offer in offers:
        if "embedding" not in offer:
            m = load_model()
            if m:
                desc = offer.get("title", "") + " " + " ".join(offer.get("skills", []))
                offer["embedding"] = m.encode(desc)
        if "embedding" in offer:
            sim = cosine_sim(student_vec, np.array(offer["embedding"]))
            results.append({"offer": offer, "score": round(sim, 3)})
    return sorted(results, key=lambda x: -x["score"])[:3]

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("cvtext", "")
        student = extract_cv(text)
        c.execute("INSERT INTO students (name,email,skills) VALUES (?,?,?)",
                  (student["name"], student["email"], ",".join(student["skills"])))
        conn.commit()

        if student.get("embedding"):
            recs = recommend_offers_semantic(student) or recommend_offers_keyword(student)
        else:
            recs = recommend_offers_keyword(student)

        highlighted = highlight_matched_skills(student["cv_text"], student["skills"])
        return render_template("index.html", student=student,
                               recommendations=recs, highlighted_skills=highlighted)
    return render_template("index.html")

@app.route("/tools/cv-extract", methods=["POST"])
def cv_extract():
    text = request.json.get("text", "")
    return jsonify(extract_cv(text))

@app.route("/tools/student-create", methods=["POST"])
def student_create():
    student = request.json
    c.execute("INSERT INTO students (name,email,skills) VALUES (?,?,?)",
              (student["name"], student["email"], ",".join(student["skills"])))
    conn.commit()
    return jsonify({"status": "ok", "student": student})

@app.route("/tools/offers-search", methods=["POST"])
def offers_search():
    profile = request.json
    if "embedding" in profile and profile["embedding"]:
        return jsonify(recommend_offers_semantic(profile))
    return jsonify(recommend_offers_keyword(profile))

if __name__ == "__main__":
    if not os.path.exists("templates/index.html"):
        raise RuntimeError("Place index.html in templates folder")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
