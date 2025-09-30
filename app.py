from flask import Flask, request, jsonify, render_template
import re, json, sqlite3, os
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import PyPDF2

app = Flask(__name__)

# --- Load offers ---
with open("offers.json") as f:
    offers = json.load(f)

# --- Skill normalization dictionary ---
SKILL_MAP = {
    "py": "python", "python": "python",
    "js": "javascript", "ts": "typescript",
    "flask": "flask", "angular": "angular",
    "sql": "sql", "pandas": "pandas", "ml": "ml", "css":"css"
}

# --- SQLite setup ---
DB_FILE = "students.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    skills TEXT
)
''')
conn.commit()

# --- Load sentence-transformers model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Precompute offer embeddings ---
for offer in offers:
    desc = offer.get("title","") + " " + " ".join(offer.get("skills",[]))
    offer["embedding"] = model.encode(desc)

# --- Helper functions ---
def normalize_skills(skills):
    return list({SKILL_MAP.get(s.lower(), s.lower()) for s in skills})

def extract_cv(text):
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    name = text.split("\n")[0].strip() if text else "Unknown"
    skills = []
    for skill in SKILL_MAP.keys():
        if skill.lower() in text.lower():
            skills.append(skill)
    normalized = normalize_skills(skills)
    embedding = model.encode(text)
    return {"name": name, "email": email.group(0) if email else None, "skills": normalized, "embedding": embedding, "cv_text": text}

def cosine_sim(a,b):
    return np.dot(a,b)/(norm(a)*norm(b)+1e-8)

def recommend_offers_semantic(student):
    student_vec = student.get("embedding")
    results = []
    for offer in offers:
        sim = cosine_sim(student_vec, offer["embedding"])
        if sim > 0:
            results.append({"offer": offer, "score": round(float(sim),3)})
    results = sorted(results, key=lambda x: -x["score"])
    return results[:3]

def highlight_matched_skills(cv_text, skills):
    highlighted = {}
    for skill in skills:
        pattern = re.compile(re.escape(skill), re.IGNORECASE)
        highlighted[skill] = bool(pattern.search(cv_text))
    return highlighted

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# --- Routes ---
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        text = request.form.get("cvtext","")
        student = extract_cv(text)
        # Save student
        c.execute("INSERT INTO students (name,email,skills) VALUES (?,?,?)",
                  (student["name"], student["email"], ",".join(student["skills"])))
        conn.commit()
        recommendations = recommend_offers_semantic(student)
        highlighted_skills = highlight_matched_skills(student["cv_text"], student["skills"])
        return render_template("index.html", student=student, recommendations=recommendations, highlighted_skills=highlighted_skills)
    return render_template("index.html")

@app.route("/upload-cv", methods=["POST"])
def upload_cv():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", file.filename)
    file.save(filename)
    text = extract_text_from_pdf(filename)
    student = extract_cv(text)
    # Save student
    c.execute("INSERT INTO students (name,email,skills) VALUES (?,?,?)",
              (student["name"], student["email"], ",".join(student["skills"])))
    conn.commit()
    recommendations = recommend_offers_semantic(student)
    highlighted_skills = highlight_matched_skills(student["cv_text"], student["skills"])
    return render_template("index.html", student=student, recommendations=recommendations, highlighted_skills=highlighted_skills)

@app.route("/tools/cv-extract", methods=["POST"])
def cv_extract():
    text = request.json.get("text","")
    return jsonify(extract_cv(text))

@app.route("/tools/student-create", methods=["POST"])
def student_create():
    student = request.json
    c.execute("INSERT INTO students (name,email,skills) VALUES (?,?,?)",
              (student["name"], student["email"], ",".join(student["skills"])))
    conn.commit()
    return jsonify({"status":"ok","student":student})

@app.route("/tools/offers-search", methods=["POST"])
def offers_search():
    profile = request.json
    if "embedding" in profile and len(profile["embedding"])>0:
        return jsonify(recommend_offers_semantic(profile))
    else:
        return jsonify({"error":"No embedding found in profile"})

@app.route("/tools/offers-search-edit", methods=["POST"])
def offers_search_edit():
    profile = request.json
    text = profile.get("name","") + " " + " ".join(profile.get("skills",[]))
    profile["embedding"] = model.encode(text)
    return jsonify(recommend_offers_semantic(profile))

if __name__ == "__main__":
    if not os.path.exists("templates/index.html"):
        raise RuntimeError("Place index.html in templates folder")
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
