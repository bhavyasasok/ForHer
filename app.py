import pickle
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# -------------------------
# Load FAISS index + articles
# -------------------------
index = faiss.read_index("model/constitution.index")

with open("model/articles.pkl", "rb") as f:
    articles = pickle.load(f)

# -------------------------
# Load embedding model
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

def get_embedding(text):
    return np.array(embed_model.encode(text)).astype("float32")

# -------------------------
# Rule-Based Legal Guidance
# -------------------------
def generate_legal_response(user_problem):
    problem = user_problem.lower()

    if "abuse" in problem or "violence" in problem or "threat" in problem:
        return """
Summary:
Your situation may involve domestic violence or abuse.

Constitutional Protections:
- Article 14: Equality before law.
- Article 21: Right to life and personal liberty.

Legal Support:
- Protection under the Protection of Women from Domestic Violence Act, 2005.

Practical Next Steps:
1. Call Women Helpline: 181
2. Emergency: 112
3. Approach nearest police station
4. Seek Protection Order from Magistrate

Disclaimer:
This is informational only and not a substitute for a lawyer.
"""

    elif "workplace" in problem or "harassment" in problem:
        return """
Summary:
This may involve workplace harassment.

Constitutional Protections:
- Article 14: Equality before law
- Article 15: Prohibition of discrimination

Legal Support:
- Sexual Harassment of Women at Workplace Act, 2013

Next Steps:
1. Report to Internal Complaints Committee
2. Document evidence
3. Contact 181 if needed

Disclaimer:
This is informational only and not a substitute for a lawyer.
"""

    elif "property" in problem:
        return """
Summary:
This may relate to property or inheritance rights.

Constitutional Protections:
- Article 14: Equal rights
- Hindu Succession Act (if applicable)

Next Steps:
1. Consult legal advisor
2. Collect ownership documents
3. Approach local court if required

Disclaimer:
This is informational only and not a substitute for a lawyer.
"""

    else:
        return """
Summary:
Your issue may involve legal rights under the Indian Constitution.

Constitutional Protections:
- Article 14: Equality before law
- Article 21: Right to life and liberty

Next Steps:
1. Contact Women Helpline: 181
2. Emergency: 112
3. Seek legal consultation

Disclaimer:
This is informational only and not a substitute for a lawyer.
"""

# -------------------------
# Homepage
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# Analyze Route
# -------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_problem = data.get("problem")

    if not user_problem:
        return jsonify({"error": "No problem provided"}), 400

    # Embed query
    query_vector = np.array([get_embedding(user_problem)])

    # Search FAISS
    distances, indices = index.search(query_vector, 3)

    matched_articles = []

    for idx in indices[0]:
        matched_articles.append(articles[idx]["FullText"])

    explanation = generate_legal_response(user_problem)

    return jsonify({
        "matched_laws": matched_articles,
        "analysis": explanation,
        "helplines": "112 (Emergency), 181 (Women Helpline India)"
    })

# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
