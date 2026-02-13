import pandas as pd
import numpy as np
import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/constitution.csv")
df = df.dropna()

articles = []

for text in df["Articles"]:
    articles.append({
        "FullText": text
    })

print("Total articles:", len(articles))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
texts = [article["FullText"] for article in articles]
embeddings = model.encode(texts)

embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save
os.makedirs("model", exist_ok=True)

faiss.write_index(index, "model/constitution.index")

with open("model/articles.pkl", "wb") as f:
    pickle.dump(articles, f)

print("FAISS index created successfully.")
