import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

st.title("🔎 Semantic Search Engine")

# Load precomputed data
embeddings = np.load("embeddings.npy")
df = pd.read_csv("unique_questions.csv")

# ChromaDB setup
client = chromadb.Client()
collection = client.get_or_create_collection("questions_streamlit")

# add data only if collection is empty
if collection.count() == 0:
    collection.add(
        ids = df["doc_id"].tolist(),
        documents = df["text"].tolist(),
        metadatas = df[["qid"]].to_dict(orient="records"),
        embeddings = embeddings.tolist()
    )

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Search input
query = st.text_input("Enter your search query:")

if query:
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=5)
    st.write("### Results:")
    for i, doc in enumerate(results["documents"][0]):
        st.write(f"{i+1}. {doc}")
