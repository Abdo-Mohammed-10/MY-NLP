import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)  # المسار اللي فيه app.py

CSV_PATH = os.path.join(BASE_DIR, "unique_questions.csv")
EMB_PATH = os.path.join(BASE_DIR, "embeddings.npy")

df = pd.read_csv(CSV_PATH).head(500)
embeddings = np.load(EMB_PATH)[:500]


st.title("🔎 Semantic Search Engine (Fast Deploy)")

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "unique_questions.csv")
EMB_PATH = os.path.join(BASE_DIR, "embeddings.npy")

# =========================
# Lazy load model & data
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH).head(500)  # subset صغير للتجربة
    embeddings = np.load(EMB_PATH)[:500]
    return df, embeddings

model = load_model()
df, embeddings = load_data()

# =========================
# User query
# =========================
query = st.text_input("Enter your search query:")

if query:
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[-5:][::-1]

    st.write("### Results:")
    for i in top_idx:
        st.write(f"{df.iloc[i]['text']} (score={sims[i]:.4f})")

