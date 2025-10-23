import sys, pathlib
from pathlib import Path
import streamlit as st
import pandas as pd

# allow "from src..." imports when running: streamlit run app/app.py
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import DATA_PATH, TOP_N
from src.yelp_pipeline import (
    normalize_df, load_default_csv, build_synonyms,
    frequency_after_merge, select_top_by_cosine, recommend_top3
)

st.set_page_config(page_title="Yelp Recommendation", layout="wide")
st.title("Yelp Recommendation")

@st.cache_data
def load_uploaded_csv(file_like):
    return normalize_df(pd.read_csv(file_like))

# First run uses the shared default CSV
if "source" not in st.session_state:
    st.session_state["source"] = "default"
    st.session_state["df"] = load_default_csv(DATA_PATH)

uploaded = st.file_uploader("CSV (restaurant, review)", type=["csv"])
if uploaded is not None:
    st.session_state["df"] = load_uploaded_csv(uploaded)
    st.session_state["source"] = "uploaded"

# Optional reset
if st.sidebar.button("Revert to default file"):
    st.session_state["df"] = load_default_csv(DATA_PATH)
    st.session_state["source"] = "default"
    st.cache_data.clear()
    st.rerun()

df = st.session_state["df"]
st.caption("Using: default data/yelp_reviews.csv" if st.session_state["source"]=="default" else "Using: uploaded file")

# Parameters
base = build_synonyms()
with st.expander("Edit attribute keywords"):
    s = st.text_input("Service", ", ".join(base["service"]))
    f = st.text_input("Food", ", ".join(base["food"]))
    c = st.text_input("Cleanliness", ", ".join(base["cleanliness"]))
    l = st.text_input("Location", ", ".join(base["location"]))
    syn = {
        "service": [w.strip() for w in s.split(",") if w.strip()],
        "food": [w.strip() for w in f.split(",") if w.strip()],
        "cleanliness": [w.strip() for w in c.split(",") if w.strip()],
        "location": [w.strip() for w in l.split(",") if w.strip()],
    }

top_n = st.number_input("Top N reviews", min_value=50, max_value=500, value=TOP_N, step=10)

# Analysis
st.subheader("Word frequencies after merge")
freq = frequency_after_merge(df, syn)
st.dataframe(freq, use_container_width=True)

st.subheader(f"Top {int(top_n)} reviews by cosine similarity")
topk = select_top_by_cosine(df, syn, top_n=int(top_n))
st.dataframe(topk[["restaurant","review","cosine_similarity"]], use_container_width=True)

st.subheader("Top 3 recommendations (avg sentiment)")
top3 = recommend_top3(topk)
st.dataframe(top3, use_container_width=True)