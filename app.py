import streamlit as st
import pandas as pd
from pathlib import Path


from yelp_pipeline import (
    build_synonyms,
    frequency_after_merge,
    select_top_by_cosine,
    recommend_top3,
)

st.set_page_config(page_title="Yelp Recommendation", layout="wide")
st.title("Yelp Recommendation")

# ---------- default file + session handling ----------
DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "yelp_reviews.csv"

@st.cache_data
def load_csv(src):
    return pd.read_csv(src)

if "source" not in st.session_state:
    st.session_state["source"] = "default"
    st.session_state["df"] = load_csv(DEFAULT_PATH)

uploaded = st.file_uploader("CSV (restaurant, review)", type=["csv"])

if uploaded is not None:
    st.session_state["df"] = load_csv(uploaded)
    st.session_state["source"] = "uploaded"

if st.sidebar.button("Revert to default file"):
    st.session_state["df"] = load_csv(DEFAULT_PATH)
    st.session_state["source"] = "default"
    st.rerun()

df = st.session_state["df"]
st.caption(
    "Using: default data/yelp_reviews.csv" if st.session_state["source"] == "default"
    else "Using: uploaded file"
)

# ---------- light schema check ----------
df.columns = [c.strip().lower() for c in df.columns]
if "restaurant" not in df.columns or "review" not in df.columns:
    st.error("CSV needs columns: restaurant, review")
    st.stop()

# ---------- parameters ----------
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

top_n = st.number_input("Top N reviews", min_value=50, max_value=500, value=200, step=10)

# ---------- analysis ----------
st.subheader("Word frequencies (merged)")
freq = frequency_after_merge(df, syn)
st.dataframe(freq, use_container_width=True)
st.download_button("Download frequency_table.csv", freq.to_csv(index=False).encode("utf-8"),
                   "frequency_table.csv", "text/csv")

st.subheader(f"Top {int(top_n)} reviews by cosine similarity")
topk = select_top_by_cosine(df, syn, top_n=int(top_n))
st.dataframe(topk[["restaurant", "review", "cosine_similarity"]], use_container_width=True)
st.download_button("Download top_reviews.csv", topk.to_csv(index=False).encode("utf-8"),
                   "top_reviews.csv", "text/csv")

st.subheader("Top 3 recommendations (avg sentiment of selected reviews)")
top3 = recommend_top3(topk)
st.dataframe(top3, use_container_width=True)
st.download_button("Download recommendations_top3.csv", top3.to_csv(index=False).encode("utf-8"),
                   "recommendations_top3.csv", "text/csv")