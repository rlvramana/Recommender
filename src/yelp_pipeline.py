import re
from collections import Counter
from pathlib import Path
import pandas as pd

# ---------------- text utils ----------------
def _tok(x: str):
    return re.findall(r"[a-zA-Z][a-zA-Z']+", str(x).lower())

# ---------------- data loading/cleaning ----------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    name_col = next((c for c in ["restaurant","name","business","business_name","restaurant_name"] if c in d.columns), d.columns[0])
    rev_col  = next((c for c in ["review","text","body","comment","reviews"] if c in d.columns), d.columns[-1])
    d = d[[name_col, rev_col]].rename(columns={name_col:"restaurant", rev_col:"review"})
    d["restaurant"] = d["restaurant"].astype(str).str.strip()
    d["review"] = d["review"].astype(str).str.replace(r"\s+"," ", regex=True).str.strip()
    d = d[(d["restaurant"]!="") & (d["review"]!="")].drop_duplicates()
    return d

def load_default_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return normalize_df(raw)

# ---------------- vocabulary ----------------
def build_synonyms(custom: dict | None = None) -> dict:
    base = {
        "service": ["service","server","staff","waiter","waitress","host","friendly","rude","attentive","quick","slow","speed","helpful","courteous"],
        "food": ["food","taste","flavor","delicious","tasty","bland","overcooked","undercooked","fresh","portion","menu","dish","burger","pizza","taco","fries","salad","sauce"],
        "cleanliness": ["clean","dirty","messy","sanitary","hygiene","spotless","sticky","tidy","restroom","bathroom","trash","smell","odor","greasy"],
        "location": ["location","parking","lot","easy","nearby","close","downtown","drive-thru","drive through","drive thru","line","wait","busy","crowded","find","distance"],
    }
    if custom:
        for k,v in custom.items():
            if k in base and isinstance(v, list) and v:
                base[k] = v
    return base

# ---------------- frequency after merge ----------------
def frequency_after_merge(df: pd.DataFrame, synonyms: dict) -> pd.DataFrame:
    vocab_to_attr = {w: a for a, ws in synonyms.items() for w in ws}
    buckets = {k: Counter() for k in synonyms}
    for txt in df["review"]:
        for w in _tok(txt):
            a = vocab_to_attr.get(w)
            if a:
                buckets[a][w] += 1
    rows = []
    for a, cnt in buckets.items():
        tot = sum(cnt.values())
        for w, n in cnt.most_common():
            rows.append({"attribute": a, "word": w, "count": int(n), "attribute_total": int(tot)})
    out = pd.DataFrame(rows).sort_values(["attribute","count"], ascending=[True, False], ignore_index=True)
    return out

# ---------------- cosine similarity selection ----------------
def select_top_by_cosine(df: pd.DataFrame, synonyms: dict, top_n: int = 200) -> pd.DataFrame:
    combined = " ".join(" ".join(ws) for ws in synonyms.values())
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(lowercase=True, token_pattern=r"[a-zA-Z][a-zA-Z']+")
        X = vec.fit_transform([combined] + df["review"].tolist())
        sims = cosine_similarity(X[0:1], X[1:]).flatten()
    except Exception:
        def tfv(t): return Counter(_tok(t))
        a = tfv(combined)
        def cos(x,y):
            ks = set(x)|set(y)
            vx = [x.get(k,0) for k in ks]; vy = [y.get(k,0) for k in ks]
            nx = (sum(u*u for u in vx) or 1) ** 0.5
            ny = (sum(v*v for v in vy) or 1) ** 0.5
            return sum(u*v for u,v in zip(vx,vy))/(nx*ny)
        sims = [cos(a, tfv(t)) for t in df["review"]]
    out = df.copy()
    out["cosine_similarity"] = sims
    return out.sort_values("cosine_similarity", ascending=False).head(top_n).reset_index(drop=True)

# ---------------- sentiment + recommendation ----------------
_POS = set("good great excellent amazing awesome friendly fast quick tasty delicious fresh clean spotless helpful courteous love lovely perfect outstanding best recommend fantastic".split())
_NEG = set("bad poor terrible awful rude slow bland cold overcooked undercooked dirty messy greasy smelly wait long disappointed worst never".split())

def sentiment_score(text: str) -> float:
    toks = _tok(text)
    if not toks: return 0.0
    s = sum(1 if t in _POS else -1 if t in _NEG else 0 for t in toks)
    return s/len(toks)

def recommend_top3(top_df: pd.DataFrame) -> pd.DataFrame:
    tmp = top_df.copy()
    tmp["sentiment"] = tmp["review"].apply(sentiment_score)
    agg = tmp.groupby("restaurant", as_index=False)["sentiment"].mean().rename(columns={"sentiment":"avg_sentiment"})
    return agg.sort_values("avg_sentiment", ascending=False).head(3).reset_index(drop=True)