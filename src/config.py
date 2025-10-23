from pathlib import Path

# Single source of truth for the default CSV path
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "yelp_reviews.csv"

# Keep the same top-N everywhere
TOP_N = 200