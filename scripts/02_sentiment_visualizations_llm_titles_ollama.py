import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "datasets" / "ollama_comments_sentiment.csv"

FIG_DIR = BASE_DIR / "outputs" / "figures"
TABLE_DIR = BASE_DIR /  "outputs" / "tables"

Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig")

mk_counts = df["llm_label_mk_c"].value_counts()
en_counts = df["llm_label_en_c"].value_counts()

distribution = pd.DataFrame({
    "MK": mk_counts,
    "EN": en_counts
}).fillna(0)

distribution.to_csv(
    f"{TABLE_DIR}/sentiment_distribution_comments.csv",
    encoding="utf-8-sig"
)

distribution.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("LLM Sentiment Distribution Comments")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/sentiment_distribution_comments.png",
    dpi=300
)

plt.close()

