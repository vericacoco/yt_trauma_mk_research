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

plt.figure(figsize=(8,5))

plt.hist(
    df["llm_confidence_en_c"],
    bins=20
)

plt.title("Confidence Distribution Comments - EN")

plt.xlabel("Confidence")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/confidence_histogram_en_comments.png",
    dpi=300
)

plt.close()