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

agreement_counts = df["llm_agreement_title_c"].value_counts()

agreement_counts.to_csv(
    f"{TABLE_DIR}/agreement_stats.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(6,6))

plt.pie(
    agreement_counts,
    labels=agreement_counts.index,
    autopct="%1.1f%%"
)

plt.title("MK vs EN Agreement")

plt.savefig(
    f"{FIG_DIR}/agreement_distribution_comments.png",
    dpi=300
)

plt.close()