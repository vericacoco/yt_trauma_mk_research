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

disagreement = df[
    df["llm_label_mk_c"] != df["llm_label_en_c"]
]

disagreement = disagreement[
    [
        "video_id",
        "text_mk",
        "text_en",
        "llm_label_mk_c",
        "llm_label_en_c"
    ]
]

disagreement.to_csv(
    f"{TABLE_DIR}/disagreement_examples_comments.csv",
    index=False,
    encoding="utf-8-sig"
)