import pandas as pd

MANUAL_IN = "datasets/predictions_comments_en_final.csv"
NLPTOWN_IN = "datasets/predictions_comments_en_nlptown_all.csv"
OUT = "datasets/predictions_comments_en_final.csv"

manual = pd.read_csv(MANUAL_IN, encoding="utf-8-sig")
nlp = pd.read_csv(NLPTOWN_IN, encoding="utf-8-sig")

# ако имаш ID колона, најдобро merge по неа
KEY = "video_id"   # смени ако кај тебе се вика поинаку

nlp_cols = [
    KEY,
    "pred_nlptown_en_raw",
    "pred_nlptown_en_3cls",
    "score_nlptown_en"
]

manual = manual.merge(
    nlp[nlp_cols],
    on=KEY,
    how="left"
)

manual.to_csv(OUT, index=False, encoding="utf-8-sig")

print("Saved:", OUT)
print(manual[["pred_nlptown_en_3cls", "score_nlptown_en"]].head())