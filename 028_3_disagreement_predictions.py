import pandas as pd

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

IN = "datasets/predictions_videos_en_final_m.csv"
OUT_ALL = "datasets/predictions_videos_en_final_d.csv"
OUT_SAMPLE = "datasets/predictions_videos_en_final_100_sample.csv"

TEXT_COL = "title_description_en"   # ако кај тебе е title_description_en, смени тука

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(IN, encoding="utf-8-sig")

print("Total rows:", len(df))
print("Columns:")
print(df.columns.tolist())

# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_cols = [
    TEXT_COL,
    "pred_cardiff_xlm_en",
    "pred_bertweet_en_3cls",
    "pred_nlptown_en_3cls",
    "agreement_mk",
    "majority_label_mk",
    "disagreement_type_mk"
]

for col in required_cols:
    if col not in df.columns:
        raise SystemExit(f"Missing column: {col}")

# --------------------------------------------------
# FILTER DISAGREEMENT ROWS
# --------------------------------------------------

# agreement_en е True ако сите 3 модели се согласуваат
# False значи дека има disagreement
disagree = df[df["agreement_mk"] == False].copy()

print("Disagreement rows:", len(disagree))

# --------------------------------------------------
# SAVE ALL DISAGREEMENTS
# --------------------------------------------------

disagree.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
print("Saved all disagreements:", OUT_ALL)

# --------------------------------------------------
# TAKE SAMPLE OF 100 EXAMPLES
# --------------------------------------------------

sample = disagree.sample(
    n=min(100, len(disagree)),
    random_state=42
)

# земи само најважни колони за рачна анализа
keep_cols = [
    TEXT_COL,
    "pred_cardiff_xlm_en",
    "pred_bertweet_en_3cls",
    "pred_nlptown_en_3cls",
    "majority_label_mk",
    "disagreement_type_mk"
]

sample = sample[keep_cols]

sample.to_csv(OUT_SAMPLE, index=False, encoding="utf-8-sig")

print("Saved sample:", OUT_SAMPLE)
print("Sample rows:", len(sample))

print("\nDisagreement type distribution:")
print(disagree["disagreement_type_mk"].value_counts())