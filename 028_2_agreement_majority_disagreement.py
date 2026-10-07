import pandas as pd
from collections import Counter

IN = "datasets/predictions_comments_en_final.csv"
OUT = "datasets/predictions_comments_en_final.csv"

LANG = "en"

df = pd.read_csv(IN, encoding="utf-8-sig")

normalize = {
    "neg": "negative",
    "neu": "neutral",
    "pos": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}

cols = [
    "pred_cardiff_en",
    "pred_bertweet_en",
    "pred_nlptown_en_3cls"
]

for c in cols:
    if c not in df.columns:
        raise SystemExit(f"Missing column: {c}")

    df[c] = (
        df[c]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(lambda x: normalize.get(x, x))
    )

def get_values(row):
    return [row[c] for c in cols]

def agreement(row):
    vals = get_values(row)
    return len(set(vals)) == 1

def majority_label(row):
    vals = get_values(row)
    counts = Counter(vals)
    most_common = counts.most_common()

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return "no_majority"

    return most_common[0][0]

def disagreement_type(row):
    vals = get_values(row)
    unique = set(vals)

    if len(unique) == 1:
        return "full_agreement"

    if len(unique) == 2 and "positive" in unique and "negative" in unique:
        return "polarized"

    if "neutral" in unique and "positive" in unique:
        return "neutral_positive"

    if "neutral" in unique and "negative" in unique:
        return "neutral_negative"

    if len(unique) == 3:
        return "all_different"

    return "mixed"

df[f"agreement_{LANG}"] = df.apply(agreement, axis=1)
df[f"majority_label_{LANG}"] = df.apply(majority_label, axis=1)
df[f"disagreement_type_{LANG}"] = df.apply(disagreement_type, axis=1)

df.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"Saved -> {OUT}")

print("\nAgreement rate:")
print(round(df[f"agreement_{LANG}"].mean() * 100, 2), "%")

print("\nMajority labels:")
print(df[f"majority_label_{LANG}"].value_counts())

print("\nDisagreement types:")
print(df[f"disagreement_type_{LANG}"].value_counts())