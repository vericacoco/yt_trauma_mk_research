import pandas as pd

IN = "datasets/predictions_nlptown_en_videos_all.csv"
OUT = "datasets/predictions_videos_en_final.csv"

df = pd.read_csv(IN, encoding="utf-8-sig")

# default сите се no
df["manual_review"] = "no"

# првите 100 реда се рачно проверени
df.loc[:200, "manual_review"] = "yes"

df.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"Saved: {OUT}")
print(df["manual_review"].value_counts())