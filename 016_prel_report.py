import pandas as pd

v = pd.read_csv("datasets/videos_clean.csv")
c = pd.read_csv("datasets/comments_fixed.csv")

print("\nTOP 10 видеа по коментари:")
print(v.sort_values("comments", ascending=False)[["title","matched_terms","comments","views"]].head(10))

print("\nTOP 10 канали по вкупни коментари:")
print(
    v.groupby("channel")["comments"]
     .sum()
     .sort_values(ascending=False)
     .head(10)
)

print("\nКоментари по клучен збор (според matched_terms):")
# секое видео може да има повеќе термини — груба пресметка
exp = v.assign(m=v["matched_terms"].fillna("").str.split("; ")).explode("m")
print(
    exp.groupby("m")["comments"]
       .sum()
       .sort_values(ascending=False)
       .head(15)
)
