import pandas as pd
from keywords import SEARCH_TERMS

SRC = "datasets/videos.csv"
DST = "datasets/videos_clean.csv"

# 1) Вчитај
df = pd.read_csv(SRC)

# 2) Normalize language field (mk / mk-MK / None)
def norm_lang(x):
    x = str(x).lower() if pd.notna(x) else ""
    if x.startswith("mk"):
        return "mk"
    return x or None

df["lang"] = df["lang"].apply(norm_lang)

# 3) Дата и година
df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
df["year"] = df["published_at"].dt.year

# 4) Игнор-листи
IGNORE_CHANNELS = {
    "Macedonian Documentary Channel",
    "Ritko Mdfk",
    "Dragi Jovanovski",
    "COMEDY BALKAN",
    "Andyyy",
    "SBS Audio",
    "Peaceful Sleeping Music",
    "VIS Memorija"
}
IGNORE_WORDS = {"song", "lyrics"}

def contains_ignore_words(text):
    t = str(text).lower()
    return any(w in t for w in IGNORE_WORDS)

# 5) Match score (како кај колешката)
def keyword_match_count(text):
    t = str(text).lower()
    c = 0
    for term in SEARCH_TERMS:
        for w in term.lower().split():
            if w and w in t:
                c += 1
    return c

df["match_count"] = df.apply(
    lambda r: keyword_match_count(r.get("title", "")) + keyword_match_count(r.get("description", "")),
    axis=1
)

# 6) Филтри
df_clean = df[
    (df["lang"] == "mk") &
    (df["year"].fillna(0) >= 2018) &
    (~df["channel"].isin(IGNORE_CHANNELS)) &
    (~df.apply(lambda r: contains_ignore_words(r.get("title", "")) or
                         contains_ignore_words(r.get("description", "")), axis=1)) &
    (df["match_count"] >= 1)
].copy()

# 7) Сортирање (прво релевантност, па коментари, па прегледи)
df_clean = df_clean.sort_values(["match_count", "comments", "views"], ascending=[False, False, False])

# 8) Колони
keep_cols = [
    "video_id","title","description","channel","published_at","year","views","likes","comments",
    "match_count","matched_terms","seed_term","lang","video_url"
]
df_clean[keep_cols].to_csv(DST, index=False, encoding="utf-8-sig")
print(f"✅ Финиш: зачувани {len(df_clean)} редови во {DST}")
