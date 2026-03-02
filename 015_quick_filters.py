import pandas as pd
import csv

SRC = "datasets/comments.csv"  # стави го твојот извор
DST = "datasets/comments_fixed.csv"  # нов чист CSV
PRINT_SAMPLE = True

def try_read(src):
    # 1) најчесто доволно:
    try:
        return pd.read_csv(src, encoding="utf-8-sig")
    except Exception:
        pass
    # 2) Python engine + skip bad lines:
    try:
        return pd.read_csv(src, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
    except Exception:
        pass
    # 3) quote/escape подесувања:
    try:
        return pd.read_csv(src, encoding="utf-8-sig", engine="python",
                           on_bad_lines="skip", quotechar='"', escapechar='\\')
    except Exception:
        pass
    # 4) ако delimiter е ; (ретко, но пробај):
    return pd.read_csv(src, encoding="utf-8-sig", engine="python",
                       on_bad_lines="skip", sep=";", quotechar='"', escapechar='\\')

df = try_read(SRC)

# Избриши целосно празни редови/колони
df = df.dropna(how="all").dropna(axis=1, how="all")

# Нормализирај очекувани колони ако ги имаш
expected = ["video_id","thread_id","comment_id","parent_id","is_reply",
            "author_display_name","author_channel_id","text","text_length",
            "like_count","published_at","updated_at"]
for c in expected:
    if c not in df.columns:
        df[c] = None

# Преуреди ги колоните (дополнителните ќе останат на крај)
df = df[[c for c in expected if c in df.columns] + [c for c in df.columns if c not in expected]]

# Конзистентни типови
df["like_count"] = pd.to_numeric(df.get("like_count", 0), errors="coerce").fillna(0).astype(int)
df["text"] = df.get("text", "").astype(str)

# Сними нов, коректно квотиран CSV
df.to_csv(DST, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
print(f"✅ Fixed CSV saved → {DST}  (rows={len(df)}, cols={len(df.columns)})")

if PRINT_SAMPLE:
    print(df.head(5))
