#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch

torch.set_num_threads(8)

import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import re

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
IN_CSV = "datasets/videos_mk_en_combined.csv"
OUT_CSV = "datasets/predictions_cardiff_xlm_en.csv"
BATCH = 24
MAX_LEN = 256


# -------------------------------------------------
# CLEAN + TRIM FUNCTIONS (best practice)
# -------------------------------------------------
def clean(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)  # Twitter mentions
    x = re.sub(r"http\S+|www\.\S+", "URL", x)  # URLs → URL
    x = re.sub(r"\s+", " ", x).strip()
    return x


def trim(x: str, max_chars=350):
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]


# -------------------------------------------------
# LOAD CSV
# -------------------------------------------------
print("📥 Reading:", IN_CSV)
df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

if "title_description_en" not in df.columns:
    if not {"title_en", "description_en"}.issubset(df.columns):
        raise SystemExit("❌ CSV мора да има 'title_description_en' или ('title_en','description_en').")
    df["title_description_en"] = (
            df["title_en"].astype(str).apply(clean) + " " +
            df["description_en"].astype(str).apply(clean)
    ).apply(trim)

texts = df["title_description_en"].fillna(" ").tolist()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
print("🧠 Loading model: cardiffnlp/twitter-xlm-roberta-base-sentiment ...")

clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    truncation=True,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

# -------------------------------------------------
# BATCH INFERENCE (stable)
# -------------------------------------------------
preds, scores = [], []

for i in tqdm(range(0, len(texts), BATCH), total=(len(texts) + BATCH - 1) // BATCH):
    batch = texts[i:i + BATCH]
    outs = clf(batch, truncation=True, max_length=MAX_LEN)

    for o in outs:
        if isinstance(o, list):
            o = o[0]
        preds.append(o["label"].lower())
        scores.append(round(float(o["score"]), 4))

# -------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------
df["pred_cardiff_xlm_en"] = preds
df["score_cardiff_xlm_en"] = scores

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("✅ Done:", OUT_CSV)
