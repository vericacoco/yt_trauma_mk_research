#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 022_cardiff_en_latest.py
# Production-grade sentiment for English video text (Cardiff Latest)

import os
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(8)

import re
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
IN_CSV  = "datasets/predictions_bertweet_en.csv"
OUT_CSV = "datasets/predictions_cardiff_en_latest_all.csv"
BATCH   = 24
MAX_LEN = 256


# ---------------------------------------------------------
# CLEAN + TRIM (Twitter-ready)
# ---------------------------------------------------------
def clean(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)
    x = re.sub(r"http\S+|www\.\S+", "URL", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def trim(x: str, max_chars=400):
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]


# ---------------------------------------------------------
# LOAD CSV
# ---------------------------------------------------------
print("📥 Reading:", IN_CSV)
df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

if "title_description_en" not in df.columns:
    raise SystemExit("❌ Missing column: 'title_description_en'.")

texts = (
    df["title_description_en"]
    .astype(str)
    .apply(clean)
    .apply(trim)
    .tolist()
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
print("🧠 Loading model: cardiffnlp/twitter-roberta-base-sentiment-latest ...")

clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)


# ---------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------
preds, scores = [], []
print("⚙️ Running sentiment (Cardiff Latest)...")

for i in tqdm(range(0, len(texts), BATCH), total=(len(texts)+BATCH-1)//BATCH):
    batch = texts[i:i+BATCH]
    outs = clf(batch, truncation=True, max_length=MAX_LEN)

    for o in outs:
        if isinstance(o, list):  # HF safety
            o = o[0]
        preds.append(o["label"].lower())
        scores.append(round(float(o["score"]), 4))


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
df["pred_cardiff_en_latest"]  = preds
df["score_cardiff_en_latest"] = scores

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("✅ Done:", OUT_CSV)
