#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 024_english_comments_cardiff_only.py
# Production-grade Cardiff sentiment for English comments

import os
import re
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import torch

# ----------------------------------------------------
# ENVIRONMENT
# ----------------------------------------------------
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
IN = "datasets/comments_mk_en.csv"
OUT = "datasets/predictions_comments_en_cardiff.csv"
MAX_LEN = 256
BATCH = 32


# ----------------------------------------------------
# CLEAN + TRIM (Twitter-ready)
# ----------------------------------------------------
def clean(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)                 # normalize mentions
    x = re.sub(r"http\S+|www\.\S+", "URL", x)       # normalize URLs
    x = re.sub(r"#\S+", " ", x)                     # remove hashtags
    x = re.sub(r"\s+", " ", x).strip()
    return x

def trim(x: str, max_chars=400):
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]


# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
df = pd.read_csv(IN, encoding="utf-8-sig")

if "text_en" not in df.columns:
    raise SystemExit("❌ Missing 'text_en' column in dataset.")

df["text_clean"] = df["text_en"].astype(str).apply(clean).apply(trim)

texts = df["text_clean"].tolist()
print(f"📥 Loaded {len(texts)} comments")


# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
print("🔄 Loading model: cardiffnlp/twitter-roberta-base-sentiment-latest ...")

sent_cardiff = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)


# ----------------------------------------------------
# INFERENCE (batch, safe)
# ----------------------------------------------------


preds = []
scores = []

print("⚙️ Running Cardiff sentiment inference...")

for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    outs = sent_cardiff(batch, truncation=True, max_length=MAX_LEN)

    for o in outs:
        if isinstance(o, list):    # handle list output
            o = o[0]

        preds.append(o["label"].lower())
        scores.append(round(float(o["score"]), 4))


# ----------------------------------------------------
# SAVE OUTPUT
# ----------------------------------------------------
df["pred_cardiff_en"] = preds
df["score_cardiff_en"] = scores

df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"🎯 Done! Saved to: {OUT}")
