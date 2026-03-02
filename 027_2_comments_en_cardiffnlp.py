#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 024_cardiff_xlm_en_comments.py
# Production-grade CardiffNLP XLM-R sentiment for EN comments

import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import re
import torch

# ----------------------------------------------------------
# Environment
# ----------------------------------------------------------
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(8)

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
IN  = "datasets/predictions_comments_en_bertweet_all.csv"   # RAW EN comments (with text_en)
OUT = "datasets/predictions_comments_en_cardiff_xlm.csv"

TEXT_COL = "text_en"

BATCH     = 32
MAX_LEN   = 256
MAX_CHARS = 300

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# ----------------------------------------------------------
# Cleaning (Twitter-style, same as BERTweet)
# ----------------------------------------------------------
def clean_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)                 # normalize mentions
    x = re.sub(r"http\S+|www\.\S+", "URL", x)       # normalize URLs
    x = re.sub(r"#\S+", " ", x)                     # remove hashtags
    x = re.sub(r"\s+", " ", x).strip()
    return x

def trim(x: str, max_chars=MAX_CHARS):
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]

# ----------------------------------------------------------
# Load CSV
# ----------------------------------------------------------
print(f"📥 Reading: {IN}")
df = pd.read_csv(IN, encoding="utf-8-sig")

if TEXT_COL not in df.columns:
    raise SystemExit(f"❌ Missing required column: {TEXT_COL}")

df["text_clean"] = df[TEXT_COL].astype(str).apply(clean_text).apply(trim)
texts = df["text_clean"].tolist()

print(f"🔎 Prepared {len(texts)} EN comments")

# ----------------------------------------------------------
# Load CardiffNLP model
# ----------------------------------------------------------
print("🧠 Loading model:", MODEL_NAME)

clf = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    truncation=True,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

# ----------------------------------------------------------
# Infer (batch)
# ----------------------------------------------------------
preds, scores = [], []

print("⚙️ Running Cardiff XLM-R sentiment inference...")

for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    outs = clf(batch, truncation=True, max_length=MAX_LEN)

    for o in outs:
        if isinstance(o, list):   # safety
            o = o[0]

        label = o["label"].lower()     # negative / neutral / positive
        score = float(o["score"])

        preds.append(label)
        scores.append(round(score, 4))

df["pred_cardiff_en"]  = preds
df["score_cardiff_en"] = scores

# ----------------------------------------------------------
# Agreement with BERTweet (optional, if exists)
# ----------------------------------------------------------
if "bertweet_mapped" in df.columns:
    df["agree_cardiff_bertweet"] = (
        df["pred_cardiff_en"] == df["bertweet_mapped"].str.lower()
    )
    print("📊 Agreement (Cardiff vs BERTweet):",
          round(df["agree_cardiff_bertweet"].mean() * 100, 2), "%")

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"✅ Saved → {OUT}")

print("\n📊 Sentiment distribution:")
print(df["pred_cardiff_en"].value_counts())
