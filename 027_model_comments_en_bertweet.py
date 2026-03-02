#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 023_bertweet_en_comments.py
# Production-grade BERTweet sentiment for EN comments

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
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IN  = "datasets/predictions_comments_en_cardiff.csv"
OUT = "datasets/predictions_comments_en_bertweet_all.csv"

# ----------------------------------------------------------
# Cleaning (Twitter-style)
# ----------------------------------------------------------
def clean_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)                 # normalize mentions
    x = re.sub(r"http\S+|www\.\S+", "URL", x)       # normalize URLs
    x = re.sub(r"#\S+", " ", x)                     # remove hashtags
    x = re.sub(r"\s+", " ", x).strip()
    return x

def trim(x: str, max_chars=300):
    """Avoid BERTweet crashes and improve stability."""
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]

# ----------------------------------------------------------
# Load CSV
# ----------------------------------------------------------
print(f"📥 Reading: {IN}")
df = pd.read_csv(IN, encoding="utf-8-sig")

if "text_en" not in df.columns:
    raise SystemExit("❌ Missing required column: text_en")

df["text_clean"] = df["text_en"].astype(str).apply(clean_text).apply(trim)
texts = df["text_clean"].tolist()

print(f"🔎 Prepared {len(texts)} comments")

# ----------------------------------------------------------
# Load BERTweet model
# ----------------------------------------------------------
print("🧠 Loading model: finiteautomata/bertweet-base-sentiment-analysis ...")

clf = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis",
    tokenizer="finiteautomata/bertweet-base-sentiment-analysis",
    truncation=True,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

# ----------------------------------------------------------
# Infer (batch)
# ----------------------------------------------------------
preds, scores = [], []

print("⚙️ Running BERTweet sentiment inference...")

BATCH = 32
for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    outs = clf(batch, truncation=True, max_length=128)

    for o in outs:
        if isinstance(o, list):      # safety
            o = o[0]

        label = o["label"].lower()   # pos / neu / neg
        score = float(o["score"])

        preds.append(label)
        scores.append(round(score, 4))

df["pred_bertweet_en"]  = preds
df["score_bertweet_en"] = scores

# ----------------------------------------------------------
# Agreement with Cardiff
# ----------------------------------------------------------
map_bt = {"neg": "negative", "neu": "neutral", "pos": "positive"}
df["bertweet_mapped"] = df["pred_bertweet_en"].map(map_bt)

if "pred_cardiff_en" in df.columns:
    df["agree_cardiff_bertweet"] = (
        df["bertweet_mapped"] == df["pred_cardiff_en"].str.lower()
    )
    print("📊 Agreement:", round(df["agree_cardiff_bertweet"].mean() * 100, 2), "%")

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"✅ Saved → {OUT}")

print("\n📊 Sentiment distribution:")
print(df["pred_bertweet_en"].value_counts())
