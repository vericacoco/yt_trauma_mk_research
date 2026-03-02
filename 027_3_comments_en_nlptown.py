#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 025_nlptown_en_comments.py
# Production-grade NLPTown sentiment for EN comments

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
IN  = "datasets/predictions_comments_en_cardiff_xlm.csv"   # RAW EN comments (with text_en)
OUT = "datasets/predictions_comments_en_nlptown_all.csv"

TEXT_COL = "text_en"

BATCH     = 32
MAX_LEN   = 256
MAX_CHARS = 300

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# ----------------------------------------------------------
# Cleaning (Twitter-style, same as Cardiff & BERTweet)
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
# NLPTown label mapping (1–5 stars → 3 classes)
# ----------------------------------------------------------
def map_nlptown_label(label: str) -> str:
    if label.startswith("1") or label.startswith("2"):
        return "negative"
    if label.startswith("3"):
        return "neutral"
    return "positive"

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
# Load NLPTown model
# ----------------------------------------------------------
print("🧠 Loading model:", MODEL_NAME)

clf = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    top_k=None,              # NLPTown returns all star classes
    truncation=True,
    device=0 if torch.cuda.is_available() else -1
)

# ----------------------------------------------------------
# Infer (batch)
# ----------------------------------------------------------
preds_raw  = []
preds_3cls = []
scores     = []

print("⚙️ Running NLPTown sentiment inference...")

for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    outs = clf(batch, truncation=True, max_length=MAX_LEN)

    for o in outs:
        if isinstance(o, list):   # NLPTown safety
            o = o[0]

        raw_label = o["label"].lower()   # e.g. "4 stars"
        score = float(o["score"])

        preds_raw.append(raw_label)
        preds_3cls.append(map_nlptown_label(raw_label))
        scores.append(round(score, 4))

df["pred_nlptown_en_raw"]  = preds_raw
df["pred_nlptown_en_3cls"] = preds_3cls
df["score_nlptown_en"]     = scores

# ----------------------------------------------------------
# Agreement with BERTweet / Cardiff (optional)
# ----------------------------------------------------------
if "pred_cardiff_en" in df.columns:
    df["agree_cardiff_nlptown"] = (
        df["pred_nlptown_en_3cls"] == df["pred_cardiff_en"].str.lower()
    )
    print("📊 Agreement (NLPTown vs Cardiff):",
          round(df["agree_cardiff_nlptown"].mean() * 100, 2), "%")

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"✅ Saved → {OUT}")

print("\n📊 Sentiment distribution:")
print(df["pred_nlptown_en_3cls"].value_counts())
