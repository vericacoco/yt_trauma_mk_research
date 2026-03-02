#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 032_nlptown_en_videos.py
# Production-grade sentiment analysis for EN videos using NLPTown

import os
import re
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline

# -------------------------------------------------
# ENVIRONMENT
# -------------------------------------------------
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(8)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
IN_CSV  = "datasets/predictions_bertweet_en_videos.csv"
OUT_CSV = "datasets/predictions_nlptown_en_videos_all.csv"

TEXT_COL = "title_description_en"

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

BATCH     = 24
MAX_LEN   = 256
MAX_CHARS = 350

DEVICE = 0 if torch.cuda.is_available() else -1

# -------------------------------------------------
# CLEAN + TRIM (same philosophy as Cardiff & BERTweet)
# -------------------------------------------------
def clean(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)
    x = re.sub(r"http\S+|www\.\S+", "URL", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def trim_chars(x: str, max_chars=MAX_CHARS):
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]

# -------------------------------------------------
# NLPTown label mapping (1–5 stars → 3 classes)
# -------------------------------------------------
def map_nlptown_label(label: str) -> str:
    if label.startswith("1") or label.startswith("2"):
        return "negative"
    if label.startswith("3"):
        return "neutral"
    return "positive"

# -------------------------------------------------
# LOAD CSV
# -------------------------------------------------
print("📥 Reading:", IN_CSV)
df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

if TEXT_COL not in df.columns:
    if not {"title_en", "description_en"}.issubset(df.columns):
        raise SystemExit("❌ CSV мора да има 'title_description_en' или ('title_en','description_en').")

    df[TEXT_COL] = (
        df["title_en"].astype(str).apply(clean) + " " +
        df["description_en"].astype(str).apply(clean)
    ).apply(trim_chars)

df[TEXT_COL] = df[TEXT_COL].astype(str)

texts = df[TEXT_COL].fillna(" ").tolist()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
print("🧠 Loading model:", MODEL_NAME)

clf = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    top_k=None,              # NLPTown returns all star classes
    truncation=True,
    device=DEVICE
)

# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
preds_raw  = []
preds_3cls = []
scores     = []

print("⚙️ Running NLPTown sentiment inference (EN videos)...")

for i in tqdm(range(0, len(texts), BATCH), total=(len(texts) + BATCH - 1) // BATCH):
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

# -------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------
df["pred_nlptown_en_raw"]  = preds_raw
df["pred_nlptown_en_3cls"] = preds_3cls
df["score_nlptown_en"]     = scores

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("✅ Done:", OUT_CSV)

print("\n📊 NLPTown EN videos sentiment distribution:")
print(df["pred_nlptown_en_3cls"].value_counts())
