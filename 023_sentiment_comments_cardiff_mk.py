#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 022_cardiff_xlm_mk_comments.py
# Production-grade sentiment analysis for Macedonian comments using Cardiff XLM-R

import os
import re
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import torch

# ---------------------------------------------------------
# 1) Environment
# ---------------------------------------------------------
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IN  = "datasets/comments_mk_en.csv"
OUT = "datasets/predictions_comments_mk_cardiff.csv"
BATCH = 32
MAX_LEN = 256

# ---------------------------------------------------------
# 2) Cleaning + Trimming (Best practices)
# ---------------------------------------------------------
def clean(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)                 # normalize mentions
    x = re.sub(r"http\S+|www\.\S+", "URL", x)       # normalize URLs
    x = re.sub(r"#\S+", " ", x)                     # remove hashtags
    x = re.sub(r"\s+", " ", x).strip()
    return x

def trim(x: str, max_chars=350):
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]

# ---------------------------------------------------------
# 3) Load dataset
# ---------------------------------------------------------
df = pd.read_csv(IN, encoding="utf-8-sig")

if "text_mk" not in df.columns:
    raise SystemExit("❌ Column 'text_mk' not found!")

df["text_clean"] = df["text_mk"].astype(str).apply(clean).apply(trim)
texts = df["text_clean"].tolist()

print(f"📥 Loaded {len(texts)} MK comments")

# ---------------------------------------------------------
# 4) Load the model
# ---------------------------------------------------------
print("🧠 Loading model: cardiffnlp/twitter-xlm-roberta-base-sentiment ...")

clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    top_k=None,
    truncation=True,
    device=0 if torch.cuda.is_available() else -1
)

# ---------------------------------------------------------
# 5) Batch inference (safe + fast)
# ---------------------------------------------------------
preds = []
scores = []

print("⚙️ Running MK sentiment inference (XLM-R)...")

for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    outs = clf(batch, truncation=True, max_length=MAX_LEN)

    for o in outs:
        if isinstance(o, list):      # safety
            o = o[0]
        preds.append(o["label"].lower())
        scores.append(round(float(o["score"]), 4))

# ---------------------------------------------------------
# 6) Save results
# ---------------------------------------------------------
df["pred_cardiff_mk_xlm"]  = preds
df["score_cardiff_mk_xlm"] = scores

df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"✅ Done! Saved → {OUT}")

print("\n📊 Sentiment distribution:")
print(df["pred_cardiff_mk_xlm"].value_counts())
