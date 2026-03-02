#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 022_bertweet_en_from_prev.py
# BEST-PRACTICES BERTweet sentiment for English video text

import os
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(8)

import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import re

# ---------------------------------------------------------
# 1) CONFIG
# ---------------------------------------------------------
IN_CSV  = "datasets/predictions_cardiff_xlm_en.csv"
OUT_CSV = "datasets/predictions_bertweet_en.csv"
BATCH   = 24
MAX_LEN = 128     # IMPORTANT: BERTweet max stable length
MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"


# ---------------------------------------------------------
# 2) CLEAN + TRIM — REQUIRED for BERTweet
# ---------------------------------------------------------
def clean(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)            # Twitter-style mention
    x = re.sub(r"http\S+|www\.\S+", "URL", x)  # normalize URLs
    x = re.sub(r"\s+", " ", x).strip()
    return x

def trim_for_bertweet(x: str, max_chars=350):
    """Avoid BERTweet position embedding crashes."""
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]


# ---------------------------------------------------------
# 3) Load CSV
# ---------------------------------------------------------
print("📥 Reading:", IN_CSV)
df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

if "title_description_en" not in df.columns:
    raise SystemExit("❌ Column 'title_description_en' not found.")

texts = (
    df["title_description_en"]
    .astype(str)
    .apply(clean)
    .apply(trim_for_bertweet)
    .tolist()
)


# ---------------------------------------------------------
# 4) Load BERTweet model
# ---------------------------------------------------------
print(f"🧠 Loading model: {MODEL_NAME} ...")

clf = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    truncation=True,
    top_k=None,                         # ensure single output
    device=0 if torch.cuda.is_available() else -1
)


# ---------------------------------------------------------
# 5) Batch prediction (stable)
# ---------------------------------------------------------
preds, scores = [], []
print("⚙️ Running BERTweet sentiment analysis...")

for i in tqdm(range(0, len(texts), BATCH), total=(len(texts)+BATCH-1)//BATCH):
    batch = texts[i:i+BATCH]

    outs = clf(batch, truncation=True, max_length=MAX_LEN)

    for o in outs:
        if isinstance(o, list):         # safety for top-k behavior
            o = o[0]
        preds.append(o["label"].lower())
        scores.append(round(float(o["score"]), 4))


# ---------------------------------------------------------
# 6) Save results
# ---------------------------------------------------------
df["pred_bertweet_en"]  = preds
df["score_bertweet_en"] = scores

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("✅ Done:", OUT_CSV)

print("\n📊 sentiment distribution:")
print(df["pred_bertweet_en"].value_counts())
