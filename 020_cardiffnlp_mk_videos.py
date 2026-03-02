#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 022_cardiff_mk_sentiment_xlmroberta.py
# Production-grade sentiment analysis for MK videos using XLM-R CardiffNLP

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------------------------------------------------------
# 1) CONFIG
# ---------------------------------------------------------
IN_CSV  = "datasets/predictions_bertweet_en_videos.csv"
OUT_CSV = "datasets/predictions_cardiff_mk_xlm_sentiment.csv"
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH   = 24
MAX_LEN = 256


# ---------------------------------------------------------
# 2) CLEANING + TRIMMING (best practice)
# ---------------------------------------------------------
def clean(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)                  # normalize mentions
    x = re.sub(r"http\S+|www\.\S+", "URL", x)        # normalize URLs
    x = re.sub(r"\s+", " ", x).strip()
    return x

def trim(x: str, max_chars=400):
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]


# ---------------------------------------------------------
# 3) LOAD DATA
# ---------------------------------------------------------
print("📥 Reading:", IN_CSV)
df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

if "title_description_mk" not in df.columns:
    df["title_description_mk"] = (
        df["title_mk"].astype(str).apply(clean).apply(trim)
        + " "
        + df["description_mk"].astype(str).apply(clean).apply(trim)
    ).str.strip()

texts = df["title_description_mk"].fillna("").tolist()


# ---------------------------------------------------------
# 4) LOAD MODEL
# ---------------------------------------------------------
print(f"🧠 Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

clf = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    truncation=True,
    device=0 if torch.cuda.is_available() else -1
)


# ---------------------------------------------------------
# 5) INFERENCE (batch, safe)
# ---------------------------------------------------------
preds, scores = [], []
print("⚙️ Running MK sentiment analysis (XLM-R Cardiff)...")

for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    outs = clf(batch, truncation=True, max_length=MAX_LEN)

    for o in outs:
        if isinstance(o, list):
            o = o[0]

        preds.append(o["label"].lower())
        scores.append(round(float(o["score"]), 4))


# ---------------------------------------------------------
# 6) SAVE RESULTS
# ---------------------------------------------------------
df["pred_cardiff_mk_xlm"]  = preds
df["score_cardiff_mk_xlm"] = scores

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ Saved: {OUT_CSV}")

print("\n📊 Sentiment distribution:")
print(df["pred_cardiff_mk_xlm"].value_counts())
