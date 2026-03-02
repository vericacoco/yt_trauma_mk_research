#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 031_bertweet_en_videos.py
# Production-grade sentiment analysis for EN videos using BERTweet

import os
import re
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer

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
IN_CSV  = "datasets/predictions_cardiff_xlm_en.csv"
OUT_CSV = "datasets/predictions_bertweet_en_videos.csv"

TEXT_COL = "title_description_en"

MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"

BATCH      = 16        # safer on CPU
MAX_TOKENS = 256
MAX_CHARS  = 350

DEVICE = 0 if torch.cuda.is_available() else -1

# -------------------------------------------------
# CLEAN + TRIM (same philosophy as Cardiff)
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
# SAFE TOKEN-LEVEL TRUNCATION (Roberta fix)
# -------------------------------------------------
def safe_truncate_for_bertweet(text: str, tokenizer, max_tokens: int):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

# -------------------------------------------------
# LABEL MAPPING (explicit)
# -------------------------------------------------
def map_bertweet_label(label: str) -> str:
    label = label.upper()
    if label == "POS":
        return "positive"
    if label == "NEU":
        return "neutral"
    if label == "NEG":
        return "negative"
    raise ValueError(f"Unknown label: {label}")

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

# -------------------------------------------------
# LOAD TOKENIZER + MODEL
# -------------------------------------------------
print("🧠 Loading model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

clf = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    truncation=True,
    device=DEVICE
)

# -------------------------------------------------
# PREPARE SAFE TEXTS
# -------------------------------------------------
texts = [
    safe_truncate_for_bertweet(t, tokenizer, MAX_TOKENS)
    for t in df[TEXT_COL].fillna(" ").tolist()
]

# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
preds_raw  = []
preds_3cls = []
scores     = []

print("⚙️ Running BERTweet sentiment inference (EN videos)...")

for i in tqdm(range(0, len(texts), BATCH), total=(len(texts) + BATCH - 1) // BATCH):
    batch = texts[i:i+BATCH]
    outs = clf(batch, truncation=True)

    for o in outs:
        raw_label = o["label"]          # POS / NEU / NEG
        score = float(o["score"])

        preds_raw.append(raw_label)
        preds_3cls.append(map_bertweet_label(raw_label))
        scores.append(round(score, 4))

# -------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------
df["pred_bertweet_en_raw"]  = preds_raw
df["pred_bertweet_en_3cls"] = preds_3cls
df["score_bertweet_en"]     = scores

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("✅ Done:", OUT_CSV)

print("\n📊 BERTweet EN videos sentiment distribution:")
print(df["pred_bertweet_en_3cls"].value_counts())
