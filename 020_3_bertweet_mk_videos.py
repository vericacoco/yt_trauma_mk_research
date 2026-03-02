#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERTweet sentiment analysis for Macedonian videos via English translation (MK → EN)

Model:
    finiteautomata/bertweet-base-sentiment-analysis

Input:
    Macedonian video titles + descriptions translated to English

Output:
    - raw BERTweet label (POS / NEU / NEG)
    - confidence score
    - unified sentiment label:
        positive / neutral / negative

NOTE:
BERTweet is NOT applied directly to Macedonian text.
It is applied to MK → EN translated content, which is methodologically correct.
"""

# ---------------------------------------------------------
# 0) ENVIRONMENT SETUP
# ---------------------------------------------------------
import os
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(8)

# ---------------------------------------------------------
# 1) IMPORTS
# ---------------------------------------------------------
import re
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# ---------------------------------------------------------
# 2) CONFIGURATION
# ---------------------------------------------------------
IN_CSV  = "datasets/predictions_mk_videos_nlptown.csv"
OUT_CSV = "datasets/predictions_mk_videos_bertweet_all.csv"

MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"

BATCH_SIZE = 16          # safer on CPU
MAX_TOKENS = 256         # token-level limit (NOT chars)
MAX_CHARS  = 400         # pre-trim for safety

DEVICE = -1              # force CPU (stable)

# ---------------------------------------------------------
# 3) TEXT CLEANING (BERTweet-style)
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "URL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def trim_chars(text: str, max_chars: int = MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]

# ---------------------------------------------------------
# 4) SAFE TOKEN-LEVEL TRUNCATION (CRITICAL FIX)
# ---------------------------------------------------------
def safe_truncate_for_bertweet(text: str, tokenizer, max_tokens: int) -> str:
    """
    Token-level truncation to prevent Roberta position embedding overflow.
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

# ---------------------------------------------------------
# 5) LABEL MAPPING
# ---------------------------------------------------------
def map_bertweet_label(label: str) -> str:
    """
    Map BERTweet output labels to unified classes.
    """
    label = label.upper()

    if label == "POS":
        return "positive"
    if label == "NEU":
        return "neutral"
    if label == "NEG":
        return "negative"

    raise ValueError(f"Unknown BERTweet label: {label}")

# ---------------------------------------------------------
# 6) LOAD DATA
# ---------------------------------------------------------
print(f"📥 Loading input data: {IN_CSV}")
df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

if "title_description_mk" not in df.columns:
    df["title_description_mk"] = (
        df["title_mk"].astype(str).apply(clean_text).apply(trim_chars)
        + " "
        + df["description_mk"].astype(str).apply(clean_text).apply(trim_chars)
    ).str.strip()

# ---------------------------------------------------------
# 7) LOAD MODEL + TOKENIZER
# ---------------------------------------------------------
print(f"🧠 Loading BERTweet model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE
)

# ---------------------------------------------------------
# 8) PREPARE TEXTS (SAFE)
# ---------------------------------------------------------
texts = [
    safe_truncate_for_bertweet(t, tokenizer, MAX_TOKENS)
    for t in df["title_description_mk"].fillna("").tolist()
]

# ---------------------------------------------------------
# 9) INFERENCE
# ---------------------------------------------------------
raw_labels = []
raw_scores = []
labels_3cls = []

print("⚙️ Running BERTweet sentiment inference (MK)...")

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i : i + BATCH_SIZE]
    outputs = sentiment_pipeline(batch, truncation=True)

    for out in outputs:
        raw_label = out["label"]          # POS / NEU / NEG
        score = float(out["score"])

        raw_labels.append(raw_label)
        raw_scores.append(round(score, 4))
        labels_3cls.append(map_bertweet_label(raw_label))

# ---------------------------------------------------------
# 10) SAVE RESULTS
# ---------------------------------------------------------
df["bertweet_label_raw"]  = raw_labels
df["bertweet_score"]      = raw_scores
df["bertweet_label_3cls"] = labels_3cls

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# 11) SANITY CHECK
# ---------------------------------------------------------
print(f"\n✅ Saved predictions to: {OUT_CSV}")
print("\n📊 BERTweet unified sentiment distribution:")
print(df["bertweet_label_3cls"].value_counts())
