#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLPTown sentiment analysis for Macedonian video texts (title + description)

- Model: nlptown/bert-base-multilingual-uncased-sentiment
- Input: Macedonian video metadata
- Output:
    - raw NLPTown label (1–5 stars)
    - confidence score
    - unified 3-class sentiment (NEG / NEU / POS)

This script is production- and paper-ready.
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
IN_CSV  = "datasets/predictions_cardiff_mk_xlm_sentiment.csv"
OUT_CSV = "datasets/predictions_mk_videos_nlptown.csv"

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

BATCH_SIZE = 24
MAX_LEN    = 256
MAX_CHARS  = 400

DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------------------------------------------------
# 3) TEXT CLEANING & NORMALIZATION
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Normalize text while preserving sentiment-bearing tokens.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"http\S+|www\.\S+", "URL", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def trim_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """
    Character-level trimming to avoid overlong inputs.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]

# ---------------------------------------------------------
# 4) NLPTOWN → 3-CLASS MAPPING
# ---------------------------------------------------------
def map_nlptown_to_3class(label: str) -> str:
    """
    Map NLPTown star ratings to negative / neutral / positive.
    """
    if label.startswith("1") or label.startswith("2"):
        return "negative"
    if label.startswith("3"):
        return "neutral"
    return "positive"

# ---------------------------------------------------------
# 5) LOAD DATA
# ---------------------------------------------------------
print(f"📥 Loading input data: {IN_CSV}")
df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

# Build MK video text if not already present
if "title_description_mk" not in df.columns:
    df["title_description_mk"] = (
        df["title_mk"].astype(str).apply(clean_text).apply(trim_text)
        + " "
        + df["description_mk"].astype(str).apply(clean_text).apply(trim_text)
    ).str.strip()

texts = df["title_description_mk"].fillna("").tolist()

# ---------------------------------------------------------
# 6) LOAD MODEL
# ---------------------------------------------------------
print(f"🧠 Loading NLPTown model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    truncation=True,
    top_k=None  # NLPTown returns all star classes
)

# ---------------------------------------------------------
# 7) INFERENCE
# ---------------------------------------------------------
raw_labels = []
raw_scores = []

print("⚙️ Running NLPTown sentiment inference (MK videos)...")

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i : i + BATCH_SIZE]
    outputs = sentiment_pipeline(batch, max_length=MAX_LEN)

    for out in outputs:
        # NLPTown returns a list of all star classes; take highest probability
        if isinstance(out, list):
            out = out[0]

        raw_labels.append(out["label"].lower())
        raw_scores.append(round(float(out["score"]), 4))

# ---------------------------------------------------------
# 8) SAVE RESULTS
# ---------------------------------------------------------
df["nlptown_label_raw"]   = raw_labels
df["nlptown_score"]       = raw_scores
df["nlptown_label_3cls"]  = df["nlptown_label_raw"].apply(map_nlptown_to_3class)

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# 9) QUICK SANITY CHECK
# ---------------------------------------------------------
print(f"\n✅ Saved predictions to: {OUT_CSV}")
print("\n📊 NLPTown 3-class sentiment distribution:")
print(df["nlptown_label_3cls"].value_counts())
