#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 023_3_bertweet_mk_comments_direct.py
# BERTweet DIRECTLY on Macedonian comments (out-of-domain / ablation experiment)

"""
SCIENTIFIC NOTE:
BERTweet is an English-only model.
This script applies it directly to Macedonian comments
as an OUT-OF-DOMAIN / ABLATION experiment.
Results are NOT interpreted as valid MK sentiment.
"""

# ---------------------------------------------------------
# 1) Environment
# ---------------------------------------------------------
import os
import re
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    pipeline,
    AutoTokenizer
)

os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(8)

# ---------------------------------------------------------
# 2) CONFIG
# ---------------------------------------------------------
IN  = "datasets/predictions_comments_mk_nlptown.csv"
OUT = "datasets/predictions_comments_mk_bertweet_all.csv"

TEXT_COL = "text_mk"   # Macedonian text (INTENTIONALLY)

MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"

BATCH     = 16          # safer on CPU
MAX_TOKENS = 256        # token-level truncation
MAX_CHARS  = 350

DEVICE = -1             # force CPU for stability

# ---------------------------------------------------------
# 3) Cleaning + Trimming (language-neutral)
# ---------------------------------------------------------
def clean(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)
    x = re.sub(r"http\S+|www\.\S+", "URL", x)
    x = re.sub(r"#\S+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def trim_chars(x: str, max_chars=MAX_CHARS):
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]

# ---------------------------------------------------------
# 4) SAFE TOKEN-LEVEL TRUNCATION (CRITICAL FIX)
# ---------------------------------------------------------
def safe_truncate_for_bertweet(text: str, tokenizer, max_tokens: int):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

# ---------------------------------------------------------
# 5) BERTweet label mapping
# ---------------------------------------------------------
def map_bertweet_label(label: str) -> str:
    label = label.upper()
    if label == "POS":
        return "positive"
    if label == "NEU":
        return "neutral"
    if label == "NEG":
        return "negative"
    raise ValueError(f"Unknown BERTweet label: {label}")

# ---------------------------------------------------------
# 6) Load dataset
# ---------------------------------------------------------
print(f"📥 Reading comments: {IN}")
df = pd.read_csv(IN, encoding="utf-8-sig")

if TEXT_COL not in df.columns:
    raise SystemExit(f"❌ Required column '{TEXT_COL}' not found!")

df["text_mk"] = df[TEXT_COL].astype(str)
df["text_clean"] = df["text_mk"].apply(clean).apply(trim_chars)

print(f"📥 Loaded {len(df)} MK comments")

# ---------------------------------------------------------
# 7) Load tokenizer + model
# ---------------------------------------------------------
print("🧠 Loading BERTweet tokenizer + model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

clf = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    truncation=True,
    device=DEVICE
)

# ---------------------------------------------------------
# 8) Prepare SAFE texts (token-truncated)
# ---------------------------------------------------------
texts = [
    safe_truncate_for_bertweet(t, tokenizer, MAX_TOKENS)
    for t in df["text_clean"].tolist()
]

# ---------------------------------------------------------
# 9) Inference
# ---------------------------------------------------------
preds_raw  = []
preds_3cls = []
scores     = []

print("⚙️ Running BERTweet DIRECTLY on MK comments (ablation)...")

for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    outs = clf(batch, truncation=True)

    for o in outs:
        raw_label = o["label"]       # POS / NEU / NEG
        score = float(o["score"])

        preds_raw.append(raw_label)
        preds_3cls.append(map_bertweet_label(raw_label))
        scores.append(round(score, 4))

# ---------------------------------------------------------
# 10) Save results (KEEP text_mk)
# ---------------------------------------------------------
df["pred_bertweet_raw"]  = preds_raw
df["pred_bertweet_3cls"] = preds_3cls
df["score_bertweet"]     = scores

df.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"✅ Done! Saved → {OUT}")
print("\n📊 BERTweet DIRECT MK sentiment distribution:")
print(df["pred_bertweet_3cls"].value_counts())
