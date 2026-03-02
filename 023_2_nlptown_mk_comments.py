#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 022_nlptown_mk_comments.py
# Production-grade sentiment analysis for Macedonian comments using NLPTown

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

torch.set_num_threads(8)

IN  = "datasets/predictions_comments_mk_cardiff.csv"   # ист input како кај Cardiff
OUT = "datasets/predictions_comments_mk_nlptown.csv"
BATCH = 32
MAX_LEN = 256

# ---------------------------------------------------------
# 2) Cleaning + Trimming (Best practices)
# ---------------------------------------------------------
def clean(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"@\w+", "@USER", x)                    # normalize mentions
    x = re.sub(r"http\S+|www\.\S+", "URL", x)          # normalize URLs
    x = re.sub(r"#\S+", " ", x)                        # remove hashtags
    x = re.sub(r"\s+", " ", x).strip()
    return x


def trim(x: str, max_chars=350):
    if len(x) <= max_chars:
        return x
    return x[:max_chars].rsplit(" ", 1)[0]

# ---------------------------------------------------------
# 3) NLPTown label mapping
# ---------------------------------------------------------
def map_nlptown_label(label: str) -> str:
    """
    Map NLPTown star labels to:
    positive / neutral / negative
    """
    if label.startswith("1") or label.startswith("2"):
        return "negative"
    if label.startswith("3"):
        return "neutral"
    return "positive"

# ---------------------------------------------------------
# 4) Load dataset
# ---------------------------------------------------------
df = pd.read_csv(IN, encoding="utf-8-sig")

if "text_mk" not in df.columns:
    raise SystemExit("❌ Column 'text_mk' not found!")

df["text_clean"] = df["text_mk"].astype(str).apply(clean).apply(trim)
texts = df["text_clean"].tolist()

print(f"📥 Loaded {len(texts)} MK comments")

# ---------------------------------------------------------
# 5) Load the model
# ---------------------------------------------------------
print("🧠 Loading model: nlptown/bert-base-multilingual-uncased-sentiment ...")

clf = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
    top_k=None,            # NLPTown returns all star classes
    truncation=True,
    device=0 if torch.cuda.is_available() else -1
)

# ---------------------------------------------------------
# 6) Batch inference (safe + fast)
# ---------------------------------------------------------
preds_raw = []
preds_3cls = []
scores = []

print("⚙️ Running MK sentiment inference (NLPTown)...")

for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    outs = clf(batch, truncation=True, max_length=MAX_LEN)

    for o in outs:
        if isinstance(o, list):      # NLPTown safety
            o = o[0]

        raw_label = o["label"].lower()   # e.g. "4 stars"
        score = float(o["score"])

        preds_raw.append(raw_label)
        preds_3cls.append(map_nlptown_label(raw_label))
        scores.append(round(score, 4))

# ---------------------------------------------------------
# 7) Save results
# ---------------------------------------------------------
df["pred_nlptown_raw"]   = preds_raw
df["pred_nlptown_3cls"]  = preds_3cls
df["score_nlptown"]      = scores

df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"✅ Done! Saved → {OUT}")

print("\n📊 Sentiment distribution (NLPTown MK comments):")
print(df["pred_nlptown_3cls"].value_counts())
