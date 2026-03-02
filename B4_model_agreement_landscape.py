#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# B4_model_agreement_landscape.py
# Model Agreement Landscape (MK/EN × Videos/Comments)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results/B4_model_agreement"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# CONFIG (ист стил како B4)
# ---------------------------------------------------------
CONFIG = {
    "MK": {
        "videos": {
            "NLPTown":  ("datasets/predictions_mk_videos_nlptown.csv", "nlptown_label_3cls"),
            "Cardiff":  ("datasets/predictions_cardiff_mk_xlm_sentiment.csv", "pred_cardiff_mk_xlm"),
            "BERTweet": ("datasets/predictions_mk_videos_bertweet_all.csv", "bertweet_label_3cls"),
        },
        "comments": {
            "NLPTown": ("datasets/predictions_comments_mk_nlptown.csv", "pred_nlptown_3cls"),
            "Cardiff": ("datasets/predictions_comments_mk_cardiff.csv", "pred_cardiff_mk_xlm"),
            "BERTweet": ("datasets/predictions_comments_mk_bertweet_all.csv", "pred_bertweet_3cls"),
        },
    },
    "EN": {
        "videos": {
            "NLPTown":  ("datasets/predictions_nlptown_en_videos_all.csv", "pred_nlptown_en_3cls"),
            "Cardiff":  ("datasets/predictions_cardiff_xlm_en.csv", "pred_cardiff_xlm_en"),
            "BERTweet": ("datasets/predictions_bertweet_en_videos.csv", "pred_bertweet_en_3cls"),
        },
        "comments": {
            "NLPTown": ("datasets/predictions_comments_en_nlptown_all.csv", "pred_nlptown_en_3cls"),
            "Cardiff": ("datasets/predictions_comments_en_cardiff_xlm.csv", "pred_cardiff_en"),
            "BERTweet": ("datasets/predictions_comments_en_bertweet_all.csv", "bertweet_mapped"),
        },
    },
}

MODELS = ["NLPTown", "Cardiff", "BERTweet"]

# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------
def load_labels(path, col):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[col].astype(str).values

def disagreement(a, b):
    return (a != b).mean() * 100

def compute_matrix(cfg_section):
    mat = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
    for i in MODELS:
        for j in MODELS:
            if i == j:
                mat.loc[i, j] = 0.0
            else:
                a = load_labels(*cfg_section[i])
                b = load_labels(*cfg_section[j])
                mat.loc[i, j] = round(disagreement(a, b), 2)
    return mat

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

panels = [
    ("EN", "videos", "EN Videos"),
    ("EN", "comments", "EN Comments"),
    ("MK", "videos", "MK Videos"),
    ("MK", "comments", "MK Comments"),
]

for ax, (lang, ctype, title) in zip(axes, panels):
    matrix = compute_matrix(CONFIG[lang][ctype])

    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        vmin=0,
        vmax=100,
        cbar=False
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.suptitle(
    "B4 — Model Agreement Landscape (% Disagreement)",
    fontsize=16
)

plt.tight_layout()
out_path = os.path.join(
    RESULTS_DIR,
    "B4_model_agreement_landscape.svg"
)
plt.savefig(out_path, format="svg")
plt.close()

print(f"✅ FIGURE A saved → {out_path}")
