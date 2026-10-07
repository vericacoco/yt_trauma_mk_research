#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# B5_crosslingual_class_flow_EN_MK_only.py
# Cross-lingual class flow ONLY for models that support MK via cross-lingual setup:
# NLPTown, Cardiff (EN → MK).  NO BERTweet panels.

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------
RESULTS_DIR = "results/B5_crosslingual_flow"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# CONFIG — NLPTown & Cardiff (EN → MK)
# ---------------------------------------------------------
CONFIG = {
    "videos": {
        "NLPTown": (
            "datasets/predictions_nlptown_en_videos_all.csv",
            "datasets/predictions_mk_videos_nlptown.csv",
            "pred_nlptown_en_3cls",
            "nlptown_label_3cls",
        ),
        "Cardiff": (
            "datasets/predictions_cardiff_xlm_en.csv",
            "datasets/predictions_cardiff_mk_xlm_sentiment.csv",
            "pred_cardiff_xlm_en",
            "pred_cardiff_mk_xlm",
        ),
    },
    "comments": {
        "NLPTown": (
            "datasets/predictions_comments_en_nlptown_all.csv",
            "datasets/predictions_comments_mk_nlptown.csv",
            "pred_nlptown_en_3cls",
            "pred_nlptown_3cls",
        ),
        "Cardiff": (
            "datasets/predictions_comments_en_cardiff_xlm.csv",
            "datasets/predictions_comments_mk_cardiff.csv",
            "pred_cardiff_en",
            "pred_cardiff_mk_xlm",
        ),
    },
}

CLASSES = ["negative", "neutral", "positive"]

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def load_pair(path_a, path_b, col_a, col_b):
    df_a = pd.read_csv(path_a, encoding="utf-8-sig")
    df_b = pd.read_csv(path_b, encoding="utf-8-sig")
    return pd.DataFrame({
        "A": df_a[col_a].astype(str),
        "B": df_b[col_b].astype(str),
    })

def class_flow_matrix(df):
    mat = pd.crosstab(df["A"], df["B"], normalize="index") * 100
    return mat.reindex(index=CLASSES, columns=CLASSES)

# ---------------------------------------------------------
# FIGURE (2 × 2)
# ---------------------------------------------------------
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)

panels = [
    ("videos", "NLPTown", 0, 0),
    ("videos", "Cardiff", 0, 1),
    ("comments", "NLPTown", 1, 0),
    ("comments", "Cardiff", 1, 1),
]

for ctype, model, r, c in panels:
    ax = fig.add_subplot(gs[r, c])
    en_path, mk_path, en_col, mk_col = CONFIG[ctype][model]

    df = load_pair(en_path, mk_path, en_col, mk_col)
    mat = class_flow_matrix(df)

    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar=False
    )

    ax.set_title(f"{model} — {ctype.capitalize()} (EN → MK)")
    ax.set_xlabel("MK sentiment")
    ax.set_ylabel("EN sentiment")

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
plt.suptitle("B5 — Cross-lingual Class Flow (EN → MK)", fontsize=16)
plt.tight_layout()

out_path = os.path.join(RESULTS_DIR, "B5_crosslingual_class_flow_EN_MK_only.png")
plt.savefig(out_path, format="png")
plt.close()

print(f"✅ B5 saved → {out_path}")
