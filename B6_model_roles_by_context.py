#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# B6 — Model Roles by Context (CSV-based, robust)

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results/B6_model_roles"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# CONFIG (ТОЧНО како што побара)
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

BERTWEET = {
    "videos": (
        "datasets/predictions_bertweet_en_videos.csv",
        "pred_bertweet_en_3cls",
    ),
    "comments": (
        "datasets/predictions_comments_en_bertweet_all.csv",
        "bertweet_mapped",
    ),
}

CLASSES = ["negative", "neutral", "positive"]

# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------
def disagreement(a, b):
    return (a != b).mean()

def load_pair_safe(en_path, mk_path, en_col, mk_col):
    df_en = pd.read_csv(en_path, encoding="utf-8-sig")
    df_mk = pd.read_csv(mk_path, encoding="utf-8-sig")

    n = min(len(df_en), len(df_mk))
    return (
        df_en[en_col].astype(str).iloc[:n].values,
        df_mk[mk_col].astype(str).iloc[:n].values,
    )

def bertweet_consistency(path, col):
    """
    Distribution-based consistency proxy for BERTweet.
    Higher entropy -> lower agreement.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    p = df[col].astype(str).value_counts(normalize=True)
    entropy = -(p * np.log(p + 1e-12)).sum()
    max_entropy = np.log(len(CLASSES))
    normalized_entropy = entropy / max_entropy
    return 1 - normalized_entropy

# ---------------------------------------------------------
# COMPUTE AGREEMENT SCORES
# ---------------------------------------------------------
rows = []

for context in ["videos", "comments"]:
    # NLPTown & Cardiff (EN ↔ MK)
    for model in ["NLPTown", "Cardiff"]:
        en_path, mk_path, en_col, mk_col = CONFIG[context][model]
        en_labels, mk_labels = load_pair_safe(
            en_path, mk_path, en_col, mk_col
        )

        agree = 1 - disagreement(en_labels, mk_labels)

        rows.append({
            "context": context.capitalize(),
            "model": model,
            "agreement": round(agree * 100, 2),
        })

    # BERTweet (EN-only baseline)
    bt_path, bt_col = BERTWEET[context]
    agree_bt = bertweet_consistency(bt_path, bt_col)

    rows.append({
        "context": context.capitalize(),
        "model": "BERTweet",
        "agreement": round(agree_bt * 100, 2),
    })

df_plot = pd.DataFrame(rows)

# ---------------------------------------------------------
# PLOT (clean & readable)
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

markers = {
    "NLPTown": "o",
    "Cardiff": "s",
    "BERTweet": "^",
}

for model, marker in markers.items():
    subset = df_plot[df_plot["model"] == model]
    plt.plot(
        subset["context"],
        subset["agreement"],
        marker=marker,
        linewidth=2,
        label=model
    )

    for _, r in subset.iterrows():
        plt.text(
            r["context"],
            r["agreement"] + 1,
            f"{r['agreement']:.1f}",
            ha="center",
            fontsize=9
        )

plt.ylabel("Agreement score (%)")
plt.xlabel("Content type")
plt.title("B6 — Model Roles by Context")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(title="Model")

plt.tight_layout()
out_path = os.path.join(
    RESULTS_DIR,
    "B6_model_roles_by_context.png"
)
plt.savefig(out_path, dpi=300)
plt.close()

print(f"✅ B6 saved → {out_path}")
