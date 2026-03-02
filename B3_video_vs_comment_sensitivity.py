#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# B3_video_vs_comment_sensitivity.py
# Improved, interpretable version (paper-ready)

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results/B3_video_vs_comment"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# CONFIG
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

MODEL_PAIRS = [
    ("NLPTown", "Cardiff"),
    ("NLPTown", "BERTweet"),
    ("Cardiff", "BERTweet"),
]

# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------
def load_labels(path, col):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[col].astype(str).values

def disagreement(a, b):
    return (a != b).mean() * 100

# ---------------------------------------------------------
# COMPUTE
# ---------------------------------------------------------
rows = []

for lang, cfg in CONFIG.items():
    for a, b in MODEL_PAIRS:
        # Videos
        v_a = load_labels(*cfg["videos"][a])
        v_b = load_labels(*cfg["videos"][b])
        d_videos = disagreement(v_a, v_b)

        # Comments
        c_a = load_labels(*cfg["comments"][a])
        c_b = load_labels(*cfg["comments"][b])
        d_comments = disagreement(c_a, c_b)

        rows.append({
            "language": lang,
            "model_pair": f"{a} ↔ {b}",
            "videos_disagreement": round(d_videos, 2),
            "comments_disagreement": round(d_comments, 2),
            "delta_comments_minus_videos": round(d_comments - d_videos, 2),
        })

df = pd.DataFrame(rows)

# ---------------------------------------------------------
# SAVE TABLE (for paper)
# ---------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "B3_video_vs_comment_sensitivity_full.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Saved table → {csv_path}")

# ---------------------------------------------------------
# FIGURE B4a: Absolute disagreement
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, lang in zip(axes, ["MK", "EN"]):
    sub = df[df["language"] == lang]
    x = range(len(sub))

    ax.bar(x, sub["videos_disagreement"], label="Videos", alpha=0.8)
    ax.bar(x, sub["comments_disagreement"],
           bottom=sub["videos_disagreement"],
           label="Comments", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sub["model_pair"], rotation=0)
    ax.set_title(f"{lang}: Absolute Disagreement")
    ax.set_ylabel("Disagreement (%)")

axes[0].legend()
plt.tight_layout()

fig_a_path = os.path.join(RESULTS_DIR, "B3a_absolute_disagreement.png")
plt.savefig(fig_a_path, dpi=300)
plt.close()
print(f"📊 Saved → {fig_a_path}")

# ---------------------------------------------------------
# FIGURE B4b: Δ sensitivity (summary)
# ---------------------------------------------------------
pivot = df.pivot(
    index="model_pair",
    columns="language",
    values="delta_comments_minus_videos"
)

ax = pivot.plot(kind="bar", figsize=(10, 5))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Δ Disagreement (%)  (comments − videos)")
ax.set_title("B3 — Video vs Comment Sensitivity")
plt.xticks(rotation=0)
plt.tight_layout()

fig_b_path = os.path.join(RESULTS_DIR, "B3b_delta_sensitivity.png")
plt.savefig(fig_b_path, dpi=300)
plt.close()
print(f"📊 Saved → {fig_b_path}")

print("\n🎯 B3 analysis completed successfully (clean & interpretable).")
