#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# B5_crosslingual_class_flow_EN_MK_only.py
#
# Cross-lingual class flow EN -> MK for:
# NLPTown, Cardiff, Gemma and Llama
#
# Row 1: Videos/Titles
# Row 2: Comments

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
# CONFIG
#
# Format:
# model: (
#     EN file,
#     MK file,
#     EN label column,
#     MK label column
# )
#
# For Gemma/Llama EN and MK are in the SAME file,
# therefore the same path is intentionally used twice.
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

        "Gemma": (
            "outputs/titles_llm_gemma_probabilities.csv",
            "outputs/titles_llm_gemma_probabilities.csv",
            "gemma_label_en",
            "gemma_label_mk",
        ),

        "Llama": (
            "outputs/titles_llm_llama_probabilities.csv",
            "outputs/titles_llm_llama_probabilities.csv",
            "llama_label_en",
            "llama_label_mk",
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

        "Gemma": (
            "outputs/gemma_comments_sentiment.csv",
            "outputs/gemma_comments_sentiment.csv",
            "gemma_label_en_c",
            "gemma_label_mk_c",
        ),

        "Llama": (
            "outputs/llama_comments_sentiment.csv",
            "outputs/llama_comments_sentiment.csv",
            "llama_label_en_c",
            "llama_label_mk_c",
        ),
    },
}

CLASSES = ["negative", "neutral", "positive"]


# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def load_pair(path_en, path_mk, col_en, col_mk):

    if not os.path.exists(path_en):
        raise FileNotFoundError(
            f"\n❌ EN file not found:\n{path_en}"
        )

    if not os.path.exists(path_mk):
        raise FileNotFoundError(
            f"\n❌ MK file not found:\n{path_mk}"
        )

    df_en = pd.read_csv(path_en, encoding="utf-8-sig")
    df_mk = pd.read_csv(path_mk, encoding="utf-8-sig")

    # Check columns
    if col_en not in df_en.columns:
        raise KeyError(
            f"\n❌ Column '{col_en}' not found in:\n"
            f"{path_en}\n\n"
            f"Available columns:\n{df_en.columns.tolist()}"
        )

    if col_mk not in df_mk.columns:
        raise KeyError(
            f"\n❌ Column '{col_mk}' not found in:\n"
            f"{path_mk}\n\n"
            f"Available columns:\n{df_mk.columns.tolist()}"
        )

    # Important:
    # EN and MK predictions must refer to the same examples
    n = min(len(df_en), len(df_mk))

    if len(df_en) != len(df_mk):
        print(
            f"⚠️ Different number of rows: "
            f"EN={len(df_en)}, MK={len(df_mk)}. "
            f"Using first {n} rows."
        )

    df = pd.DataFrame({
        "EN": df_en[col_en].iloc[:n].astype(str).str.lower().str.strip(),
        "MK": df_mk[col_mk].iloc[:n].astype(str).str.lower().str.strip(),
    })

    # Keep only valid sentiment labels
    df = df[
        df["EN"].isin(CLASSES)
        & df["MK"].isin(CLASSES)
    ].copy()

    return df


def class_flow_matrix(df):
    """
    Rows = EN prediction
    Columns = MK prediction

    Each row sums to 100%.

    Example:
    EN negative -> MK negative / neutral / positive
    """

    mat = pd.crosstab(
        df["EN"],
        df["MK"],
        normalize="index"
    ) * 100

    mat = mat.reindex(
        index=CLASSES,
        columns=CLASSES,
        fill_value=0
    )

    return mat


# ---------------------------------------------------------
# FIGURE: 2 x 4
# ---------------------------------------------------------
sns.set_theme(style="white")

fig, axes = plt.subplots(
    2,
    4,
    figsize=(20, 10)
)

models = [
    "NLPTown",
    "Cardiff",
    "Gemma",
    "Llama"
]

content_types = [
    ("videos", "Videos / Titles"),
    ("comments", "Comments")
]


# ---------------------------------------------------------
# CREATE PANELS
# ---------------------------------------------------------
for row, (ctype, display_name) in enumerate(content_types):

    for col, model in enumerate(models):

        ax = axes[row, col]

        en_path, mk_path, en_col, mk_col = CONFIG[ctype][model]

        print()
        print("=" * 60)
        print(f"{model} — {display_name}")
        print(f"EN: {en_path} -> {en_col}")
        print(f"MK: {mk_path} -> {mk_col}")

        df = load_pair(
            en_path,
            mk_path,
            en_col,
            mk_col
        )

        print(f"Valid paired observations: {len(df)}")

        mat = class_flow_matrix(df)

        print("\nCross-lingual matrix:")
        print(mat.round(1))

        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            vmin=0,
            vmax=100,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            square=True,
            annot_kws={"fontsize": 10}
        )

        ax.set_title(
            f"{model} — {display_name}\n(EN → MK)",
            fontsize=12,
            fontweight="bold"
        )

        # Only bottom row needs x labels
        if row == 1:
            ax.set_xlabel(
                "Macedonian sentiment",
                fontsize=10
            )
        else:
            ax.set_xlabel("")

        # Only first column needs y labels
        if col == 0:
            ax.set_ylabel(
                "English sentiment",
                fontsize=10
            )
        else:
            ax.set_ylabel("")

        ax.tick_params(
            axis="x",
            rotation=0
        )

        ax.tick_params(
            axis="y",
            rotation=0
        )


# ---------------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------------
fig.suptitle(
    "B5 — Cross-lingual Sentiment Class Flow (EN → MK)",
    fontsize=18,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
out_path = os.path.join(
    RESULTS_DIR,
    "B5_crosslingual_class_flow_EN_MK_all_models.png"
)

plt.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("=" * 60)
print(f"✅ B5 saved → {out_path}")