#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# B4_cohens_kappa_landscape_all_models.py
#
# Pairwise Cohen's Kappa Landscape
# Languages: EN / MK
# Content: Videos / Comments
# Models: NLPTown, Cardiff, BERTweet, Gemma, Llama

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------
RESULTS_DIR = "results/B4_model_agreement"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CONFIG = {

    # =====================================================
    # MACEDONIAN
    # =====================================================
    "MK": {

        "videos": {

            "NLPTown": (
                "datasets/predictions_mk_videos_nlptown.csv",
                "nlptown_label_3cls"
            ),

            "Cardiff": (
                "datasets/predictions_cardiff_mk_xlm_sentiment.csv",
                "pred_cardiff_mk_xlm"
            ),

            "BERTweet": (
                "datasets/predictions_mk_videos_bertweet_all.csv",
                "bertweet_label_3cls"
            ),

            "Gemma": (
                "outputs/titles_llm_gemma_probabilities.csv",
                "gemma_label_mk"
            ),

            "Llama": (
                "outputs/titles_llm_llama_probabilities.csv",
                "llama_label_mk"
            ),
        },

        "comments": {

            "NLPTown": (
                "datasets/predictions_comments_mk_nlptown.csv",
                "pred_nlptown_3cls"
            ),

            "Cardiff": (
                "datasets/predictions_comments_mk_cardiff.csv",
                "pred_cardiff_mk_xlm"
            ),

            "BERTweet": (
                "datasets/predictions_comments_mk_bertweet_all.csv",
                "pred_bertweet_3cls"
            ),

            "Gemma": (
                "outputs/gemma_comments_sentiment.csv",
                "gemma_label_mk_c"
            ),

            "Llama": (
                "outputs/llama_comments_sentiment.csv",
                "llama_label_mk_c"
            ),
        },
    },

    # =====================================================
    # ENGLISH
    # =====================================================
    "EN": {

        "videos": {

            "NLPTown": (
                "datasets/predictions_nlptown_en_videos_all.csv",
                "pred_nlptown_en_3cls"
            ),

            "Cardiff": (
                "datasets/predictions_cardiff_xlm_en.csv",
                "pred_cardiff_xlm_en"
            ),

            "BERTweet": (
                "datasets/predictions_bertweet_en_videos.csv",
                "pred_bertweet_en_3cls"
            ),

            "Gemma": (
                "outputs/titles_llm_gemma_probabilities.csv",
                "gemma_label_en"
            ),

            "Llama": (
                "outputs/titles_llm_llama_probabilities.csv",
                "llama_label_en"
            ),
        },

        "comments": {

            "NLPTown": (
                "datasets/predictions_comments_en_nlptown_all.csv",
                "pred_nlptown_en_3cls"
            ),

            "Cardiff": (
                "datasets/predictions_comments_en_cardiff_xlm.csv",
                "pred_cardiff_en"
            ),

            "BERTweet": (
                "datasets/predictions_comments_en_bertweet_all.csv",
                "bertweet_mapped"
            ),

            "Gemma": (
                "outputs/gemma_comments_sentiment.csv",
                "gemma_label_en_c"
            ),

            "Llama": (
                "outputs/llama_comments_sentiment.csv",
                "llama_label_en_c"
            ),
        },
    },
}


MODELS = [
    "NLPTown",
    "Cardiff",
    "BERTweet",
    "Gemma",
    "Llama"
]

VALID_LABELS = {
    "negative",
    "neutral",
    "positive"
}


# ---------------------------------------------------------
# LOAD LABELS
# ---------------------------------------------------------
def load_labels(path, col):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌ File not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig"
    )

    if col not in df.columns:
        raise KeyError(
            f"\n❌ Column '{col}' not found in:\n"
            f"{path}\n\n"
            f"Available columns:\n"
            f"{df.columns.tolist()}"
        )

    labels = (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .reset_index(drop=True)
    )

    return labels


# ---------------------------------------------------------
# COHEN'S KAPPA
# ---------------------------------------------------------
def calculate_kappa(a, b):

    # Use same number of rows
    n = min(
        len(a),
        len(b)
    )

    if len(a) != len(b):
        print(
            f"⚠️ Different lengths: "
            f"{len(a)} vs {len(b)} "
            f"→ using first {n} rows."
        )

    a = a.iloc[:n].reset_index(drop=True)
    b = b.iloc[:n].reset_index(drop=True)

    # Keep only valid sentiment labels
    valid_mask = (
        a.isin(VALID_LABELS)
        &
        b.isin(VALID_LABELS)
    )

    a = a[valid_mask]
    b = b[valid_mask]

    if len(a) == 0:
        return np.nan

    kappa = cohen_kappa_score(
        a,
        b,
        labels=[
            "negative",
            "neutral",
            "positive"
        ]
    )

    return kappa


# ---------------------------------------------------------
# COMPUTE MATRIX
# ---------------------------------------------------------
def compute_matrix(cfg_section):

    matrix = pd.DataFrame(
        index=MODELS,
        columns=MODELS,
        dtype=float
    )

    loaded = {}

    # Load every model only once
    for model in MODELS:

        path, col = cfg_section[model]

        loaded[model] = load_labels(
            path,
            col
        )

    # Pairwise Cohen's kappa
    for model_a in MODELS:

        for model_b in MODELS:

            if model_a == model_b:

                # Perfect agreement with itself
                matrix.loc[
                    model_a,
                    model_b
                ] = 1.0

            else:

                kappa = calculate_kappa(
                    loaded[model_a],
                    loaded[model_b]
                )

                matrix.loc[
                    model_a,
                    model_b
                ] = round(
                    kappa,
                    3
                )

    return matrix


# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------
sns.set_theme(
    style="white"
)


# ---------------------------------------------------------
# FIGURE
# ---------------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(17, 14)
)

axes = axes.flatten()


panels = [

    (
        "EN",
        "videos",
        "EN Videos"
    ),

    (
        "EN",
        "comments",
        "EN Comments"
    ),

    (
        "MK",
        "videos",
        "MK Videos"
    ),

    (
        "MK",
        "comments",
        "MK Comments"
    ),
]


# ---------------------------------------------------------
# DRAW
# ---------------------------------------------------------
for ax, (
    lang,
    content_type,
    title
) in zip(
    axes,
    panels
):

    print()
    print("=" * 70)
    print(
        f"Cohen's kappa: "
        f"{lang} / {content_type}"
    )
    print("=" * 70)

    matrix = compute_matrix(
        CONFIG[lang][content_type]
    )

    print(matrix)
    print()


    sns.heatmap(

        matrix,

        ax=ax,

        annot=True,

        fmt=".2f",

        cmap="RdYlGn",

        vmin=-1,

        vmax=1,

        center=0,

        cbar=False,

        square=True,

        linewidths=0.5,

        linecolor="white",

        annot_kws={
            "fontsize": 9
        }
    )


    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(
        axis="x",
        labelrotation=30,
        labelsize=10
    )

    ax.tick_params(
        axis="y",
        labelrotation=0,
        labelsize=10
    )


# ---------------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------------
fig.suptitle(

    "Model Agreement Landscape — Cohen's κ",

    fontsize=19,

    fontweight="bold",

    y=0.98
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.95
    ]
)


# ---------------------------------------------------------
# SAVE PNG
# ---------------------------------------------------------
png_path = os.path.join(

    RESULTS_DIR,

    "B4_cohens_kappa_landscape_all_models.png"
)

plt.savefig(

    png_path,

    dpi=300,

    bbox_inches="tight"
)


# ---------------------------------------------------------
# SAVE SVG
# ---------------------------------------------------------
svg_path = os.path.join(

    RESULTS_DIR,

    "B4_cohens_kappa_landscape_all_models.svg"
)

plt.savefig(

    svg_path,

    format="svg",

    bbox_inches="tight"
)


plt.close()


print()
print("=" * 70)

print(
    f"✅ PNG saved → "
    f"{png_path}"
)

print(
    f"✅ SVG saved → "
    f"{svg_path}"
)