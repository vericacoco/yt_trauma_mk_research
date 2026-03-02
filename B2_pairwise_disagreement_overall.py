#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# B2_pairwise_disagreement_overall.py
# Pairwise disagreement (%) + heatmaps for all datasets

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# CONFIG: INPUT FILES + candidate label columns
# ---------------------------------------------------------
DATASETS = {
    "mk_videos": {
        "NLPTown": (
            "datasets/predictions_mk_videos_nlptown.csv",
            ["nlptown_label_3cls"]
        ),
        "Cardiff": (
            "datasets/predictions_cardiff_mk_xlm_sentiment.csv",
            ["pred_cardiff_mk_xlm"]
        ),
        "BERTweet": (
            "datasets/predictions_mk_videos_bertweet_all.csv",
            ["bertweet_label_3cls"]
        ),
    },
    "en_videos": {
        "NLPTown": (
            "datasets/predictions_nlptown_en_videos_all.csv",
            ["pred_nlptown_en_3cls"]
        ),
        "Cardiff": (
            "datasets/predictions_cardiff_xlm_en.csv",
            ["pred_cardiff_xlm_en"]
        ),
        "BERTweet": (
            "datasets/predictions_bertweet_en_videos.csv",
            ["pred_bertweet_en_3cls"]
        ),
    },
    "mk_comments": {
        "NLPTown": (
            "datasets/predictions_comments_mk_nlptown.csv",
            ["pred_nlptown_3cls"]
        ),
        "Cardiff": (
            "datasets/predictions_comments_mk_cardiff.csv",
            ["pred_cardiff_mk_xlm"]
        ),
        "BERTweet": (
            "datasets/predictions_comments_mk_bertweet_all.csv",
            ["pred_bertweet_3cls"]
        ),
    },
    "en_comments": {
        "NLPTown": (
            "datasets/predictions_comments_en_nlptown_all.csv",
            ["pred_nlptown_en_3cls"]
        ),
        "Cardiff": (
            "datasets/predictions_comments_en_cardiff_xlm.csv",
            ["pred_cardiff_en"]
        ),
        "BERTweet": (
            "datasets/predictions_comments_en_bertweet_all.csv",
            ["bertweet_mapped"]
        ),
    },
}

RESULTS_DIR = "results"

# ---------------------------------------------------------
# UTIL: find sentiment column
# ---------------------------------------------------------
def find_sentiment_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"None of the sentiment columns found. Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )

# ---------------------------------------------------------
# UTIL: pairwise disagreement (%)
# ---------------------------------------------------------
def disagreement_percentage(a, b):
    a = a.astype(str)
    b = b.astype(str)
    return round((a != b).mean() * 100, 2)

# ---------------------------------------------------------
# UTIL: heatmap
# ---------------------------------------------------------
def plot_heatmap(matrix, title, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        vmin=0,
        vmax=100,
        square=True,
        cbar_kws={"label": "Disagreement (%)"}
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for dataset_name, models in DATASETS.items():
        print(f"\n📊 Processing B2: {dataset_name}")

        dataset_dir = os.path.join(RESULTS_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        predictions = {}

        # Load predictions for each model
        for model_name, (csv_path, col_candidates) in models.items():
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Missing file: {csv_path}")

            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            sent_col = find_sentiment_column(df, col_candidates)

            predictions[model_name] = df[sent_col].astype(str).values

        model_names = list(predictions.keys())
        n = len(model_names)

        # Build disagreement matrix
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = disagreement_percentage(
                        predictions[model_names[i]],
                        predictions[model_names[j]]
                    )

        df_matrix = pd.DataFrame(
            matrix,
            index=model_names,
            columns=model_names
        )

        # Save matrix as CSV
        matrix_csv = os.path.join(dataset_dir, "pairwise_disagreement_matrix.csv")
        df_matrix.to_csv(matrix_csv, encoding="utf-8-sig")
        print(f"✅ Saved matrix → {matrix_csv}")

        # Save heatmap
        heatmap_path = os.path.join(dataset_dir, "pairwise_disagreement_heatmap.png")
        plot_heatmap(
            df_matrix,
            title=f"Pairwise disagreement (%) – {dataset_name.replace('_', ' ').title()}",
            out_path=heatmap_path
        )
        print(f"🔥 Saved heatmap → {heatmap_path}")

    print("\n🎯 B2 analysis (overall disagreement) completed successfully.")

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
