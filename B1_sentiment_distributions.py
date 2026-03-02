#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# B1_sentiment_distributions.py
# Compute NEG / NEU / POS distributions and stacked bar charts
# for MK/EN videos and comments

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIG: INPUT FILES
# ---------------------------------------------------------
DATASETS = {
    "mk_videos": {
        "NLPTown":  ("datasets/predictions_mk_videos_nlptown.csv", "nlptown_label_3cls"),
        "Cardiff":  ("datasets/predictions_cardiff_mk_xlm_sentiment.csv", "pred_cardiff_mk_xlm"),
        "BERTweet": ("datasets/predictions_mk_videos_bertweet_all.csv", "bertweet_label_3cls"),
    },
    "en_videos": {
        "NLPTown":  ("datasets/predictions_nlptown_en_videos_all.csv", "pred_nlptown_en_3cls"),
        "Cardiff":  ("datasets/predictions_cardiff_xlm_en.csv", "pred_cardiff_xlm_en"),
        "BERTweet": ("datasets/predictions_bertweet_en_videos.csv", "pred_bertweet_en_3cls"),
    },
    "mk_comments": {
        "NLPTown":  ("datasets/predictions_comments_mk_nlptown.csv", "pred_nlptown_3cls"),
        "Cardiff":  ("datasets/predictions_comments_mk_cardiff.csv", "pred_cardiff_mk_xlm"),
        "BERTweet": ("datasets/predictions_comments_mk_bertweet_all.csv", "pred_bertweet_3cls"),
    },
    "en_comments": {
        "NLPTown":  ("datasets/predictions_comments_en_nlptown_all.csv", "pred_nlptown_en_3cls"),
        "Cardiff":  ("datasets/predictions_comments_en_cardiff_xlm.csv", "pred_cardiff_en"),
        "BERTweet": ("datasets/predictions_comments_en_bertweet_all.csv", "bertweet_mapped"),
    },
}

RESULTS_DIR = "results"

# ---------------------------------------------------------
# UTIL: sentiment percentages
# ---------------------------------------------------------
def sentiment_percentages(df, label_col):
    counts = df[label_col].value_counts(normalize=True) * 100
    return {
        "NEG%": round(counts.get("negative", 0), 2),
        "NEU%": round(counts.get("neutral", 0), 2),
        "POS%": round(counts.get("positive", 0), 2),
    }

# ---------------------------------------------------------
# UTIL: stacked bar plot
# ---------------------------------------------------------
def plot_stacked_bar(table, title, out_path):
    plot_df = table.set_index("model")[["NEG%", "NEU%", "POS%"]]

    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(7, 5)
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")

    # ---- MANUAL ANNOTATION (ROBUST)
    for i, model in enumerate(plot_df.index):
        y_offset = 0
        for label in ["NEG%", "NEU%", "POS%"]:
            value = plot_df.loc[model, label]
            if value >= 5:  # avoid clutter
                ax.text(
                    i,
                    y_offset + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold"
                )
            y_offset += value

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for dataset_name, models in DATASETS.items():
        print(f"\n📊 Processing: {dataset_name}")

        dataset_dir = os.path.join(RESULTS_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        rows = []

        for model_name, (csv_path, label_col) in models.items():
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Missing file: {csv_path}")

            df = pd.read_csv(csv_path, encoding="utf-8-sig")

            if label_col not in df.columns:
                raise ValueError(f"Column '{label_col}' not found in {csv_path}")

            perc = sentiment_percentages(df, label_col)

            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                **perc
            })

        table = pd.DataFrame(rows)

        # Save table
        table_csv = os.path.join(dataset_dir, "sentiment_distribution_table.csv")
        table.to_csv(table_csv, index=False, encoding="utf-8-sig")
        print(f"✅ Saved table → {table_csv}")

        # Save plot
        plot_path = os.path.join(dataset_dir, "sentiment_distribution_stacked_bar.png")
        plot_stacked_bar(
            table,
            title=f"Sentiment distribution – {dataset_name.replace('_', ' ').title()}",
            out_path=plot_path
        )
        print(f"📈 Saved plot → {plot_path}")

    print("\n🎯 B1 analysis completed successfully.")

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
