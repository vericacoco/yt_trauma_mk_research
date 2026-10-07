from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# Избери што споредуваш:
# "titles" или "comments"
# ------------------------------------------------------------
DATA_TYPE = "comments"

# ------------------------------------------------------------
# Влезни датотеки
# Промени ги имињата ако твоите датотеки се викаат поинаку.
# ------------------------------------------------------------
if DATA_TYPE == "comments":
    GEMMA_FILE = BASE_DIR / "outputs" / "gemma_comments_sentiment.csv"
    LLAMA_FILE = BASE_DIR / "outputs" / "llama_comments_sentiment.csv"

    # Најдобро е спојувањето да биде преку video_id.
    ID_COLUMNS = ["comment_id"]

else:
    GEMMA_FILE = BASE_DIR / "outputs" / "comments_llm_gemma.csv"
    LLAMA_FILE = BASE_DIR / "outputs" / "comments_llm_llama.csv"

    # Ако имаш comment_id, користи го.
    # Ако немаш, може ["video_id", "comment_text"].
    ID_COLUMNS = ["comment_id"]


# ------------------------------------------------------------
# Колони со label и confidence
#
# Пример:
# sentiment_label
# sentiment
# llm_label
# label
#
# Намести ги според твоите CSV датотеки.
# ------------------------------------------------------------
GEMMA_LABEL_COLUMN = "gemma_label_mk_c"
GEMMA_CONFIDENCE_COLUMN = "gemma_confidence_mk_c"

LLAMA_LABEL_COLUMN = "llama_label_mk_c"
LLAMA_CONFIDENCE_COLUMN = "llama_confidence_mk_c"


# ------------------------------------------------------------
# Output директориум
# ------------------------------------------------------------
OUTPUT_DIR = BASE_DIR / "outputs" / "comments_gemma_llama_comparison_mk_c"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Вчитува CSV со поддршка за UTF-8 и UTF-8-SIG.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"\nНе е пронајдена датотеката:\n{file_path}\n"
            f"Провери ја патеката во CONFIGURATION делот."
        )

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="utf-8")

    print(f"Вчитана датотека: {file_path.name}")
    print(f"Број на редови: {len(df)}")
    print(f"Колони: {list(df.columns)}")
    print("-" * 80)

    return df


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    model_name: str,
) -> None:
    """
    Проверува дали сите потребни колони постојат.
    """
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nВо {model_name} датотеката недостигаат колони:\n"
            f"{missing_columns}\n\n"
            f"Постоечки колони:\n{list(df.columns)}"
        )


def normalize_label(value) -> str | None:
    """
    Ги нормализира sentiment labels во:
    positive, neutral, negative.
    """
    if pd.isna(value):
        return None

    label = str(value).strip().lower()

    mapping = {
        "positive": "positive",
        "pos": "positive",
        "позитивно": "positive",
        "позитивен": "positive",
        "1": "positive",

        "neutral": "neutral",
        "neu": "neutral",
        "неутрално": "neutral",
        "неутрален": "neutral",
        "0": "neutral",

        "negative": "negative",
        "neg": "negative",
        "негативно": "negative",
        "негативен": "negative",
        "-1": "negative",
    }

    return mapping.get(label, label)


def normalize_confidence(value) -> float:
    """
    Го претвора confidence во број од 0 до 1.

    Поддржува:
    0.85
    85
    "85%"
    "0,85"
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace("%", "").replace(",", ".")

    try:
        confidence = float(value)
    except ValueError:
        return np.nan

    # Ако confidence е даден како процент, на пример 85,
    # го претвора во 0.85.
    if confidence > 1:
        confidence = confidence / 100

    return confidence


def prepare_model_dataframe(
    df: pd.DataFrame,
    id_columns: list[str],
    label_column: str,
    confidence_column: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Ги задржува потребните колони и ги преименува според моделот.
    """
    required_columns = id_columns + [label_column, confidence_column]
    validate_columns(df, required_columns, model_name)

    result = df[required_columns].copy()

    result[label_column] = result[label_column].apply(normalize_label)
    result[confidence_column] = result[confidence_column].apply(
        normalize_confidence
    )

    result = result.rename(
        columns={
            label_column: f"{model_name}_label",
            confidence_column: f"{model_name}_confidence",
        }
    )

    return result


def save_dataframe(df: pd.DataFrame, file_name: str) -> None:
    """
    Зачувува DataFrame како CSV.
    """
    output_path = OUTPUT_DIR / file_name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Зачувано: {output_path}")


def save_plot(file_name: str) -> None:
    """
    Го зачувува тековниот matplotlib график.
    """
    output_path = PLOTS_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Зачуван график: {output_path}")


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def create_label_distribution(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Креира споредба на distribution на labels.
    """
    labels = ["negative", "neutral", "positive"]

    rows = []

    for label in labels:
        gemma_count = (
            comparison_df["gemma_label"].eq(label).sum()
        )
        llama_count = (
            comparison_df["llama_label"].eq(label).sum()
        )

        rows.append(
            {
                "label": label,
                "gemma_count": gemma_count,
                "llama_count": llama_count,
                "gemma_percentage": (
                    gemma_count / len(comparison_df) * 100
                    if len(comparison_df) > 0
                    else 0
                ),
                "llama_percentage": (
                    llama_count / len(comparison_df) * 100
                    if len(comparison_df) > 0
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def create_confidence_summary(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Основна статистика за confidence на двата модели.
    """
    rows = []

    for model in ["gemma", "llama"]:
        confidence_column = f"{model}_confidence"

        rows.append(
            {
                "model": model.capitalize(),
                "count": comparison_df[confidence_column].count(),
                "mean_confidence": comparison_df[
                    confidence_column
                ].mean(),
                "median_confidence": comparison_df[
                    confidence_column
                ].median(),
                "std_confidence": comparison_df[
                    confidence_column
                ].std(),
                "min_confidence": comparison_df[
                    confidence_column
                ].min(),
                "max_confidence": comparison_df[
                    confidence_column
                ].max(),
            }
        )

    return pd.DataFrame(rows)


def create_confidence_by_agreement(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Confidence според тоа дали моделите се согласуваат.
    """
    summary = (
        comparison_df.groupby("agreement")
        .agg(
            number_of_examples=("agreement", "size"),
            gemma_mean_confidence=("gemma_confidence", "mean"),
            llama_mean_confidence=("llama_confidence", "mean"),
            mean_absolute_confidence_difference=(
                "confidence_difference_absolute",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


def create_confidence_by_label(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Просечен confidence за секоја класа и за секој модел.
    """
    gemma_summary = (
        comparison_df.groupby("gemma_label")
        .agg(
            number_of_examples=("gemma_label", "size"),
            mean_confidence=("gemma_confidence", "mean"),
            median_confidence=("gemma_confidence", "median"),
        )
        .reset_index()
    )

    gemma_summary["model"] = "Gemma"
    gemma_summary = gemma_summary.rename(
        columns={"gemma_label": "label"}
    )

    llama_summary = (
        comparison_df.groupby("llama_label")
        .agg(
            number_of_examples=("llama_label", "size"),
            mean_confidence=("llama_confidence", "mean"),
            median_confidence=("llama_confidence", "median"),
        )
        .reset_index()
    )

    llama_summary["model"] = "Llama"
    llama_summary = llama_summary.rename(
        columns={"llama_label": "label"}
    )

    return pd.concat(
        [gemma_summary, llama_summary],
        ignore_index=True,
    )


def create_disagreement_table(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Групира кои различни предвидувања најчесто се појавуваат.
    """
    disagreement_df = comparison_df[
        comparison_df["agreement"] == "disagreement"
    ]

    summary = (
        disagreement_df.groupby(
            ["gemma_label", "llama_label"],
            dropna=False,
        )
        .agg(
            count=("agreement", "size"),
            gemma_mean_confidence=("gemma_confidence", "mean"),
            llama_mean_confidence=("llama_confidence", "mean"),
            mean_absolute_confidence_difference=(
                "confidence_difference_absolute",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    return summary


# ============================================================
# PLOT FUNCTIONS
# ============================================================

def plot_label_distribution(
    label_distribution_df: pd.DataFrame,
) -> None:
    labels = label_distribution_df["label"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 6))

    plt.bar(
        x - width / 2,
        label_distribution_df["gemma_count"],
        width,
        label="Gemma",
    )

    plt.bar(
        x + width / 2,
        label_distribution_df["llama_count"],
        width,
        label="Llama",
    )

    plt.xticks(x, labels)
    plt.xlabel("Sentiment label")
    plt.ylabel("Number of examples")
    plt.title("Sentiment Distribution: Gemma vs Llama")
    plt.legend()

    save_plot("01_label_distribution.png")


def plot_agreement_distribution(
    comparison_df: pd.DataFrame,
) -> None:
    counts = (
        comparison_df["agreement"]
        .value_counts()
        .reindex(["agreement", "disagreement"], fill_value=0)
    )

    plt.figure(figsize=(8, 6))
    bars = plt.bar(counts.index, counts.values)

    plt.xlabel("Comparison result")
    plt.ylabel("Number of examples")
    plt.title("Gemma and Llama Agreement")

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    save_plot("02_agreement_distribution.png")


def plot_average_confidence(
    confidence_summary_df: pd.DataFrame,
) -> None:
    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        confidence_summary_df["model"],
        confidence_summary_df["mean_confidence"],
    )

    plt.xlabel("Model")
    plt.ylabel("Mean confidence")
    plt.title("Average Confidence: Gemma vs Llama")
    plt.ylim(0, 1)

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    save_plot("03_average_confidence.png")


def plot_confidence_boxplot(
    comparison_df: pd.DataFrame,
) -> None:
    gemma_confidence = (
        comparison_df["gemma_confidence"]
        .dropna()
        .to_numpy()
    )

    llama_confidence = (
        comparison_df["llama_confidence"]
        .dropna()
        .to_numpy()
    )

    plt.figure(figsize=(8, 6))

    plt.boxplot(
        [gemma_confidence, llama_confidence],
        tick_labels=["Gemma", "Llama"],
    )

    plt.xlabel("Model")
    plt.ylabel("Confidence")
    plt.title("Confidence Distribution: Gemma vs Llama")
    plt.ylim(0, 1)

    save_plot("04_confidence_boxplot.png")


def plot_confidence_histogram(
    comparison_df: pd.DataFrame,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.hist(
        comparison_df["gemma_confidence"].dropna(),
        bins=20,
        alpha=0.6,
        label="Gemma",
    )

    plt.hist(
        comparison_df["llama_confidence"].dropna(),
        bins=20,
        alpha=0.6,
        label="Llama",
    )

    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Confidence Histogram: Gemma vs Llama")
    plt.legend()

    save_plot("05_confidence_histogram.png")


def plot_confidence_scatter(
    comparison_df: pd.DataFrame,
) -> None:
    plot_df = comparison_df.dropna(
        subset=["gemma_confidence", "llama_confidence"]
    )

    plt.figure(figsize=(8, 8))

    plt.scatter(
        plot_df["gemma_confidence"],
        plot_df["llama_confidence"],
        alpha=0.5,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("Gemma confidence")
    plt.ylabel("Llama confidence")
    plt.title("Confidence Comparison: Gemma vs Llama")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    save_plot("06_confidence_scatter.png")


def plot_confidence_difference(
    comparison_df: pd.DataFrame,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.hist(
        comparison_df["confidence_difference"].dropna(),
        bins=30,
    )

    plt.axvline(
        x=0,
        linestyle="--",
    )

    plt.xlabel("Gemma confidence - Llama confidence")
    plt.ylabel("Frequency")
    plt.title("Confidence Difference Between Models")

    save_plot("07_confidence_difference.png")


def plot_confusion_matrix(
    comparison_df: pd.DataFrame,
) -> None:
    """
    Ова не е accuracy confusion matrix со ground truth.
    Ја прикажува Gemma како редови, а Llama како колони.
    """
    labels = ["negative", "neutral", "positive"]

    valid_df = comparison_df.dropna(
        subset=["gemma_label", "llama_label"]
    )

    matrix = confusion_matrix(
        valid_df["gemma_label"],
        valid_df["llama_label"],
        labels=labels,
    )

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix)

    plt.xticks(
        np.arange(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        np.arange(len(labels)),
        labels,
    )

    plt.xlabel("Llama prediction")
    plt.ylabel("Gemma prediction")
    plt.title("Gemma vs Llama Prediction Matrix")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                matrix[row_index, column_index],
                ha="center",
                va="center",
            )

    plt.colorbar()

    save_plot("08_prediction_matrix.png")


def plot_confidence_by_agreement(
    confidence_by_agreement_df: pd.DataFrame,
) -> None:
    categories = confidence_by_agreement_df["agreement"]
    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(9, 6))

    plt.bar(
        x - width / 2,
        confidence_by_agreement_df[
            "gemma_mean_confidence"
        ],
        width,
        label="Gemma",
    )

    plt.bar(
        x + width / 2,
        confidence_by_agreement_df[
            "llama_mean_confidence"
        ],
        width,
        label="Llama",
    )

    plt.xticks(x, categories)
    plt.xlabel("Agreement category")
    plt.ylabel("Mean confidence")
    plt.title("Confidence for Agreement and Disagreement")
    plt.ylim(0, 1)
    plt.legend()

    save_plot("09_confidence_by_agreement.png")


# ============================================================
# TEXT REPORT
# ============================================================

def create_text_report(
    comparison_df: pd.DataFrame,
    confidence_summary_df: pd.DataFrame,
    kappa: float,
) -> str:
    total = len(comparison_df)

    agreement_count = (
        comparison_df["agreement"] == "agreement"
    ).sum()

    disagreement_count = (
        comparison_df["agreement"] == "disagreement"
    ).sum()

    agreement_percentage = (
        agreement_count / total * 100 if total > 0 else 0
    )

    disagreement_percentage = (
        disagreement_count / total * 100 if total > 0 else 0
    )

    gemma_mean_confidence = confidence_summary_df.loc[
        confidence_summary_df["model"] == "Gemma",
        "mean_confidence",
    ].iloc[0]

    llama_mean_confidence = confidence_summary_df.loc[
        confidence_summary_df["model"] == "Llama",
        "mean_confidence",
    ].iloc[0]

    higher_gemma = (
        comparison_df["confidence_difference"] > 0
    ).sum()

    higher_llama = (
        comparison_df["confidence_difference"] < 0
    ).sum()

    equal_confidence = (
        comparison_df["confidence_difference"] == 0
    ).sum()

    report = f"""
GEMMA VS LLAMA SENTIMENT COMPARISON
===================================

Dataset type: {DATA_TYPE}

Total compared examples: {total}

LABEL AGREEMENT
---------------
Agreement count: {agreement_count}
Agreement percentage: {agreement_percentage:.2f}%

Disagreement count: {disagreement_count}
Disagreement percentage: {disagreement_percentage:.2f}%

Cohen's Kappa: {kappa:.4f}

CONFIDENCE COMPARISON
---------------------
Gemma mean confidence: {gemma_mean_confidence:.4f}
Llama mean confidence: {llama_mean_confidence:.4f}

Gemma has higher confidence in: {higher_gemma} examples
Llama has higher confidence in: {higher_llama} examples
Equal confidence in: {equal_confidence} examples

Mean absolute confidence difference:
{comparison_df["confidence_difference_absolute"].mean():.4f}

Maximum absolute confidence difference:
{comparison_df["confidence_difference_absolute"].max():.4f}
"""

    return report.strip()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("GEMMA VS LLAMA COMPARISON")
    print("=" * 80)

    # 1. Load data
    gemma_df = load_csv(GEMMA_FILE)
    llama_df = load_csv(LLAMA_FILE)

    # 2. Prepare data
    gemma_prepared = prepare_model_dataframe(
        df=gemma_df,
        id_columns=ID_COLUMNS,
        label_column=GEMMA_LABEL_COLUMN,
        confidence_column=GEMMA_CONFIDENCE_COLUMN,
        model_name="gemma",
    )

    llama_prepared = prepare_model_dataframe(
        df=llama_df,
        id_columns=ID_COLUMNS,
        label_column=LLAMA_LABEL_COLUMN,
        confidence_column=LLAMA_CONFIDENCE_COLUMN,
        model_name="llama",
    )

    # 3. Check duplicate IDs
    gemma_duplicates = gemma_prepared.duplicated(
        subset=ID_COLUMNS
    ).sum()

    llama_duplicates = llama_prepared.duplicated(
        subset=ID_COLUMNS
    ).sum()

    if gemma_duplicates > 0:
        print(
            f"WARNING: Gemma има {gemma_duplicates} duplicate IDs."
        )

    if llama_duplicates > 0:
        print(
            f"WARNING: Llama има {llama_duplicates} duplicate IDs."
        )

    # Се задржува првиот резултат за секој ID.
    gemma_prepared = gemma_prepared.drop_duplicates(
        subset=ID_COLUMNS,
        keep="first",
    )

    llama_prepared = llama_prepared.drop_duplicates(
        subset=ID_COLUMNS,
        keep="first",
    )

    # 4. Merge
    comparison_df = pd.merge(
        gemma_prepared,
        llama_prepared,
        on=ID_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if comparison_df.empty:
        raise ValueError(
            "Нема совпаднати редови меѓу Gemma и Llama. "
            "Провери ги ID_COLUMNS."
        )

    print(f"\nСпоени редови: {len(comparison_df)}")

    # 5. Create comparison columns
    comparison_df["agreement"] = np.where(
        comparison_df["gemma_label"]
        == comparison_df["llama_label"],
        "agreement",
        "disagreement",
    )

    comparison_df["confidence_difference"] = (
        comparison_df["gemma_confidence"]
        - comparison_df["llama_confidence"]
    )

    comparison_df["confidence_difference_absolute"] = (
        comparison_df["confidence_difference"].abs()
    )

    comparison_df["higher_confidence_model"] = np.select(
        [
            comparison_df["confidence_difference"] > 0,
            comparison_df["confidence_difference"] < 0,
        ],
        [
            "Gemma",
            "Llama",
        ],
        default="Equal",
    )

    comparison_df["agreement_with_high_confidence"] = (
        (
            comparison_df["agreement"] == "agreement"
        )
        & (
            comparison_df[
                ["gemma_confidence", "llama_confidence"]
            ].min(axis=1)
            >= 0.80
        )
    )

    # 6. Metrics
    valid_labels_df = comparison_df.dropna(
        subset=["gemma_label", "llama_label"]
    )

    raw_agreement_accuracy = accuracy_score(
        valid_labels_df["gemma_label"],
        valid_labels_df["llama_label"],
    )

    kappa = cohen_kappa_score(
        valid_labels_df["gemma_label"],
        valid_labels_df["llama_label"],
    )

    print(f"Agreement accuracy: {raw_agreement_accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    # Llama се третира само како споредбен модел,
    # не како ground truth.
    report_dictionary = classification_report(
        valid_labels_df["gemma_label"],
        valid_labels_df["llama_label"],
        labels=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0,
    )

    classification_df = (
        pd.DataFrame(report_dictionary)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )

    # 7. Summary tables
    label_distribution_df = create_label_distribution(
        comparison_df
    )

    confidence_summary_df = create_confidence_summary(
        comparison_df
    )

    confidence_by_agreement_df = (
        create_confidence_by_agreement(comparison_df)
    )

    confidence_by_label_df = create_confidence_by_label(
        comparison_df
    )

    disagreement_summary_df = create_disagreement_table(
        comparison_df
    )

    # 8. Sort disagreements for manual analysis
    disagreements_df = comparison_df[
        comparison_df["agreement"] == "disagreement"
    ].copy()

    disagreements_df = disagreements_df.sort_values(
        by="confidence_difference_absolute",
        ascending=False,
    )

    # Cases where both models are confident but disagree
    high_confidence_disagreements_df = comparison_df[
        (comparison_df["agreement"] == "disagreement")
        & (comparison_df["gemma_confidence"] >= 0.80)
        & (comparison_df["llama_confidence"] >= 0.80)
    ].copy()

    high_confidence_disagreements_df = (
        high_confidence_disagreements_df.sort_values(
            by="confidence_difference_absolute",
            ascending=False,
        )
    )

    # Cases with largest confidence difference
    largest_confidence_differences_df = (
        comparison_df.sort_values(
            by="confidence_difference_absolute",
            ascending=False,
        )
        .head(100)
        .copy()
    )

    # 9. Save CSV outputs
    save_dataframe(
        comparison_df,
        f"{DATA_TYPE}_gemma_llama_full_comparison.csv",
    )

    save_dataframe(
        disagreements_df,
        f"{DATA_TYPE}_gemma_llama_disagreements.csv",
    )

    save_dataframe(
        high_confidence_disagreements_df,
        f"{DATA_TYPE}_high_confidence_disagreements.csv",
    )

    save_dataframe(
        largest_confidence_differences_df,
        f"{DATA_TYPE}_largest_confidence_differences.csv",
    )

    save_dataframe(
        label_distribution_df,
        f"{DATA_TYPE}_label_distribution.csv",
    )

    save_dataframe(
        confidence_summary_df,
        f"{DATA_TYPE}_confidence_summary.csv",
    )

    save_dataframe(
        confidence_by_agreement_df,
        f"{DATA_TYPE}_confidence_by_agreement.csv",
    )

    save_dataframe(
        confidence_by_label_df,
        f"{DATA_TYPE}_confidence_by_label.csv",
    )

    save_dataframe(
        disagreement_summary_df,
        f"{DATA_TYPE}_disagreement_summary.csv",
    )

    save_dataframe(
        classification_df,
        f"{DATA_TYPE}_classification_report.csv",
    )

    # 10. Generate plots
    plot_label_distribution(label_distribution_df)
    plot_agreement_distribution(comparison_df)
    plot_average_confidence(confidence_summary_df)
    plot_confidence_boxplot(comparison_df)
    plot_confidence_histogram(comparison_df)
    plot_confidence_scatter(comparison_df)
    plot_confidence_difference(comparison_df)
    plot_confusion_matrix(comparison_df)
    plot_confidence_by_agreement(
        confidence_by_agreement_df
    )

    # 11. Text report
    text_report = create_text_report(
        comparison_df=comparison_df,
        confidence_summary_df=confidence_summary_df,
        kappa=kappa,
    )

    report_path = OUTPUT_DIR / f"{DATA_TYPE}_comparison_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(text_report)

    print(f"Зачуван извештај: {report_path}")

    print("\n" + "=" * 80)
    print(text_report)
    print("=" * 80)

    print("\nАнализата е успешно завршена.")
    print(f"Резултатите се наоѓаат во:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()