from pathlib import Path
from itertools import combinations
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Избери:
# DATASET = "titles" или "comments"
# LANGUAGE = "mk" или "en"
DATASET = "comments"
LANGUAGE = "en"

VALID_DATASETS = {"titles", "comments"}
VALID_LANGUAGES = {"mk", "en"}

if DATASET not in VALID_DATASETS:
    raise ValueError(f"DATASET мора да биде едно од: {VALID_DATASETS}")

if LANGUAGE not in VALID_LANGUAGES:
    raise ValueError(f"LANGUAGE мора да биде едно од: {VALID_LANGUAGES}")


# ============================================================
# INPUT CONFIGURATION
# ============================================================

CONFIG = {
    ("titles", "mk"): {
        "id_column": "video_id",
        "text_columns": ["title_mk", "title_en", "title"],
        "traditional_file": BASE_DIR / "datasets" / "predictions_videos_mk_final.csv",
        "traditional_columns": {
            "CardiffNLP": "pred_cardiff_mk_xlm",
            "NLPTown": "nlptown_label_3cls",
            "BERTweet": "bertweet_label_3cls",
        },
        "gemma_file": BASE_DIR / "outputs" / "titles_llm_gemma_probabilities.csv",
        "gemma_label": "gemma_label_mk",
        "gemma_confidence": "gemma_confidence_mk",
        "llama_file": BASE_DIR / "outputs" / "titles_llm_llama_probabilities.csv",
        "llama_label": "llama_label_mk",
        "llama_confidence": "llama_confidence_mk",
    },

    ("titles", "en"): {
        "id_column": "video_id",
        "text_columns": ["title_en", "title_mk", "title"],
        "traditional_file": BASE_DIR / "datasets" / "predictions_videos_en_final.csv",
        "traditional_columns": {
            "CardiffNLP": "pred_cardiff_xlm_en",
            "NLPTown": "pred_nlptown_en_3cls",
            "BERTweet": "pred_bertweet_en_3cls",
        },
        "gemma_file": BASE_DIR / "outputs" / "titles_llm_gemma_probabilities.csv",
        "gemma_label": "gemma_label_en",
        "gemma_confidence": "gemma_confidence_en",
        "llama_file": BASE_DIR / "outputs" / "titles_llm_llama_probabilities.csv",
        "llama_label": "llama_label_en",
        "llama_confidence": "llama_confidence_en",
    },

    ("comments", "mk"): {
        "id_column": "comment_id",
        "text_columns": ["text_mk", "text_en", "text"],
        "traditional_file": BASE_DIR / "datasets" / "predictions_comments_mk_final.csv",
        "traditional_columns": {
            "CardiffNLP": "pred_cardiff_mk_xlm",
            "NLPTown": "pred_nlptown_3cls",
            "BERTweet": "pred_bertweet_3cls",
        },
        "gemma_file": BASE_DIR / "outputs" / "gemma_comments_sentiment.csv",
        "gemma_label": "gemma_label_mk_c",
        "gemma_confidence": "gemma_confidence_mk_c",
        "llama_file": BASE_DIR / "outputs" / "llama_comments_sentiment.csv",
        "llama_label": "llama_label_mk_c",
        "llama_confidence": "llama_confidence_mk_c",
    },

    ("comments", "en"): {
        "id_column": "comment_id",
        "text_columns": ["text_en", "text_mk", "text"],
        "traditional_file": BASE_DIR / "datasets" / "predictions_comments_en_final.csv",
        "traditional_columns": {
            "CardiffNLP": "pred_cardiff_en",
            "NLPTown": "pred_nlptown_en_3cls",
            # Во final датотеката оваа колона е веќе мапирана во 3 класи.
            "BERTweet": "bertweet_mapped",
        },
        "gemma_file": BASE_DIR / "outputs" / "gemma_comments_sentiment.csv",
        "gemma_label": "gemma_label_en_c",
        "gemma_confidence": "gemma_confidence_en_c",
        "llama_file": BASE_DIR / "outputs" / "llama_comments_sentiment.csv",
        "llama_label": "llama_label_en_c",
        "llama_confidence": "llama_confidence_en_c",
    },
}

CURRENT_CONFIG = CONFIG[(DATASET, LANGUAGE)]

ID_COLUMN = CURRENT_CONFIG["id_column"]
MODEL_COLUMNS = ["CardiffNLP", "NLPTown", "BERTweet", "Gemma", "Llama"]
LABEL_ORDER = ["negative", "neutral", "positive"]

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / f"all_models_{DATASET}_{LANGUAGE}"
)

PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не е пронајдена датотеката:\n{path}")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)

    print(f"Вчитана: {path.name} | редови: {len(df)}")
    return df


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    file_name: str,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"\nВо {file_name} недостигаат колони:\n{missing}\n\n"
            f"Постоечки колони:\n{list(df.columns)}"
        )


def normalize_label(value) -> str | None:
    if pd.isna(value):
        return None

    label = str(value).strip().lower()

    mapping = {
        # Positive
        "positive": "positive",
        "pos": "positive",
        "позитивно": "positive",
        "позитивен": "positive",
        "label_2": "positive",
        "2": "positive",
        "4 stars": "positive",
        "5 stars": "positive",
        "4 star": "positive",
        "5 star": "positive",

        # Neutral
        "neutral": "neutral",
        "neu": "neutral",
        "неутрално": "neutral",
        "неутрален": "neutral",
        "label_1": "neutral",
        "1": "neutral",
        "3 stars": "neutral",
        "3 star": "neutral",

        # Negative
        "negative": "negative",
        "neg": "negative",
        "негативно": "negative",
        "негативен": "negative",
        "label_0": "negative",
        "0": "negative",
        "-1": "negative",
        "1 star": "negative",
        "2 stars": "negative",
        "1 stars": "negative",
        "2 star": "negative",
    }

    normalized = mapping.get(label, label)

    if normalized not in LABEL_ORDER:
        return None

    return normalized


def normalize_confidence(value) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace("%", "").replace(",", ".")

    try:
        confidence = float(text)
    except ValueError:
        return np.nan

    if confidence > 1:
        confidence /= 100

    if confidence < 0 or confidence > 1:
        return np.nan

    return confidence


def choose_text_column(df: pd.DataFrame) -> str | None:
    for column in CURRENT_CONFIG["text_columns"]:
        if column in df.columns:
            return column
    return None


def save_csv(df: pd.DataFrame, file_name: str) -> None:
    output_path = TABLES_DIR / file_name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"CSV: {output_path}")


def save_matrix_csv(df: pd.DataFrame, file_name: str) -> None:
    output_path = TABLES_DIR / file_name
    df.to_csv(output_path, encoding="utf-8-sig")
    print(f"CSV: {output_path}")


def save_plot(file_name: str) -> None:
    output_path = PLOTS_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PNG: {output_path}")


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_traditional_data() -> pd.DataFrame:
    file_path = CURRENT_CONFIG["traditional_file"]
    df = load_csv(file_path)

    required = [ID_COLUMN] + list(
        CURRENT_CONFIG["traditional_columns"].values()
    )
    validate_columns(df, required, file_path.name)

    text_column = choose_text_column(df)

    selected_columns = [ID_COLUMN]

    if text_column:
        selected_columns.append(text_column)

    selected_columns.extend(
        CURRENT_CONFIG["traditional_columns"].values()
    )

    result = df[selected_columns].copy()

    rename_map = {
        source_column: model_name
        for model_name, source_column
        in CURRENT_CONFIG["traditional_columns"].items()
    }

    if text_column:
        rename_map[text_column] = "text"

    result = result.rename(columns=rename_map)

    for model in ["CardiffNLP", "NLPTown", "BERTweet"]:
        result[model] = result[model].apply(normalize_label)

    duplicate_count = result.duplicated(subset=[ID_COLUMN]).sum()

    if duplicate_count:
        print(
            f"WARNING: традиционалната датотека има "
            f"{duplicate_count} duplicate IDs. Се задржува првиот."
        )

    return result.drop_duplicates(subset=[ID_COLUMN], keep="first")


def prepare_llm_data(
    file_path: Path,
    model_name: str,
    label_column: str,
    confidence_column: str,
) -> pd.DataFrame:
    df = load_csv(file_path)

    required = [ID_COLUMN, label_column, confidence_column]
    validate_columns(df, required, file_path.name)

    result = df[required].copy()

    result[label_column] = result[label_column].apply(normalize_label)
    result[confidence_column] = result[confidence_column].apply(
        normalize_confidence
    )

    result = result.rename(
        columns={
            label_column: model_name,
            confidence_column: f"{model_name}_confidence",
        }
    )

    duplicate_count = result.duplicated(subset=[ID_COLUMN]).sum()

    if duplicate_count:
        print(
            f"WARNING: {model_name} има {duplicate_count} duplicate IDs. "
            f"Се задржува првиот."
        )

    return result.drop_duplicates(subset=[ID_COLUMN], keep="first")


def merge_all_models() -> pd.DataFrame:
    traditional = prepare_traditional_data()

    gemma = prepare_llm_data(
        CURRENT_CONFIG["gemma_file"],
        "Gemma",
        CURRENT_CONFIG["gemma_label"],
        CURRENT_CONFIG["gemma_confidence"],
    )

    llama = prepare_llm_data(
        CURRENT_CONFIG["llama_file"],
        "Llama",
        CURRENT_CONFIG["llama_label"],
        CURRENT_CONFIG["llama_confidence"],
    )

    merged = traditional.merge(
        gemma,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    merged = merged.merge(
        llama,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "Нема заеднички редови меѓу датотеките. "
            f"Провери го ID_COLUMN = {ID_COLUMN!r}."
        )

    print(f"\nЗаеднички споредени редови: {len(merged)}")

    for model in MODEL_COLUMNS:
        invalid_count = merged[model].isna().sum()
        print(f"{model}: missing/invalid labels = {invalid_count}")

    return merged


# ============================================================
# ANALYSIS
# ============================================================

def calculate_majority(labels: pd.Series) -> str | None:
    valid = [
        label
        for label in labels.tolist()
        if label in LABEL_ORDER
    ]

    if not valid:
        return None

    counts = Counter(valid)
    top_count = max(counts.values())
    winners = [
        label for label, count in counts.items()
        if count == top_count
    ]

    # Со 5 модели нормално нема tie ако сите имаат валиден label.
    if len(winners) != 1:
        return None

    return winners[0]


def add_consensus_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["valid_model_count"] = result[MODEL_COLUMNS].notna().sum(axis=1)

    result["majority_label"] = result[MODEL_COLUMNS].apply(
        calculate_majority,
        axis=1,
    )

    result["number_of_unique_labels"] = result[MODEL_COLUMNS].nunique(
        axis=1,
        dropna=True,
    )

    result["full_agreement"] = (
        (result["valid_model_count"] == len(MODEL_COLUMNS))
        & (result["number_of_unique_labels"] == 1)
    )

    result["majority_count"] = result.apply(
        lambda row: sum(
            row[model] == row["majority_label"]
            for model in MODEL_COLUMNS
            if pd.notna(row[model])
        )
        if pd.notna(row["majority_label"])
        else 0,
        axis=1,
    )

    result["consensus_strength"] = (
        result["majority_count"] / result["valid_model_count"]
    )

    for model in MODEL_COLUMNS:
        result[f"{model}_agrees_with_majority"] = (
            result[model] == result["majority_label"]
        )

    result["outlier_models"] = result.apply(
        lambda row: ", ".join(
            model
            for model in MODEL_COLUMNS
            if pd.notna(row[model])
            and pd.notna(row["majority_label"])
            and row[model] != row["majority_label"]
        ),
        axis=1,
    )

    return result


def create_agreement_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(
        index=MODEL_COLUMNS,
        columns=MODEL_COLUMNS,
        dtype=float,
    )

    for model_a in MODEL_COLUMNS:
        for model_b in MODEL_COLUMNS:

            # Моделот секогаш има 100% agreement со самиот себе
            if model_a == model_b:
                matrix.loc[model_a, model_b] = 1.0
                continue

            valid = df[[model_a, model_b]].dropna()

            if valid.empty:
                matrix.loc[model_a, model_b] = np.nan
            else:
                agreement_value = (
                    valid[model_a] == valid[model_b]
                ).mean()

                matrix.loc[model_a, model_b] = float(
                    agreement_value
                )

    return matrix


def create_kappa_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(
        index=MODEL_COLUMNS,
        columns=MODEL_COLUMNS,
        dtype=float,
    )

    for model_a in MODEL_COLUMNS:
        for model_b in MODEL_COLUMNS:
            if model_a == model_b:
                matrix.loc[model_a, model_b] = 1.0
                continue

            valid = df[[model_a, model_b]].dropna()

            if valid.empty:
                matrix.loc[model_a, model_b] = np.nan
                continue

            try:
                matrix.loc[model_a, model_b] = cohen_kappa_score(
                    valid[model_a],
                    valid[model_b],
                    labels=LABEL_ORDER,
                )
            except Exception:
                matrix.loc[model_a, model_b] = np.nan

    return matrix


def create_pairwise_table(
    agreement_matrix: pd.DataFrame,
    kappa_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model_a, model_b in combinations(MODEL_COLUMNS, 2):
        rows.append(
            {
                "model_1": model_a,
                "model_2": model_b,
                "agreement": agreement_matrix.loc[model_a, model_b],
                "agreement_percentage": (
                    agreement_matrix.loc[model_a, model_b] * 100
                ),
                "cohen_kappa": kappa_matrix.loc[model_a, model_b],
            }
        )

    return pd.DataFrame(rows).sort_values(
        "agreement",
        ascending=False,
    )


def create_majority_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model in MODEL_COLUMNS:
        valid = df[[model, "majority_label"]].dropna()

        rows.append(
            {
                "model": model,
                "valid_examples": len(valid),
                "agreement_with_majority_count": (
                    valid[model] == valid["majority_label"]
                ).sum(),
                "agreement_with_majority_percentage": (
                    (valid[model] == valid["majority_label"]).mean() * 100
                    if len(valid)
                    else np.nan
                ),
                "outlier_count": (
                    valid[model] != valid["majority_label"]
                ).sum(),
                "outlier_percentage": (
                    (valid[model] != valid["majority_label"]).mean() * 100
                    if len(valid)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "agreement_with_majority_percentage",
        ascending=False,
    )


def create_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model in MODEL_COLUMNS:
        valid_count = df[model].notna().sum()

        for label in LABEL_ORDER:
            count = df[model].eq(label).sum()

            rows.append(
                {
                    "model": model,
                    "label": label,
                    "count": count,
                    "percentage": (
                        count / valid_count * 100
                        if valid_count
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def create_confidence_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model in ["Gemma", "Llama"]:
        column = f"{model}_confidence"
        series = df[column].dropna()

        rows.append(
            {
                "model": model,
                "count": series.count(),
                "mean_confidence": series.mean(),
                "median_confidence": series.median(),
                "std_confidence": series.std(),
                "min_confidence": series.min(),
                "max_confidence": series.max(),
                "mean_confidence_when_agrees_with_majority": df.loc[
                    df[f"{model}_agrees_with_majority"],
                    column,
                ].mean(),
                "mean_confidence_when_disagrees_with_majority": df.loc[
                    ~df[f"{model}_agrees_with_majority"],
                    column,
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def create_consensus_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    rows = [
        {
            "metric": "total_examples",
            "value": total,
            "percentage": 100.0,
        },
        {
            "metric": "full_agreement",
            "value": int(df["full_agreement"].sum()),
            "percentage": df["full_agreement"].mean() * 100,
        },
        {
            "metric": "four_or_more_models_agree",
            "value": int((df["majority_count"] >= 4).sum()),
            "percentage": (df["majority_count"] >= 4).mean() * 100,
        },
        {
            "metric": "exactly_three_models_agree",
            "value": int((df["majority_count"] == 3).sum()),
            "percentage": (df["majority_count"] == 3).mean() * 100,
        },
        {
            "metric": "no_unique_majority",
            "value": int(df["majority_label"].isna().sum()),
            "percentage": df["majority_label"].isna().mean() * 100,
        },
    ]

    return pd.DataFrame(rows)


def create_disagreement_examples(df: pd.DataFrame) -> pd.DataFrame:
    disagreement = df[~df["full_agreement"]].copy()

    sort_columns = [
        "number_of_unique_labels",
        "consensus_strength",
    ]

    disagreement = disagreement.sort_values(
        sort_columns,
        ascending=[False, True],
    )

    wanted_columns = [ID_COLUMN]

    if "text" in disagreement.columns:
        wanted_columns.append("text")

    wanted_columns.extend(MODEL_COLUMNS)
    wanted_columns.extend(
        [
            "majority_label",
            "majority_count",
            "number_of_unique_labels",
            "consensus_strength",
            "outlier_models",
            "Gemma_confidence",
            "Llama_confidence",
        ]
    )

    return disagreement[wanted_columns].head(200)


# ============================================================
# PLOTS
# ============================================================

def plot_matrix(
    matrix: pd.DataFrame,
    title: str,
    file_name: str,
    value_format: str,
    vmin: float,
    vmax: float,
) -> None:
    values = matrix.to_numpy(dtype=float)

    plt.figure(figsize=(9, 8))
    image = plt.imshow(values, vmin=vmin, vmax=vmax)

    plt.xticks(
        np.arange(len(MODEL_COLUMNS)),
        MODEL_COLUMNS,
        rotation=35,
        ha="right",
    )
    plt.yticks(np.arange(len(MODEL_COLUMNS)), MODEL_COLUMNS)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]

            if not np.isnan(value):
                plt.text(
                    column,
                    row,
                    format(value, value_format),
                    ha="center",
                    va="center",
                )

    plt.title(title)
    plt.colorbar(image)
    save_plot(file_name)


def plot_label_distribution(label_distribution: pd.DataFrame) -> None:
    x = np.arange(len(LABEL_ORDER))
    width = 0.15

    plt.figure(figsize=(12, 7))

    for index, model in enumerate(MODEL_COLUMNS):
        model_data = (
            label_distribution[
                label_distribution["model"] == model
            ]
            .set_index("label")
            .reindex(LABEL_ORDER)
        )

        positions = x + (
            index - (len(MODEL_COLUMNS) - 1) / 2
        ) * width

        plt.bar(
            positions,
            model_data["percentage"],
            width=width,
            label=model,
        )

    plt.xticks(x, LABEL_ORDER)
    plt.xlabel("Sentiment label")
    plt.ylabel("Percentage of predictions")
    plt.title(
        f"Label Distribution — {DATASET.capitalize()} {LANGUAGE.upper()}"
    )
    plt.legend()
    save_plot("03_label_distribution.png")


def plot_majority_agreement(majority_summary: pd.DataFrame) -> None:
    data = majority_summary.sort_values(
        "agreement_with_majority_percentage",
        ascending=False,
    )

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        data["model"],
        data["agreement_with_majority_percentage"],
    )

    plt.ylabel("Agreement with majority (%)")
    plt.xlabel("Model")
    plt.title(
        f"Agreement with Five-Model Majority — "
        f"{DATASET.capitalize()} {LANGUAGE.upper()}"
    )
    plt.ylim(0, 105)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    save_plot("04_agreement_with_majority.png")


def plot_outlier_percentage(majority_summary: pd.DataFrame) -> None:
    data = majority_summary.sort_values(
        "outlier_percentage",
        ascending=False,
    )

    plt.figure(figsize=(10, 6))
    bars = plt.bar(data["model"], data["outlier_percentage"])

    plt.ylabel("Outlier predictions (%)")
    plt.xlabel("Model")
    plt.title(
        f"Model Outlier Rate — "
        f"{DATASET.capitalize()} {LANGUAGE.upper()}"
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    save_plot("05_model_outlier_rate.png")


def plot_pairwise_agreement(pairwise: pd.DataFrame) -> None:
    data = pairwise.copy()
    data["pair"] = data["model_1"] + " vs " + data["model_2"]
    data = data.sort_values("agreement_percentage")

    plt.figure(figsize=(11, 7))
    bars = plt.barh(data["pair"], data["agreement_percentage"])

    plt.xlabel("Agreement (%)")
    plt.ylabel("Model pair")
    plt.title(
        f"Pairwise Model Agreement — "
        f"{DATASET.capitalize()} {LANGUAGE.upper()}"
    )
    plt.xlim(0, 105)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
        )

    save_plot("06_pairwise_agreement.png")


def plot_llm_confidence(confidence_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        confidence_summary["model"],
        confidence_summary["mean_confidence"],
    )

    plt.ylabel("Mean confidence")
    plt.xlabel("LLM")
    plt.title(
        f"Gemma vs Llama Mean Confidence — "
        f"{DATASET.capitalize()} {LANGUAGE.upper()}"
    )
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

    save_plot("07_llm_mean_confidence.png")


def plot_consensus_strength(df: pd.DataFrame) -> None:
    counts = (
        df["majority_count"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(9, 6))
    bars = plt.bar(counts.index.astype(str), counts.values)

    plt.xlabel("Number of models supporting majority label")
    plt.ylabel("Number of examples")
    plt.title(
        f"Consensus Strength — "
        f"{DATASET.capitalize()} {LANGUAGE.upper()}"
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
        )

    save_plot("08_consensus_strength.png")


def plot_majority_distribution(df: pd.DataFrame) -> None:
    counts = (
        df["majority_label"]
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
    )

    plt.figure(figsize=(9, 6))
    bars = plt.bar(counts.index, counts.values)

    plt.xlabel("Majority sentiment")
    plt.ylabel("Number of examples")
    plt.title(
        f"Five-Model Majority Distribution — "
        f"{DATASET.capitalize()} {LANGUAGE.upper()}"
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
        )

    save_plot("09_majority_distribution.png")


# ============================================================
# REPORT
# ============================================================

def create_report(
    df: pd.DataFrame,
    agreement_matrix: pd.DataFrame,
    kappa_matrix: pd.DataFrame,
    majority_summary: pd.DataFrame,
    confidence_summary: pd.DataFrame,
) -> str:
    full_agreement_count = int(df["full_agreement"].sum())
    full_agreement_percentage = df["full_agreement"].mean() * 100

    strongest_pair = None
    strongest_agreement = -1

    for model_a, model_b in combinations(MODEL_COLUMNS, 2):
        value = agreement_matrix.loc[model_a, model_b]

        if pd.notna(value) and value > strongest_agreement:
            strongest_agreement = value
            strongest_pair = f"{model_a} vs {model_b}"

    weakest_pair = None
    weakest_agreement = 2

    for model_a, model_b in combinations(MODEL_COLUMNS, 2):
        value = agreement_matrix.loc[model_a, model_b]

        if pd.notna(value) and value < weakest_agreement:
            weakest_agreement = value
            weakest_pair = f"{model_a} vs {model_b}"

    best_majority_model = majority_summary.iloc[0]
    worst_majority_model = majority_summary.iloc[-1]

    gemma_conf = confidence_summary.loc[
        confidence_summary["model"] == "Gemma",
        "mean_confidence",
    ].iloc[0]

    llama_conf = confidence_summary.loc[
        confidence_summary["model"] == "Llama",
        "mean_confidence",
    ].iloc[0]

    report = f"""
ALL MODELS SENTIMENT COMPARISON
===============================

Dataset: {DATASET}
Language: {LANGUAGE.upper()}
Compared examples: {len(df)}

MODELS
------
CardiffNLP
NLPTown
BERTweet
Gemma
Llama

FULL CONSENSUS
--------------
Full agreement count: {full_agreement_count}
Full agreement percentage: {full_agreement_percentage:.2f}%

PAIRWISE AGREEMENT
------------------
Strongest pair: {strongest_pair}
Strongest agreement: {strongest_agreement * 100:.2f}%

Weakest pair: {weakest_pair}
Weakest agreement: {weakest_agreement * 100:.2f}%

MAJORITY AGREEMENT
------------------
Highest agreement with majority:
{best_majority_model["model"]}: {best_majority_model["agreement_with_majority_percentage"]:.2f}%

Lowest agreement with majority:
{worst_majority_model["model"]}: {worst_majority_model["agreement_with_majority_percentage"]:.2f}%

LLM CONFIDENCE
--------------
Gemma mean confidence: {gemma_conf:.4f}
Llama mean confidence: {llama_conf:.4f}

IMPORTANT
---------
The majority label is a five-model consensus label, not a manually
annotated ground-truth label. Agreement with majority therefore measures
model consistency with the ensemble, not true predictive accuracy.
"""

    return report.strip()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 90)
    print("ALL MODELS SENTIMENT COMPARISON")
    print(f"DATASET: {DATASET.upper()} | LANGUAGE: {LANGUAGE.upper()}")
    print("=" * 90)

    merged = merge_all_models()
    comparison = add_consensus_columns(merged)

    valid_for_five_model_analysis = comparison[
        comparison[MODEL_COLUMNS].notna().all(axis=1)
    ].copy()

    dropped = len(comparison) - len(valid_for_five_model_analysis)

    if dropped:
        print(
            f"\nWARNING: {dropped} редови немаат валидни labels од сите "
            f"5 модели и се исклучени од majority анализата."
        )

    if valid_for_five_model_analysis.empty:
        raise ValueError(
            "Нема редови со валидни labels од сите пет модели."
        )

    comparison = valid_for_five_model_analysis

    agreement_matrix = create_agreement_matrix(comparison)
    kappa_matrix = create_kappa_matrix(comparison)
    pairwise_table = create_pairwise_table(
        agreement_matrix,
        kappa_matrix,
    )
    majority_summary = create_majority_summary(comparison)
    label_distribution = create_label_distribution(comparison)
    confidence_summary = create_confidence_summary(comparison)
    consensus_summary = create_consensus_summary(comparison)
    disagreement_examples = create_disagreement_examples(comparison)

    # Save tables
    save_csv(comparison, "01_all_models_predictions.csv")
    save_matrix_csv(agreement_matrix, "02_agreement_matrix.csv")
    save_matrix_csv(kappa_matrix, "03_kappa_matrix.csv")
    save_csv(pairwise_table, "04_pairwise_agreement.csv")
    save_csv(majority_summary, "05_agreement_with_majority.csv")
    save_csv(label_distribution, "06_label_distribution.csv")
    save_csv(confidence_summary, "07_llm_confidence_summary.csv")
    save_csv(consensus_summary, "08_consensus_summary.csv")
    save_csv(disagreement_examples, "09_disagreement_examples.csv")

    # Plots
    plot_matrix(
        agreement_matrix,
        title=(
            f"Pairwise Agreement — "
            f"{DATASET.capitalize()} {LANGUAGE.upper()}"
        ),
        file_name="01_agreement_heatmap.png",
        value_format=".2f",
        vmin=0,
        vmax=1,
    )

    plot_matrix(
        kappa_matrix,
        title=(
            f"Cohen's Kappa — "
            f"{DATASET.capitalize()} {LANGUAGE.upper()}"
        ),
        file_name="02_kappa_heatmap.png",
        value_format=".2f",
        vmin=-1,
        vmax=1,
    )

    plot_label_distribution(label_distribution)
    plot_majority_agreement(majority_summary)
    plot_outlier_percentage(majority_summary)
    plot_pairwise_agreement(pairwise_table)
    plot_llm_confidence(confidence_summary)
    plot_consensus_strength(comparison)
    plot_majority_distribution(comparison)

    report = create_report(
        comparison,
        agreement_matrix,
        kappa_matrix,
        majority_summary,
        confidence_summary,
    )

    report_path = OUTPUT_DIR / "comparison_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 90)
    print(report)
    print("=" * 90)

    print(f"\nГотово.")
    print(f"Резултати: {OUTPUT_DIR}")
    print(f"PNG слики: {PLOTS_DIR}")
    print(f"CSV табели: {TABLES_DIR}")


if __name__ == "__main__":
    main()