import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# =========================================================
# SETTINGS
# =========================================================

MODEL = "gemma3:4b"

OLLAMA_URL = "http://localhost:11434/api/generate"

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "datasets" / "videos_mk_en_combined.csv"
OUTPUT = BASE_DIR / "outputs" / "titles_llm_llama_probabilities.csv"

ID_COL = "video_id"
MK_COL = "title_description_mk"
EN_COL = "title_description_en"

SAVE_EVERY = 10
TIMEOUT = 180
MAX_RETRIES = 3

VALID_LABELS = {
    "positive",
    "neutral",
    "negative"
}

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# JSON SCHEMA
# =========================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "positive",
                "neutral",
                "negative"
            ]
        },
        "probabilities": {
            "type": "object",
            "properties": {
                "positive": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "neutral": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "negative": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": [
                "positive",
                "neutral",
                "negative"
            ],
            "additionalProperties": False
        }
    },
    "required": [
        "label",
        "probabilities"
    ],
    "additionalProperties": False
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def is_empty_text(text):
    """
    Проверува дали текстот е празен.
    """

    return (
        pd.isna(text)
        or str(text).strip() == ""
    )


def save_csv_safely(dataframe, output_path):
    """
    Безбедно го зачувува CSV фајлот преку привремен фајл.
    """

    output_path = Path(output_path)

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig"
    )

    os.replace(
        temporary_path,
        output_path
    )


def normalize_probabilities(
    positive_probability,
    neutral_probability,
    negative_probability
):
    """
    Ги ограничува probability вредностите меѓу 0 и 1
    и ги нормализира така што нивниот збир да биде 1.
    """

    positive_probability = float(
        positive_probability
    )

    neutral_probability = float(
        neutral_probability
    )

    negative_probability = float(
        negative_probability
    )

    positive_probability = max(
        0.0,
        min(positive_probability, 1.0)
    )

    neutral_probability = max(
        0.0,
        min(neutral_probability, 1.0)
    )

    negative_probability = max(
        0.0,
        min(negative_probability, 1.0)
    )

    total = (
        positive_probability
        + neutral_probability
        + negative_probability
    )

    if total <= 0:
        raise ValueError(
            "The probability sum is zero."
        )

    return {
        "positive": (
            positive_probability / total
        ),
        "neutral": (
            neutral_probability / total
        ),
        "negative": (
            negative_probability / total
        )
    }


# =========================================================
# GEMMA SENTIMENT CLASSIFICATION
# =========================================================

def classify_sentiment(text):
    """
    Класифицира еден текст со Gemma.

    Враќа:
    - label
    - confidence
    - positive probability
    - neutral probability
    - negative probability
    - status
    - error
    """

    if is_empty_text(text):
        return {
            "label": "",
            "confidence": pd.NA,
            "positive_probability": pd.NA,
            "neutral_probability": pd.NA,
            "negative_probability": pd.NA,
            "status": "missing",
            "error": ""
        }

    text = str(text).strip()

    prompt = f"""
You are an expert sentiment evaluator for political YouTube
video titles and descriptions.

Classify the sentiment expressed in the text as exactly one of:

- positive
- neutral
- negative

Classification rules:

- Negative sentiment includes criticism, attacks, mockery,
  blame, insults, anger, fear, dissatisfaction and disapproval.

- Positive sentiment includes support, praise, approval,
  satisfaction, encouragement and hope.

- Neutral sentiment includes factual, descriptive, unclear,
  balanced or mixed content without a clear emotional position.

- Consider political tone, sarcasm, irony, insults,
  criticism, support and emotional meaning.

- Classify the sentiment expressed by the author.
  Do not classify whether the topic itself is good or bad.

Probability rules:

- Return a probability for positive, neutral and negative.
- Every probability must be between 0.0 and 1.0.
- The probabilities should sum approximately to 1.0.
- Use decimal values such as 0.72, 0.18 and 0.10.
- Do not return only 0 and 1 unless the sentiment is
  exceptionally clear.
- The label must correspond to the class with the highest
  probability.
- Return only valid JSON.

TEXT START

{text}

TEXT END
"""

    request_data = {
        "model": MODEL,
        "prompt": prompt,
        "format": OUTPUT_SCHEMA,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 120
        }
    }

    last_error = ""

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=request_data,
                timeout=TIMEOUT
            )

            response.raise_for_status()

            response_data = response.json()

            raw_result = response_data[
                "response"
            ].strip()

            result = json.loads(
                raw_result
            )

            model_label = str(
                result["label"]
            ).strip().lower()

            if model_label not in VALID_LABELS:
                raise ValueError(
                    f"Invalid label: {model_label}"
                )

            raw_probabilities = result[
                "probabilities"
            ]

            probabilities = normalize_probabilities(
                raw_probabilities["positive"],
                raw_probabilities["neutral"],
                raw_probabilities["negative"]
            )

            # Label се избира според најголемата probability
            final_label = max(
                probabilities,
                key=probabilities.get
            )

            confidence = probabilities[
                final_label
            ]

            return {
                "label": final_label,

                "confidence": round(
                    confidence,
                    4
                ),

                "positive_probability": round(
                    probabilities["positive"],
                    4
                ),

                "neutral_probability": round(
                    probabilities["neutral"],
                    4
                ),

                "negative_probability": round(
                    probabilities["negative"],
                    4
                ),

                "status": "ok",

                "error": ""
            }

        except Exception as error:
            last_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    2 ** (attempt - 1)
                )

    return {
        "label": "",
        "confidence": pd.NA,
        "positive_probability": pd.NA,
        "neutral_probability": pd.NA,
        "negative_probability": pd.NA,
        "status": "error",
        "error": last_error[:500]
    }


# =========================================================
# OUTPUT COLUMNS
# =========================================================

OUTPUT_COLUMNS = [
    "gemma_label_mk",
    "gemma_confidence_mk",
    "gemma_positive_probability_mk",
    "gemma_neutral_probability_mk",
    "gemma_negative_probability_mk",
    "gemma_status_mk",
    "gemma_error_mk",

    "gemma_label_en",
    "gemma_confidence_en",
    "gemma_positive_probability_en",
    "gemma_neutral_probability_en",
    "gemma_negative_probability_en",
    "gemma_status_en",
    "gemma_error_en",

    "gemma_agreement_title"
]


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    """
    Го вчитува оригиналниот dataset.

    Ако OUTPUT веќе постои, продолжува од претходно
    зачуваните резултати.
    """

    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: "
            f"{input_path.resolve()}"
        )

    original_df = pd.read_csv(
        input_path,
        encoding="utf-8-sig"
    )

    required_input_columns = [
        ID_COL,
        MK_COL,
        EN_COL
    ]

    missing_columns = [
        column
        for column in required_input_columns
        if column not in original_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
            + "\nAvailable columns: "
            + ", ".join(
                original_df.columns.astype(str)
            )
        )

    if output_path.exists():
        df_out = pd.read_csv(
            output_path,
            encoding="utf-8-sig"
        )

        if len(df_out) != len(original_df):
            raise ValueError(
                "Input and output have different "
                "numbers of rows."
            )

        original_ids = (
            original_df[ID_COL]
            .astype(str)
            .tolist()
        )

        output_ids = (
            df_out[ID_COL]
            .astype(str)
            .tolist()
        )

        if original_ids != output_ids:
            raise ValueError(
                "The video IDs or row order changed."
            )

        # Ги освежува оригиналните колони
        for column in original_df.columns:
            df_out[column] = original_df[column]

    else:
        df_out = original_df.copy()

    for column in OUTPUT_COLUMNS:
        if column not in df_out.columns:
            df_out[column] = pd.NA

    return df_out


# =========================================================
# RESULT HELPERS
# =========================================================

def is_completed(
    dataframe,
    row_index,
    language
):
    """
    Проверува дали редот веќе е обработен.
    """

    status_column = (
        f"gemma_status_{language}"
    )

    status = str(
        dataframe.loc[
            row_index,
            status_column
        ]
    ).strip().lower()

    return status in {
        "ok",
        "missing"
    }


def save_language_result(
    dataframe,
    row_index,
    language,
    result
):
    """
    Го запишува резултатот за MK или EN.
    """

    dataframe.loc[
        row_index,
        f"gemma_label_{language}"
    ] = result["label"]

    dataframe.loc[
        row_index,
        f"gemma_confidence_{language}"
    ] = result["confidence"]

    dataframe.loc[
        row_index,
        f"gemma_positive_probability_{language}"
    ] = result["positive_probability"]

    dataframe.loc[
        row_index,
        f"gemma_neutral_probability_{language}"
    ] = result["neutral_probability"]

    dataframe.loc[
        row_index,
        f"gemma_negative_probability_{language}"
    ] = result["negative_probability"]

    dataframe.loc[
        row_index,
        f"gemma_status_{language}"
    ] = result["status"]

    dataframe.loc[
        row_index,
        f"gemma_error_{language}"
    ] = result["error"]


def calculate_agreement(
    dataframe,
    row_index
):
    """
    Проверува дали MK и EN имаат ист label.
    """

    mk_status = str(
        dataframe.loc[
            row_index,
            "gemma_status_mk"
        ]
    ).strip().lower()

    en_status = str(
        dataframe.loc[
            row_index,
            "gemma_status_en"
        ]
    ).strip().lower()

    if (
        mk_status == "ok"
        and en_status == "ok"
    ):
        mk_label = dataframe.loc[
            row_index,
            "gemma_label_mk"
        ]

        en_label = dataframe.loc[
            row_index,
            "gemma_label_en"
        ]

        if mk_label == en_label:
            agreement = "agree"
        else:
            agreement = "disagree"

    elif (
        mk_status == "missing"
        or en_status == "missing"
    ):
        agreement = "missing"

    else:
        agreement = "error"

    dataframe.loc[
        row_index,
        "gemma_agreement_title"
    ] = agreement


# =========================================================
# MAIN
# =========================================================

def main():
    df_out = load_data()

    processed_since_save = 0

    for index in tqdm(
        range(len(df_out)),
        desc="Gemma sentiment analysis"
    ):

        # Македонски текст
        if not is_completed(
            df_out,
            index,
            "mk"
        ):
            mk_result = classify_sentiment(
                df_out.loc[
                    index,
                    MK_COL
                ]
            )

            save_language_result(
                df_out,
                index,
                "mk",
                mk_result
            )

            processed_since_save += 1

        # Англиски текст
        if not is_completed(
            df_out,
            index,
            "en"
        ):
            en_result = classify_sentiment(
                df_out.loc[
                    index,
                    EN_COL
                ]
            )

            save_language_result(
                df_out,
                index,
                "en",
                en_result
            )

            processed_since_save += 1

        calculate_agreement(
            df_out,
            index
        )

        # Зачувување по секои 10 класификации
        if processed_since_save >= SAVE_EVERY:
            save_csv_safely(
                df_out,
                OUTPUT
            )

            processed_since_save = 0

        time.sleep(0.1)

    # Финално зачувување
    save_csv_safely(
        df_out,
        OUTPUT
    )

    print("\nDone.")

    print(
        "\nOutput file:"
    )

    print(
        Path(OUTPUT).resolve()
    )

    print(
        "\nMK sentiment distribution:"
    )

    print(
        df_out.loc[
            df_out["gemma_status_mk"] == "ok",
            "gemma_label_mk"
        ].value_counts()
    )

    print(
        "\nEN sentiment distribution:"
    )

    print(
        df_out.loc[
            df_out["gemma_status_en"] == "ok",
            "gemma_label_en"
        ].value_counts()
    )

    print(
        "\nMK vs EN agreement:"
    )

    valid_agreement = df_out[
        "gemma_agreement_title"
    ].isin(
        [
            "agree",
            "disagree"
        ]
    )

    print(
        df_out.loc[
            valid_agreement,
            "gemma_agreement_title"
        ].value_counts(
            normalize=True
        ) * 100
    )

    print(
        "\nMK vs EN confusion matrix:"
    )

    valid_rows = (
        (
            df_out["gemma_status_mk"]
            == "ok"
        )
        &
        (
            df_out["gemma_status_en"]
            == "ok"
        )
    )

    confusion_matrix = pd.crosstab(
        df_out.loc[
            valid_rows,
            "gemma_label_mk"
        ],
        df_out.loc[
            valid_rows,
            "gemma_label_en"
        ],
        rownames=["MK"],
        colnames=["EN"]
    )

    print(
        confusion_matrix
    )

    print(
        "\nAverage MK confidence:"
    )

    print(
        pd.to_numeric(
            df_out.loc[
                df_out["gemma_status_mk"] == "ok",
                "gemma_confidence_mk"
            ],
            errors="coerce"
        ).mean()
    )

    print(
        "\nAverage EN confidence:"
    )

    print(
        pd.to_numeric(
            df_out.loc[
                df_out["gemma_status_en"] == "ok",
                "gemma_confidence_en"
            ],
            errors="coerce"
        ).mean()
    )

    print(
        "\nMK errors:"
    )

    print(
        (
            df_out["gemma_status_mk"]
            == "error"
        ).sum()
    )

    print(
        "\nEN errors:"
    )

    print(
        (
            df_out["gemma_status_en"]
            == "error"
        ).sum()
    )


if __name__ == "__main__":
    main()