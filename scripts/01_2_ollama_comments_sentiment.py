import pandas as pd
import requests
import json
import time
from tqdm import tqdm
from pathlib import Path

MODEL = "llama3.1:8b"

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "datasets" / "comments_mk_en.csv"
OUTPUT = BASE_DIR / "datasets" / "ollama_comments_sentiment.csv"

df = pd.read_csv(INPUT, encoding="utf-8-sig")

ID_COL = "video_id"
MK_COL = "text_mk"
EN_COL = "text_en"

Path("outputs").mkdir(exist_ok=True)


def classify_sentiment(text):
    if pd.isna(text) or str(text).strip() == "":
        return "neutral", 0.0

    prompt = f"""
You are a sentiment analysis evaluator for political YouTube comments.

Classify the sentiment as exactly one of:
positive, neutral, negative.

Guidelines:
- Consider political tone, sarcasm, criticism, insults, irony, support, blame, anger, and approval.
- If the text attacks, mocks, blames, insults, or expresses anger, classify it as negative.
- If the text expresses support, praise, approval, or hope, classify it as positive.
- If the text is factual, unclear, mixed, or mainly descriptive, classify it as neutral.

Return only valid JSON in this exact format:
{{
  "label": "positive",
  "confidence": 0.0
}}

Text:
{text}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 80
                }
            },
            timeout=180
        )

        response.raise_for_status()
        raw = response.json()["response"].strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1
        raw_json = raw[start:end]

        result = json.loads(raw_json)

        label = result.get("label", "neutral").lower().strip()
        confidence = float(result.get("confidence", 0.0))

        if label not in ["positive", "neutral", "negative"]:
            label = "neutral"

        confidence = max(0.0, min(confidence, 1.0))

        return label, confidence

    except Exception as e:
        print("Error:", e)
        return "neutral", 0.0


df = pd.read_csv(INPUT, encoding="utf-8-sig")

# ако веќе постои output, продолжува од него
if Path(OUTPUT).exists():
    df_out = pd.read_csv(OUTPUT, encoding="utf-8-sig")
else:
    df_out = df.copy()
    df_out["llm_label_mk_c"] = ""
    df_out["llm_confidence_mk_c"] = ""
    df_out["llm_label_en_c"] = ""
    df_out["llm_confidence_en_c"] = ""

for i in tqdm(range(len(df_out))):
    if pd.isna(df_out.loc[i, "llm_label_mk_c"]) or df_out.loc[i, "llm_label_mk_c"] == "":
        label, conf = classify_sentiment(df_out.loc[i, MK_COL])
        df_out.loc[i, "llm_label_mk_c"] = label
        df_out.loc[i, "llm_confidence_mk_c"] = conf

    if pd.isna(df_out.loc[i, "llm_label_en_c"]) or df_out.loc[i, "llm_label_en_c"] == "":
        label, conf = classify_sentiment(df_out.loc[i, EN_COL])
        df_out.loc[i, "llm_label_en_c"] = label
        df_out.loc[i, "llm_confidence_en_c"] = conf

    df_out.loc[i, "llm_agreement_title_c"] = (
        "agree"
        if df_out.loc[i, "llm_label_mk_c"] == df_out.loc[i, "llm_label_en_c"]
        else "disagree"
    )

    if i % 10 == 0:
        df_out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    time.sleep(0.1)

df_out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("Done.")
print("MK distribution:")
print(df_out["llm_label_mk_c"].value_counts())

print("\nEN distribution:")
print(df_out["llm_label_en_c"].value_counts())

print("\nMK vs EN agreement:")
print(df_out["llm_agreement_title_c"].value_counts(normalize=True) * 100)

print("\nConfusion matrix:")
print(pd.crosstab(df_out["llm_label_mk_c"], df_out["llm_label_en_c"]))