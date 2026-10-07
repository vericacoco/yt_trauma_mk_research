from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

SEARCH_TERMS = [
    "cardiff",
    "nlptown",
    "bertweet",
    "sentiment",
    "label",
    "positive",
    "negative",
    "neutral",
]


print("=" * 110)
print(f"ПРЕБАРУВАЊЕ НА ЦЕЛ ПРОЕКТ: {BASE_DIR}")
print("=" * 110)

csv_files = sorted(BASE_DIR.rglob("*.csv"))

print(f"\nВкупно CSV датотеки во проектот: {len(csv_files)}")

found_files = 0

for csv_file in csv_files:
    try:
        df = pd.read_csv(
            csv_file,
            encoding="utf-8-sig",
            nrows=2,
            low_memory=False,
        )
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(
                csv_file,
                encoding="utf-8",
                nrows=2,
                low_memory=False,
            )
        except Exception:
            continue
    except Exception:
        continue

    file_name_lower = csv_file.name.lower()
    columns_lower = [str(column).lower() for column in df.columns]

    matching_columns = [
        column
        for column in df.columns
        if any(
            term in str(column).lower()
            for term in SEARCH_TERMS
        )
    ]

    matching_file_name = any(
        term in file_name_lower
        for term in ["cardiff", "nlptown", "bertweet"]
    )

    if matching_file_name or matching_columns:
        found_files += 1

        print("\n" + "-" * 110)
        print(f"ДАТОТЕКА: {csv_file}")
        print(f"БРОЈ КОЛОНИ: {len(df.columns)}")

        print("\nРЕЛЕВАНТНИ КОЛОНИ:")

        if matching_columns:
            for column in matching_columns:
                print(f"  - {column}")
        else:
            print("  Нема директно препознаени sentiment колони.")

        print("\nСИТЕ КОЛОНИ:")

        for column in df.columns:
            print(f"  - {column}")


print("\n" + "=" * 110)
print(f"Пронајдени релевантни датотеки: {found_files}")
print("=" * 110)