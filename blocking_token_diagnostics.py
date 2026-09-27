import pandas as pd
import re
import unicodedata
from collections import Counter

BASE = "sample_dataset/dataset"

s1 = pd.read_csv(
    f"{BASE}/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    f"{BASE}/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    f"{BASE}/train/train_source3.tsv",
    sep="\t"
)


# ---------------------------------------------------------
# Unicode-aware normalization
# ---------------------------------------------------------

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove accents but preserve other Unicode scripts
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # Keep Unicode letters/numbers
    result = []

    for c in text:
        if c.isalnum():
            result.append(c)
        else:
            result.append(" ")

    text = "".join(result)

    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------
# Extract name tokens
# ---------------------------------------------------------

def get_tokens(text):
    value = normalize(text)

    if not value:
        return set()

    return set(value.split())


# ---------------------------------------------------------
# Count how many records contain each token
# ---------------------------------------------------------

def count_tokens(df, source_name):

    counter = Counter()

    for name in df["business_name"]:
        tokens = get_tokens(name)

        for token in tokens:

            # Ignore extremely short tokens
            if len(token) >= 3:
                counter[token] += 1

    print("\n" + "=" * 80)
    print(f"{source_name} NAME TOKEN DIAGNOSTICS")
    print("=" * 80)

    print(f"\nTotal records: {len(df):,}")
    print(f"Unique tokens: {len(counter):,}")

    print("\nTOP 30 MOST COMMON TOKENS:")
    print("-" * 60)

    for token, count in counter.most_common(30):
        print(f"{token:30} {count:6,}")

    print("\n")

    # Frequency buckets
    buckets = {
        "1 record": 0,
        "2-5 records": 0,
        "6-10 records": 0,
        "11-25 records": 0,
        "26-50 records": 0,
        "51-100 records": 0,
        "101-500 records": 0,
        "501+ records": 0
    }

    for count in counter.values():

        if count == 1:
            buckets["1 record"] += 1

        elif count <= 5:
            buckets["2-5 records"] += 1

        elif count <= 10:
            buckets["6-10 records"] += 1

        elif count <= 25:
            buckets["11-25 records"] += 1

        elif count <= 50:
            buckets["26-50 records"] += 1

        elif count <= 100:
            buckets["51-100 records"] += 1

        elif count <= 500:
            buckets["101-500 records"] += 1

        else:
            buckets["501+ records"] += 1

    print("TOKEN FREQUENCY DISTRIBUTION:")
    print("-" * 60)

    for bucket, count in buckets.items():
        print(f"{bucket:20} {count:8,}")


# ---------------------------------------------------------
# Run diagnostics
# ---------------------------------------------------------

count_tokens(s1, "SOURCE 1")
count_tokens(s2, "SOURCE 2")
count_tokens(s3, "SOURCE 3")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)