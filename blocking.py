import pandas as pd
import re
import unicodedata
from collections import defaultdict


# ============================================================
# USE ONLY SAMPLE DATA FOR NOW
# ============================================================

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


print("Loaded:")
print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def tokens(text):

    value = normalize(text)

    if not value:
        return set()

    return set(value.split())


def numbers(text):

    if pd.isna(text):
        return set()

    return set(
        re.findall(r"\d+", str(text))
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].apply(
        normalize
    )

    df["name_tokens"] = df["business_name"].apply(
        tokens
    )

    df["address_tokens"] = df["business_address"].apply(
        tokens
    )

    df["address_numbers"] = df["business_address"].apply(
        numbers
    )

    return df


s1 = prepare(s1)
s2 = prepare(s2)
s3 = prepare(s3)


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(df):

    index = {
        "name": defaultdict(set),
        "name_token": defaultdict(set),
        "address_token": defaultdict(set),
        "number": defaultdict(set)
    }

    for idx, row in df.iterrows():

        name = row["name_norm"]

        if name:
            index["name"][name].add(idx)

        for token in row["name_tokens"]:
            if len(token) >= 3:
                index["name_token"][token].add(idx)

        for token in row["address_tokens"]:
            if len(token) >= 4:
                index["address_token"][token].add(idx)

        for number in row["address_numbers"]:
            index["number"][number].add(idx)

    return index


print("Building S2 index...")
s2_index = build_index(s2)

print("Building S3 index...")
s3_index = build_index(s3)

print("Indexes ready!")


# ============================================================
# CANDIDATES
# ============================================================

def candidates(row, index):

    result = set()

    # Exact normalized name
    if row["name_norm"]:
        result.update(
            index["name"].get(
                row["name_norm"],
                set()
            )
        )

    # Name tokens
    for token in row["name_tokens"]:

        if len(token) >= 3:
            result.update(
                index["name_token"].get(
                    token,
                    set()
                )
            )

    # Address tokens
    for token in row["address_tokens"]:

        if len(token) >= 4:
            result.update(
                index["address_token"].get(
                    token,
                    set()
                )
            )

    # Address numbers
    for number in row["address_numbers"]:

        result.update(
            index["number"].get(
                number,
                set()
            )
        )

    return result


# ============================================================
# TEST ONLY 20 S1 RECORDS
# ============================================================

print("\n" + "=" * 70)
print("BLOCKING TEST")
print("=" * 70)

total = 0
maximum = 0

for _, row in s1.head(20).iterrows():

    c2 = candidates(row, s2_index)
    c3 = candidates(row, s3_index)

    count = len(c2) + len(c3)

    total += count
    maximum = max(maximum, count)

    print(
        row["entity_id"],
        "->",
        count,
        "candidates"
    )


print("\nAverage candidates:", total / 20)
print("Maximum candidates:", maximum)

print("\nDONE")