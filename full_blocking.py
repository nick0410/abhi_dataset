import pandas as pd
import re
import unicodedata
from collections import defaultdict

BASE = "dataset/train"

print("Loading datasets...")

s1 = pd.read_csv(f"{BASE}/train_source1.tsv", sep="\t")
s2 = pd.read_csv(f"{BASE}/train_source2.tsv", sep="\t")
s3 = pd.read_csv(f"{BASE}/train_source3.tsv", sep="\t")

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_tokens(text):
    return text.split()


def get_numbers(text):
    return re.findall(r"\d+", text)


def build_index(df, source_name):

    name_exact = defaultdict(list)
    name_token = defaultdict(list)
    address_token = defaultdict(list)
    number_index = defaultdict(list)

    print(f"\nBuilding {source_name} index...")

    for idx, row in df.iterrows():

        name = normalize(row["business_name"])
        address = normalize(row["business_address"])

        if name:
            name_exact[name].append(idx)

        for token in set(get_tokens(name)):
            if len(token) >= 3:
                name_token[token].append(idx)

        for token in set(get_tokens(address)):
            if len(token) >= 4:
                address_token[token].append(idx)

        for number in set(get_numbers(address)):
            number_index[number].append(idx)

        if idx % 100000 == 0:
            print(f"{source_name}: {idx}/{len(df)}")

    return (
        name_exact,
        name_token,
        address_token,
        number_index
    )


print("\nBuilding S2 index...")
s2_index = build_index(s2, "S2")

print("\nBuilding S3 index...")
s3_index = build_index(s3, "S3")

print("\nIndexes built successfully.")


def get_candidates(row, index):

    name_exact, name_token, address_token, number_index = index

    name = normalize(row["business_name"])
    address = normalize(row["business_address"])

    candidates = set()

    # Exact name
    candidates.update(
        name_exact.get(name, [])
    )

    # Name tokens
    for token in set(get_tokens(name)):
        if len(token) >= 3:
            candidates.update(
                name_token.get(token, [])
            )

    # Address tokens
    for token in set(get_tokens(address)):
        if len(token) >= 4:
            candidates.update(
                address_token.get(token, [])
            )

    # Numbers
    for number in set(get_numbers(address)):
        candidates.update(
            number_index.get(number, [])
        )

    return candidates


total_candidates = 0
max_candidates = 0

print("\nTesting blocker on S1...")

for i, row in s1.iterrows():

    c2 = get_candidates(row, s2_index)
    c3 = get_candidates(row, s3_index)

    total = len(c2) + len(c3)

    total_candidates += total
    max_candidates = max(max_candidates, total)

    if i % 10000 == 0:
        print(
            f"S1: {i}/{len(s1)} | "
            f"Avg candidates: {total_candidates/(i+1):.2f} | "
            f"Max: {max_candidates}"
        )

print("\n========== DONE ==========")
print("Total S1:", len(s1))
print("Average candidates:", total_candidates / len(s1))
print("Maximum candidates:", max_candidates)