import pandas as pd
import re
import unicodedata
from collections import defaultdict

BASE = "sample_dataset/dataset"

# =========================================================
# LOAD
# =========================================================

s1 = pd.read_csv(f"{BASE}/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv(f"{BASE}/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv(f"{BASE}/train/train_source3.tsv", sep="\t")
gt = pd.read_csv(f"{BASE}/train/train_ground_truth.tsv", sep="\t")


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    result = []

    for c in text:
        if c.isalnum():
            result.append(c)
        else:
            result.append(" ")

    return re.sub(r"\s+", " ", "".join(result)).strip()


def get_tokens(text):
    value = normalize(text)
    return set(value.split()) if value else set()


def get_numbers(text):
    if pd.isna(text):
        return set()

    return set(re.findall(r"\d+", str(text)))


def prepare(df):
    df = df.copy()

    df["name_norm"] = df["business_name"].apply(normalize)

    df["name_tokens"] = df["business_name"].apply(get_tokens)

    df["address_tokens"] = df["business_address"].apply(get_tokens)

    df["address_numbers"] = df["business_address"].apply(
        get_numbers
    )

    return df


s1 = prepare(s1)
s2 = prepare(s2)
s3 = prepare(s3)


# =========================================================
# BUILD INDICES
# =========================================================

def build_index(df):

    index = {
        "name_exact": defaultdict(set),
        "name_token": defaultdict(set),
        "address_token": defaultdict(set),
        "number": defaultdict(set)
    }

    for idx, row in df.iterrows():

        # Exact normalized name
        if row["name_norm"]:
            index["name_exact"][
                row["name_norm"]
            ].add(idx)

        # Name tokens
        for token in row["name_tokens"]:

            if len(token) >= 3:
                index["name_token"][token].add(idx)

        # Address tokens
        for token in row["address_tokens"]:

            if len(token) >= 4:
                index["address_token"][token].add(idx)

        # Numbers
        for number in row["address_numbers"]:
            index["number"][number].add(idx)

    return index


s2_index = build_index(s2)
s3_index = build_index(s3)


# =========================================================
# GET TRUE MATCHES
# =========================================================

def true_ids(gt_row):

    value = gt_row["matched_entity_ids"]

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return set(value.split(","))


# =========================================================
# EVALUATE A STRATEGY
# =========================================================

def evaluate(strategy_name, candidate_function):

    total_true = 0
    total_found = 0

    total_candidates = 0
    max_candidates = 0

    missed_entities = 0

    for i, row in s1.iterrows():

        s2_candidates = candidate_function(
            row,
            s2_index
        )

        s3_candidates = candidate_function(
            row,
            s3_index
        )

        # Convert indexes to entity IDs
        candidate_ids = set()

        for idx in s2_candidates:
            candidate_ids.add(
                s2.iloc[idx]["entity_id"]
            )

        for idx in s3_candidates:
            candidate_ids.add(
                s3.iloc[idx]["entity_id"]
            )

        total_candidates += len(candidate_ids)

        max_candidates = max(
            max_candidates,
            len(candidate_ids)
        )

        true = true_ids(gt.iloc[i])

        found = true.intersection(candidate_ids)

        total_true += len(true)
        total_found += len(found)

        if len(found) < len(true):
            missed_entities += 1

    recall = (
        total_found / total_true * 100
        if total_true
        else 0
    )

    average_candidates = (
        total_candidates / len(s1)
    )

    print("\n" + "=" * 80)
    print(strategy_name)
    print("=" * 80)

    print(f"True matches       : {total_true:,}")
    print(f"Found              : {total_found:,}")
    print(f"Missed             : {total_true - total_found:,}")
    print(f"Recall              : {recall:.2f}%")
    print(f"Average candidates : {average_candidates:.2f}")
    print(f"Maximum candidates : {max_candidates:,}")
    print(f"Entities with miss: {missed_entities:,}")

    return {
        "strategy": strategy_name,
        "recall": recall,
        "avg_candidates": average_candidates,
        "max_candidates": max_candidates
    }


# =========================================================
# STRATEGY 1
# EXACT NAME
# =========================================================

def exact_name(row, index):

    if not row["name_norm"]:
        return set()

    return index["name_exact"].get(
        row["name_norm"],
        set()
    )


# =========================================================
# STRATEGY 2
# NAME TOKEN + ADDRESS TOKEN
#
# Candidate must share BOTH:
# at least one name token
# AND
# at least one address token
# =========================================================

def name_and_address(row, index):

    name_candidates = set()
    address_candidates = set()

    for token in row["name_tokens"]:

        if len(token) >= 3:
            name_candidates.update(
                index["name_token"].get(
                    token,
                    set()
                )
            )

    for token in row["address_tokens"]:

        if len(token) >= 4:
            address_candidates.update(
                index["address_token"].get(
                    token,
                    set()
                )
            )

    return name_candidates.intersection(
        address_candidates
    )


# =========================================================
# STRATEGY 3
# NAME TOKEN + ADDRESS NUMBER
# =========================================================

def name_and_number(row, index):

    name_candidates = set()
    number_candidates = set()

    for token in row["name_tokens"]:

        if len(token) >= 3:
            name_candidates.update(
                index["name_token"].get(
                    token,
                    set()
                )
            )

    for number in row["address_numbers"]:

        number_candidates.update(
            index["number"].get(
                number,
                set()
            )
        )

    return name_candidates.intersection(
        number_candidates
    )


# =========================================================
# STRATEGY 4
# ADDRESS TOKEN + NUMBER
# =========================================================

def address_and_number(row, index):

    address_candidates = set()
    number_candidates = set()

    for token in row["address_tokens"]:

        if len(token) >= 4:
            address_candidates.update(
                index["address_token"].get(
                    token,
                    set()
                )
            )

    for number in row["address_numbers"]:

        number_candidates.update(
            index["number"].get(
                number,
                set()
            )
        )

    return address_candidates.intersection(
        number_candidates
    )


# =========================================================
# STRATEGY 5
# TWO NAME TOKENS
# =========================================================

def two_name_tokens(row, index):

    token_candidates = defaultdict(int)

    for token in row["name_tokens"]:

        if len(token) < 3:
            continue

        for idx in index["name_token"].get(
            token,
            set()
        ):
            token_candidates[idx] += 1

    return {
        idx
        for idx, count in token_candidates.items()
        if count >= 2
    }


# =========================================================
# STRATEGY 6
# TWO ADDRESS TOKENS
# =========================================================

def two_address_tokens(row, index):

    token_candidates = defaultdict(int)

    for token in row["address_tokens"]:

        if len(token) < 4:
            continue

        for idx in index["address_token"].get(
            token,
            set()
        ):
            token_candidates[idx] += 1

    return {
        idx
        for idx, count in token_candidates.items()
        if count >= 2
    }


# =========================================================
# STRATEGY 7
# EXACT NAME OR NAME+ADDRESS
# =========================================================

def exact_or_name_address(row, index):

    result = set()

    # Exact name
    if row["name_norm"]:

        result.update(
            index["name_exact"].get(
                row["name_norm"],
                set()
            )
        )

    # Name + address
    result.update(
        name_and_address(row, index)
    )

    return result


# =========================================================
# STRATEGY 8
# EXACT NAME OR NAME+NUMBER
# =========================================================

def exact_or_name_number(row, index):

    result = set()

    # Exact name
    if row["name_norm"]:

        result.update(
            index["name_exact"].get(
                row["name_norm"],
                set()
            )
        )

    # Name + number
    result.update(
        name_and_number(row, index)
    )

    return result


# =========================================================
# RUN ALL STRATEGIES
# =========================================================

results = []

strategies = [
    ("1. Exact Name", exact_name),
    ("2. Name AND Address", name_and_address),
    ("3. Name AND Number", name_and_number),
    ("4. Address AND Number", address_and_number),
    ("5. Two Name Tokens", two_name_tokens),
    ("6. Two Address Tokens", two_address_tokens),
    ("7. Exact Name OR Name+Address", exact_or_name_address),
    ("8. Exact Name OR Name+Number", exact_or_name_number),
]


for name, function in strategies:

    result = evaluate(
        name,
        function
    )

    results.append(result)


# =========================================================
# FINAL TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 100)
print("FINAL STRATEGY COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)

print("\nDONE")