import pandas as pd
import re
import unicodedata
from collections import defaultdict

BASE = "sample_dataset/dataset"

# =========================================================
# LOAD DATA
# =========================================================

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

gt = pd.read_csv(
    f"{BASE}/train/train_ground_truth.tsv",
    sep="\t"
)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = unicodedata.normalize(
        "NFKD",
        text
    )

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

    text = "".join(result)

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def get_tokens(text):

    value = normalize(text)

    if not value:
        return set()

    return set(value.split())


def get_numbers(text):

    if pd.isna(text):
        return set()

    return set(
        re.findall(
            r"\d+",
            str(text)
        )
    )


# =========================================================
# PREPARE
# =========================================================

def prepare(df):

    df = df.copy()

    df["name_norm"] = (
        df["business_name"]
        .apply(normalize)
    )

    df["name_tokens"] = (
        df["business_name"]
        .apply(get_tokens)
    )

    df["address_tokens"] = (
        df["business_address"]
        .apply(get_tokens)
    )

    df["address_numbers"] = (
        df["business_address"]
        .apply(get_numbers)
    )

    return df


s1 = prepare(s1)
s2 = prepare(s2)
s3 = prepare(s3)


# =========================================================
# INDICES
# =========================================================

def build_index(df):

    index = {
        "exact_name": defaultdict(set),
        "name_token": defaultdict(set),
        "address_token": defaultdict(set),
        "number": defaultdict(set)
    }

    for idx, row in df.iterrows():

        # Exact name
        if row["name_norm"]:

            index["exact_name"][
                row["name_norm"]
            ].add(idx)

        # Name tokens
        for token in row["name_tokens"]:

            if len(token) >= 3:

                index["name_token"][
                    token
                ].add(idx)

        # Address tokens
        for token in row["address_tokens"]:

            if len(token) >= 4:

                index["address_token"][
                    token
                ].add(idx)

        # Numbers
        for number in row["address_numbers"]:

            index["number"][
                number
            ].add(idx)

    return index


s2_index = build_index(s2)
s3_index = build_index(s3)


# =========================================================
# GROUND TRUTH
# =========================================================

def get_true_ids(row):

    value = row["matched_entity_ids"]

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return set(
        value.split(",")
    )


# =========================================================
# RULE CANDIDATES
# =========================================================

def exact_name_candidates(row, index):

    if not row["name_norm"]:
        return set()

    return index["exact_name"].get(
        row["name_norm"],
        set()
    )


def name_token_candidates(row, index):

    result = set()

    for token in row["name_tokens"]:

        if len(token) < 3:
            continue

        result.update(
            index["name_token"].get(
                token,
                set()
            )
        )

    return result


def address_token_candidates(row, index):

    result = set()

    for token in row["address_tokens"]:

        if len(token) < 4:
            continue

        result.update(
            index["address_token"].get(
                token,
                set()
            )
        )

    return result


def number_candidates(row, index):

    result = set()

    for number in row["address_numbers"]:

        result.update(
            index["number"].get(
                number,
                set()
            )
        )

    return result


# =========================================================
# CONVERT INDEX -> ENTITY IDs
# =========================================================

def ids_from_indices(
    s2_indices,
    s3_indices
):

    result = set()

    for idx in s2_indices:

        result.add(
            s2.iloc[idx]["entity_id"]
        )

    for idx in s3_indices:

        result.add(
            s3.iloc[idx]["entity_id"]
        )

    return result


# =========================================================
# ANALYZE ALL TRUE MATCHES
# =========================================================

rule_counts = {
    "exact_name": 0,
    "name_token": 0,
    "address_token": 0,
    "number": 0,

    "name_only": 0,
    "address_only": 0,
    "number_only": 0,

    "name_and_address": 0,
    "name_and_number": 0,
    "address_and_number": 0,

    "multiple_rules": 0,

    "caught_by_any_rule": 0,
    "missed_by_all_rules": 0
}


total_true = 0


for i, row in s1.iterrows():

    true_ids = get_true_ids(
        gt.iloc[i]
    )

    if not true_ids:
        continue

    # S2
    s2_exact = ids_from_indices(
        exact_name_candidates(row, s2_index),
        set()
    )

    s2_name = ids_from_indices(
        name_token_candidates(row, s2_index),
        set()
    )

    s2_address = ids_from_indices(
        address_token_candidates(row, s2_index),
        set()
    )

    s2_number = ids_from_indices(
        number_candidates(row, s2_index),
        set()
    )

    # S3
    s3_exact = ids_from_indices(
        set(),
        exact_name_candidates(row, s3_index)
    )

    s3_name = ids_from_indices(
        set(),
        name_token_candidates(row, s3_index)
    )

    s3_address = ids_from_indices(
        set(),
        address_token_candidates(row, s3_index)
    )

    s3_number = ids_from_indices(
        set(),
        number_candidates(row, s3_index)
    )

    exact_ids = s2_exact | s3_exact
    name_ids = s2_name | s3_name
    address_ids = s2_address | s3_address
    number_ids = s2_number | s3_number

    total_true += len(true_ids)

    for true_id in true_ids:

        rules = []

        if true_id in exact_ids:
            rules.append("exact")

        if true_id in name_ids:
            rules.append("name")

        if true_id in address_ids:
            rules.append("address")

        if true_id in number_ids:
            rules.append("number")

        # Individual rule counts
        if "exact" in rules:
            rule_counts["exact_name"] += 1

        if "name" in rules:
            rule_counts["name_token"] += 1

        if "address" in rules:
            rule_counts["address_token"] += 1

        if "number" in rules:
            rule_counts["number"] += 1

        # Combination counts
        if len(rules) == 1:

            if rules[0] == "name":
                rule_counts["name_only"] += 1

            elif rules[0] == "address":
                rule_counts["address_only"] += 1

            elif rules[0] == "number":
                rule_counts["number_only"] += 1

        if "name" in rules and "address" in rules:
            rule_counts["name_and_address"] += 1

        if "name" in rules and "number" in rules:
            rule_counts["name_and_number"] += 1

        if "address" in rules and "number" in rules:
            rule_counts["address_and_number"] += 1

        if len(rules) >= 2:
            rule_counts["multiple_rules"] += 1

        if len(rules) >= 1:
            rule_counts["caught_by_any_rule"] += 1

        else:
            rule_counts["missed_by_all_rules"] += 1


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 80)
print("V4 TRUE MATCH BLOCKING DIAGNOSTICS")
print("=" * 80)

print(
    f"\nTotal true matches: {total_true:,}"
)

print("\nINDIVIDUAL RULE COVERAGE")
print("-" * 60)

for key in [
    "exact_name",
    "name_token",
    "address_token",
    "number"
]:

    count = rule_counts[key]

    percentage = (
        count / total_true * 100
        if total_true
        else 0
    )

    print(
        f"{key:20} "
        f"{count:7,} "
        f"({percentage:6.2f}%)"
    )


print("\nCOMBINATION COVERAGE")
print("-" * 60)

for key in [
    "name_only",
    "address_only",
    "number_only",
    "name_and_address",
    "name_and_number",
    "address_and_number",
    "multiple_rules"
]:

    count = rule_counts[key]

    percentage = (
        count / total_true * 100
        if total_true
        else 0
    )

    print(
        f"{key:20} "
        f"{count:7,} "
        f"({percentage:6.2f}%)"
    )


print("\nOVERALL")
print("-" * 60)

count = rule_counts["caught_by_any_rule"]

print(
    f"Caught by any rule : "
    f"{count:,} "
    f"({count / total_true * 100:.2f}%)"
)

count = rule_counts["missed_by_all_rules"]

print(
    f"Missed by all rules: "
    f"{count:,} "
    f"({count / total_true * 100:.2f}%)"
)

print("\nDONE")