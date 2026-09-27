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
# SAME NORMALIZATION AS ORIGINAL BLOCKER
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
        .apply(tokens)
    )

    df["address_tokens"] = (
        df["business_address"]
        .apply(tokens)
    )

    df["address_numbers"] = (
        df["business_address"]
        .apply(numbers)
    )

    return df


s1 = prepare(s1)
s2 = prepare(s2)
s3 = prepare(s3)


# =========================================================
# ORIGINAL INDEX
# =========================================================

def build_indices(df):

    indices = {
        "exact_name": defaultdict(set),
        "name_token": defaultdict(set),
        "address_token": defaultdict(set),
        "number": defaultdict(set)
    }

    for idx, row in df.iterrows():

        # Exact name
        if row["name_norm"]:

            indices["exact_name"][
                row["name_norm"]
            ].add(idx)

        # Name tokens
        for token in row["name_tokens"]:

            if len(token) >= 3:

                indices["name_token"][
                    token
                ].add(idx)

        # Address tokens
        for token in row["address_tokens"]:

            if len(token) >= 4:

                indices["address_token"][
                    token
                ].add(idx)

        # Numbers
        for number in row["address_numbers"]:

            indices["number"][
                number
            ].add(idx)

    return indices


s2_index = build_indices(s2)
s3_index = build_indices(s3)


# =========================================================
# ORIGINAL BLOCKING
# =========================================================

def get_rule_candidates(row, index):

    exact = set()
    name = set()
    address = set()
    number = set()

    # -------------------------
    # Exact name
    # -------------------------

    if row["name_norm"]:

        exact = index["exact_name"].get(
            row["name_norm"],
            set()
        )

    # -------------------------
    # Name token
    # -------------------------

    for token in row["name_tokens"]:

        if len(token) >= 3:

            name.update(
                index["name_token"].get(
                    token,
                    set()
                )
            )

    # -------------------------
    # Address token
    # -------------------------

    for token in row["address_tokens"]:

        if len(token) >= 4:

            address.update(
                index["address_token"].get(
                    token,
                    set()
                )
            )

    # -------------------------
    # Number
    # -------------------------

    for number_value in row["address_numbers"]:

        number.update(
            index["number"].get(
                number_value,
                set()
            )
        )

    union = (
        exact
        | name
        | address
        | number
    )

    return (
        exact,
        name,
        address,
        number,
        union
    )


# =========================================================
# GET TRUE IDS
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
# STATISTICS
# =========================================================

stats = {
    "total_true": 0,

    "exact_only": 0,
    "name_only": 0,
    "address_only": 0,
    "number_only": 0,

    "exact_name_combo": 0,
    "name_address_combo": 0,
    "name_number_combo": 0,
    "address_number_combo": 0,

    "two_or_more_rules": 0,

    "caught": 0,
    "missed": 0
}


total_candidates = 0
max_candidates = 0


# =========================================================
# MAIN ANALYSIS
# =========================================================

# Create GT lookup by S1 entity ID
gt_lookup = {
    row["source1_entity_id"]: row["matched_entity_ids"]
    for _, row in gt.iterrows()
}


for _, row in s1.iterrows():

    value = gt_lookup.get(
        row["entity_id"],
        ""
    )

    if pd.isna(value) or not str(value).strip():
        true_ids = set()
    else:
        true_ids = set(
            str(value).strip().split(",")
        )

    # S2
    s2_exact, s2_name, s2_address, s2_number, s2_union = \
        get_rule_candidates(row, s2_index)

    # S3
    s3_exact, s3_name, s3_address, s3_number, s3_union = \
        get_rule_candidates(row, s3_index)

    # Convert indexes to IDs
    exact_ids = set(
        s2.iloc[list(s2_exact)]["entity_id"]
    ) | set(
        s3.iloc[list(s3_exact)]["entity_id"]
    )

    name_ids = set(
        s2.iloc[list(s2_name)]["entity_id"]
    ) | set(
        s3.iloc[list(s3_name)]["entity_id"]
    )

    address_ids = set(
        s2.iloc[list(s2_address)]["entity_id"]
    ) | set(
        s3.iloc[list(s3_address)]["entity_id"]
    )

    number_ids = set(
        s2.iloc[list(s2_number)]["entity_id"]
    ) | set(
        s3.iloc[list(s3_number)]["entity_id"]
    )

    candidate_ids = (
        exact_ids
        | name_ids
        | address_ids
        | number_ids
    )

    total_candidates += len(candidate_ids)

    max_candidates = max(
        max_candidates,
        len(candidate_ids)
    )

    # ----------------------------------------
    # Analyze EVERY true match
    # ----------------------------------------

    for true_id in true_ids:

        stats["total_true"] += 1

        rules = []

        if true_id in exact_ids:
            rules.append("exact")

        if true_id in name_ids:
            rules.append("name")

        if true_id in address_ids:
            rules.append("address")

        if true_id in number_ids:
            rules.append("number")

        if not rules:

            stats["missed"] += 1
            continue

        stats["caught"] += 1

        # One rule only
        if len(rules) == 1:

            if rules[0] == "exact":
                stats["exact_only"] += 1

            elif rules[0] == "name":
                stats["name_only"] += 1

            elif rules[0] == "address":
                stats["address_only"] += 1

            elif rules[0] == "number":
                stats["number_only"] += 1

        # Combinations
        if "exact" in rules and "name" in rules:
            stats["exact_name_combo"] += 1

        if "name" in rules and "address" in rules:
            stats["name_address_combo"] += 1

        if "name" in rules and "number" in rules:
            stats["name_number_combo"] += 1

        if "address" in rules and "number" in rules:
            stats["address_number_combo"] += 1

        if len(rules) >= 2:
            stats["two_or_more_rules"] += 1


# =========================================================
# REPORT
# =========================================================

print("\n")
print("=" * 85)
print("V5 ORIGINAL BLOCKER DIAGNOSTICS")
print("=" * 85)

print(
    f"\nTotal true matches : "
    f"{stats['total_true']:,}"
)

print(
    f"Caught             : "
    f"{stats['caught']:,}"
)

print(
    f"Missed             : "
    f"{stats['missed']:,}"
)

print(
    f"Recall             : "
    f"{stats['caught'] / stats['total_true'] * 100:.2f}%"
)


print("\n")
print("SINGLE-RULE COVERAGE")
print("-" * 65)

for key in [
    "exact_only",
    "name_only",
    "address_only",
    "number_only"
]:

    count = stats[key]

    pct = (
        count
        / stats["total_true"]
        * 100
    )

    print(
        f"{key:25} "
        f"{count:7,} "
        f"({pct:6.2f}%)"
    )


print("\n")
print("MULTI-RULE COVERAGE")
print("-" * 65)

for key in [
    "exact_name_combo",
    "name_address_combo",
    "name_number_combo",
    "address_number_combo",
    "two_or_more_rules"
]:

    count = stats[key]

    pct = (
        count
        / stats["total_true"]
        * 100
    )

    print(
        f"{key:25} "
        f"{count:7,} "
        f"({pct:6.2f}%)"
    )


print("\n")
print("CANDIDATE SIZE")
print("-" * 65)

print(
    f"Average candidates : "
    f"{total_candidates / len(s1):.2f}"
)

print(
    f"Maximum candidates : "
    f"{max_candidates:,}"
)

print("\nDONE")