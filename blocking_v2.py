import pandas as pd
import re
import unicodedata
from collections import defaultdict, Counter

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
# UNICODE-AWARE NORMALIZATION
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

    text = "".join(result)

    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    value = normalize(text)

    if not value:
        return set()

    return set(value.split())


def numbers(text):
    if pd.isna(text):
        return set()

    return set(re.findall(r"\d+", str(text)))


# =========================================================
# PREPARE DATA
# =========================================================

def prepare(df):
    df = df.copy()

    df["name_norm"] = df["business_name"].apply(normalize)

    df["name_tokens"] = df["business_name"].apply(tokens)

    df["address_tokens"] = df["business_address"].apply(tokens)

    df["address_numbers"] = df["business_address"].apply(numbers)

    return df


s1 = prepare(s1)
s2 = prepare(s2)
s3 = prepare(s3)


# =========================================================
# BUILD TOKEN FREQUENCY
# =========================================================

def token_frequency(df):

    counter = Counter()

    for token_set in df["name_tokens"]:

        for token in token_set:

            if len(token) >= 3:
                counter[token] += 1

    return counter


s2_name_freq = token_frequency(s2)
s3_name_freq = token_frequency(s3)


# =========================================================
# BUILD BLOCKING INDICES
# =========================================================

def build_indices(df, token_freq, max_token_frequency):

    indices = {
        "exact_name": defaultdict(set),
        "rare_name_token": defaultdict(set),
        "address_token": defaultdict(set),
        "number": defaultdict(set)
    }

    for idx, row in df.iterrows():

        # -----------------------------
        # Exact name
        # -----------------------------

        if row["name_norm"]:
            indices["exact_name"][
                row["name_norm"]
            ].add(idx)

        # -----------------------------
        # Rare name tokens
        # -----------------------------

        for token in row["name_tokens"]:

            if len(token) < 3:
                continue

            frequency = token_freq.get(token, 0)

            if frequency <= max_token_frequency:
                indices["rare_name_token"][
                    token
                ].add(idx)

        # -----------------------------
        # Address tokens
        # -----------------------------

        for token in row["address_tokens"]:

            if len(token) >= 4:
                indices["address_token"][
                    token
                ].add(idx)

        # -----------------------------
        # Address numbers
        # -----------------------------

        for number in row["address_numbers"]:

            indices["number"][
                number
            ].add(idx)

    return indices


# =========================================================
# CANDIDATE GENERATION
# =========================================================

def get_candidates(row, index):

    candidates = set()

    # Exact name
    if row["name_norm"]:

        candidates.update(
            index["exact_name"].get(
                row["name_norm"],
                set()
            )
        )

    # Rare name tokens
    for token in row["name_tokens"]:

        if len(token) >= 3:

            candidates.update(
                index["rare_name_token"].get(
                    token,
                    set()
                )
            )

    # Address tokens
    for token in row["address_tokens"]:

        if len(token) >= 4:

            candidates.update(
                index["address_token"].get(
                    token,
                    set()
                )
            )

    # Address numbers
    for number in row["address_numbers"]:

        candidates.update(
            index["number"].get(
                number,
                set()
            )
        )

    return candidates


# =========================================================
# TRUE MATCH IDs
# =========================================================

def get_true_ids(row):

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        return set()

    return set(
        str(value).split(",")
    )


# =========================================================
# EVALUATE ONE THRESHOLD
# =========================================================

def evaluate_threshold(max_frequency):

    print("\n" + "=" * 80)
    print(
        f"TESTING NAME TOKEN FREQUENCY <= {max_frequency}"
    )
    print("=" * 80)

    s2_index = build_indices(
        s2,
        s2_name_freq,
        max_frequency
    )

    s3_index = build_indices(
        s3,
        s3_name_freq,
        max_frequency
    )

    total_true = 0
    found_true = 0

    total_candidates = 0
    max_candidates = 0

    missed_entities = 0

    for i, (_, row) in enumerate(s1.iterrows()):

        candidates_s2 = get_candidates(
            row,
            s2_index
        )

        candidates_s3 = get_candidates(
            row,
            s3_index
        )

        candidates = (
            len(candidates_s2)
            + len(candidates_s3)
        )

        total_candidates += candidates

        max_candidates = max(
            max_candidates,
            candidates
        )

        # Ground truth
        gt_row = gt.iloc[i]

        true_ids = get_true_ids(gt_row)

        total_true += len(true_ids)

        # Convert candidate indices -> entity IDs
        candidate_ids = set()

        for idx in candidates_s2:
            candidate_ids.add(
                s2.iloc[idx]["entity_id"]
            )

        for idx in candidates_s3:
            candidate_ids.add(
                s3.iloc[idx]["entity_id"]
            )

        found = true_ids.intersection(
            candidate_ids
        )

        found_true += len(found)

        if len(found) < len(true_ids):
            missed_entities += 1

    recall = (
        found_true / total_true
        if total_true > 0
        else 0
    )

    avg_candidates = (
        total_candidates / len(s1)
    )

    print(
        f"Total true matches : {total_true:,}"
    )

    print(
        f"True matches found : {found_true:,}"
    )

    print(
        f"Missed matches     : "
        f"{total_true - found_true:,}"
    )

    print(
        f"Blocking recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Average candidates : "
        f"{avg_candidates:.2f}"
    )

    print(
        f"Maximum candidates: "
        f"{max_candidates:,}"
    )

    print(
        f"Entities with misses: "
        f"{missed_entities:,}"
    )

    return {
        "frequency": max_frequency,
        "recall": recall * 100,
        "avg_candidates": avg_candidates,
        "max_candidates": max_candidates
    }


# =========================================================
# RUN EXPERIMENT
# =========================================================

results = []

thresholds = [
    1,
    2,
    5,
    10,
    25,
    50
]

for threshold in thresholds:

    result = evaluate_threshold(
        threshold
    )

    results.append(result)


# =========================================================
# FINAL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

print("\nDONE")