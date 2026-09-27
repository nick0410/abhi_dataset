import pandas as pd
import re
import unicodedata
from collections import defaultdict


# ============================================================
# LOAD SAMPLE DATA
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

gt = pd.read_csv(
    f"{BASE}/train/train_ground_truth.tsv",
    sep="\t"
)


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
# PREPARE
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

        if row["name_norm"]:
            index["name"][
                row["name_norm"]
            ].add(idx)

        for token in row["name_tokens"]:

            if len(token) >= 3:
                index["name_token"][token].add(idx)

        for token in row["address_tokens"]:

            if len(token) >= 4:
                index["address_token"][token].add(idx)

        for number in row["address_numbers"]:

            index["number"][number].add(idx)

    return index


s2_index = build_index(s2)
s3_index = build_index(s3)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(row, index):

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
# CREATE ENTITY ID LOOKUPS
# ============================================================

s2_ids = s2["entity_id"].tolist()
s3_ids = s3["entity_id"].tolist()


# ============================================================
# EVALUATE BLOCKING RECALL
# ============================================================

total_true = 0
total_found = 0

entities_with_missed_matches = []

candidate_counts = []


for _, row in s1.iterrows():

    s2_candidates = generate_candidates(
        row,
        s2_index
    )

    s3_candidates = generate_candidates(
        row,
        s3_index
    )

    # Convert dataframe indexes -> entity IDs

    candidate_ids = set()

    for idx in s2_candidates:
        candidate_ids.add(s2_ids[idx])

    for idx in s3_candidates:
        candidate_ids.add(s3_ids[idx])

    candidate_counts.append(len(candidate_ids))


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_row = gt[
        gt["source1_entity_id"] == row["entity_id"]
    ]

    if len(gt_row) == 0:
        continue

    value = gt_row.iloc[0]["matched_entity_ids"]

    if pd.isna(value) or not str(value).strip():
        continue

    true_ids = set(
        str(value).split(",")
    )

    found = true_ids.intersection(
        candidate_ids
    )

    total_true += len(true_ids)
    total_found += len(found)

    if len(found) < len(true_ids):

        entities_with_missed_matches.append({
            "source1": row["entity_id"],
            "true": true_ids,
            "found": found,
            "missed": true_ids - found,
            "candidate_count": len(candidate_ids)
        })


# ============================================================
# RESULTS
# ============================================================

recall = (
    total_found / total_true
    if total_true > 0
    else 0
)

print("\n" + "=" * 80)
print("BLOCKING RECALL")
print("=" * 80)

print("Total true matches :", total_true)
print("True matches found :", total_found)
print("Missed matches     :", total_true - total_found)

print(
    "Blocking recall    :",
    round(recall * 100, 2),
    "%"
)

print(
    "Average candidates :",
    round(sum(candidate_counts) / len(candidate_counts), 2)
)

print(
    "Maximum candidates:",
    max(candidate_counts)
)

print(
    "Entities with misses:",
    len(entities_with_missed_matches)
)


# ============================================================
# SHOW MISSED MATCHES
# ============================================================

print("\n" + "=" * 80)
print("FIRST 10 MISSED CASES")
print("=" * 80)

for item in entities_with_missed_matches[:10]:

    print("\nS1:", item["source1"])

    print("True matches:")
    print(item["true"])

    print("Found:")
    print(item["found"])

    print("MISSED:")
    print(item["missed"])

    print(
        "Candidate count:",
        item["candidate_count"]
    )

print("\nDONE")