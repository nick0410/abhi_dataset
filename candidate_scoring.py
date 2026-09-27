import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher


BASE = "sample_dataset/dataset"

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

    text = re.sub(r"[^a-z0-9]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    value = normalize(text)
    return set(value.split()) if value else set()


def numbers(text):
    if pd.isna(text):
        return set()

    return set(re.findall(r"\d+", str(text)))


# =========================================================
# FEATURES
# =========================================================

def jaccard(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def similarity(a, b):
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def make_features(a, b):

    name_a = normalize(a["business_name"])
    name_b = normalize(b["business_name"])

    addr_a = normalize(a["business_address"])
    addr_b = normalize(b["business_address"])

    name_tokens_a = tokens(a["business_name"])
    name_tokens_b = tokens(b["business_name"])

    addr_tokens_a = tokens(a["business_address"])
    addr_tokens_b = tokens(b["business_address"])

    nums_a = numbers(a["business_address"])
    nums_b = numbers(b["business_address"])

    return {
        "name_exact": int(
            bool(name_a) and name_a == name_b
        ),

        "name_similarity": similarity(
            name_a,
            name_b
        ),

        "name_jaccard": jaccard(
            name_tokens_a,
            name_tokens_b
        ),

        "address_similarity": similarity(
            addr_a,
            addr_b
        ),

        "address_jaccard": jaccard(
            addr_tokens_a,
            addr_tokens_b
        ),

        "number_overlap": jaccard(
            nums_a,
            nums_b
        ),

        "country_match": int(
            str(a["country"]).lower()
            == str(b["country"]).lower()
        )
    }


# =========================================================
# LOAD BLOCKER
# =========================================================

exec(open("blocking_v5.py", encoding="utf-8").read().split(
    "# =========================================================\n# STATISTICS"
)[0])


# =========================================================
# GT LOOKUP
# =========================================================

gt_lookup = {
    row["source1_entity_id"]: row["matched_entity_ids"]
    for _, row in gt.iterrows()
}


# =========================================================
# SCORE CANDIDATES
# =========================================================

results = []

for _, row in s1.iterrows():

    _, _, _, _, s2_candidates = get_rule_candidates(
        row,
        s2_index
    )

    _, _, _, _, s3_candidates = get_rule_candidates(
        row,
        s3_index
    )

    candidate_ids = set()

    for idx in s2_candidates:
        candidate_ids.add(s2.iloc[idx]["entity_id"])

    for idx in s3_candidates:
        candidate_ids.add(s3.iloc[idx]["entity_id"])

    true_value = gt_lookup.get(
        row["entity_id"],
        ""
    )

    if pd.isna(true_value) or not str(true_value).strip():
        true_ids = set()
    else:
        true_ids = set(
            str(true_value).strip().split(",")
        )

    # S2 candidates
    for idx in s2_candidates:

        candidate = s2.iloc[idx]

        features = make_features(
            row,
            candidate
        )

        features["source1_entity_id"] = row["entity_id"]
        features["candidate_entity_id"] = candidate["entity_id"]
        features["label"] = int(
            candidate["entity_id"] in true_ids
        )

        results.append(features)

    # S3 candidates
    for idx in s3_candidates:

        candidate = s3.iloc[idx]

        features = make_features(
            row,
            candidate
        )

        features["source1_entity_id"] = row["entity_id"]
        features["candidate_entity_id"] = candidate["entity_id"]
        features["label"] = int(
            candidate["entity_id"] in true_ids
        )

        results.append(features)


# =========================================================
# SAVE
# =========================================================

df = pd.DataFrame(results)

df.to_csv(
    "candidate_features.csv",
    index=False
)

print("\nDONE")
print("Candidate pairs :", len(df))
print("True pairs      :", df["label"].sum())
print("False pairs     :", (df["label"] == 0).sum())
print("\nSaved: candidate_features.csv")