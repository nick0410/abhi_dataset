import pandas as pd
import sqlite3
import re
import unicodedata
import time

BASE = "dataset/train"
SAMPLE_N = 10000

TOKEN_DB = "output/v9_tokens.sqlite"
ADDRESS_DB = "output/address_exact.sqlite"

def normalize(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = unicodedata.normalize("NFKC", x)

    result = []
    for c in x:
        if c.isalnum() or c.isspace():
            result.append(c)
        else:
            result.append(" ")

    return re.sub(r"\s+", " ", "".join(result)).strip()


def toks(x, min_len):
    return {
        t for t in normalize(x).split()
        if len(t) >= min_len
    }


print("Loading...")

s1 = pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\t",
    dtype=str,
    nrows=SAMPLE_N
).fillna("")

s2 = pd.read_csv(
    f"{BASE}/train_source2.tsv",
    sep="\t",
    dtype=str
).fillna("")

s3 = pd.read_csv(
    f"{BASE}/train_source3.tsv",
    sep="\t",
    dtype=str
).fillna("")

gt = pd.read_csv(
    f"{BASE}/train_ground_truth.tsv",
    sep="\t",
    dtype=str
).fillna("")

print("Loaded")


# ---------------------------------------------
# Prepare S1
# ---------------------------------------------

s1["name_norm"] = s1["business_name"].map(normalize)
s1["address_norm"] = s1["business_address"].map(normalize)

s1["name_tokens"] = s1["name_norm"].map(
    lambda x: toks(x, 3)
)

s1["address_tokens"] = s1["address_norm"].map(
    lambda x: toks(x, 4)
)


# ---------------------------------------------
# SQLite
# ---------------------------------------------

token_conn = sqlite3.connect(TOKEN_DB)
addr_conn = sqlite3.connect(ADDRESS_DB)


def token_candidates(table, token, source):

    cur = token_conn.execute(
        f"""
        SELECT idx
        FROM {table}
        WHERE token = ?
        AND source = ?
        """,
        (token, source)
    )

    return {x[0] for x in cur.fetchall()}


def address_candidates(address, source):

    cur = addr_conn.execute(
        """
        SELECT idx
        FROM address_exact
        WHERE address = ?
        AND source = ?
        """,
        (address, source)
    )

    return {x[0] for x in cur.fetchall()}


# ---------------------------------------------
# Candidate generation
# ---------------------------------------------

def candidates(row, source):

    result = set()

    # EXACT ADDRESS
    if row["address_norm"]:
        result.update(
            address_candidates(
                row["address_norm"],
                source
            )
        )

    # NAME TOKENS
    for t in row["name_tokens"]:
        result.update(
            token_candidates(
                "name_token",
                t,
                source
            )
        )

    # ADDRESS TOKENS
    for t in row["address_tokens"]:
        result.update(
            token_candidates(
                "address_token",
                t,
                source
            )
        )

    return result


# ---------------------------------------------
# GT
# ---------------------------------------------

gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)


# ---------------------------------------------
# Evaluation
# ---------------------------------------------

total_true = 0
found_true = 0
total_candidates = 0
max_candidates = 0

missed = []

start = time.time()


for i, row in s1.iterrows():

    if i % 500 == 0:
        print(
            f"{i}/{len(s1)} "
            f"elapsed={time.time()-start:.1f}s"
        )

    c2 = candidates(row, "S2")
    c3 = candidates(row, "S3")

    candidate_ids = set()

    candidate_ids.update(
        s2.iloc[idx]["entity_id"]
        for idx in c2
    )

    candidate_ids.update(
        s3.iloc[idx]["entity_id"]
        for idx in c3
    )

    total_candidates += len(candidate_ids)

    max_candidates = max(
        max_candidates,
        len(candidate_ids)
    )

    true_ids = {
        x.strip()
        for x in str(
            gt_lookup.get(row["entity_id"], "")
        ).split(",")
        if x.strip()
    }

    total_true += len(true_ids)

    found = true_ids & candidate_ids

    found_true += len(found)

    missing = true_ids - candidate_ids

    if missing:
        missed.append(
            (
                row["entity_id"],
                list(missing)
            )
        )


print("\n" + "=" * 60)
print("V10 BASELINE")
print("=" * 60)

print("True matches :", total_true)
print("Found        :", found_true)
print("Missed       :", total_true - found_true)

print(
    "Recall       :",
    found_true / total_true
)

print(
    "Avg candidates:",
    total_candidates / len(s1)
)

print(
    "Max candidates:",
    max_candidates
)

print("\nMissed:", len(missed))

for x in missed[:30]:
    print(x)

token_conn.close()
addr_conn.close()