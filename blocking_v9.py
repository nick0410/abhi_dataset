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

    x = "".join(result)
    return re.sub(r"\s+", " ", x).strip()


def tokens(x, min_len=3):
    return {
        t for t in normalize(x).split()
        if len(t) >= min_len
    }


def load_file(path, nrows=None):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        nrows=nrows
    ).fillna("")


print("Loading data...")

s1 = load_file(f"{BASE}/train_source1.tsv", SAMPLE_N)
s2 = load_file(f"{BASE}/train_source2.tsv")
s3 = load_file(f"{BASE}/train_source3.tsv")

gt = pd.read_csv(
    f"{BASE}/train_ground_truth.tsv",
    sep="\t",
    dtype=str
).fillna("")

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


# --------------------------------------------------
# Build S1 normalized fields
# --------------------------------------------------

print("Preparing S1...")

s1["name_norm"] = s1["business_name"].map(normalize)
s1["address_norm"] = s1["business_address"].map(normalize)

s1["name_tokens"] = s1["name_norm"].map(
    lambda x: [t for t in x.split() if len(t) >= 3]
)

s1["address_tokens"] = s1["address_norm"].map(
    lambda x: [t for t in x.split() if len(t) >= 4]
)


# --------------------------------------------------
# SQLite connections
# --------------------------------------------------

token_conn = sqlite3.connect(TOKEN_DB)
token_cur = token_conn.cursor()

addr_conn = sqlite3.connect(ADDRESS_DB)
addr_cur = addr_conn.cursor()


# --------------------------------------------------
# Source-aware token lookup
# --------------------------------------------------

def get_token_candidates(table, token, source):
    cur = token_conn.execute(
        f"""
        SELECT idx
        FROM {table}
        WHERE token = ?
          AND source = ?
        """,
        (token, source)
    )
    return [r[0] for r in cur.fetchall()]


def get_address_candidates(address, source):
    cur = addr_conn.execute(
        """
        SELECT idx
        FROM address_exact
        WHERE address = ?
          AND source = ?
        """,
        (address, source)
    )
    return [r[0] for r in cur.fetchall()]


# --------------------------------------------------
# Candidate generation
# --------------------------------------------------

def get_candidates(row, source):

    candidates = set()

    name_tokens = row["name_tokens"]
    address_tokens = row["address_tokens"]
    address_norm = row["address_norm"]

    # Exact normalized address
    if address_norm:
        for idx in get_address_candidates(address_norm, source):
            candidates.add(idx)

    # Name tokens
    for token in name_tokens:
        for idx in get_token_candidates(
            "name_token",
            token,
            source
        ):
            candidates.add(idx)

    # Address tokens
    for token in address_tokens:
        for idx in get_token_candidates(
            "address_token",
            token,
            source
        ):
            candidates.add(idx)

    return candidates


# --------------------------------------------------
# Ground truth
# --------------------------------------------------

gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

total_true = 0
found_true = 0
total_candidates = 0
max_candidates = 0

missed = []

start = time.time()

for i, row in s1.iterrows():

    if i % 500 == 0:
        print(
            f"Processed {i}/{len(s1)} | "
            f"Elapsed {time.time()-start:.1f}s"
        )

    all_candidates = set()

    s2_candidates = get_candidates(row, "S2")
    s3_candidates = get_candidates(row, "S3")

    all_candidates.update(
        ("S2", idx) for idx in s2_candidates
    )

    all_candidates.update(
        ("S3", idx) for idx in s3_candidates
    )

    total_candidates += len(all_candidates)
    max_candidates = max(
        max_candidates,
        len(all_candidates)
    )

    true_ids_str = gt_lookup.get(
        row["entity_id"],
        ""
    )

    true_ids = {
        x.strip()
        for x in str(true_ids_str).split(",")
        if x.strip()
    }

    total_true += len(true_ids)

    found_ids = set()

    for source, idx in all_candidates:

        if source == "S2":
            found_ids.add(
                s2.iloc[idx]["entity_id"]
            )
        else:
            found_ids.add(
                s3.iloc[idx]["entity_id"]
            )

    found = true_ids.intersection(found_ids)

    found_true += len(found)

    missing = true_ids - found_ids

    if missing:
        missed.append(
            (
                row["entity_id"],
                list(missing)
            )
        )


# --------------------------------------------------
# Results
# --------------------------------------------------

recall = (
    found_true / total_true
    if total_true else 0
)

avg_candidates = (
    total_candidates / len(s1)
    if len(s1) else 0
)

print("\n" + "=" * 50)
print("V9 BLOCKING RESULTS")
print("=" * 50)

print("S1:", len(s1))
print("True matches:", total_true)
print("Found:", found_true)
print("Missed:", total_true - found_true)
print("Blocking recall:", recall)
print("Average candidates:", avg_candidates)
print("Maximum candidates:", max_candidates)

print("\nFirst 20 missed:")

for x in missed[:20]:
    print(x)

token_conn.close()
addr_conn.close()