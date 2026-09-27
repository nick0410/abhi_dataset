import pandas as pd
import sqlite3
import re
import unicodedata
import time

BASE = "dataset/train"

# IMPORTANT: pehle sirf 1000 S1 test karenge
SAMPLE_N = 1000

TOKEN_DB = "output/v9_tokens.sqlite"
ADDRESS_DB = "output/address_exact.sqlite"


# ============================================================
# NORMALIZATION
# ============================================================

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


def get_tokens(x, min_len):
    return {
        t for t in normalize(x).split()
        if len(t) >= min_len
    }


# ============================================================
# LOAD DATA
# ============================================================

print("Loading S1...")

s1 = pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\t",
    dtype=str,
    nrows=SAMPLE_N
).fillna("")

print("Loading S2...")

s2 = pd.read_csv(
    f"{BASE}/train_source2.tsv",
    sep="\t",
    dtype=str
).fillna("")

print("Loading S3...")

s3 = pd.read_csv(
    f"{BASE}/train_source3.tsv",
    sep="\t",
    dtype=str
).fillna("")

print("Loading GT...")

gt = pd.read_csv(
    f"{BASE}/train_ground_truth.tsv",
    sep="\t",
    dtype=str
).fillna("")


# ============================================================
# PREPARE S1
# ============================================================

print("Preparing S1...")

s1_name_tokens = []
s1_address_tokens = []
s1_address_norm = []

for _, row in s1.iterrows():

    s1_name_tokens.append(
        get_tokens(row["business_name"], 3)
    )

    s1_address_tokens.append(
        get_tokens(row["business_address"], 4)
    )

    s1_address_norm.append(
        normalize(row["business_address"])
    )


# ============================================================
# BUILD FAST IN-MEMORY POSTING MAPS
#
# Instead of querying SQLite for every token,
# load only the relevant postings for tokens appearing
# in our 1000 S1 records.
# ============================================================

print("Opening token database...")

token_conn = sqlite3.connect(TOKEN_DB)

address_conn = sqlite3.connect(ADDRESS_DB)


# ============================================================
# Collect all tokens required by S1
# ============================================================

all_name_tokens = set()

all_address_tokens = set()

for x in s1_name_tokens:
    all_name_tokens.update(x)

for x in s1_address_tokens:
    all_address_tokens.update(x)

print("Unique S1 name tokens:", len(all_name_tokens))
print("Unique S1 address tokens:", len(all_address_tokens))


# ============================================================
# FAST BATCH TOKEN LOAD
# ============================================================

def load_postings(table, tokens, source):

    result = {}

    tokens = list(tokens)

    if not tokens:
        return result

    # SQLite variable limit safety
    chunk_size = 500

    for start in range(0, len(tokens), chunk_size):

        chunk = tokens[start:start + chunk_size]

        placeholders = ",".join(
            ["?"] * len(chunk)
        )

        query = f"""
            SELECT token, idx
            FROM {table}
            WHERE source = ?
            AND token IN ({placeholders})
        """

        rows = token_conn.execute(
            query,
            [source] + chunk
        ).fetchall()

        for token, idx in rows:

            if token not in result:
                result[token] = set()

            result[token].add(idx)

    return result


print("Loading S2 name postings...")
s2_name = load_postings(
    "name_token",
    all_name_tokens,
    "S2"
)

print("Loading S3 name postings...")
s3_name = load_postings(
    "name_token",
    all_name_tokens,
    "S3"
)

print("Loading S2 address postings...")
s2_address = load_postings(
    "address_token",
    all_address_tokens,
    "S2"
)

print("Loading S3 address postings...")
s3_address = load_postings(
    "address_token",
    all_address_tokens,
    "S3"
)


# ============================================================
# EXACT ADDRESS POSTINGS
# ============================================================

all_addresses = set(s1_address_norm)

print(
    "Unique S1 addresses:",
    len(all_addresses)
)


def load_exact_addresses(addresses, source):

    result = {}

    addresses = list(addresses)

    chunk_size = 500

    for start in range(
        0,
        len(addresses),
        chunk_size
    ):

        chunk = addresses[start:start + chunk_size]

        placeholders = ",".join(
            ["?"] * len(chunk)
        )

        query = f"""
            SELECT address, idx
            FROM address_exact
            WHERE source = ?
            AND address IN ({placeholders})
        """

        rows = address_conn.execute(
            query,
            [source] + chunk
        ).fetchall()

        for address, idx in rows:

            if address not in result:
                result[address] = set()

            result[address].add(idx)

    return result


print("Loading exact S2 addresses...")

s2_exact_address = load_exact_addresses(
    all_addresses,
    "S2"
)

print("Loading exact S3 addresses...")

s3_exact_address = load_exact_addresses(
    all_addresses,
    "S3"
)


# ============================================================
# GROUND TRUTH
# ============================================================

gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)


# ============================================================
# EVALUATION
# ============================================================

total_true = 0
found_true = 0

total_candidates = 0
max_candidates = 0

missed = []

start_time = time.time()


for i, row in s1.iterrows():

    if i % 100 == 0:

        elapsed = time.time() - start_time

        print(
            f"Processed {i}/{len(s1)} "
            f"| elapsed {elapsed:.1f}s"
        )

    name_tokens = s1_name_tokens[i]

    address_tokens = s1_address_tokens[i]

    address_norm = s1_address_norm[i]


    # --------------------------------------------------------
    # S2 candidates
    # --------------------------------------------------------

    c2 = set()

    # Name tokens
    for token in name_tokens:

        c2.update(
            s2_name.get(token, set())
        )

    # Address tokens
    for token in address_tokens:

        c2.update(
            s2_address.get(token, set())
        )

    # Exact address
    c2.update(
        s2_exact_address.get(
            address_norm,
            set()
        )
    )


    # --------------------------------------------------------
    # S3 candidates
    # --------------------------------------------------------

    c3 = set()

    # Name tokens
    for token in name_tokens:

        c3.update(
            s3_name.get(token, set())
        )

    # Address tokens
    for token in address_tokens:

        c3.update(
            s3_address.get(token, set())
        )

    # Exact address
    c3.update(
        s3_exact_address.get(
            address_norm,
            set()
        )
    )


    # --------------------------------------------------------
    # Convert indexes to entity IDs
    # --------------------------------------------------------

    candidate_ids = set()

    for idx in c2:

        candidate_ids.add(
            s2.iloc[idx]["entity_id"]
        )

    for idx in c3:

        candidate_ids.add(
            s3.iloc[idx]["entity_id"]
        )


    total_candidates += len(candidate_ids)

    max_candidates = max(
        max_candidates,
        len(candidate_ids)
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    true_ids = {
        x.strip()
        for x in str(
            gt_lookup.get(
                row["entity_id"],
                ""
            )
        ).split(",")
        if x.strip()
    }


    total_true += len(true_ids)


    found = true_ids.intersection(
        candidate_ids
    )

    found_true += len(found)


    missing = true_ids - candidate_ids

    if missing:

        missed.append(
            (
                row["entity_id"],
                list(missing)
            )
        )


# ============================================================
# RESULTS
# ============================================================

elapsed = time.time() - start_time

print()
print("=" * 70)
print("V10 FAST RESULTS")
print("=" * 70)

print("S1:", len(s1))

print("True matches:", total_true)

print("Found:", found_true)

print(
    "Missed:",
    total_true - found_true
)

print(
    "Recall:",
    found_true / total_true
    if total_true
    else 0
)

print(
    "Average candidates:",
    total_candidates / len(s1)
)

print(
    "Maximum candidates:",
    max_candidates
)

print(
    "Runtime:",
    elapsed,
    "seconds"
)

print()
print("First 30 missed:")

for item in missed[:30]:
    print(item)


token_conn.close()
address_conn.close()