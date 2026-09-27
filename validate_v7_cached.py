import sqlite3
import pickle
import pandas as pd
import re
import unicodedata
import time

BASE = "dataset/train"
CACHE = "output/v7_cache"
N = 10_000

MAX_NAME_FREQ = 100
MAX_ADDRESS_FREQ = 150
MAX_NUMBER_FREQ = 300
MAX_PAIR_TOKEN_FREQ = 500


def normalize(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value).lower()
    )

    result = []

    for c in text:
        if c.isalnum() or c.isspace():
            result.append(c)
        else:
            result.append(" ")

    return re.sub(r"\s+", " ", "".join(result)).strip()


def tokenize(text):
    return set(text.split()) if text else set()


def numbers(text):
    return set(re.findall(r"\d+", text)) if text else set()


print("Loading frequencies...")

with open(
    f"{CACHE}/frequencies.pkl",
    "rb"
) as f:
    freq = pickle.load(f)

name_freq = freq["name_freq"]
address_freq = freq["address_freq"]
number_freq = freq["number_freq"]

print("Loading S1...")

s1 = pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\t"
).iloc[:N].copy()

print("Loading S2...")

s2 = pd.read_csv(
    f"{BASE}/train_source2.tsv",
    sep="\t"
)

print("Loading S3...")

s3 = pd.read_csv(
    f"{BASE}/train_source3.tsv",
    sep="\t"
)

print("Loading GT...")

gt = pd.read_csv(
    f"{BASE}/train_ground_truth.tsv",
    sep="\t"
)

print("Loading S2 index...")

with open(
    f"{CACHE}/S2_index.pkl",
    "rb"
) as f:
    s2_index = pickle.load(f)

print("S2 index loaded.")

print("Opening S3 SQLite...")

conn = sqlite3.connect(
    f"{CACHE}/S3_index.sqlite"
)

cur = conn.cursor()

# ------------------------------------------------------------
# Prepare S1
# ------------------------------------------------------------

print("Preparing S1...")

s1["name_norm"] = s1["business_name"].map(normalize)
s1["address_norm"] = s1["business_address"].map(normalize)

s1["name_tokens"] = s1["name_norm"].map(tokenize)
s1["address_tokens"] = s1["address_norm"].map(tokenize)
s1["numbers"] = s1["address_norm"].map(numbers)

# Entity IDs
s2_ids = s2["entity_id"].astype(str).tolist()
s3_ids = s3["entity_id"].astype(str).tolist()

# Ground truth
gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)


def s2_candidates(row):

    candidates = set()

    # Exact name
    if row.name_norm:

        candidates.update(
            s2_index["exact_name"].get(
                row.name_norm,
                []
            )
        )

    # Name tokens
    for token in row.name_tokens:

        if (
            len(token) >= 3
            and name_freq[token] <= MAX_NAME_FREQ
        ):

            candidates.update(
                s2_index["name_token"].get(
                    token,
                    []
                )
            )

    # Address tokens
    for token in row.address_tokens:

        if (
            len(token) >= 4
            and address_freq[token] <= MAX_ADDRESS_FREQ
        ):

            candidates.update(
                s2_index["address_token"].get(
                    token,
                    []
                )
            )

    # Numbers
    for number in row.numbers:

        if (
            number_freq[number] <= MAX_NUMBER_FREQ
        ):

            candidates.update(
                s2_index["number"].get(
                    number,
                    []
                )
            )

    # Address pairs
    pair_tokens = sorted([
        token
        for token in row.address_tokens
        if (
            len(token) >= 4
            and address_freq[token]
            <= MAX_PAIR_TOKEN_FREQ
        )
    ])

    for i in range(len(pair_tokens)):

        for j in range(i + 1, len(pair_tokens)):

            pair = (
                pair_tokens[i],
                pair_tokens[j]
            )

            candidates.update(
                s2_index["address_pair"].get(
                    pair,
                    []
                )
            )

    return candidates


def s3_candidates(row):

    candidates = set()

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    if row.name_norm:

        cur.execute(
            "SELECT idx FROM exact_name WHERE name=?",
            (row.name_norm,)
        )

        for (idx,) in cur.fetchall():
            candidates.add(idx)

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    for token in row.address_tokens:

        if (
            len(token) >= 4
            and address_freq[token]
            <= MAX_ADDRESS_FREQ
        ):

            cur.execute(
                """
                SELECT idx
                FROM address_token
                WHERE token=?
                """,
                (token,)
            )

            for (idx,) in cur.fetchall():
                candidates.add(idx)

    # --------------------------------------------------------
    # Numbers
    # --------------------------------------------------------

    for number in row.numbers:

        if (
            number_freq[number]
            <= MAX_NUMBER_FREQ
        ):

            cur.execute(
                """
                SELECT idx
                FROM number_index
                WHERE number=?
                """,
                (number,)
            )

            for (idx,) in cur.fetchall():
                candidates.add(idx)

    # --------------------------------------------------------
    # Address pairs
    # --------------------------------------------------------

    pair_tokens = sorted([
        token
        for token in row.address_tokens
        if (
            len(token) >= 4
            and address_freq[token]
            <= MAX_PAIR_TOKEN_FREQ
        )
    ])

    for i in range(len(pair_tokens)):

        for j in range(i + 1, len(pair_tokens)):

            cur.execute(
                """
                SELECT idx
                FROM address_pair
                WHERE token1=? AND token2=?
                """,
                (
                    pair_tokens[i],
                    pair_tokens[j]
                )
            )

            for (idx,) in cur.fetchall():
                candidates.add(idx)

    return candidates


# ============================================================
# VALIDATION
# ============================================================

print("\nValidating first 10,000 S1...\n")

start = time.time()

total_true = 0
total_found = 0
total_candidates = 0
max_candidates = 0

missed_examples = []

for count, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    c2 = s2_candidates(row)
    c3 = s3_candidates(row)

    total_candidates += (
        len(c2) + len(c3)
    )

    max_candidates = max(
        max_candidates,
        len(c2) + len(c3)
    )

    candidate_ids = set()

    for idx in c2:
        candidate_ids.add(
            s2_ids[idx]
        )

    for idx in c3:
        candidate_ids.add(
            s3_ids[idx]
        )

    truth = gt_lookup.get(
        row.entity_id,
        ""
    )

    true_ids = set()

    if (
        pd.notna(truth)
        and str(truth).strip()
    ):

        true_ids = {
            x.strip()
            for x in str(truth).split(",")
            if x.strip()
        }

    found = true_ids & candidate_ids

    missed = true_ids - candidate_ids

    total_true += len(true_ids)
    total_found += len(found)

    if missed and len(missed_examples) < 20:

        missed_examples.append(
            (
                row.entity_id,
                missed
            )
        )

    if count % 500 == 0:

        elapsed = time.time() - start

        recall = (
            total_found /
            max(total_true, 1)
        )

        avg = (
            total_candidates /
            count
        )

        print(
            f"{count:,}/{N:,} | "
            f"Recall={recall:.6f} | "
            f"AvgCandidates={avg:.2f} | "
            f"Time={elapsed:.1f}s"
        )


# ============================================================
# RESULT
# ============================================================

recall = (
    total_found /
    max(total_true, 1)
)

avg_candidates = (
    total_candidates /
    N
)

elapsed = time.time() - start

print("\n" + "=" * 65)
print("V7 CACHED VALIDATION RESULT")
print("=" * 65)

print(f"S1 tested:          {N:,}")
print(f"True matches:       {total_true:,}")
print(f"Found:              {total_found:,}")
print(f"Missed:             {total_true-total_found:,}")
print(f"Blocking recall:    {recall:.6f}")
print(f"Average candidates: {avg_candidates:.2f}")
print(f"Maximum candidates: {max_candidates:,}")
print(f"Validation time:    {elapsed:.1f} sec")

print("=" * 65)

if missed_examples:

    print("\nMISSED EXAMPLES:")

    for entity_id, missed in missed_examples:

        print(
            entity_id,
            "->",
            sorted(missed)
        )

else:

    print("\n🔥 ZERO MISSED MATCHES 🔥")

conn.close()