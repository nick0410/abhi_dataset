import sqlite3
import pandas as pd
import re
import unicodedata

BASE = "dataset/train"

DBS = {
    "S2": "output/v6_full/S2.sqlite",
    "S3": "output/v6_full/S3.sqlite"
}


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


def tokens(x):
    return set(normalize(x).split())


def numbers(x):
    return set(re.findall(r"\d+", normalize(x)))


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

s1 = pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    f"{BASE}/train_source2.tsv",
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    f"{BASE}/train_source3.tsv",
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    f"{BASE}/train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

# Same first 10k validation used by V6
s1 = s1.head(10000)

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


# ============================================================
# LOOKUPS
# ============================================================

s2_lookup = s2.set_index("entity_id").to_dict("index")
s3_lookup = s3.set_index("entity_id").to_dict("index")

# idx -> entity_id
s2_idx_to_id = s2["entity_id"].to_dict()
s3_idx_to_id = s3["entity_id"].to_dict()

gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)


# ============================================================
# SQLITE CANDIDATE RETRIEVAL
# ============================================================

def get_candidates(conn, row):

    cand = set()

    cur = conn.cursor()

    name = normalize(row["business_name"])
    addr = normalize(row["business_address"])

    name_tokens = tokens(name)
    addr_tokens = tokens(addr)
    nums = numbers(addr)

    # --------------------------------------------------------
    # 1. EXACT NAME
    # --------------------------------------------------------

    if name:

        cur.execute(
            """
            SELECT idx
            FROM exact_name
            WHERE name = ?
            """,
            (name,)
        )

        cand.update(
            x[0]
            for x in cur.fetchall()
        )

    # --------------------------------------------------------
    # 2. NAME TOKENS
    # --------------------------------------------------------

    for token in name_tokens:

        if len(token) < 3:
            continue

        cur.execute(
            """
            SELECT idx
            FROM name_token
            WHERE token = ?
            """,
            (token,)
        )

        cand.update(
            x[0]
            for x in cur.fetchall()
        )

    # --------------------------------------------------------
    # 3. ADDRESS TOKENS
    # --------------------------------------------------------

    for token in addr_tokens:

        if len(token) < 4:
            continue

        cur.execute(
            """
            SELECT idx
            FROM address_token
            WHERE token = ?
            """,
            (token,)
        )

        cand.update(
            x[0]
            for x in cur.fetchall()
        )

    # --------------------------------------------------------
    # 4. NUMBERS
    # --------------------------------------------------------

    for num in nums:

        cur.execute(
            """
            SELECT idx
            FROM number_index
            WHERE number = ?
            """,
            (num,)
        )

        cand.update(
            x[0]
            for x in cur.fetchall()
        )

    # --------------------------------------------------------
    # 5. ADDRESS TOKEN PAIRS
    # --------------------------------------------------------

    addr_tokens_sorted = sorted(
        token
        for token in addr_tokens
        if len(token) >= 4
    )

    for i in range(len(addr_tokens_sorted)):

        for j in range(i + 1, len(addr_tokens_sorted)):

            token1 = addr_tokens_sorted[i]
            token2 = addr_tokens_sorted[j]

            cur.execute(
                """
                SELECT idx
                FROM address_pair
                WHERE token1 = ?
                  AND token2 = ?
                """,
                (token1, token2)
            )

            cand.update(
                x[0]
                for x in cur.fetchall()
            )

    return cand


# ============================================================
# VALIDATION
# ============================================================

missed = []

total_true = 0
total_found = 0

print("\nRunning diagnostic...\n")


for counter, (_, row) in enumerate(s1.iterrows()):

    if counter % 500 == 0:
        print(
            f"{counter}/{len(s1)} | "
            f"missed pairs: {len(missed)}"
        )

    source1_id = row["entity_id"]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    value = gt_lookup.get(source1_id, "")

    if pd.isna(value) or not value:
        continue

    true_ids = set(
        str(value).split(",")
    )

    true_ids.discard("")

    total_true += len(true_ids)

    # --------------------------------------------------------
    # Retrieve candidates from S2 + S3
    # --------------------------------------------------------

    found = set()

    for source in ["S2", "S3"]:

        conn = sqlite3.connect(
            DBS[source]
        )

        candidate_indexes = get_candidates(
            conn,
            row
        )

        conn.close()

        # idx -> entity_id
        if source == "S2":

            candidate_ids = {
                s2_idx_to_id[int(idx)]
                for idx in candidate_indexes
                if int(idx) in s2_idx_to_id
            }

        else:

            candidate_ids = {
                s3_idx_to_id[int(idx)]
                for idx in candidate_indexes
                if int(idx) in s3_idx_to_id
            }

        found.update(candidate_ids)

    # --------------------------------------------------------
    # Calculate misses
    # --------------------------------------------------------

    matched = true_ids.intersection(found)

    missing = true_ids - found

    total_found += len(matched)

    # --------------------------------------------------------
    # Inspect missed TRUE pairs
    # --------------------------------------------------------

    for true_id in missing:

        if true_id.startswith("S2-"):

            r = s2_lookup.get(true_id)

        else:

            r = s3_lookup.get(true_id)

        if r is None:
            continue

        name1 = tokens(
            row["business_name"]
        )

        name2 = tokens(
            r["business_name"]
        )

        addr1 = tokens(
            row["business_address"]
        )

        addr2 = tokens(
            r["business_address"]
        )

        num1 = numbers(
            row["business_address"]
        )

        num2 = numbers(
            r["business_address"]
        )

        missed.append(
            {
                "s1": source1_id,
                "true": true_id,

                "common_name":
                    sorted(name1 & name2),

                "common_addr":
                    sorted(addr1 & addr2),

                "common_num":
                    sorted(num1 & num2),

                "s1_name":
                    row["business_name"],

                "true_name":
                    r["business_name"],

                "s1_address":
                    row["business_address"],

                "true_address":
                    r["business_address"]
            }
        )

        # Only first 30
        if len(missed) >= 30:
            break

    if len(missed) >= 30:
        break


# ============================================================
# RESULT
# ============================================================

print("\n")
print("=" * 90)
print("V6 FULL DATA DIAGNOSTIC")
print("=" * 90)

print("True matches checked:", total_true)
print("Found:", total_found)

if total_true:
    print(
        "Recall:",
        f"{total_found / total_true:.6f}"
    )

print("Missed examples:", len(missed))


# ============================================================
# PRINT MISSED PAIRS
# ============================================================

print("\n")
print("=" * 90)
print("FIRST 30 MISSED TRUE PAIRS")
print("=" * 90)

for i, x in enumerate(missed, 1):

    print("\n" + "-" * 90)

    print(f"#{i}")

    print("S1:", x["s1"])
    print("TRUE:", x["true"])

    print(
        "Common name tokens:",
        x["common_name"]
    )

    print(
        "Common address tokens:",
        x["common_addr"]
    )

    print(
        "Common numbers:",
        x["common_num"]
    )

    print(
        "S1 name:",
        x["s1_name"]
    )

    print(
        "TRUE name:",
        x["true_name"]
    )

    print(
        "S1 address:",
        x["s1_address"]
    )

    print(
        "TRUE address:",
        x["true_address"]
    )

print("\nDone.")