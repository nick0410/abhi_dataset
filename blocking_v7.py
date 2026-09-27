import sqlite3
import pandas as pd
import re
import unicodedata


# ============================================================
# CONFIG
# ============================================================

BASE = "dataset/train"

S2_DB = "output/v6_full/S2.sqlite"
S3_DB = "output/v6_full/S3.sqlite"

ADDRESS_DB = "output/address_exact.sqlite"

N = 10000

MAX_POSTINGS = 5000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):

    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = unicodedata.normalize("NFKC", x)

    out = []

    for c in x:

        if c.isalnum() or c.isspace():
            out.append(c)

        else:
            out.append(" ")

    return re.sub(
        r"\s+",
        " ",
        "".join(out)
    ).strip()


def tokens(x):
    return set(
        normalize(x).split()
    )


def numbers(x):
    return set(
        re.findall(
            r"\d+",
            normalize(x)
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

s1 = pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\t",
    dtype=str
).head(N)

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

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))


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
# INDEX -> ENTITY ID
# ============================================================

s2_idx_to_id = s2[
    "entity_id"
].to_dict()

s3_idx_to_id = s3[
    "entity_id"
].to_dict()


# ============================================================
# CANDIDATE FUNCTION
# ============================================================

def get_candidates(
    conn,
    address_conn,
    row
):

    cur = conn.cursor()

    candidates = set()

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    name = normalize(
        row["business_name"]
    )

    address = normalize(
        row["business_address"]
    )

    name_tokens = [
        x
        for x in tokens(name)
        if len(x) >= 3
    ]

    address_tokens = [
        x
        for x in tokens(address)
        if len(x) >= 4
    ]

    nums = list(
        numbers(address)
    )


    # ========================================================
    # RULE 0: EXACT NORMALIZED ADDRESS
    # ========================================================

    if address:

        address_cur = address_conn.cursor()

        address_cur.execute(
            """
            SELECT source, idx
            FROM address_exact
            WHERE address = ?
            """,
            (address,)
        )

        rows = address_cur.fetchall()

        for source, idx in rows:

            # Since this function is called separately
            # for S2/S3, only return matching source.
            if source == "S2" or source == "S3":

                candidates.add(
                    int(idx)
                )


    # ========================================================
    # RULE 1: EXACT NAME
    # ========================================================

    if name:

        cur.execute(
            """
            SELECT idx
            FROM exact_name
            WHERE name = ?
            """,
            (name,)
        )

        candidates.update(
            int(x[0])
            for x in cur.fetchall()
        )


    # ========================================================
    # RULE 2: RAREST NAME TOKENS
    # ========================================================

    name_freq = []

    for token in name_tokens:

        cur.execute(
            """
            SELECT COUNT(*)
            FROM name_token
            WHERE token = ?
            """,
            (token,)
        )

        freq = cur.fetchone()[0]

        if freq > 0:

            name_freq.append(
                (freq, token)
            )

    name_freq.sort()


    # Top 3 rarest name tokens
    for freq, token in name_freq[:3]:

        if freq > MAX_POSTINGS:
            continue

        cur.execute(
            """
            SELECT idx
            FROM name_token
            WHERE token = ?
            """,
            (token,)
        )

        candidates.update(
            int(x[0])
            for x in cur.fetchall()
        )


    # ========================================================
    # RULE 3: RAREST ADDRESS TOKENS
    # ========================================================

    addr_freq = []

    for token in address_tokens:

        cur.execute(
            """
            SELECT COUNT(*)
            FROM address_token
            WHERE token = ?
            """,
            (token,)
        )

        freq = cur.fetchone()[0]

        if freq > 0:

            addr_freq.append(
                (freq, token)
            )

    addr_freq.sort()


    # Top 4 rarest address tokens
    for freq, token in addr_freq[:4]:

        if freq > MAX_POSTINGS:
            continue

        cur.execute(
            """
            SELECT idx
            FROM address_token
            WHERE token = ?
            """,
            (token,)
        )

        candidates.update(
            int(x[0])
            for x in cur.fetchall()
        )


    # ========================================================
    # RULE 4: NUMBERS
    # ========================================================

    for num in nums:

        cur.execute(
            """
            SELECT COUNT(*)
            FROM number_index
            WHERE number = ?
            """,
            (num,)
        )

        freq = cur.fetchone()[0]

        if freq == 0:
            continue

        if freq > MAX_POSTINGS:
            continue

        cur.execute(
            """
            SELECT idx
            FROM number_index
            WHERE number = ?
            """,
            (num,)
        )

        candidates.update(
            int(x[0])
            for x in cur.fetchall()
        )


    # ========================================================
    # RULE 5: ADDRESS TOKEN PAIRS
    # ========================================================

    rare_addr = [
        token
        for _, token in addr_freq[:6]
    ]

    for i in range(
        len(rare_addr)
    ):

        for j in range(
            i + 1,
            len(rare_addr)
        ):

            a = rare_addr[i]
            b = rare_addr[j]

            # SQLite index stores token1/token2
            # in sorted order.
            if a > b:
                a, b = b, a

            cur.execute(
                """
                SELECT idx
                FROM address_pair
                WHERE token1 = ?
                  AND token2 = ?
                """,
                (a, b)
            )

            candidates.update(
                int(x[0])
                for x in cur.fetchall()
            )


    return candidates


# ============================================================
# VALIDATION
# ============================================================

print("\nTesting V8 blocker...\n")

total_candidates = 0
total_true = 0
total_found = 0

missed_examples = []


# Open once
conn_s2 = sqlite3.connect(
    S2_DB
)

conn_s3 = sqlite3.connect(
    S3_DB
)

address_conn = sqlite3.connect(
    ADDRESS_DB
)


# ============================================================
# LOOP
# ============================================================

for counter, (_, row) in enumerate(
    s1.iterrows()
):

    if counter % 100 == 0:

        recall = (
            total_found / total_true
            if total_true
            else 0
        )

        avg = (
            total_candidates /
            (counter + 1)
        )

        print(
            f"{counter}/{len(s1)} | "
            f"avg candidates: {avg:.2f} | "
            f"recall: {recall:.4f}"
        )


    # --------------------------------------------------------
    # TRUE MATCHES
    # --------------------------------------------------------

    value = gt_lookup.get(
        row["entity_id"],
        ""
    )

    if pd.isna(value) or not value:
        continue

    true_ids = set(
        str(value).split(",")
    )

    true_ids.discard("")

    total_true += len(
        true_ids
    )


    # --------------------------------------------------------
    # S2 CANDIDATES
    # --------------------------------------------------------

    c2 = get_candidates(
        conn_s2,
        address_conn,
        row
    )


    # --------------------------------------------------------
    # S3 CANDIDATES
    # --------------------------------------------------------

    c3 = get_candidates(
        conn_s3,
        address_conn,
        row
    )


    total_candidates += (
        len(c2) +
        len(c3)
    )


    # --------------------------------------------------------
    # CONVERT INDEX -> ENTITY ID
    # --------------------------------------------------------

    found = set()


    for idx in c2:

        if idx in s2_idx_to_id:

            found.add(
                s2_idx_to_id[idx]
            )


    for idx in c3:

        if idx in s3_idx_to_id:

            found.add(
                s3_idx_to_id[idx]
            )


    # --------------------------------------------------------
    # MATCH
    # --------------------------------------------------------

    matched = (
        true_ids &
        found
    )

    total_found += len(
        matched
    )


    # --------------------------------------------------------
    # MISSED
    # --------------------------------------------------------

    missing = (
        true_ids -
        found
    )

    for true_id in missing:

        if len(missed_examples) < 20:

            missed_examples.append(
                (
                    row["entity_id"],
                    true_id
                )
            )


# ============================================================
# CLOSE
# ============================================================

conn_s2.close()
conn_s3.close()
address_conn.close()


# ============================================================
# RESULT
# ============================================================

print("\n")

print("=" * 70)
print("V8 BLOCKING RESULT")
print("=" * 70)

print(
    "True matches:",
    total_true
)

print(
    "Found:",
    total_found
)

print(
    "Missed:",
    total_true - total_found
)

print(
    "Blocking recall:",
    f"{total_found / total_true:.6f}"
)

print(
    "Average candidates:",
    f"{total_candidates / len(s1):.2f}"
)

print(
    "Maximum candidates:",
    "calculation skipped"
)


# ============================================================
# MISSED PAIRS
# ============================================================

print("\nFirst missed pairs:")

for s1_id, true_id in missed_examples:

    print(
        s1_id,
        "->",
        true_id
    )

print("\nDone.")