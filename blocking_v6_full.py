import sqlite3
import pandas as pd
import re
import unicodedata
import os
import time
from collections import Counter

BASE = "dataset/train"
CACHE = "output/v6_full"

os.makedirs(CACHE, exist_ok=True)

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

    for char in text:
        if char.isalnum() or char.isspace():
            result.append(char)
        else:
            result.append(" ")

    return re.sub(
        r"\s+",
        " ",
        "".join(result)
    ).strip()


def tokenize(text):
    return set(text.split()) if text else set()


def extract_numbers(text):
    return set(
        re.findall(r"\d+", text)
    )


# ============================================================
# LOAD
# ============================================================

print("Loading datasets...")

s1 = pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\t"
).iloc[:N].copy()

s2 = pd.read_csv(
    f"{BASE}/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    f"{BASE}/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    f"{BASE}/train_ground_truth.tsv",
    sep="\t"
)

print(
    f"S1: {len(s1):,}"
)

print(
    f"S2: {len(s2):,}"
)

print(
    f"S3: {len(s3):,}"
)


# ============================================================
# FREQUENCIES
# ============================================================

freq_file = f"{CACHE}/frequencies.pkl"

if os.path.exists(freq_file):

    print("\nLoading cached frequencies...")

    import pickle

    with open(freq_file, "rb") as f:
        freq = pickle.load(f)

    name_freq = freq["name"]
    address_freq = freq["address"]
    number_freq = freq["number"]

else:

    print("\nCalculating frequencies...")

    name_freq = Counter()
    address_freq = Counter()
    number_freq = Counter()

    for source_name, df in [
        ("S2", s2),
        ("S3", s3)
    ]:

        print(
            f"Processing {source_name}..."
        )

        for row in df.itertuples(index=False):

            name = normalize(
                row.business_name
            )

            address = normalize(
                row.business_address
            )

            for token in tokenize(name):

                if len(token) >= 3:
                    name_freq[token] += 1

            for token in tokenize(address):

                if len(token) >= 4:
                    address_freq[token] += 1

            for number in extract_numbers(address):

                number_freq[number] += 1

    freq = {
        "name": name_freq,
        "address": address_freq,
        "number": number_freq
    }

    import pickle

    with open(
        freq_file,
        "wb"
    ) as f:
        pickle.dump(
            freq,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print("Frequencies saved.")


# ============================================================
# SQLITE INDEX BUILDER
# ============================================================

def build_sqlite(
    df,
    source_name
):

    db_path = (
        f"{CACHE}/{source_name}.sqlite"
    )

    if os.path.exists(db_path):

        print(
            f"\n{source_name} SQLite already exists."
        )

        return sqlite3.connect(
            db_path
        )

    print(
        f"\nBuilding {source_name} SQLite..."
    )

    conn = sqlite3.connect(
        db_path
    )

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE exact_name (
            name TEXT,
            idx INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE name_token (
            token TEXT,
            idx INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE address_token (
            token TEXT,
            idx INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE number_index (
            number TEXT,
            idx INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE address_pair (
            token1 TEXT,
            token2 TEXT,
            idx INTEGER
        )
    """)

    conn.commit()

    BATCH = 50_000

    exact_batch = []
    name_batch = []
    address_batch = []
    number_batch = []
    pair_batch = []

    total = len(df)

    for idx, row in enumerate(
        df.itertuples(index=False)
    ):

        name = normalize(
            row.business_name
        )

        address = normalize(
            row.business_address
        )

        name_tokens = tokenize(name)
        address_tokens = tokenize(address)
        numbers = extract_numbers(address)

        # ----------------------------------------------------
        # Exact name
        # ----------------------------------------------------

        if name:

            exact_batch.append(
                (name, idx)
            )

        # ----------------------------------------------------
        # Name tokens
        # ----------------------------------------------------

        for token in name_tokens:

            if (
                len(token) >= 3
                and name_freq[token]
                <= MAX_NAME_FREQ
            ):

                name_batch.append(
                    (token, idx)
                )

        # ----------------------------------------------------
        # Address tokens + pair tokens
        # ----------------------------------------------------

        pair_tokens = []

        for token in address_tokens:

            if (
                len(token) >= 4
                and address_freq[token]
                <= MAX_ADDRESS_FREQ
            ):

                address_batch.append(
                    (token, idx)
                )

            if (
                len(token) >= 4
                and address_freq[token]
                <= MAX_PAIR_TOKEN_FREQ
            ):

                pair_tokens.append(token)

        # ----------------------------------------------------
        # Numbers
        # ----------------------------------------------------

        for number in numbers:

            if (
                number_freq[number]
                <= MAX_NUMBER_FREQ
            ):

                number_batch.append(
                    (number, idx)
                )

        # ----------------------------------------------------
        # Address pairs
        # ----------------------------------------------------

        pair_tokens = sorted(
            set(pair_tokens)
        )

        for i in range(
            len(pair_tokens)
        ):

            for j in range(
                i + 1,
                len(pair_tokens)
            ):

                pair_batch.append(
                    (
                        pair_tokens[i],
                        pair_tokens[j],
                        idx
                    )
                )

        # ----------------------------------------------------
        # Flush
        # ----------------------------------------------------

        if len(exact_batch) >= BATCH:

            cur.executemany(
                """
                INSERT INTO exact_name
                VALUES (?, ?)
                """,
                exact_batch
            )

            exact_batch.clear()

        if len(name_batch) >= BATCH:

            cur.executemany(
                """
                INSERT INTO name_token
                VALUES (?, ?)
                """,
                name_batch
            )

            name_batch.clear()

        if len(address_batch) >= BATCH:

            cur.executemany(
                """
                INSERT INTO address_token
                VALUES (?, ?)
                """,
                address_batch
            )

            address_batch.clear()

        if len(number_batch) >= BATCH:

            cur.executemany(
                """
                INSERT INTO number_index
                VALUES (?, ?)
                """,
                number_batch
            )

            number_batch.clear()

        if len(pair_batch) >= BATCH:

            cur.executemany(
                """
                INSERT INTO address_pair
                VALUES (?, ?, ?)
                """,
                pair_batch
            )

            pair_batch.clear()

        if (
            (idx + 1) % 500_000 == 0
        ):

            conn.commit()

            print(
                f"{source_name}: "
                f"{idx+1:,}/{total:,}"
            )

    # --------------------------------------------------------
    # Final flush
    # --------------------------------------------------------

    if exact_batch:

        cur.executemany(
            """
            INSERT INTO exact_name
            VALUES (?, ?)
            """,
            exact_batch
        )

    if name_batch:

        cur.executemany(
            """
            INSERT INTO name_token
            VALUES (?, ?)
            """,
            name_batch
        )

    if address_batch:

        cur.executemany(
            """
            INSERT INTO address_token
            VALUES (?, ?)
            """,
            address_batch
        )

    if number_batch:

        cur.executemany(
            """
            INSERT INTO number_index
            VALUES (?, ?)
            """,
            number_batch
        )

    if pair_batch:

        cur.executemany(
            """
            INSERT INTO address_pair
            VALUES (?, ?, ?)
            """,
            pair_batch
        )

    conn.commit()

    print(
        f"Creating indexes for {source_name}..."
    )

    cur.execute("""
        CREATE INDEX idx_exact_name
        ON exact_name(name)
    """)

    cur.execute("""
        CREATE INDEX idx_name_token
        ON name_token(token)
    """)

    cur.execute("""
        CREATE INDEX idx_address_token
        ON address_token(token)
    """)

    cur.execute("""
        CREATE INDEX idx_number
        ON number_index(number)
    """)

    cur.execute("""
        CREATE INDEX idx_address_pair
        ON address_pair(token1, token2)
    """)

    conn.commit()

    print(
        f"{source_name} SQLite COMPLETE."
    )

    return conn


# ============================================================
# BUILD BOTH
# ============================================================

s2_conn = build_sqlite(
    s2,
    "S2"
)

s3_conn = build_sqlite(
    s3,
    "S3"
)

s2_cur = s2_conn.cursor()
s3_cur = s3_conn.cursor()


# ============================================================
# PREPARE S1
# ============================================================

print("\nPreparing S1...")

s1["name_norm"] = (
    s1["business_name"]
    .map(normalize)
)

s1["address_norm"] = (
    s1["business_address"]
    .map(normalize)
)

s1["name_tokens"] = (
    s1["name_norm"]
    .map(tokenize)
)

s1["address_tokens"] = (
    s1["address_norm"]
    .map(tokenize)
)

s1["numbers"] = (
    s1["address_norm"]
    .map(extract_numbers)
)


# ============================================================
# GT
# ============================================================

gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)


# ============================================================
# CANDIDATE QUERY
# ============================================================

def get_candidates(
    row,
    cur
):

    candidates = set()

    # --------------------------------------------------------
    # EXACT NAME
    # --------------------------------------------------------

    if row.name_norm:

        cur.execute(
            """
            SELECT idx
            FROM exact_name
            WHERE name = ?
            """,
            (row.name_norm,)
        )

        candidates.update(
            idx for (idx,)
            in cur.fetchall()
        )

    # --------------------------------------------------------
    # NAME TOKENS
    # --------------------------------------------------------

    for token in row.name_tokens:

        if (
            len(token) >= 3
            and name_freq[token]
            <= MAX_NAME_FREQ
        ):

            cur.execute(
                """
                SELECT idx
                FROM name_token
                WHERE token = ?
                """,
                (token,)
            )

            candidates.update(
                idx for (idx,)
                in cur.fetchall()
            )

    # --------------------------------------------------------
    # ADDRESS TOKENS
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
                WHERE token = ?
                """,
                (token,)
            )

            candidates.update(
                idx for (idx,)
                in cur.fetchall()
            )

    # --------------------------------------------------------
    # NUMBERS
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
                WHERE number = ?
                """,
                (number,)
            )

            candidates.update(
                idx for (idx,)
                in cur.fetchall()
            )

    # --------------------------------------------------------
    # ADDRESS PAIRS
    # --------------------------------------------------------

    valid_tokens = [
        token
        for token in row.address_tokens
        if (
            len(token) >= 4
            and address_freq[token]
            <= MAX_PAIR_TOKEN_FREQ
        )
    ]

    valid_tokens = sorted(
        set(valid_tokens)
    )

    for i in range(
        len(valid_tokens)
    ):

        for j in range(
            i + 1,
            len(valid_tokens)
        ):

            cur.execute(
                """
                SELECT idx
                FROM address_pair
                WHERE token1 = ?
                AND token2 = ?
                """,
                (
                    valid_tokens[i],
                    valid_tokens[j]
                )
            )

            candidates.update(
                idx for (idx,)
                in cur.fetchall()
            )

    return candidates


# ============================================================
# VALIDATE
# ============================================================

print(
    "\n========================================"
)

print(
    "VALIDATING V6 FULL-DATA BLOCKER"
)

print(
    "========================================\n"
)

start = time.time()

total_candidates = 0
max_candidates = 0

total_true = 0
total_found = 0

missed_examples = []

for count, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    c2 = get_candidates(
        row,
        s2_cur
    )

    c3 = get_candidates(
        row,
        s3_cur
    )

    candidates = c2 | c3

    total_candidates += len(
        candidates
    )

    max_candidates = max(
        max_candidates,
        len(candidates)
    )

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    candidate_ids = set()

    for idx in c2:

        candidate_ids.add(
            str(
                s2.iloc[idx]["entity_id"]
            )
        )

    for idx in c3:

        candidate_ids.add(
            str(
                s3.iloc[idx]["entity_id"]
            )
        )

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

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

    found = (
        true_ids &
        candidate_ids
    )

    missed = (
        true_ids -
        candidate_ids
    )

    total_true += len(
        true_ids
    )

    total_found += len(
        found
    )

    if (
        missed
        and len(missed_examples) < 20
    ):

        missed_examples.append(
            (
                row.entity_id,
                missed
            )
        )

    if count % 1000 == 0:

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
            f"Time={time.time()-start:.1f}s"
        )


# ============================================================
# RESULT
# ============================================================

recall = (
    total_found /
    max(total_true, 1)
)

average = (
    total_candidates /
    N
)

print(
    "\n"
    + "=" * 65
)

print(
    "V6 FULL-DATA RESULT"
)

print(
    "=" * 65
)

print(
    f"True matches:       {total_true:,}"
)

print(
    f"Found:              {total_found:,}"
)

print(
    f"Missed:             "
    f"{total_true-total_found:,}"
)

print(
    f"Blocking recall:    "
    f"{recall:.6f}"
)

print(
    f"Average candidates: "
    f"{average:.2f}"
)

print(
    f"Maximum candidates: "
    f"{max_candidates:,}"
)

print(
    f"Time:               "
    f"{time.time()-start:.1f}s"
)

print(
    "=" * 65
)

if missed_examples:

    print(
        "\nMISSED EXAMPLES:"
    )

    for entity_id, missed in missed_examples:

        print(
            f"{entity_id} -> "
            f"{sorted(missed)}"
        )

else:

    print(
        "\n🔥 ZERO MISSED MATCHES 🔥"
    )

s2_conn.close()
s3_conn.close()