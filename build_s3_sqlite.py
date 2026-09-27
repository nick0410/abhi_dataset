import sqlite3
import pandas as pd
import re
import unicodedata
import os
from collections import Counter

BASE = "dataset/train"
DB = "output/v7_cache/S3_index.sqlite"

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

    return re.sub(
        r"\s+",
        " ",
        "".join(result)
    ).strip()


def tokenize(text):
    return set(text.split()) if text else set()


def numbers(text):
    return set(re.findall(r"\d+", text)) if text else set()


print("Loading S3...")

s3 = pd.read_csv(
    f"{BASE}/train_source3.tsv",
    sep="\t"
)

print(f"S3 rows: {len(s3):,}")

print("Loading frequencies...")

import pickle

with open(
    "output/v7_cache/frequencies.pkl",
    "rb"
) as f:
    freq = pickle.load(f)

address_freq = freq["address_freq"]
number_freq = freq["number_freq"]

print("Frequencies loaded.")

# Remove old corrupt DB if present
if os.path.exists(DB):
    os.remove(DB)

print("Creating SQLite database...")

conn = sqlite3.connect(DB)

cur = conn.cursor()

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

cur.execute("""
CREATE TABLE exact_name (
    name TEXT,
    idx INTEGER
)
""")

conn.commit()

print("Building indexes...")

batch_address = []
batch_number = []
batch_pair = []
batch_name = []

BATCH = 50_000

for idx, row in enumerate(
    s3.itertuples(index=False)
):

    name = normalize(row.business_name)
    address = normalize(row.business_address)

    address_tokens = tokenize(address)
    row_numbers = numbers(address)

    # ----------------------------
    # Exact name
    # ----------------------------

    if name:
        batch_name.append(
            (name, idx)
        )

    # ----------------------------
    # Address tokens
    # ----------------------------

    pair_tokens = []

    for token in address_tokens:

        if (
            len(token) >= 4
            and address_freq[token]
            <= MAX_ADDRESS_FREQ
        ):
            batch_address.append(
                (token, idx)
            )

        if (
            len(token) >= 4
            and address_freq[token]
            <= MAX_PAIR_TOKEN_FREQ
        ):
            pair_tokens.append(token)

    # ----------------------------
    # Numbers
    # ----------------------------

    for number in row_numbers:

        if (
            number_freq[number]
            <= MAX_NUMBER_FREQ
        ):
            batch_number.append(
                (number, idx)
            )

    # ----------------------------
    # Address pairs
    # ----------------------------

    pair_tokens = sorted(
        set(pair_tokens)
    )

    for i in range(len(pair_tokens)):

        for j in range(
            i + 1,
            len(pair_tokens)
        ):

            batch_pair.append(
                (
                    pair_tokens[i],
                    pair_tokens[j],
                    idx
                )
            )

    # ----------------------------
    # Flush periodically
    # ----------------------------

    if len(batch_address) >= BATCH:

        cur.executemany(
            "INSERT INTO address_token VALUES (?, ?)",
            batch_address
        )

        batch_address.clear()

    if len(batch_number) >= BATCH:

        cur.executemany(
            "INSERT INTO number_index VALUES (?, ?)",
            batch_number
        )

        batch_number.clear()

    if len(batch_pair) >= BATCH:

        cur.executemany(
            "INSERT INTO address_pair VALUES (?, ?, ?)",
            batch_pair
        )

        batch_pair.clear()

    if len(batch_name) >= BATCH:

        cur.executemany(
            "INSERT INTO exact_name VALUES (?, ?)",
            batch_name
        )

        batch_name.clear()

    if (idx + 1) % 500_000 == 0:

        conn.commit()

        print(
            f"{idx + 1:,}/{len(s3):,}"
        )


# Final flush

if batch_address:
    cur.executemany(
        "INSERT INTO address_token VALUES (?, ?)",
        batch_address
    )

if batch_number:
    cur.executemany(
        "INSERT INTO number_index VALUES (?, ?)",
        batch_number
    )

if batch_pair:
    cur.executemany(
        "INSERT INTO address_pair VALUES (?, ?, ?)",
        batch_pair
    )

if batch_name:
    cur.executemany(
        "INSERT INTO exact_name VALUES (?, ?)",
        batch_name
    )

conn.commit()

print("Creating indexes...")

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

cur.execute("""
CREATE INDEX idx_exact_name
ON exact_name(name)
""")

conn.commit()

conn.close()

print()
print("======================================")
print("S3 SQLITE INDEX COMPLETE")
print("======================================")
print(f"Database: {DB}")
print("Done.")