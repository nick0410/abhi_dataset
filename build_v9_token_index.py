import sqlite3
import pandas as pd
import re
import unicodedata
import os
from collections import Counter

BASE = "dataset/train"
OUT = "output/v9_tokens.sqlite"

MAX_FREQ = 10000
CHUNK = 100000


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


def get_tokens(x, min_len):

    return {
        t
        for t in normalize(x).split()
        if len(t) >= min_len
    }


# ============================================================
# REMOVE OLD DB
# ============================================================

if os.path.exists(OUT):
    os.remove(OUT)


# ============================================================
# PASS 1: COUNT TOKENS
# ============================================================

name_counter = Counter()
address_counter = Counter()

print("PASS 1: counting tokens...")


for source in [2, 3]:

    path = f"{BASE}/train_source{source}.tsv"

    print("\nReading:", path)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK
    ):

        for value in chunk["business_name"]:

            name_counter.update(
                get_tokens(value, 3)
            )

        for value in chunk["business_address"]:

            address_counter.update(
                get_tokens(value, 4)
            )

        print(
            "Processed:",
            chunk.index[-1] + 1
        )


print("\nName vocabulary:", len(name_counter))
print("Address vocabulary:", len(address_counter))


# ============================================================
# SELECT TOKENS
# ============================================================

name_tokens = {
    token
    for token, count in name_counter.items()
    if count <= MAX_FREQ
}

address_tokens = {
    token
    for token, count in address_counter.items()
    if count <= MAX_FREQ
}


print(
    "Selected name tokens:",
    len(name_tokens)
)

print(
    "Selected address tokens:",
    len(address_tokens)
)


# ============================================================
# CREATE SQLITE
# ============================================================

print("\nCreating SQLite...")

conn = sqlite3.connect(OUT)

conn.execute("""
CREATE TABLE name_token (
    token TEXT,
    source TEXT,
    idx INTEGER
)
""")

conn.execute("""
CREATE TABLE address_token (
    token TEXT,
    source TEXT,
    idx INTEGER
)
""")

conn.execute("""
CREATE INDEX idx_name_token
ON name_token(token)
""")

conn.execute("""
CREATE INDEX idx_address_token
ON address_token(token)
""")


# ============================================================
# PASS 2: BUILD INDEX
# ============================================================

for source in [2, 3]:

    path = f"{BASE}/train_source{source}.tsv"

    print("\nBuilding:", path)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK
    ):

        name_rows = []
        address_rows = []

        for idx, row in chunk.iterrows():

            # NAME
            nt = get_tokens(
                row["business_name"],
                3
            )

            for token in nt:

                if token in name_tokens:

                    name_rows.append(
                        (
                            token,
                            f"S{source}",
                            int(idx)
                        )
                    )

            # ADDRESS
            at = get_tokens(
                row["business_address"],
                4
            )

            for token in at:

                if token in address_tokens:

                    address_rows.append(
                        (
                            token,
                            f"S{source}",
                            int(idx)
                        )
                    )


        if name_rows:

            conn.executemany(
                """
                INSERT INTO name_token
                (token, source, idx)
                VALUES (?, ?, ?)
                """,
                name_rows
            )


        if address_rows:

            conn.executemany(
                """
                INSERT INTO address_token
                (token, source, idx)
                VALUES (?, ?, ?)
                """,
                address_rows
            )


        conn.commit()

        print(
            "Processed:",
            chunk.index[-1] + 1
        )


# ============================================================
# DONE
# ============================================================

conn.execute(
    "ANALYZE"
)

conn.commit()
conn.close()

print("\n========================================")
print("V9 TOKEN INDEX COMPLETE")
print("========================================")
print("Database:", OUT)