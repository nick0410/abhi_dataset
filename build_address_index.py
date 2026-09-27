import sqlite3
import pandas as pd
import re
import unicodedata
import os

BASE = "dataset/train"
OUT = "output/address_exact.sqlite"

os.makedirs("output", exist_ok=True)

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

    return re.sub(r"\s+", " ", "".join(out)).strip()


if os.path.exists(OUT):
    os.remove(OUT)

conn = sqlite3.connect(OUT)

conn.execute("""
CREATE TABLE address_exact (
    source TEXT,
    address TEXT,
    idx INTEGER
)
""")

conn.execute("""
CREATE INDEX idx_address_exact
ON address_exact(address)
""")

for source in ["S2", "S3"]:

    print("Loading", source)

    path = f"{BASE}/train_source{source[-1]}.tsv"

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100000
    ):

        rows = []

        for idx, address in zip(
            chunk.index,
            chunk["business_address"]
        ):

            addr = normalize(address)

            if addr:
                rows.append(
                    (source, addr, int(idx))
                )

        conn.executemany(
            """
            INSERT INTO address_exact
            (source, address, idx)
            VALUES (?, ?, ?)
            """,
            rows
        )

        conn.commit()

        print(
            source,
            "processed:",
            chunk.index[-1] + 1
        )

conn.close()

print("DONE:", OUT)