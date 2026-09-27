import pandas as pd
import sqlite3
import re
import unicodedata

BASE = "dataset/train"
TOKEN_DB = "output/v9_tokens.sqlite"
ADDRESS_DB = "output/address_exact.sqlite"

S1_IDS = [
    "S1-765734254",
    "S1-597762257",
    "S1-252242682",
    "S1-883973644",
    "S1-17666546",
    "S1-681392030",
    "S1-454111437",
    "S1-630223771",
    "S1-366329737",
    "S1-912508322",
]

s1 = pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\t",
    dtype=str
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

    return re.sub(
        r"\s+",
        " ",
        "".join(result)
    ).strip()


def get_tokens(x, min_len):
    return {
        t for t in normalize(x).split()
        if len(t) >= min_len
    }


token_conn = sqlite3.connect(TOKEN_DB)
addr_conn = sqlite3.connect(ADDRESS_DB)


def token_hits(table, token, source):
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


def address_hits(address, source):
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


gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)

s1_lookup = {
    row["entity_id"]: row
    for _, row in s1.iterrows()
}


for s1_id in S1_IDS:

    print("\n" + "=" * 80)
    print("S1:", s1_id)
    print("=" * 80)

    row = s1_lookup[s1_id]

    print("NAME   :", row["business_name"])
    print("ADDRESS:", row["business_address"])
    print("COUNTRY:", row["country"])

    true_ids = [
        x.strip()
        for x in gt_lookup[s1_id].split(",")
        if x.strip()
    ]

    print("\nTRUE MATCHES:")
    print(true_ids)

    name_tokens = get_tokens(
        row["business_name"],
        3
    )

    address_tokens = get_tokens(
        row["business_address"],
        4
    )

    print("\nNAME TOKENS:")
    print(sorted(name_tokens))

    print("\nADDRESS TOKENS:")
    print(sorted(address_tokens))

    for true_id in true_ids:

        if true_id.startswith("S2-"):
            idx = s2.index[
                s2["entity_id"] == true_id
            ][0]

            target = s2.iloc[idx]
            source = "S2"

        else:
            idx = s3.index[
                s3["entity_id"] == true_id
            ][0]

            target = s3.iloc[idx]
            source = "S3"

        print("\n" + "-" * 70)
        print("TRUE:", true_id, source)

        print("NAME   :", target["business_name"])
        print("ADDRESS:", target["business_address"])

        target_name_tokens = get_tokens(
            target["business_name"],
            3
        )

        target_address_tokens = get_tokens(
            target["business_address"],
            4
        )

        common_name = name_tokens & target_name_tokens
        common_address = address_tokens & target_address_tokens

        print("COMMON NAME TOKENS:")
        print(sorted(common_name))

        print("COMMON ADDRESS TOKENS:")
        print(sorted(common_address))

        # Check whether V9 token index would retrieve it
        name_index_hit = False
        address_index_hit = False

        for token in name_tokens:
            if idx in token_hits(
                "name_token",
                token,
                source
            ):
                name_index_hit = True
                break

        for token in address_tokens:
            if idx in token_hits(
                "address_token",
                token,
                source
            ):
                address_index_hit = True
                break

        exact_address = normalize(
            row["business_address"]
        ) == normalize(
            target["business_address"]
        )

        print("V9 NAME TOKEN HIT    :", name_index_hit)
        print("V9 ADDRESS TOKEN HIT :", address_index_hit)
        print("EXACT ADDRESS HIT    :", exact_address)


token_conn.close()
addr_conn.close()