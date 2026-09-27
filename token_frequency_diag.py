import sqlite3
import pandas as pd
import re
import unicodedata

BASE = "dataset/train"

S2_DB = "output/v6_full/S2.sqlite"
S3_DB = "output/v6_full/S3.sqlite"

N = 10000


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


def tokens(x):
    return set(normalize(x).split())


def numbers(x):
    return set(re.findall(r"\d+", normalize(x)))


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

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

gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)

s2_lookup = s2.set_index(
    "entity_id"
).to_dict("index")

s3_lookup = s3.set_index(
    "entity_id"
).to_dict("index")


# ------------------------------------------------------------
# OPEN DBs
# ------------------------------------------------------------

dbs = {
    "S2": sqlite3.connect(S2_DB),
    "S3": sqlite3.connect(S3_DB)
}


def freq(conn, table, column, value):

    cur = conn.cursor()

    cur.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        WHERE {column} = ?
        """,
        (value,)
    )

    return cur.fetchone()[0]


def show_token_frequencies(
    conn,
    table,
    column,
    values
):

    result = []

    for value in sorted(values):

        f = freq(
            conn,
            table,
            column,
            value
        )

        result.append(
            (value, f)
        )

    return result


# ------------------------------------------------------------
# FIRST 20 MISSED PAIRS FROM V8
# ------------------------------------------------------------

missed_pairs = [
    ("S1-925783039", "S3-698172821"),
    ("S1-773889195", "S2-970528089"),
    ("S1-377745466", "S3-402918963"),
    ("S1-377745466", "S3-70942743"),
    ("S1-377745466", "S3-836170816"),
    ("S1-377745466", "S2-203037501"),
    ("S1-133037285", "S3-33144658"),
    ("S1-133037285", "S2-407207105"),
    ("S1-133037285", "S2-103437512"),
    ("S1-851869949", "S2-915265864"),
    ("S1-851869949", "S2-961342067"),
    ("S1-851869949", "S2-801813753"),
    ("S1-851869949", "S3-949828938"),
    ("S1-27541239", "S2-199341916"),
    ("S1-27541239", "S3-739757636"),
    ("S1-629417405", "S3-606982601"),
    ("S1-629417405", "S3-438606634"),
    ("S1-629417405", "S3-786131946"),
    ("S1-629417405", "S2-928426462"),
    ("S1-629417405", "S3-613837325"),
]


# ------------------------------------------------------------
# DIAGNOSTIC
# ------------------------------------------------------------

for s1_id, true_id in missed_pairs:

    s1_row = s1[
        s1["entity_id"] == s1_id
    ]

    if len(s1_row) == 0:
        continue

    s1_row = s1_row.iloc[0]

    if true_id.startswith("S2-"):
        true_row = s2_lookup[true_id]
        source = "S2"
    else:
        true_row = s3_lookup[true_id]
        source = "S3"

    name1 = tokens(
        s1_row["business_name"]
    )

    name2 = tokens(
        true_row["business_name"]
    )

    addr1 = tokens(
        s1_row["business_address"]
    )

    addr2 = tokens(
        true_row["business_address"]
    )

    nums1 = numbers(
        s1_row["business_address"]
    )

    nums2 = numbers(
        true_row["business_address"]
    )

    common_name = name1 & name2
    common_addr = addr1 & addr2
    common_nums = nums1 & nums2

    print("\n" + "=" * 90)

    print(
        "S1:",
        s1_id,
        "| TRUE:",
        true_id,
        "|",
        source
    )

    print(
        "S1 NAME:",
        s1_row["business_name"]
    )

    print(
        "TRUE NAME:",
        true_row["business_name"]
    )

    print(
        "S1 ADDRESS:",
        s1_row["business_address"]
    )

    print(
        "TRUE ADDRESS:",
        true_row["business_address"]
    )

    print("\nCOMMON NAME TOKEN FREQUENCIES:")

    conn = dbs[source]

    for token, f in show_token_frequencies(
        conn,
        "name_token",
        "token",
        common_name
    ):
        print(
            f"  {token:<25} {f}"
        )

    print("\nCOMMON ADDRESS TOKEN FREQUENCIES:")

    for token, f in show_token_frequencies(
        conn,
        "address_token",
        "token",
        common_addr
    ):
        print(
            f"  {token:<25} {f}"
        )

    print("\nCOMMON NUMBER FREQUENCIES:")

    for num in sorted(common_nums):

        f = freq(
            conn,
            "number_index",
            "number",
            num
        )

        print(
            f"  {num:<25} {f}"
        )


for conn in dbs.values():
    conn.close()

print("\nDONE")