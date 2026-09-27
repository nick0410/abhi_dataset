import pandas as pd
import re
import unicodedata
from collections import defaultdict


BASE = "sample_dataset/dataset"

s1 = pd.read_csv(
    f"{BASE}/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    f"{BASE}/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    f"{BASE}/train/train_source3.tsv",
    sep="\t"
)


def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def tokens(text):

    value = normalize(text)

    if not value:
        return set()

    return set(value.split())


def numbers(text):

    if pd.isna(text):
        return set()

    return set(
        re.findall(r"\d+", str(text))
    )


def prepare(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].apply(
        normalize
    )

    df["name_tokens"] = df["business_name"].apply(
        tokens
    )

    df["address_tokens"] = df["business_address"].apply(
        tokens
    )

    df["address_numbers"] = df["business_address"].apply(
        numbers
    )

    return df


s1 = prepare(s1)
s2 = prepare(s2)
s3 = prepare(s3)


# ============================================================
# BUILD SEPARATE INDICES
# ============================================================

def build_indices(df):

    indices = {
        "exact_name": defaultdict(set),
        "name_token": defaultdict(set),
        "address_token": defaultdict(set),
        "number": defaultdict(set)
    }

    for idx, row in df.iterrows():

        # Exact name
        if row["name_norm"]:
            indices["exact_name"][
                row["name_norm"]
            ].add(idx)

        # Name tokens
        for token in row["name_tokens"]:

            if len(token) >= 3:
                indices["name_token"][token].add(idx)

        # Address tokens
        for token in row["address_tokens"]:

            if len(token) >= 4:
                indices["address_token"][token].add(idx)

        # Numbers
        for number in row["address_numbers"]:
            indices["number"][number].add(idx)

    return indices


s2_idx = build_indices(s2)
s3_idx = build_indices(s3)


# ============================================================
# DIAGNOSTICS
# ============================================================

def analyze(row, index):

    exact_name = set()

    name_token = set()

    address_token = set()

    number = set()


    # Exact name
    if row["name_norm"]:

        exact_name = index["exact_name"].get(
            row["name_norm"],
            set()
        )


    # Name tokens
    for token in row["name_tokens"]:

        if len(token) >= 3:

            name_token.update(
                index["name_token"].get(
                    token,
                    set()
                )
            )


    # Address tokens
    for token in row["address_tokens"]:

        if len(token) >= 4:

            address_token.update(
                index["address_token"].get(
                    token,
                    set()
                )
            )


    # Numbers
    for n in row["address_numbers"]:

        number.update(
            index["number"].get(
                n,
                set()
            )
        )


    return {
        "exact_name": len(exact_name),
        "name_token": len(name_token),
        "address_token": len(address_token),
        "number": len(number),
        "union": len(
            exact_name |
            name_token |
            address_token |
            number
        )
    }


# ============================================================
# RUN ON 100 S1 RECORDS
# ============================================================

results = []

for _, row in s1.head(100).iterrows():

    a = analyze(row, s2_idx)
    b = analyze(row, s3_idx)

    results.append({
        "s1": row["entity_id"],

        "exact_name":
            a["exact_name"] + b["exact_name"],

        "name_token":
            a["name_token"] + b["name_token"],

        "address_token":
            a["address_token"] + b["address_token"],

        "number":
            a["number"] + b["number"],

        "union":
            len(
                set()
            )
    })


df = pd.DataFrame(results)


print("\n" + "=" * 80)
print("BLOCKING DIAGNOSTICS")
print("=" * 80)

print(
    "\nAverage candidates contributed by each rule:"
)

print(
    df[
        [
            "exact_name",
            "name_token",
            "address_token",
            "number"
        ]
    ].mean()
)

print(
    "\nMaximum candidates contributed by each rule:"
)

print(
    df[
        [
            "exact_name",
            "name_token",
            "address_token",
            "number"
        ]
    ].max()
)

print("\nSample:")
print(
    df.head(20).to_string(index=False)
)

print("\nDONE")