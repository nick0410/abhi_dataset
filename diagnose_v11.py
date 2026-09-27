import pandas as pd
import sqlite3
import re
import unicodedata

BASE = "dataset/train"
TOKEN_DB = "output/v9_tokens.sqlite"
ADDRESS_DB = "output/address_exact.sqlite"

# Missed pairs from V10
MISSED = {
    "S1-765734254": ["S3-799567902"],
    "S1-597762257": ["S2-535895394", "S3-426660831", "S2-152865025"],
    "S1-252242682": ["S2-84842746"],
    "S1-883973644": ["S2-346262429"],
    "S1-17666546": ["S2-341313901"],
    "S1-681392030": ["S3-340243016"],
    "S1-454111437": ["S2-790403370"],
    "S1-630223771": ["S3-697578371", "S3-587424087", "S3-815538600"],
    "S1-366329737": ["S2-23560385"],
    "S1-912508322": ["S2-276782865"],
    "S1-800482427": ["S3-355590735"],
    "S1-21084577": ["S3-50929802"],
    "S1-978421811": ["S2-847503139"],
    "S1-499302167": ["S2-148775637"],
    "S1-639096068": ["S2-435057007"],
    "S1-17506850": ["S3-917774125"],
    "S1-419483044": ["S2-497881051"],
    "S1-787617931": ["S2-863761700"],
    "S1-535605149": ["S2-751164711"],
    "S1-387694500": ["S3-9126403"],
    "S1-615455408": ["S3-156495960"],
    "S1-974991176": ["S3-766777210"],
    "S1-536948776": ["S2-376450162"],
    "S1-684505968": ["S3-300894176"],
    "S1-386654734": ["S3-508283569"],
    "S1-973233920": ["S3-110090880"],
    "S1-761585587": ["S2-586234138"],
    "S1-602775749": ["S2-872504799"],
    "S1-501296085": ["S3-328999650"],
    "S1-98579303": ["S3-719484948", "S3-690119410"],
}


def normalize(x):
    if pd.isna(x):
        return ""

    x = unicodedata.normalize("NFKC", str(x).lower())

    out = []

    for c in x:
        if c.isalnum() or c.isspace():
            out.append(c)
        else:
            out.append(" ")

    return re.sub(r"\s+", " ", "".join(out)).strip()


def tokens(x, min_len):
    return {
        x for x in normalize(x).split()
        if len(x) >= min_len
    }


def numbers(x):
    x = normalize(x)
    return set(re.findall(r"\d+", x))


print("Loading data...")

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


s1_lookup = {
    x["entity_id"]: x
    for _, x in s1.iterrows()
}

s2_lookup = {
    x["entity_id"]: x
    for _, x in s2.iterrows()
}

s3_lookup = {
    x["entity_id"]: x
    for _, x in s3.iterrows()
}


# ============================================================
# ANALYZE MISSED PAIRS
# ============================================================

total = 0

name_token_shared = 0
address_token_shared = 0
number_shared = 0
name_char_shared = 0
address_char_shared = 0
city_like_shared = 0

no_common_token = 0


def char_ngrams(text, n=3):
    text = normalize(text).replace(" ", "")
    if len(text) < n:
        return set()
    return {
        text[i:i+n]
        for i in range(len(text) - n + 1)
    }


for s1_id, target_ids in MISSED.items():

    source1 = s1_lookup[s1_id]

    s1_name = source1["business_name"]
    s1_addr = source1["business_address"]

    s1_name_tokens = tokens(s1_name, 2)
    s1_addr_tokens = tokens(s1_addr, 2)

    s1_nums = numbers(s1_addr)

    s1_name_3 = char_ngrams(s1_name, 3)
    s1_addr_3 = char_ngrams(s1_addr, 3)

    for target_id in target_ids:

        total += 1

        if target_id.startswith("S2-"):
            target = s2_lookup[target_id]
        else:
            target = s3_lookup[target_id]

        t_name = target["business_name"]
        t_addr = target["business_address"]

        t_name_tokens = tokens(t_name, 2)
        t_addr_tokens = tokens(t_addr, 2)

        t_nums = numbers(t_addr)

        common_name = s1_name_tokens & t_name_tokens
        common_addr = s1_addr_tokens & t_addr_tokens
        common_nums = s1_nums & t_nums

        common_name_3 = (
            s1_name_3 &
            char_ngrams(t_name, 3)
        )

        common_addr_3 = (
            s1_addr_3 &
            char_ngrams(t_addr, 3)
        )

        if common_name:
            name_token_shared += 1

        if common_addr:
            address_token_shared += 1

        if common_nums:
            number_shared += 1

        if common_name_3:
            name_char_shared += 1

        if common_addr_3:
            address_char_shared += 1

        # Print useful cases
        print("\n" + "=" * 90)

        print("S1:", s1_id)
        print("TARGET:", target_id)

        print("S1 NAME   :", s1_name)
        print("TGT NAME  :", t_name)

        print("S1 ADDRESS:", s1_addr)
        print("TGT ADDRESS:", t_addr)

        print("COMMON NAME TOKENS:", sorted(common_name))
        print("COMMON ADDR TOKENS:", sorted(common_addr))
        print("COMMON NUMBERS:", sorted(common_nums))

        print(
            "COMMON NAME 3-GRAMS:",
            len(common_name_3)
        )

        print(
            "COMMON ADDR 3-GRAMS:",
            len(common_addr_3)
        )


print("\n")
print("=" * 70)
print("V11 MISSED-PAIR ANALYSIS")
print("=" * 70)

print("Total missed pairs analysed:", total)

print(
    "Shared name tokens:",
    name_token_shared,
    "/",
    total
)

print(
    "Shared address tokens:",
    address_token_shared,
    "/",
    total
)

print(
    "Shared numbers:",
    number_shared,
    "/",
    total
)

print(
    "Shared name 3-grams:",
    name_char_shared,
    "/",
    total
)

print(
    "Shared address 3-grams:",
    address_char_shared,
    "/",
    total
)