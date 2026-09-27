import pandas as pd
import re
import unicodedata
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "sample_dataset/dataset/train"

MAX_NAME_FREQ = 100
MAX_ADDRESS_FREQ = 150
MAX_NUMBER_FREQ = 300

# Address pair should be relatively selective.
# A pair is only indexed if both tokens are not extremely common.
MAX_PAIR_TOKEN_FREQ = 500


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

s1 = pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\t"
)

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

print(f"S1: {len(s1)}")
print(f"S2: {len(s2)}")
print(f"S3: {len(s3)}")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):
    """
    Unicode-aware normalization.

    Important:
    - Preserves non-Latin scripts such as Hindi.
    - Converts punctuation to spaces.
    - Lowercases text.
    """

    if pd.isna(value):
        return ""

    text = str(value).lower()

    # Keep Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    result = []

    for char in text:
        if char.isalnum() or char.isspace():
            result.append(char)
        else:
            result.append(" ")

    text = "".join(result)

    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    if not text:
        return set()

    return set(text.split())


def extract_numbers(text):
    return set(re.findall(r"\d+", text))


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["name_norm"] = (
        df["business_name"]
        .map(normalize)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize)
    )

    df["name_tokens"] = (
        df["name_norm"]
        .map(tokenize)
    )

    df["address_tokens"] = (
        df["address_norm"]
        .map(tokenize)
    )

    df["numbers"] = (
        df["address_norm"]
        .map(extract_numbers)
    )

    return df


print("\nPreparing data...")

s1 = prepare_dataframe(s1)
s2 = prepare_dataframe(s2)
s3 = prepare_dataframe(s3)


# ============================================================
# BUILD GLOBAL TOKEN FREQUENCIES
# ============================================================

print("\nCalculating token frequencies...")

name_freq = Counter()
address_freq = Counter()
number_freq = Counter()

source_data = pd.concat(
    [s2, s3],
    ignore_index=True
)

for _, row in source_data.iterrows():

    # Name token frequency
    for token in row["name_tokens"]:

        if len(token) >= 3:
            name_freq[token] += 1

    # Address token frequency
    for token in row["address_tokens"]:

        if len(token) >= 4:
            address_freq[token] += 1

    # Number frequency
    for number in row["numbers"]:
        number_freq[number] += 1


# ============================================================
# BUILD SOURCE INDEX
# ============================================================

def build_index(df):

    exact_name_index = defaultdict(list)
    name_token_index = defaultdict(list)
    address_token_index = defaultdict(list)
    number_index = defaultdict(list)
    address_pair_index = defaultdict(list)

    print(f"Building index for {len(df)} records...")

    for idx, row in df.iterrows():

        # ----------------------------------------------------
        # Exact normalized business name
        # ----------------------------------------------------

        name = row["name_norm"]

        if name:
            exact_name_index[name].append(idx)

        # ----------------------------------------------------
        # Name token index
        # ----------------------------------------------------

        for token in row["name_tokens"]:

            if (
                len(token) >= 3
                and name_freq[token] <= MAX_NAME_FREQ
            ):
                name_token_index[token].append(idx)

        # ----------------------------------------------------
        # Address token index
        # ----------------------------------------------------

        valid_address_tokens = []

        for token in row["address_tokens"]:

            if (
                len(token) >= 4
                and address_freq[token] <= MAX_ADDRESS_FREQ
            ):
                address_token_index[token].append(idx)

            # Used for address-pair blocking.
            #
            # We allow somewhat more common tokens here
            # because two tokens together are much more selective.

            if (
                len(token) >= 4
                and address_freq[token] <= MAX_PAIR_TOKEN_FREQ
            ):
                valid_address_tokens.append(token)

        # ----------------------------------------------------
        # Address number index
        # ----------------------------------------------------

        for number in row["numbers"]:

            if number_freq[number] <= MAX_NUMBER_FREQ:
                number_index[number].append(idx)

        # ----------------------------------------------------
        # Address token-pair index
        # ----------------------------------------------------

        valid_address_tokens = sorted(
            set(valid_address_tokens)
        )

        token_count = len(valid_address_tokens)

        for i in range(token_count):

            for j in range(i + 1, token_count):

                pair = (
                    valid_address_tokens[i],
                    valid_address_tokens[j]
                )

                address_pair_index[pair].append(idx)

    return {
        "exact_name": exact_name_index,
        "name_token": name_token_index,
        "address_token": address_token_index,
        "number": number_index,
        "address_pair": address_pair_index
    }


print("\nBuilding S2 index...")

s2_index = build_index(s2)

print("\nBuilding S3 index...")

s3_index = build_index(s3)

print("\nIndexes built successfully.")


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def get_candidates(row, index):

    candidates = set()

    exact_name_index = index["exact_name"]
    name_token_index = index["name_token"]
    address_token_index = index["address_token"]
    number_index = index["number"]
    address_pair_index = index["address_pair"]

    # ========================================================
    # RULE 1 — EXACT NORMALIZED NAME
    # ========================================================

    candidates.update(
        exact_name_index.get(
            row["name_norm"],
            []
        )
    )

    # ========================================================
    # RULE 2 — RARE NAME TOKENS
    # ========================================================

    for token in row["name_tokens"]:

        if (
            len(token) >= 3
            and name_freq[token] <= MAX_NAME_FREQ
        ):

            candidates.update(
                name_token_index.get(
                    token,
                    []
                )
            )

    # ========================================================
    # RULE 3 — RARE ADDRESS TOKENS
    # ========================================================

    for token in row["address_tokens"]:

        if (
            len(token) >= 4
            and address_freq[token] <= MAX_ADDRESS_FREQ
        ):

            candidates.update(
                address_token_index.get(
                    token,
                    []
                )
            )

    # ========================================================
    # RULE 4 — RARE ADDRESS NUMBERS
    # ========================================================

    for number in row["numbers"]:

        if number_freq[number] <= MAX_NUMBER_FREQ:

            candidates.update(
                number_index.get(
                    number,
                    []
                )
            )

    # ========================================================
    # RULE 5 — ADDRESS TOKEN PAIRS
    # ========================================================
    #
    # Important for cases such as:
    #
    # S1:
    # G-1 ... Mumbai, Maharashtra
    #
    # S2:
    # G-1, CHURU, Mumbai, Maharashtra
    #
    # Even if the business names are in different scripts,
    # the address can still provide a strong blocking signal.
    #
    # We require TWO shared address tokens.
    #
    # ========================================================

    valid_tokens = [
        token
        for token in row["address_tokens"]
        if (
            len(token) >= 4
            and address_freq[token] <= MAX_PAIR_TOKEN_FREQ
        )
    ]

    valid_tokens = sorted(set(valid_tokens))

    token_count = len(valid_tokens)

    for i in range(token_count):

        for j in range(i + 1, token_count):

            pair = (
                valid_tokens[i],
                valid_tokens[j]
            )

            candidates.update(
                address_pair_index.get(
                    pair,
                    []
                )
            )

    return candidates


# ============================================================
# GROUND TRUTH LOOKUP
# ============================================================

gt_lookup = {
    row["source1_entity_id"]:
        row["matched_entity_ids"]
    for _, row in gt.iterrows()
}


# ============================================================
# EVALUATION
# ============================================================

print("\nTesting V6 blocker...\n")

total_candidates = 0
max_candidates = 0

total_true = 0
total_found = 0

coverage = Counter()


for i, row in s1.iterrows():

    # --------------------------------------------------------
    # Generate candidates from S2
    # --------------------------------------------------------

    candidates_s2 = get_candidates(
        row,
        s2_index
    )

    # --------------------------------------------------------
    # Generate candidates from S3
    # --------------------------------------------------------

    candidates_s3 = get_candidates(
        row,
        s3_index
    )

    candidates = (
        candidates_s2 |
        candidates_s3
    )

    total_candidates += len(candidates)

    max_candidates = max(
        max_candidates,
        len(candidates)
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    truth = gt_lookup.get(
        row["entity_id"],
        ""
    )

    true_ids = set()

    if (
        pd.notna(truth)
        and str(truth).strip()
    ):

        true_ids = {
            value.strip()
            for value in str(truth).split(",")
            if value.strip()
        }

    # --------------------------------------------------------
    # Candidate entity IDs
    # --------------------------------------------------------

    candidate_ids = set()

    for idx in candidates_s2:

        candidate_ids.add(
            str(
                s2.iloc[idx]["entity_id"]
            )
        )

    for idx in candidates_s3:

        candidate_ids.add(
            str(
                s3.iloc[idx]["entity_id"]
            )
        )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    found = true_ids & candidate_ids

    total_true += len(true_ids)
    total_found += len(found)

    if true_ids:

        if found == true_ids:
            coverage["all_true_found"] += 1
        else:
            coverage["missed"] += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if i % 100 == 0:

        recall = (
            total_found /
            max(total_true, 1)
        )

        average = (
            total_candidates /
            (i + 1)
        )

        print(
            f"{i}/{len(s1)} | "
            f"avg candidates: {average:.2f} | "
            f"recall: {recall:.4f}"
        )


# ============================================================
# FIND MISSED TRUE MATCHES
# ============================================================

print("\nSearching for missed matches...\n")

missed_count = 0

for _, row in s1.iterrows():

    candidates_s2 = get_candidates(
        row,
        s2_index
    )

    candidates_s3 = get_candidates(
        row,
        s3_index
    )

    candidate_ids = set()

    for idx in candidates_s2:

        candidate_ids.add(
            str(
                s2.iloc[idx]["entity_id"]
            )
        )

    for idx in candidates_s3:

        candidate_ids.add(
            str(
                s3.iloc[idx]["entity_id"]
            )
        )

    truth = gt_lookup.get(
        row["entity_id"],
        ""
    )

    if (
        pd.isna(truth)
        or not str(truth).strip()
    ):
        continue

    true_ids = {
        value.strip()
        for value in str(truth).split(",")
        if value.strip()
    }

    missed = true_ids - candidate_ids

    if not missed:
        continue

    missed_count += 1

    print("\n" + "=" * 60)
    print("MISSED MATCH")
    print("=" * 60)

    print("S1:", row["entity_id"])
    print("Name:", row["business_name"])
    print("Address:", row["business_address"])
    print("Country:", row["country"])

    print("Missed IDs:", missed)

    print("\nS1 address tokens:")
    print(row["address_tokens"])

    print("\nS1 numbers:")
    print(row["numbers"])

    # --------------------------------------------------------
    # Inspect missed records
    # --------------------------------------------------------

    for entity_id in missed:

        match_s2 = s2[
            s2["entity_id"].astype(str) ==
            entity_id
        ]

        match_s3 = s3[
            s3["entity_id"].astype(str) ==
            entity_id
        ]

        match = pd.concat(
            [match_s2, match_s3],
            ignore_index=True
        )

        if match.empty:
            continue

        matched_row = match.iloc[0]

        print("\nTRUE MATCH:")
        print(
            match[
                [
                    "entity_id",
                    "business_name",
                    "business_address",
                    "country"
                ]
            ].to_string(index=False)
        )

        print("\nMatch address tokens:")
        print(
            matched_row["address_tokens"]
        )

        print("\nMatch numbers:")
        print(
            matched_row["numbers"]
        )

        print("\nCommon address tokens:")
        print(
            row["address_tokens"] &
            matched_row["address_tokens"]
        )

        print("\nCommon numbers:")
        print(
            row["numbers"] &
            matched_row["numbers"]
        )

    # Only display the first missed S1
    break


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("V6 BLOCKING RESULT")
print("=" * 60)

print(
    "Total candidates:",
    total_candidates
)

print(
    "Average candidates:",
    total_candidates / len(s1)
)

print(
    "Maximum candidates:",
    max_candidates
)

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
    f"{total_found / max(total_true, 1):.6f}"
)

print("\nS1 coverage:")
print(dict(coverage))