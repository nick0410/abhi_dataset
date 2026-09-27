import os
import pickle
import pandas as pd
import re
import unicodedata
from collections import Counter, defaultdict

BASE = "dataset/train"
CACHE_DIR = "output/v7_cache"

os.makedirs(CACHE_DIR, exist_ok=True)

N = 10_000

MAX_NAME_FREQ = 100
MAX_ADDRESS_FREQ = 150
MAX_NUMBER_FREQ = 300
MAX_PAIR_TOKEN_FREQ = 500


# ============================================================
# NORMALIZATION
# ============================================================

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

    if not text:
        return set()

    return set(text.split())


def numbers(text):

    if not text:
        return set()

    return set(
        re.findall(r"\d+", text)
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].map(normalize)
    df["address_norm"] = df["business_address"].map(normalize)

    df["name_tokens"] = df["name_norm"].map(tokenize)
    df["address_tokens"] = df["address_norm"].map(tokenize)

    df["numbers"] = df["address_norm"].map(numbers)

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("Loading...")

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
    f"Loaded: {len(s1)} {len(s2)} {len(s3)}"
)

print("\nPreparing source data...")

s1 = prepare(s1)
s2 = prepare(s2)
s3 = prepare(s3)


# ============================================================
# FREQUENCY CACHE
# ============================================================

freq_file = os.path.join(
    CACHE_DIR,
    "frequencies.pkl"
)


if os.path.exists(freq_file):

    print("\nLoading cached frequencies...")

    with open(freq_file, "rb") as f:
        frequencies = pickle.load(f)

    name_freq = frequencies["name_freq"]
    address_freq = frequencies["address_freq"]
    number_freq = frequencies["number_freq"]

    print("Frequencies loaded from cache.")

else:

    print("\nCalculating frequencies...")

    name_freq = Counter()
    address_freq = Counter()
    number_freq = Counter()

    for source_name, df in [
        ("S2", s2),
        ("S3", s3)
    ]:

        print(f"Processing {source_name}...")

        for row in df.itertuples(index=False):

            for token in row.name_tokens:

                if len(token) >= 3:
                    name_freq[token] += 1

            for token in row.address_tokens:

                if len(token) >= 4:
                    address_freq[token] += 1

            for number in row.numbers:

                number_freq[number] += 1

    frequencies = {
        "name_freq": name_freq,
        "address_freq": address_freq,
        "number_freq": number_freq
    }

    with open(freq_file, "wb") as f:
        pickle.dump(
            frequencies,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print("Frequencies saved.")


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(df, source_name):

    cache_file = os.path.join(
        CACHE_DIR,
        f"{source_name}_index.pkl"
    )

    if os.path.exists(cache_file):

        print(
            f"\nLoading cached {source_name} index..."
        )

        with open(cache_file, "rb") as f:
            index = pickle.load(f)

        print(
            f"{source_name} index loaded."
        )

        return index

    print(
        f"\nBuilding {source_name} index..."
    )

    exact_name = defaultdict(list)
    name_token = defaultdict(list)
    address_token = defaultdict(list)
    number_index = defaultdict(list)
    address_pair = defaultdict(list)

    total = len(df)

    for idx, row in enumerate(
        df.itertuples(index=False)
    ):

        # ----------------------------------------------------
        # Exact name
        # ----------------------------------------------------

        if row.name_norm:

            exact_name[
                row.name_norm
            ].append(idx)

        # ----------------------------------------------------
        # Name tokens
        # ----------------------------------------------------

        for token in row.name_tokens:

            if (
                len(token) >= 3
                and name_freq[token] <= MAX_NAME_FREQ
            ):

                name_token[
                    token
                ].append(idx)

        # ----------------------------------------------------
        # Address tokens
        # ----------------------------------------------------

        pair_tokens = []

        for token in row.address_tokens:

            if (
                len(token) >= 4
                and address_freq[token] <= MAX_ADDRESS_FREQ
            ):

                address_token[
                    token
                ].append(idx)

            if (
                len(token) >= 4
                and address_freq[token] <= MAX_PAIR_TOKEN_FREQ
            ):

                pair_tokens.append(token)

        # ----------------------------------------------------
        # Numbers
        # ----------------------------------------------------

        for number in row.numbers:

            if (
                number_freq[number]
                <= MAX_NUMBER_FREQ
            ):

                number_index[
                    number
                ].append(idx)

        # ----------------------------------------------------
        # Address token pairs
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

                pair = (
                    pair_tokens[i],
                    pair_tokens[j]
                )

                address_pair[
                    pair
                ].append(idx)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (idx + 1) % 500_000 == 0
        ):

            print(
                f"{source_name}: "
                f"{idx + 1:,}/{total:,}"
            )

    index = {
        "exact_name": dict(exact_name),
        "name_token": dict(name_token),
        "address_token": dict(address_token),
        "number": dict(number_index),
        "address_pair": dict(address_pair)
    }

    print(
        f"Saving {source_name} index..."
    )

    with open(
        cache_file,
        "wb"
    ) as f:

        pickle.dump(
            index,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        f"{source_name} index saved."
    )

    return index


# ============================================================
# BUILD / LOAD INDEXES
# ============================================================

s2_index = build_index(
    s2,
    "S2"
)

s3_index = build_index(
    s3,
    "S3"
)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def get_candidates(
    row,
    index
):

    candidates = set()

    name_tokens = row.name_tokens
    address_tokens = row.address_tokens

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    if row.name_norm:

        candidates.update(
            index["exact_name"].get(
                row.name_norm,
                []
            )
        )

    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    for token in name_tokens:

        if (
            len(token) >= 3
            and name_freq[token]
            <= MAX_NAME_FREQ
        ):

            candidates.update(
                index["name_token"].get(
                    token,
                    []
                )
            )

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    for token in address_tokens:

        if (
            len(token) >= 4
            and address_freq[token]
            <= MAX_ADDRESS_FREQ
        ):

            candidates.update(
                index["address_token"].get(
                    token,
                    []
                )
            )

    # --------------------------------------------------------
    # Numbers
    # --------------------------------------------------------

    for number in row.numbers:

        if (
            number_freq[number]
            <= MAX_NUMBER_FREQ
        ):

            candidates.update(
                index["number"].get(
                    number,
                    []
                )
            )

    # --------------------------------------------------------
    # Address pairs
    # --------------------------------------------------------

    pair_tokens = sorted([
        token
        for token in address_tokens
        if (
            len(token) >= 4
            and address_freq[token]
            <= MAX_PAIR_TOKEN_FREQ
        )
    ])

    for i in range(
        len(pair_tokens)
    ):

        for j in range(
            i + 1,
            len(pair_tokens)
        ):

            pair = (
                pair_tokens[i],
                pair_tokens[j]
            )

            candidates.update(
                index["address_pair"].get(
                    pair,
                    []
                )
            )

    return candidates


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
# VALIDATION
# ============================================================

print(
    "\nValidating V7...\n"
)

total_true = 0
total_found = 0
total_candidates = 0
max_candidates = 0

missed_examples = []

s2_ids = s2["entity_id"].astype(str).tolist()
s3_ids = s3["entity_id"].astype(str).tolist()


for counter, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    c2 = get_candidates(
        row,
        s2_index
    )

    c3 = get_candidates(
        row,
        s3_index
    )

    candidate_count = (
        len(c2) + len(c3)
    )

    total_candidates += candidate_count

    max_candidates = max(
        max_candidates,
        candidate_count
    )

    # --------------------------------------------------------
    # Convert indexes -> entity IDs
    # --------------------------------------------------------

    candidate_ids = set()

    for idx in c2:

        candidate_ids.add(
            s2_ids[idx]
        )

    for idx in c3:

        candidate_ids.add(
            s3_ids[idx]
        )

    # --------------------------------------------------------
    # Ground truth
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
        true_ids
        & candidate_ids
    )

    missed = (
        true_ids
        - candidate_ids
    )

    total_true += len(true_ids)
    total_found += len(found)

    # --------------------------------------------------------
    # Store missed examples
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if counter % 500 == 0:

        recall = (
            total_found
            / max(total_true, 1)
        )

        avg_candidates = (
            total_candidates
            / counter
        )

        print(
            f"{counter:,}/{N:,} | "
            f"Recall: {recall:.6f} | "
            f"Avg candidates: "
            f"{avg_candidates:.2f}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

recall = (
    total_found
    / max(total_true, 1)
)

avg_candidates = (
    total_candidates
    / N
)

print(
    "\n"
    + "=" * 65
)

print(
    "V7 VALIDATION RESULT"
)

print(
    "=" * 65
)

print(
    f"S1 tested:           {N:,}"
)

print(
    f"True matches:        {total_true:,}"
)

print(
    f"Found:               {total_found:,}"
)

print(
    f"Missed:              "
    f"{total_true - total_found:,}"
)

print(
    f"Blocking recall:     {recall:.6f}"
)

print(
    f"Average candidates:  {avg_candidates:.2f}"
)

print(
    f"Maximum candidates:  {max_candidates:,}"
)

print(
    "=" * 65
)


# ============================================================
# MISSED DETAILS
# ============================================================

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


# ============================================================
# SAVE RESULT
# ============================================================

result_file = (
    "output/blocking_v7_validation.txt"
)

with open(
    result_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"S1 tested: {N:,}\n"
    )

    f.write(
        f"True matches: {total_true:,}\n"
    )

    f.write(
        f"Found: {total_found:,}\n"
    )

    f.write(
        f"Missed: "
        f"{total_true - total_found:,}\n"
    )

    f.write(
        f"Recall: {recall:.6f}\n"
    )

    f.write(
        f"Average candidates: "
        f"{avg_candidates:.2f}\n"
    )

    f.write(
        f"Maximum candidates: "
        f"{max_candidates:,}\n"
    )

    if missed_examples:

        f.write(
            "\nMISSED EXAMPLES:\n"
        )

        for entity_id, missed in missed_examples:

            f.write(
                f"{entity_id} -> "
                f"{sorted(missed)}\n"
            )

    else:

        f.write(
            "\nZERO MISSED MATCHES\n"
        )

print(
    f"\nResult saved to: "
    f"{result_file}"
)