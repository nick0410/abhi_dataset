import pandas as pd


# =========================
# Load data
# =========================

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# =========================
# Create lookup dictionaries
# =========================

s1_lookup = s1.set_index("entity_id").to_dict("index")
s2_lookup = s2.set_index("entity_id").to_dict("index")
s3_lookup = s3.set_index("entity_id").to_dict("index")


# =========================
# Print 10 real examples
# =========================

examples = gt[gt["matched_entity_ids"].notna()].head(10)

for _, row in examples.iterrows():

    s1_id = row["source1_entity_id"]
    matched_ids = row["matched_entity_ids"].split(",")

    print("\n")
    print("=" * 100)
    print("SOURCE 1")
    print("=" * 100)

    r1 = s1_lookup[s1_id]

    print("ID      :", s1_id)
    print("NAME    :", r1["business_name"])
    print("ADDRESS :", r1["business_address"])
    print("COUNTRY :", r1["country"])

    print("\nMATCHED RECORDS")
    print("-" * 100)

    for mid in matched_ids:

        if mid.startswith("S2-"):
            r = s2_lookup[mid]
        else:
            r = s3_lookup[mid]

        print("\nID      :", mid)
        print("NAME    :", r["business_name"])
        print("ADDRESS :", r["business_address"])
        print("COUNTRY :", r["country"])

print("\nDONE")