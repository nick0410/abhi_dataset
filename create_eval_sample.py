import pandas as pd
import os


# ============================================================
# LOAD FULL TRAIN DATA
# ============================================================

print("Loading training data...")

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

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))
print("GT:", len(gt))


# ============================================================
# SELECT 1000 S1 IDs FROM GROUND TRUTH
# ============================================================

selected_gt = gt.head(1000).copy()

selected_s1_ids = set(
    selected_gt["source1_entity_id"]
)


# ============================================================
# GET CORRESPONDING S1 RECORDS
# ============================================================

sample_s1 = s1[
    s1["entity_id"].isin(selected_s1_ids)
].copy()


# ============================================================
# GET ALL TRUE S2/S3 IDS
# ============================================================

matched_s2_ids = set()
matched_s3_ids = set()

for value in selected_gt["matched_entity_ids"].dropna():

    for entity_id in str(value).split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            matched_s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            matched_s3_ids.add(entity_id)


# ============================================================
# GET CORRESPONDING S2/S3 RECORDS
# ============================================================

sample_s2 = s2[
    s2["entity_id"].isin(matched_s2_ids)
].copy()

sample_s3 = s3[
    s3["entity_id"].isin(matched_s3_ids)
].copy()


# ============================================================
# IMPORTANT:
# Add extra random S2/S3 records
# ============================================================

# We don't want a toy dataset where every record is guaranteed
# to be a match. Add unrelated records as distractors.

extra_s2 = s2[
    ~s2["entity_id"].isin(matched_s2_ids)
].head(1000)

extra_s3 = s3[
    ~s3["entity_id"].isin(matched_s3_ids)
].head(1000)


sample_s2 = pd.concat(
    [sample_s2, extra_s2]
).drop_duplicates(
    subset=["entity_id"]
)

sample_s3 = pd.concat(
    [sample_s3, extra_s3]
).drop_duplicates(
    subset=["entity_id"]
)


# ============================================================
# CREATE FOLDERS
# ============================================================

folders = [
    "sample_dataset/dataset/train"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# SAVE
# ============================================================

sample_s1.to_csv(
    "sample_dataset/dataset/train/train_source1.tsv",
    sep="\t",
    index=False
)

sample_s2.to_csv(
    "sample_dataset/dataset/train/train_source2.tsv",
    sep="\t",
    index=False
)

sample_s3.to_csv(
    "sample_dataset/dataset/train/train_source3.tsv",
    sep="\t",
    index=False
)

selected_gt.to_csv(
    "sample_dataset/dataset/train/train_ground_truth.tsv",
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE CREATED")
print("=" * 70)

print("S1 rows:", len(sample_s1))
print("S2 rows:", len(sample_s2))
print("S3 rows:", len(sample_s3))
print("GT rows:", len(selected_gt))

print("\nTrue S2 IDs:", len(matched_s2_ids))
print("True S3 IDs:", len(matched_s3_ids))

print("\nDone!")