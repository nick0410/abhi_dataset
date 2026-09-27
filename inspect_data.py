import pandas as pd

files = {
    "S1": "dataset/train/train_source1.tsv",
    "S2": "dataset/train/train_source2.tsv",
    "S3": "dataset/train/train_source3.tsv",
    "GT": "dataset/train/train_ground_truth.tsv"
}

for name, path in files.items():
    df = pd.read_csv(path, sep="\t", nrows=1000)

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))


# ---------------------------------------------------------
# Ground truth analysis
# ---------------------------------------------------------

gt = pd.read_csv(
    files["GT"],
    sep="\t",
    nrows=1000
)

print("\n" + "=" * 80)
print("GROUND TRUTH ANALYSIS")
print("=" * 80)

gt["match_count"] = gt["matched_entity_ids"].fillna("").apply(
    lambda x: 0 if not x.strip() else len(x.split(","))
)

print("Total GT rows:", len(gt))
print("Singletons / no-match:", (gt["match_count"] == 0).sum())
print("With matches:", (gt["match_count"] > 0).sum())

print("\nMatch count distribution:")
print(gt["match_count"].value_counts().sort_index())

print("\nExample ground truth:")
print(gt.head(20).to_string(index=False))

print("\nDONE")