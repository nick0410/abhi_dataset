import pandas as pd
import os

files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv"
]

for f in files:
    out = "sample_dataset/" + f
    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = pd.read_csv(f, sep="\t", nrows=1000)
    df.to_csv(out, sep="\t", index=False)

    print(f"{f} -> {len(df)} rows")

print("\nDONE!")