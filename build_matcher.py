import pandas as pd

df = pd.read_csv("candidate_features.csv")


def score(row):
    return (
        0.25 * row["name_similarity"]
        + 0.15 * row["name_jaccard"]
        + 0.30 * row["address_similarity"]
        + 0.20 * row["address_jaccard"]
        + 0.10 * row["number_overlap"]
    )


df["score"] = df.apply(score, axis=1)

THRESHOLD = 0.485

selected = df[df["score"] >= THRESHOLD].copy()

print("=" * 70)
print("MATCHER")
print("=" * 70)

print("Threshold       :", THRESHOLD)
print("Total candidates:", len(df))
print("Selected pairs  :", len(selected))
print("True matches    :", selected["label"].sum())
print("False matches   :", (selected["label"] == 0).sum())

print("\nS1 coverage:")

coverage = (
    selected
    .groupby("source1_entity_id")
    .size()
)

print("S1 with matches :", len(coverage))
print("Total S1        :", df["source1_entity_id"].nunique())

print("\nDONE")