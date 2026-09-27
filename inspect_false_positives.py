import pandas as pd

df = pd.read_csv("candidate_features.csv")


def score(row):
    return (
        0.30 * row["name_similarity"]
        + 0.20 * row["name_jaccard"]
        + 0.25 * row["address_similarity"]
        + 0.15 * row["address_jaccard"]
        + 0.10 * row["number_overlap"]
    )


df["score"] = df.apply(score, axis=1)

fp = df[
    (df["score"] >= 0.50) &
    (df["label"] == 0)
].sort_values(
    "score",
    ascending=False
)

print("\nFALSE POSITIVES:", len(fp))
print("=" * 100)

print(
    fp[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "score",
            "name_similarity",
            "name_jaccard",
            "address_similarity",
            "address_jaccard",
            "number_overlap",
            "country_match"
        ]
    ].to_string(index=False)
)

print("\nDONE")