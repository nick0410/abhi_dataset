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

true_df = df[df["label"] == 1].copy()

missed = true_df[true_df["score"] < 0.485].sort_values(
    ["source1_entity_id", "score"],
    ascending=[True, False]
)

print("\n" + "=" * 100)
print("MISSED TRUE MATCHES")
print("=" * 100)

print(
    missed[
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

print("\nTotal missed true pairs:", len(missed))
print("DONE")