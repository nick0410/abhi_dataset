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

selected = df[df["score"] >= 0.485]

# Ground-truth matches by S1
true_counts = (
    df[df["label"] == 1]
    .groupby("source1_entity_id")
    .size()
)

# Predicted matches by S1
pred_counts = (
    selected
    .groupby("source1_entity_id")
    .size()
)

all_s1 = df["source1_entity_id"].unique()

missed_entities = []

for s1_id in all_s1:

    true_count = true_counts.get(s1_id, 0)
    pred_count = pred_counts.get(s1_id, 0)

    if true_count > 0 and pred_count == 0:
        missed_entities.append(
            (s1_id, true_count)
        )

print("\n" + "=" * 70)
print("MISSED TRUE-MATCH ENTITIES")
print("=" * 70)

print("Entities with true matches but zero prediction:",
      len(missed_entities))

print("Total true matches missed:",
      sum(x[1] for x in missed_entities))

print("\nFirst 30:")

for x in missed_entities[:30]:
    print(x)

print("\nDONE")