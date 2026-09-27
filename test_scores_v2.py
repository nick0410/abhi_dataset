import pandas as pd

df = pd.read_csv("candidate_features.csv")


def calculate_score(row):
    name = (
        0.25 * row["name_similarity"]
        + 0.15 * row["name_jaccard"]
    )

    address = (
        0.30 * row["address_similarity"]
        + 0.20 * row["address_jaccard"]
    )

    number = 0.10 * row["number_overlap"]

    return name + address + number


df["score"] = df.apply(calculate_score, axis=1)

total_true = df["label"].sum()

best = None

for threshold in [x / 1000 for x in range(400, 701, 5)]:

    selected = df[df["score"] >= threshold]

    tp = selected["label"].sum()
    fp = len(selected) - tp

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / total_true if total_true else 0

    f05 = (
        1.25 * precision * recall
        / (0.25 * precision + recall)
        if precision + recall > 0
        else 0
    )

    if best is None or f05 > best["f05"]:
        best = {
            "threshold": threshold,
            "candidates": len(selected),
            "tp": tp,
            "fp": fp,
            "precision": precision,
            "recall": recall,
            "f05": f05
        }


print("\n" + "=" * 60)
print("V2 SCORE")
print("=" * 60)

for key, value in best.items():
    print(f"{key:12}: {value}")