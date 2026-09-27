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

total_true = df["label"].sum()

print("\nTHRESHOLD DETAIL")
print("=" * 75)

for threshold in [
    0.450,
    0.455,
    0.460,
    0.465,
    0.470,
    0.475,
    0.480,
    0.485,
    0.490,
    0.495,
    0.500,
    0.505,
    0.510,
    0.515,
    0.520,
    0.525,
    0.530
]:

    selected = df[df["score"] >= threshold]

    tp = selected["label"].sum()
    fp = len(selected) - tp

    precision = tp / len(selected) if len(selected) else 0
    recall = tp / total_true if total_true else 0

    f05 = (
        1.25 * precision * recall
        / (0.25 * precision + recall)
        if precision + recall > 0
        else 0
    )

    print(
        f"{threshold:.3f} | "
        f"cand={len(selected):4} | "
        f"TP={tp:4} | "
        f"FP={fp:3} | "
        f"P={precision:.4f} | "
        f"R={recall:.4f} | "
        f"F0.5={f05:.4f}"
    )