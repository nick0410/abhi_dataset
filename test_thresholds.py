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


print("\n" + "=" * 75)
print("THRESHOLD ANALYSIS")
print("=" * 75)

for threshold in [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]:

    selected = df[df["score"] >= threshold]

    tp = selected["label"].sum()
    fp = len(selected) - tp

    total_true = df["label"].sum()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / total_true if total_true else 0

    f05 = (
        1.25 * precision * recall
        / (0.25 * precision + recall)
        if precision + recall > 0
        else 0
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"Candidates={len(selected):7} | "
        f"TP={tp:5} | "
        f"FP={fp:7} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F0.5={f05:.4f}"
    )


print("\nDONE")