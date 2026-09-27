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

total_true = df["label"].sum()

best = None

for threshold in [x / 1000 for x in range(400, 601, 5)]:

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

    result = (
        f05,
        threshold,
        len(selected),
        tp,
        fp,
        precision,
        recall
    )

    if best is None or f05 > best[0]:
        best = result


print("\nBEST THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best[1]:.3f}\n"
    f"Candidates: {best[2]}\n"
    f"TP        : {best[3]}\n"
    f"FP        : {best[4]}\n"
    f"Precision : {best[5]:.4f}\n"
    f"Recall    : {best[6]:.4f}\n"
    f"F0.5      : {best[0]:.4f}"
)