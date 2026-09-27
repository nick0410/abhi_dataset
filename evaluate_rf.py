import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import fbeta_score


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("candidate_features.csv")

features = [
    "name_exact",
    "name_similarity",
    "name_jaccard",
    "address_similarity",
    "address_jaccard",
    "number_overlap",
    "country_match"
]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        df,
        df["label"],
        groups=df["source1_entity_id"]
    )
)

train = df.iloc[train_idx]
val = df.iloc[val_idx].copy()


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(
    train[features],
    train["label"]
)


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

val["probability"] = model.predict_proba(
    val[features]
)[:, 1]


# ============================================================
# MACRO F0.5 THRESHOLD SWEEP
# ============================================================

print("=" * 70)
print("MACRO F0.5 THRESHOLD SWEEP")
print("=" * 70)

thresholds = [
    0.755,
    0.756,
    0.757,
    0.758,
    0.759,
    0.760,
    0.761,
    0.762,
    0.763,
    0.764,
    0.765,
    0.766,
    0.767,
    0.768,
    0.769,
    0.770
]

results = []

for threshold in thresholds:

    scores = []

    for s1_id, group in val.groupby("source1_entity_id"):

        y_true = group["label"].values

        y_pred = (
            group["probability"].values >= threshold
        ).astype(int)

        score = fbeta_score(
            y_true,
            y_pred,
            beta=0.5,
            zero_division=0
        )

        scores.append(score)

    macro_f05 = sum(scores) / len(scores)

    results.append(
        (threshold, macro_f05)
    )

    print(
        f"Threshold {threshold:.3f} | "
        f"Macro F0.5 = {macro_f05:.4f}"
    )


# ============================================================
# BEST THRESHOLD
# ============================================================

best_threshold, best_score = max(
    results,
    key=lambda x: x[1]
)

print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Macro F0.5 : {best_score:.4f}"
)


# ============================================================
# FINAL VALIDATION STATISTICS
# ============================================================

predicted = val[
    val["probability"] >= best_threshold
]

true_pairs = predicted["label"].sum()

false_pairs = (
    predicted["label"] == 0
).sum()

print("\n" + "=" * 70)
print("RANDOM FOREST VALIDATION")
print("=" * 70)

print(
    "Predicted pairs :",
    len(predicted)
)

print(
    "True pairs      :",
    true_pairs
)

print(
    "False pairs     :",
    false_pairs
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature importance:")

for name, importance in sorted(
    zip(
        features,
        model.feature_importances_
    ),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{name:22} {importance:.4f}"
    )


    # ============================================================
# MISSED TRUE MATCHES
# ============================================================

print("\n" + "=" * 70)
print("MISSED TRUE MATCHES")
print("=" * 70)

missed = val[
    (val["label"] == 1) &
    (val["probability"] < best_threshold)
].copy()

print("Total missed true pairs :", len(missed))
print(
    "S1 entities affected    :",
    missed["source1_entity_id"].nunique()
)

print("\nTop missed pairs:")

cols = [
    "source1_entity_id",
    "candidate_entity_id",
    "name_similarity",
    "name_jaccard",
    "address_similarity",
    "address_jaccard",
    "number_overlap",
    "probability"
]

print(
    missed.sort_values(
        "probability",
        ascending=False
    )[cols].head(30).to_string(index=False)
)

# ============================================================
# FALSE POSITIVES
# ============================================================

print("\n" + "=" * 70)
print("FALSE POSITIVES")
print("=" * 70)

false_positive = val[
    (val["label"] == 0) &
    (val["probability"] >= best_threshold)
].copy()

print("Total false positives :", len(false_positive))

print("\nFalse positive pairs:")

print(
    false_positive.sort_values(
        "probability",
        ascending=False
    )[cols].to_string(index=False)
)


# ============================================================
# CANDIDATES MATCHED TO MULTIPLE S1s
# ============================================================

print("\n" + "=" * 70)
print("MULTI-S1 CANDIDATE ANALYSIS")
print("=" * 70)

predicted = val[
    val["probability"] >= best_threshold
].copy()

candidate_counts = (
    predicted
    .groupby("candidate_entity_id")["source1_entity_id"]
    .nunique()
    .sort_values(ascending=False)
)

multi = candidate_counts[
    candidate_counts > 1
]

print(
    "Candidates matched to multiple S1s :",
    len(multi)
)

print("\nTop multi-S1 candidates:")

print(
    multi.head(20).to_string()
)


# ============================================================
# AFFECTED S1 ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("AFFECTED S1 ANALYSIS")
print("=" * 70)

affected_s1 = missed["source1_entity_id"].unique()

for s1_id in affected_s1:

    group = val[
        val["source1_entity_id"] == s1_id
    ].copy()

    print("\nS1:", s1_id)

    print(
        group.sort_values(
            "probability",
            ascending=False
        )[[
            "candidate_entity_id",
            "label",
            "name_similarity",
            "name_jaccard",
            "address_similarity",
            "address_jaccard",
            "number_overlap",
            "probability"
        ]].head(10).to_string(index=False)
    )

# ============================================================
# DONE
# ============================================================

print("\nDONE")