import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, fbeta_score

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

# Split by S1 entity, not individual candidate pairs.
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
val = df.iloc[val_idx]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(train[features], train["label"])

probabilities = model.predict_proba(
    val[features]
)[:, 1]

print("\nTHRESHOLD | PRECISION | RECALL | F0.5")

for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:

    pred = probabilities >= threshold

    print(
        f"{threshold:.2f} | "
        f"{precision_score(val['label'], pred, zero_division=0):.4f} | "
        f"{recall_score(val['label'], pred):.4f} | "
        f"{fbeta_score(val['label'], pred, beta=0.5):.4f}"
    )

print("\nVALIDATION S1:", val["source1_entity_id"].nunique())
print("DONE")