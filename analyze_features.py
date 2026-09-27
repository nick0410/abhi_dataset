import pandas as pd

df = pd.read_csv("candidate_features.csv")

print("\n" + "=" * 70)
print("TRUE vs FALSE FEATURE ANALYSIS")
print("=" * 70)

features = [
    "name_exact",
    "name_similarity",
    "name_jaccard",
    "address_similarity",
    "address_jaccard",
    "number_overlap",
    "country_match"
]

for f in features:

    true_mean = df.loc[df["label"] == 1, f].mean()
    false_mean = df.loc[df["label"] == 0, f].mean()

    print(
        f"{f:22} "
        f"TRUE={true_mean:.4f}   "
        f"FALSE={false_mean:.4f}"
    )


print("\n" + "=" * 70)
print("TRUE MATCH FEATURE DISTRIBUTION")
print("=" * 70)

print(
    df[df["label"] == 1][features]
    .describe()
    .T[
        ["min", "25%", "50%", "75%", "max"]
    ]
)


print("\nDONE")