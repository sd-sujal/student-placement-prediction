import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/student_data.csv")

# Basic info
print("Shape:", df.shape)

print("\nPlacement Distribution:")
print(df["placed"].value_counts())

print("\nCorrelation with placement:")
print(df.corr(numeric_only=True)["placed"].sort_values(ascending=False))

# Sanity checks (IMPORTANT)
assert df.isnull().sum().sum() == 0, "Dataset contains null values"
assert df["cgpa"].between(5.0, 10.0).all(), "CGPA out of range"
assert (df.loc[df["placed"] == 0, "salary_lpa"] == 0).all(), "Unplaced students have salary"

# Save processed dataset (no transformations yet)
df.to_csv("data/processed/student_data_clean.csv", index=False)

print("\nProcessed data saved to data/processed/student_data_clean.csv")
