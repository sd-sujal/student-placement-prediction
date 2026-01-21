import pandas as pd

df = pd.read_csv("data/processed/student_data_clean.csv")
print(df.columns)
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load processed dataset
df = pd.read_csv("data/processed/student_data_clean.csv")

print("\nColumns in dataset:")
print(df.columns.tolist())

# 2. Automatically detect target column
possible_targets = [
    "PlacementStatus",
    "placed",
    "placement",
    "Placement",
    "is_placed",
    "status"
]

target_column = None
for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError(
        "No placement target column found.\n"
        f"Available columns: {df.columns.tolist()}"
    )

print(f"\nUsing target column: {target_column}")

# 3. Split features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nTarget value counts:")
print(y.value_counts())

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Performance")
print("-----------------")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save trained model

with open("models/placement_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nPlacement prediction model saved at:")
print("models/placement_model.pkl")
