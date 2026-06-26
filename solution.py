import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def first_existing(possible_names):
    """Return the first filename that exists in the current folder."""
    for name in possible_names:
        if os.path.exists(name):
            return name
    raise FileNotFoundError(f"None of these files were found: {possible_names}")


train_file = first_existing(["train_3.csv", "train.csv"])
test_file = first_existing(["test_3.csv", "test.csv"])
sample_file = first_existing(["samp_subm_3.csv", "sample_subm_3.csv", "sample_submission.csv"])

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
sample_submission = pd.read_csv(sample_file)

print("Train file:", train_file, train.shape)
print("Test file:", test_file, test.shape)
print("Sample submission file:", sample_file, sample_submission.shape)


id_col = sample_submission.columns[0]
target_col = sample_submission.columns[1]

print("ID column:", id_col)
print("Target column:", target_col)

if target_col not in train.columns:
    raise ValueError(f"Target column '{target_col}' was not found in training data.")


drop_cols = [target_col]
for col in [id_col, "User_id"]:
    if col in train.columns:
        drop_cols.append(col)

X = train.drop(columns=drop_cols)
y = train[target_col]

test_features = test.drop(columns=[col for col in [id_col, "User_id"] if col in test.columns])

feature_cols = [col for col in X.columns if col in test_features.columns]
X = X[feature_cols]
test_features = test_features[feature_cols]

print("Features used:", feature_cols)


model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("regressor", LinearRegression())
])


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model.fit(X_train, y_train)
valid_predictions = model.predict(X_valid)

mae = mean_absolute_error(y_valid, valid_predictions)
rmse = np.sqrt(mean_squared_error(y_valid, valid_predictions))
r2 = r2_score(y_valid, valid_predictions)

print("\nValidation Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = np.sqrt(-cross_val_score(
    model, X, y, scoring="neg_mean_squared_error", cv=cv
))

print("\n5-Fold CV RMSE Scores:", cv_rmse_scores)
print("Average CV RMSE:", cv_rmse_scores.mean())


model.fit(X, y)


test_predictions = model.predict(test_features)


submission = sample_submission.copy()
submission[id_col] = test[id_col].values
submission[target_col] = test_predictions

submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully!")
print(submission.head(10))