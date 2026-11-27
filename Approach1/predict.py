import pandas as pd
import numpy as np

from catboost import CatBoostRegressor

test_data = pd.read_csv("test.csv")
project_ids = test_data["project_id"].copy()

seeds = [23, 42, 100]

models = []

for seed in seeds:
    model = CatBoostRegressor()
    model.load_model(f"catboost_seed_{seed}.cbm")
    models.append(model)

feature_columns = models[0].feature_names_

qualitative_features = [
    feature_columns[i]
    for i in models[0].get_cat_feature_indices()
]

input_test = test_data.drop(columns=[
    "project_id",
    "hours_spent",
    "log_hours",
    "hours_bucket"
], errors="ignore")

input_test = input_test.reindex(columns=feature_columns)

for col in qualitative_features:
    input_test[col] = input_test[col].fillna("Unknown").astype(str)

all_predictions = []

for model in models:
    predictions = model.predict(input_test)
    all_predictions.append(predictions)

predictions = np.mean(all_predictions, axis=0)

predictions = np.expm1(predictions)
predictions = np.maximum(predictions, 0)

output = pd.DataFrame({
    "project_id": project_ids,
    "hours_spent": predictions
})

output.to_csv("predictions.csv", index=False)

print("Predictions generated:", len(output))
print("\nPrediction statistics:")
print(output["hours_spent"].describe())
