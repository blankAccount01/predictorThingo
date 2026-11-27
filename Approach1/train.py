import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error as RMSE

data = pd.read_csv("train.csv")

target = np.log1p(data["hours_spent"])

input_data = data.drop(columns=[
    "project_id",
    "hours_spent",
    "log_hours",
    "hours_bucket"
], errors="ignore")

qualitative_features = input_data.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

for col in qualitative_features:
    input_data[col] = input_data[col].fillna("Unknown").astype(str)

input_training, input_validation, output_training, output_validation = train_test_split(
    input_data,
    target,
    test_size=0.2,
    random_state=42
)

magicModel = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    leaf_estimation_method="Gradient",
    leaf_estimation_iterations=5,
    random_seed=23,
    verbose=100,
    early_stopping_rounds=200,
    allow_writing_files=False,
    thread_count=-1
)

magicModel.fit(
    input_training,
    output_training,
    cat_features=qualitative_features,
    eval_set=(input_validation, output_validation)
)

best_iteration = max(magicModel.get_best_iteration() + 1, 1)

predictions = np.expm1(magicModel.predict(input_validation))
predictions = np.maximum(predictions, 0)

true_hours = np.expm1(output_validation)
score = RMSE(true_hours, predictions)

print("\nValidation score:", score)
print("Best iteration:", best_iteration)

seeds = [23, 42, 100]

for seed in seeds:

    print(f"\nTraining final model with seed: {seed}")

    finalModel = CatBoostRegressor(
        iterations=best_iteration,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        leaf_estimation_method="Gradient",
        leaf_estimation_iterations=5,
        random_seed=seed,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1
    )

    finalModel.fit(
        input_data,
        target,
        cat_features=qualitative_features
    )

    finalModel.save_model(f"catboost_seed_{seed}.cbm")

    print(f"Saved catboost_seed_{seed}.cbm")

print("\nAll models trained and saved successfully!")
