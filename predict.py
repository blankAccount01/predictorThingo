
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error as RMSE

data_path = "train.csv"
data = pd.read_csv(data_path)

test_data = pd.read_csv("test.csv")
project_ids = test_data["project_id"].copy()

target_thing = np.log1p(data["hours_spent"])

input_data = data.drop(columns=[
    "project_id",
    "hours_spent",
    "log_hours",
    "hours_bucket"
])

input_test = test_data.drop(columns=[
    "project_id",
    "hours_spent",
    "log_hours",
    "hours_bucket"
], errors="ignore")

input_test = input_test.reindex(columns=input_data.columns)


qualitative_features = input_data.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

for col in qualitative_features:
    input_data[col] = input_data[col].fillna("Unknown").astype(str)
    input_test[col] = input_test[col].fillna("Unknown").astype(str)


input_training, input_validation, output_training, output_validation = train_test_split(
    input_data,
    target_thing,
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

best_iteration = magicModel.get_best_iteration() + 1
best_iteration = max(best_iteration, 1)

print("\nBest iteration:", best_iteration)

predictionsG = np.expm1(magicModel.predict(input_validation))
predictionsG = np.maximum(predictionsG, 0)

true_hours = np.expm1(output_validation)

scoreG = RMSE(true_hours, predictionsG)

print("Validation score:", scoreG)

seeds = [23, 42, 100]

all_predictions = []

for seed in seeds:

    print("\nTraining final model with seed:", seed)

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
        target_thing,
        cat_features=qualitative_features
    )

    test_predictions = finalModel.predict(input_test)

    all_predictions.append(test_predictions)


# Average in log space
predictions = np.mean(all_predictions, axis=0)

predictions = np.expm1(predictions)
predictions = np.maximum(predictions, 0)


output = pd.DataFrame({
    "project_id": project_ids,
    "hours_spent": predictions
})

output.to_csv("predictions.csv", index=False)

print("Validation score:", scoreG)
print("Best iteration:", best_iteration)
print("Models trained:", len(seeds))
print("Predictions generated:", len(output))

print("\nPrediction statistics:")
print(output["hours_spent"].describe())
