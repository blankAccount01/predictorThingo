
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from catboost import CatBoostRegressor


DEVICE = torch.device("cpu") # no gpu :(

class Autoencoder(nn.Module): #just standard

    def __init__(self, input_dim, embedding_dim=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )#outputs 8 dimensions

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        ) #put it back to validate

    def forward(self, x): #just runs everything
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction

test_data = pd.read_csv("test.csv")

project_ids = test_data["project_id"].copy()

X = test_data.drop(columns=["project_id"]) #data


preprocessing = joblib.load(
    "models/preprocessing.pkl"
) #remove missing values n stuff

cat_features = preprocessing["cat_features"]
num_features = preprocessing["num_features"]
medians = preprocessing["medians"] #fills missing values with the median so training still works
feature_columns = preprocessing["feature_columns"]
scaler = preprocessing["scaler"] #don't know what this tbh

X = X.replace([np.inf, -np.inf], np.nan)

for col in num_features:
    X[col] = X[col].fillna(medians[col])

for col in cat_features:
    X[col] = X[col].fillna("Unknown")

X = pd.get_dummies(
    X,
    columns=cat_features
) #categorical to binary?

X = X.reindex(
    columns=feature_columns,
    fill_value=0
) #fixes things

X[num_features] = scaler.transform(
    X[num_features]
)#standardises normal dist

X = X.astype(np.float32) 

print("Test shape:", X.shape)

checkpoint = torch.load(
    "models/autoencoder.pth",
    map_location=DEVICE #important, to run on cpu
)

autoencoder = Autoencoder(
    input_dim=checkpoint["input_dim"],
    embedding_dim=checkpoint["embedding_dim"]
).to(DEVICE)

autoencoder.load_state_dict(
    checkpoint["state_dict"]
)

autoencoder.eval()


X_tensor = torch.tensor(
    X.values,
    dtype=torch.float32
)

with torch.no_grad():

    embeddings = autoencoder.encoder(
        X_tensor.to(DEVICE)
    ).cpu().numpy()

print("Embedding shape:", embeddings.shape)

gbdt = CatBoostRegressor()

gbdt.load_model(
    "models/catboost.cbm"
)

predictions_log = gbdt.predict(embeddings)

predictions = np.expm1(
    np.maximum(predictions_log, 0)
)


submission = pd.DataFrame({
    "project_id": project_ids,
    "hours_spent": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print(submission.head())
print("\nSaved submission.csv")
