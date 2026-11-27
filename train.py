
import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor 


seed = 67 #random
embeddingthing = 8
EPOCHS = 100
BATCH_SIZE = 64

os.makedirs("models", exist_ok=True) #make sure models exist

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cpu")

data = pd.read_csv("train.csv")

X = data.drop(
    columns=["project_id", "hours_spent", "log_hours", "hours_bucket"], #no log hours
    errors="ignore"
)
y = np.log1p(data["hours_spent"])


X = X.replace([np.inf, -np.inf], np.nan)

cat_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

num_features  = X.select_dtypes(
    include=np.number
).columns.tolist()

medians = {}#whoops

for col in num_features:
    median = X[col].median()

    if pd.isna(median):
        median = 0

    medians[col] = median
    X[col] = X[col].fillna(median)

for col in cat_features:
    X[col] = X[col].fillna("Unknown")

X = pd.get_dummies(
    X,
    columns=cat_features
)

feature_columns = X.columns.tolist()

scaler = StandardScaler()

X[num_features] =  scaler.fit_transform(
    X[num_features]
)

X = X.astype(np.float32)

print("Input shape:", X.shape)

joblib.dump(
    {
        "cat_features": cat_features,
        "num_features": num_features,
        "medians": medians,
        "feature_columns": feature_columns,
        "scaler": scaler
    },
    "models/preprocessing.pkl"
)

class Autoencoder(nn.Module):

    def __init__(self, input_dim, embedding_dim=8):
        super().__init__()

        self.encoder =nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction


X_tensor= torch.tensor(
    X.values,
    dtype=torch.float32
)

loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

autoencoder  = Autoencoder(
    input_dim=X.shape[1],
    embedding_dim=embeddingthing
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    autoencoder.parameters(),
    lr=0.001
)


print("\nTraini ng autoencoder...")

for epoch in range(EPOCHS):

    autoencoder.train()
    total_loss = 0

    for ( batch,) in loader:

        batch = batch.to(device)

        reconstructed = autoencoder(batch)

        loss = criterion(reconstructed, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss: {total_loss / len(loader):.6f}"
        )

torch.save(
    {
        "state_dict": autoencoder.state_dict(),
        "input_dim": X.shape[1],
        "embedding_dim": embeddingthing
    },
    "models/autoencoder.pth"
)

autoencoder.eval()

with torch.no_grad():

    embeddings = autoencoder.encoder(
        X_tensor.to(device)
    ).cpu().numpy()

print("\nEmbedding shape:", embeddings.shape)


print("\nTraining CatBoost...")

gbdt = CatBoostRegressor(
    iterations=1400,
    depth=6,
    learning_rate=0.03,
    loss_function="RMSE",
    random_seed=seed,
    verbose=100
)

gbdt.fit(
    embeddings,
    y
 )

gbdt.save_model("models/catboost.cbm")

print("Projects:", len(X))
print("Original features:", X.shape[1])
print("Embedding dimensions:", embeddings.shape[1])
print("Saved autoencoder")
print("Saved CatBoost")
print("Saved preprocessing")
