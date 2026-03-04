import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from models.hybrid_model import HybridModel
from utils.sequence import create_sequences


def train_model():

    # -----------------------
    # 0️⃣ Device Setup
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------
    # 1️⃣ Load Dataset
    # -----------------------
    df = pd.read_csv("data/sensor_data.csv")

    X = df.drop("failure", axis=1).values
    y = df["failure"].values

    # -----------------------
    # 2️⃣ Normalize Features
    # -----------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.save")

    # -----------------------
    # 3️⃣ Create Sequences
    # -----------------------
    X_seq, y_seq = create_sequences(X, y, seq_length=30)  
    # ↓ Reduced from 50 to 30 for faster training

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=128,   # Increased batch size
        shuffle=True,
        num_workers=2,    # Faster loading
        pin_memory=True
    )

    # -----------------------
    # 4️⃣ Model Setup
    # -----------------------
    model = HybridModel(input_dim=6).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -----------------------
    # 5️⃣ Training Loop
    # -----------------------
    epochs = 10

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            output = model(xb)

            loss = criterion(output, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    # -----------------------
    # 6️⃣ Save Model
    # -----------------------
    torch.save(model.state_dict(), "hybrid_model.pth")

    print("Training complete. Model saved.")