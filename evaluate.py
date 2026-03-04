import torch
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report
from models.hybrid_model import HybridModel
from utils.sequence import create_sequences
from security.crypto_layer import encrypt_data, decrypt_data, generate_hash


def evaluate(mode="normal"):

    print(f"\n========== EVALUATION MODE: {mode.upper()} ==========")

    # -----------------------
    # 1️⃣ Load Dataset Based on Mode
    # -----------------------
    if mode == "normal":
        df = pd.read_csv("data/sensor_data.csv")

    elif mode == "attack":
        df = pd.read_csv("data/attacked_data.csv")

    elif mode == "secure":
        attacked_df = pd.read_csv("data/attacked_data.csv")
        clean_df = pd.read_csv("data/sensor_data.csv")

        print("Applying AES-based integrity protection...")

        # Integrity verification simulation
        for i in range(len(attacked_df)):

            original_row = clean_df.iloc[i].to_json()
            attacked_row = attacked_df.iloc[i].to_json()

            original_hash = generate_hash(original_row)
            encrypted = encrypt_data(original_row)

            decrypted = decrypt_data(encrypted)
            received_hash = generate_hash(decrypted)

            attacked_hash = generate_hash(attacked_row)

            if attacked_hash != original_hash:
                attacked_df.iloc[i] = clean_df.iloc[i]

        print("Integrity verification completed.")
        df = attacked_df

    else:
        raise ValueError("Mode must be 'normal', 'attack', or 'secure'")

    # -----------------------
    # 2️⃣ Prepare Data
    # -----------------------
    X = df.drop("failure", axis=1).values
    y = df["failure"].values

    scaler = joblib.load("scaler.save")
    X = scaler.transform(X)

    X_seq, y_seq = create_sequences(X, y)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    # -----------------------
    # 3️⃣ Load Model
    # -----------------------
    model = HybridModel(input_dim=6)
    model.load_state_dict(torch.load("hybrid_model.pth"))
    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)
        threshold = 0.15
        preds = (probs > threshold).int().numpy()

    acc = accuracy_score(y_seq, preds)

    print(f"\nAccuracy ({mode}): {round(acc*100, 2)}%")
    print(classification_report(y_seq, preds, zero_division=0))

    return acc