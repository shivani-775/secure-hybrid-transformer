import numpy as np

def create_sequences(X, y, seq_length=70):
    Xs, ys = [], []
    for i in range(len(X)-seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)