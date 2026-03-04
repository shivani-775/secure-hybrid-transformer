import numpy as np
import pandas as pd

def generate_data(n=65000):
    np.random.seed(42)

    data = pd.DataFrame({
        "voltage": np.random.normal(230, 5, n),
        "current": np.random.normal(10, 2, n),
        "temperature": np.random.normal(60, 10, n),
        "vibration": np.random.normal(5, 1, n),
        "load": np.random.normal(75, 10, n),
        "oil_level": np.random.normal(80, 5, n)
    })

    data["failure"] = ((data["temperature"] > 75) |
                       (data["vibration"] > 6)).astype(int)
    print(data["failure"].value_counts())
    data.to_csv("data/sensor_data.csv", index=False)
    return data