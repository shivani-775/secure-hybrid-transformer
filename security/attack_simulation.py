import pandas as pd
import numpy as np

def false_data_injection():

    df = pd.read_csv("data/sensor_data.csv")

    # Select 30% of rows randomly
    n = int(0.3 * len(df))
    indices = np.random.choice(df.index, n, replace=False)

    # Modify temperature heavily
    df.loc[indices, "temperature"] += np.random.normal(40, 10, n)

    df.to_csv("data/attacked_data.csv", index=False)

    print("Attack injected. Attacked dataset saved.")