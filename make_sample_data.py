import numpy as np
import pandas as pd

np.random.seed(42)

# Reference (training) data - "old behaviour"
n_ref = 1000
ref = pd.DataFrame({
    "age": np.random.normal(30, 5, n_ref),          # avg 30
    "salary": np.random.normal(50000, 8000, n_ref), # avg 50k
    "click_rate": np.random.beta(2, 5, n_ref)       # between 0 and 1
})
ref.to_csv("reference_data.csv", index=False)

# Current (production) data - "drifted behaviour"
n_cur = 1000
cur = pd.DataFrame({
    "age": np.random.normal(36, 6, n_cur),          # shifted avg to 36
    "salary": np.random.normal(60000, 10000, n_cur),# shifted avg to 60k
    "click_rate": np.random.beta(3, 4, n_cur)       # distribution changed
})
cur.to_csv("current_data.csv", index=False)

print("Created reference_data.csv and current_data.csv")
