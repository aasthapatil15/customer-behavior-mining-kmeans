import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 5000

# Generating realistic customer purchase metrics across 5,000 shoppers
age = np.random.randint(18, 70, size=n_samples)
annual_income_k = np.random.randint(15, 140, size=n_samples)
spending_score = np.random.randint(1, 100, size=n_samples)
purchase_frequency = np.random.randint(1, 50, size=n_samples)

data = pd.DataFrame({
    "Customer_ID": [f"CUST_{i+1:05d}" for i in range(n_samples)],
    "Age": age,
    "Annual_Income_k$": annual_income_k,
    "Spending_Score_1_to_100": spending_score,
    "Annual_Purchases": purchase_frequency
})

data.to_csv("dataset_5000.csv", index=False)
print("5,000 Data Mining Customer Records Generated Successfully!")
