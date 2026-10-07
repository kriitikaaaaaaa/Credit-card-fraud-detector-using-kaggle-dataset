import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the full Kaggle dataset
df = pd.read_csv("creditcard.csv")

# Fit the SAME scalers on the full dataset
time_scaler = StandardScaler()
amount_scaler = StandardScaler()

time_scaler.fit(df[["Time"]])
amount_scaler.fit(df[["Amount"]])

# Save the scalers
joblib.dump(
    {
        "time_scaler": time_scaler,
        "amount_scaler": amount_scaler
    },
    "scalers.joblib"
)

# Create a small demo dataset.
# Keep some fraud + legitimate transactions.
fraud = df[df["Class"] == 1].head(100)
legitimate = df[df["Class"] == 0].head(900)

demo = pd.concat(
    [fraud, legitimate],
    ignore_index=True
)

# Shuffle demo rows
demo = demo.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

demo.to_csv(
    "demo_transactions.csv",
    index=False
)

print("Deployment preparation complete.")
print("Created: scalers.joblib")
print("Created: demo_transactions.csv")
print("Demo transactions:", len(demo))