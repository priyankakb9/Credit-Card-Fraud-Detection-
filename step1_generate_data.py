# ================================
# STEP 1 - DATA ENRICHMENT
# ================================

import pandas as pd
import numpy as np
import random
import os

print("\n🔄 Loading dataset...")

# ✅ FIX: LOAD ORIGINAL DATA FIRST (THIS WAS MISSING)
df = pd.read_csv("data/creditcard.csv")

print(f"Loaded dataset: {df.shape}")

# ================================
# ENRICH DATASET
# ================================
print("\n🔄 Enriching dataset with transaction metadata...")

card_types = ["Visa", "MasterCard", "RuPay"]
payment_modes = ["Card Swipe", "Online", "UPI"]
merchants = ["Amazon", "Flipkart", "Restaurant", "Electronics", "Grocery", "Petrol Pump"]
locations = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "USA", "UK", "Dubai"]
devices = ["Mobile", "Laptop", "POS"]

N = len(df)

df["txn_id"] = [f"TXN-{100000+i}" for i in range(N)]

# Better customer distribution
customers = [f"CUST-{i}" for i in range(1000, 5000)]
df["customer_id"] = np.random.choice(customers, N)

# Bias for fraud rows
def assign_features(row):
    if row["Class"] == 1:
        return pd.Series([
            np.random.choice(["Visa","MasterCard"]),
            np.random.choice(["Online","UPI"], p=[0.7,0.3]),
            np.random.choice(["Amazon","Electronics"]),
            np.random.choice(["USA","UK","Dubai"]),
            np.random.choice(["Mobile","Laptop"])
        ])
    else:
        return pd.Series([
            np.random.choice(card_types),
            np.random.choice(payment_modes),
            np.random.choice(merchants),
            np.random.choice(locations),
            np.random.choice(devices)
        ])

df[["card_type","payment_mode","merchant","location","device"]] = df.apply(assign_features, axis=1)

df["is_international"] = df["location"].apply(
    lambda x: 1 if x in ["USA","UK","Dubai"] else 0
)

# save enriched dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/creditcard_enriched.csv", index=False)

print("✅ Enriched dataset saved -> data/creditcard_enriched.csv")