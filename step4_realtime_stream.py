"""
STEP 4 - Real-Time Fraud Detection Stream (FIXED + FULL VERSION)
=================================================================
- SQLite stores ONLY core metrics (8 columns)
- CSV stores full enriched output (Power BI ready)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import pickle, time, os, sqlite3, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("database", exist_ok=True)

DB_PATH = "database/fraud_detection.db"

# ─────────────────────────────────────────────
# DATABASE SETUP (FIXED 8-COLUMN SCHEMA)
# ─────────────────────────────────────────────
def create_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        txn_id TEXT,
        amount REAL,
        score REAL,
        predicted INTEGER,
        actual INTEGER,
        correct INTEGER,
        latency_ms REAL,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_db(row):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?)
    """, row)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
BG, CARD = "#0e1117", "#1e2130"
GREEN, RED = "#00d26a", "#ff4b4b"
BLUE, GOLD = "#1f77b4", "#ffd700"
GRAY = "#aaaaaa"

print("=" * 60)
print("  STEP 4 : Real-Time Fraud Detection Stream (FIXED)")
print("=" * 60)

create_db()

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
rf       = pickle.load(open("models/rf_model.pkl", "rb"))
xm       = pickle.load(open("models/xgb_model.pkl", "rb"))
scaler   = pickle.load(open("models/scaler.pkl", "rb"))
FEATURES = pickle.load(open("models/features.pkl", "rb"))
meta     = pickle.load(open("models/meta.pkl", "rb"))

THRESHOLD = meta["threshold"]

print(f"\nModels loaded (AUC={meta['auc']:.4f})")

# ─────────────────────────────────────────────
# LOAD ENRICHED DATASET (KEEP ALL COLUMNS)
# ─────────────────────────────────────────────
df = pd.read_csv("data/creditcard_enriched.csv").fillna(0)

# metadata columns (KEEP FOR OUTPUT ONLY)
META_COLS = [
    "txn_id",
    "customer_id",
    "card_type",
    "payment_mode",
    "merchant",
    "location",
    "device",
    "is_international"
]

# encode ONLY for ML
df_ml = df.copy()
for col in df_ml.columns:
    if df_ml[col].dtype == "object":
        df_ml[col] = df_ml[col].astype("category").cat.codes

# ML dataset
stream_df = df_ml[FEATURES + ["Class"]]

# ─────────────────────────────────────────────
# CREATE STREAM
# ─────────────────────────────────────────────
fraud = stream_df[stream_df.Class == 1].sample(20, random_state=7)
normal = stream_df[stream_df.Class == 0].sample(180, random_state=7)

stream_idx = pd.concat([fraud, normal]).sample(frac=1, random_state=42).index

print("Stream size:", len(stream_idx))

# ─────────────────────────────────────────────
# STREAM PROCESSING
# ─────────────────────────────────────────────
results = []

for i, idx in enumerate(stream_idx):

    row_ml = df_ml.loc[idx]
    row_meta = df.loc[idx]

    txn_id = row_meta["txn_id"]
    amount = row_meta["Amount"]
    actual = int(row_ml["Class"])

    # inference timing
    t0 = time.time()

    X = scaler.transform([row_ml[FEATURES].values.astype(float)])

    score = (rf.predict_proba(X)[0,1] + xm.predict_proba(X)[0,1]) / 2

    latency = (time.time() - t0) * 1000
    predicted = int(score >= THRESHOLD)
    correct = int(predicted == actual)

    timestamp = pd.Timestamp.now()

    # ─────────────────────────────────────────────
    # SQL INSERT (ONLY 8 COLUMNS)
    # ─────────────────────────────────────────────
    db_row = (
        txn_id,
        float(amount),
        float(score),
        int(predicted),
        int(actual),
        int(correct),
        float(latency),
        str(timestamp)
    )

    insert_db(db_row)

    # ─────────────────────────────────────────────
    # FULL CSV OUTPUT (ENRICHED)
    # ─────────────────────────────────────────────
    results.append({
        "txn_id": txn_id,
        "customer_id": row_meta["customer_id"],
        "card_type": row_meta["card_type"],
        "payment_mode": row_meta["payment_mode"],
        "merchant": row_meta["merchant"],
        "location": row_meta["location"],
        "device": row_meta["device"],
        "is_international": row_meta["is_international"],

        "amount": amount,
        "score": score,
        "predicted": predicted,
        "actual": actual,
        "correct": correct,
        "latency_ms": latency,
        "timestamp": timestamp
    })

    status = "🚨 FRAUD" if predicted else "OK"
    print(f"{txn_id} | ${amount:.2f} | {score:.4f} | {status}")

    time.sleep(0.01)

# ─────────────────────────────────────────────
# SAVE OUTPUT
# ─────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv("outputs/stream_results.csv", index=False)

print("\n" + "=" * 60)
print("✅ STEP 4 COMPLETE")
print("✔ SQLite saved (clean 8-column schema)")
print("✔ CSV saved (FULL enriched dataset)")
print("=" * 60)