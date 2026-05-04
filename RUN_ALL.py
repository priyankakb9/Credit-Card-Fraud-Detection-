import subprocess, sys, time
import sqlite3
import os

DB_PATH = "database/fraud_detection.db"
os.makedirs("database", exist_ok=True)

print("🚀 Running Full Pipeline...\n")

# DB INIT
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
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

STEPS = [
    "step1_generate_data.py",
    "step2_eda.py",
    "step3_train_model.py",
    "step4_realtime_stream.py",
    "step5_final_report.py",
]

for step in STEPS:
    print(f"\n▶ Running {step}")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print("❌ Failed:", step)
        sys.exit(1)

print("\n✅ ALL STEPS COMPLETED")