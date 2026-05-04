import sqlite3
import os

DB_NAME = "fraud_detection.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    # Main transactions table
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
    print("✅ SQL Database & Table Created")

def insert_transaction(data):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO transactions 
    (txn_id, amount, score, predicted, actual, correct, latency_ms, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()
def fetch_all():
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect("database/fraud_detection.db")

    df = pd.read_sql("SELECT * FROM transactions", conn)

    conn.close()
    return df