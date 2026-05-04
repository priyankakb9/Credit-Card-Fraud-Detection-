from db_utils import cursor, conn

print("📦 Initializing Database...")

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cursor.fetchall())

conn.commit()
conn.close()

print("✅ DB Ready")