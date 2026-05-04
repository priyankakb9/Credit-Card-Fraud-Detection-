from db_utils import fetch_all

df = fetch_all()

# Save for Power BI
df.to_csv("outputs/powerbi_data.csv", index=False)

print("✅ Data exported to outputs/powerbi_data.csv")