"""
STEP 3 - Train RF + XGBoost Ensemble Model
============================================
Run:    python step3_train_model.py
Output: models/  +  outputs/02_model_performance.png
        + outputs/powerbi_dataset.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle, os, warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    average_precision_score
)
import xgboost as xgb

BG, CARD = "#0e1117", "#1e2130"
GREEN, RED = "#00d26a", "#ff4b4b"
BLUE, GOLD = "#1f77b4", "#ffd700"
GRAY = "#aaaaaa"

print("=" * 60)
print("  STEP 3 : Model Training (UPDATED)")
print("=" * 60)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("data/creditcard_enriched.csv")
df = df.fillna(0)

# 🔥 FIX 1: encode categorical columns safely
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

# ─────────────────────────────────────────────
# USE ONLY ORIGINAL MODEL FEATURES
# ─────────────────────────────────────────────
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

# safety check
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required features: {missing}")

X = df[FEATURES].values
y = df["Class"].values

print(f"\n  Loaded : {len(df):,} rows | Fraud: {y.sum()} ({y.mean()*100:.4f}%)")

# ─────────────────────────────────────────────
# SCALE
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────
Xtr, Xte, ytr, yte = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {len(Xtr):,} | Test: {len(Xte):,}")

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
print("\n  Training Random Forest ...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(Xtr, ytr)
rp = rf.predict_proba(Xte)[:, 1]
rf_auc = roc_auc_score(yte, rp)

print(f"  RF AUC = {rf_auc:.4f}")

print("\n  Training XGBoost ...")

xm = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1
)
xm.fit(Xtr, ytr)
xp = xm.predict_proba(Xte)[:, 1]
xgb_auc = roc_auc_score(yte, xp)

print(f"  XGB AUC = {xgb_auc:.4f}")

# ─────────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────────
ep = (rp + xp) / 2
yp = (ep >= 0.5).astype(int)

auc  = roc_auc_score(yte, ep)
ap   = average_precision_score(yte, ep)
f1   = f1_score(yte, yp)
prec = precision_score(yte, yp)
rec  = recall_score(yte, yp)
cm   = confusion_matrix(yte, yp)

print("\n  ENSEMBLE RESULTS")
print(f"  AUC  : {auc:.4f}")
print(f"  F1   : {f1:.4f}")
print(f"  Prec : {prec:.4f}")
print(f"  Rec  : {rec:.4f}")

# ─────────────────────────────────────────────
# POWER BI EXPORT
# ─────────────────────────────────────────────
Xtr_raw, Xte_raw, _, _ = train_test_split(
    df[FEATURES], y, test_size=0.2, random_state=42, stratify=y
)

df_results = Xte_raw.copy()
df_results["actual"] = yte
df_results["score"] = ep
df_results["rf_score"] = rp
df_results["xgb_score"] = xp
df_results["predicted"] = yp
df_results["correct"] = (df_results["actual"] == df_results["predicted"]).astype(int)
df_results["fraud_flag"] = df_results["predicted"].map({0:"Normal",1:"Fraud"})

df_results.to_csv("outputs/powerbi_dataset.csv", index=False)

print("  Saved Power BI dataset")

# ─────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────
pickle.dump(rf, open("models/rf_model.pkl", "wb"))
pickle.dump(xm, open("models/xgb_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(FEATURES, open("models/features.pkl", "wb"))

pickle.dump({
    "auc": auc,
    "ap": ap,
    "f1": f1,
    "precision": prec,
    "recall": rec,
    "threshold": 0.5,
    "cm": cm.tolist(),
    "rf_auc": rf_auc,
    "xgb_auc": xgb_auc
}, open("models/meta.pkl", "wb"))

print("  Models saved")

# ─────────────────────────────────────────────
# VISUALIZATION (UNCHANGED)
# ─────────────────────────────────────────────
def style(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color="white", fontweight="bold")
    ax.tick_params(colors=GRAY)
    for s in ax.spines.values():
        s.set_visible(False)

fig = plt.figure(figsize=(20,11), facecolor=BG)
fig.suptitle("Model Performance - Ensemble", color="white")

gs = gridspec.GridSpec(2,3, figure=fig)

ax = fig.add_subplot(gs[0,0])
style(ax,"Confusion Matrix")
ax.imshow(cm,cmap="Greens")

ax = fig.add_subplot(gs[0,1])
style(ax,"ROC Curve")
fpr,tpr,_ = roc_curve(yte,ep)
ax.plot(fpr,tpr,color=GOLD)

ax = fig.add_subplot(gs[0,2])
style(ax,"PR Curve")
from sklearn.metrics import precision_recall_curve
p,r,_ = precision_recall_curve(yte,ep)
ax.plot(r,p,color=RED)

plt.savefig("outputs/02_model_performance.png", dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()

print("  Saved visualization")
print("STEP 3 COMPLETE")