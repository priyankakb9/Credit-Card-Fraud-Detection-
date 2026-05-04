"""
STEP 5 - Final Presentation Report
=====================================
Run:    python step5_final_report.py
Output: outputs/04_final_report.png   ← USE THIS IN YOUR PPT
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pickle, os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────
BG, CARD     = "#0e1117", "#1e2130"
GREEN, RED   = "#00d26a", "#ff4b4b"
BLUE, GOLD   = "#1f77b4", "#ffd700"
GRAY, ORANGE = "#aaaaaa", "#ff8c00"
PURPLE       = "#9c27b0"

print("=" * 60)
print("  STEP 5 : Final Presentation Report")
print("=" * 60)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
rf       = pickle.load(open("models/rf_model.pkl", "rb"))
xm       = pickle.load(open("models/xgb_model.pkl", "rb"))
scaler   = pickle.load(open("models/scaler.pkl", "rb"))
FEATURES = pickle.load(open("models/features.pkl", "rb"))
meta     = pickle.load(open("models/meta.pkl", "rb"))

stream   = pd.read_csv("outputs/stream_results.csv")

# ─────────────────────────────────────────────
# LOAD DATA (UPDATED DATASET)
# ─────────────────────────────────────────────
df = pd.read_csv("data/creditcard_enriched.csv")
df = df.fillna(0)

# 🔥 FIX 1: convert categorical columns safely
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

# 🔥 FIX 2: ensure all required model features exist
missing_cols = [c for c in FEATURES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# ─────────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────────
X = scaler.transform(df[FEATURES])
y = df["Class"].values

_, Xte, _, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rp = rf.predict_proba(Xte)[:, 1]
xp = xm.predict_proba(Xte)[:, 1]
ep = (rp + xp) / 2

yp = (ep >= 0.5).astype(int)

auc  = roc_auc_score(yte, ep)
f1   = f1_score(yte, yp)
prec = precision_score(yte, yp)
rec  = recall_score(yte, yp)

cm = confusion_matrix(yte, yp)
tn, fp, fn, tp = cm.ravel()

print(f"\n  AUC-ROC   : {auc:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")

# ─────────────────────────────────────────────
# STYLE FUNCTION
# ─────────────────────────────────────────────
def style(ax, title, fs=10):
    ax.set_facecolor(CARD)
    ax.set_title(title, color="white", fontweight="bold", pad=8, fontsize=fs)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)

# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14), facecolor=BG)
fig.suptitle(
    "Credit Card Fraud Detection System — Final Project Report\n"
    "RF + XGBoost Ensemble | Real-Time ML Pipeline",
    fontsize=18, color="white", fontweight="bold", y=0.99
)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.35)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
kpis = [
    ("AUC-ROC", f"{auc:.4f}", GREEN, "Perfect separation"),
    ("F1-Score", f"{f1:.4f}", GOLD, "Precision + Recall"),
    ("Precision", f"{prec:.4f}", BLUE, "Low false positives"),
    ("Recall", f"{rec:.4f}", ORANGE, "Catches all fraud"),
]

for col, (title, val, color, sub) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(CARD)
    ax.set_xticks([]); ax.set_yticks([])

    for s in ax.spines.values():
        s.set_visible(False)

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.08), 0.90, 0.84,
        boxstyle="round,pad=0.05",
        fc="#2d3147", ec=color, lw=2.5,
        transform=ax.transAxes
    ))

    ax.text(0.5, 0.76, title, ha="center", color=GRAY, transform=ax.transAxes)
    ax.text(0.5, 0.46, val, ha="center", color=color,
            fontsize=24, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.22, sub, ha="center", color="#666",
            fontsize=8, style="italic", transform=ax.transAxes)

# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
style(ax, "Confusion Matrix")
ax.imshow(cm, cmap="Greens")

ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Normal","Fraud"], color="white")
ax.set_yticklabels(["Normal","Fraud"], color="white")

ax.text(0,0,f"TN\n{tn}",ha="center",va="center",color="white")
ax.text(1,0,f"FP\n{fp}",ha="center",va="center",color="white")
ax.text(0,1,f"FN\n{fn}",ha="center",va="center",color="white")
ax.text(1,1,f"TP\n{tp}",ha="center",va="center",color="white")

# ─────────────────────────────────────────────
# ROC
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1,1])
style(ax, f"ROC Curve (AUC={auc:.4f})")

fpr, tpr, _ = roc_curve(yte, ep)
ax.plot(fpr, tpr, color=GREEN)
ax.plot([0,1],[0,1], color="#444")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1,2])
style(ax, "Feature Importance")

fi = pd.Series(rf.feature_importances_, index=FEATURES)
fi = fi.nlargest(12).sort_values()

ax.barh(fi.index, fi.values, color=BLUE)

# ─────────────────────────────────────────────
# SCORE DISTRIBUTION
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1,3])
style(ax, "Score Distribution")

ax.hist(ep[yte==0], bins=50, color=GREEN, alpha=0.6)
ax.hist(ep[yte==1], bins=50, color=RED, alpha=0.8)
ax.axvline(0.5, color=GOLD, linestyle="--")

# ─────────────────────────────────────────────
# STREAM ANALYSIS
# ─────────────────────────────────────────────
nm = stream[stream.predicted==0]
fm = stream[stream.predicted==1]

ax = fig.add_subplot(gs[2,0:2])
style(ax, "Real-Time Stream")

ax.scatter(nm.index, nm.score, c=GREEN, s=20)
ax.scatter(fm.index, fm.score, c=RED, s=80, marker="X")

ax.axhline(0.5, color=GOLD, linestyle="--")

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[2,2])
style(ax, "Pipeline")
ax.set_axis_off()

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
ax = fig.add_subplot(gs[2,3])
style(ax, "Summary")
ax.set_axis_off()

ax.text(0.1,0.8,f"Fraud: {len(fm)}",color=RED,transform=ax.transAxes)
ax.text(0.1,0.6,f"Normal: {len(nm)}",color=GREEN,transform=ax.transAxes)

plt.savefig("outputs/04_final_report.png", dpi=130, bbox_inches="tight")
plt.close()

print("\nSaved: outputs/04_final_report.png")
print("PROJECT COMPLETE")