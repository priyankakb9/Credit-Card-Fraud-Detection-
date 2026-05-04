"""
STEP 2 - Exploratory Data Analysis
====================================
Run:    python step2_eda.py
Output: outputs/01_eda_report.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

BG,CARD      = "#0e1117","#1e2130"
GREEN,RED    = "#00d26a","#ff4b4b"
BLUE,GOLD    = "#1f77b4","#ffd700"
GRAY,PURPLE  = "#aaaaaa","#9c27b0"

print("=" * 60)
print("  STEP 2 : Exploratory Data Analysis")
print("=" * 60)

df = pd.read_csv("data/creditcard.csv")
VCOLS = [f"V{i}" for i in range(1, 29)]

print(f"\n  Rows         : {len(df):,}")
print(f"  Normal       : {(df.Class==0).sum():,}  ({(df.Class==0).mean()*100:.3f}%)")
print(f"  Fraud        : {(df.Class==1).sum():,}  ({(df.Class==1).mean()*100:.4f}%)")
print(f"  Missing vals : {df.isnull().sum().sum()}")
print(f"  Avg amount (normal) : ${df[df.Class==0].Amount.mean():.2f}")
print(f"  Avg amount (fraud)  : ${df[df.Class==1].Amount.mean():.2f}")

corr = df[VCOLS + ["Amount","Class"]].corr()["Class"].drop("Class").sort_values()
print(f"\n  Strongest fraud indicators (positive):")
for f,v in corr.tail(5)[::-1].items():
    print(f"    {f:>8}: {v:+.4f}")
print(f"  Strongest fraud indicators (negative):")
for f,v in corr.head(5).items():
    print(f"    {f:>8}: {v:+.4f}")

fig = plt.figure(figsize=(20,12), facecolor=BG)
fig.suptitle(
    f"Credit Card Fraud Detection  —  EDA\n"
    f"Dataset: {len(df):,} transactions  |  Fraud: {(df.Class==1).sum():,} ({df.Class.mean()*100:.4f}%)",
    fontsize=16, color="white", fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

def style(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color="white", fontweight="bold", pad=8, fontsize=10)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ax.spines.values(): s.set_visible(False)

# 1) Class imbalance
ax = fig.add_subplot(gs[0,0])
style(ax, "Class Imbalance (log scale)")
cnts = df.Class.value_counts().sort_index()
bars = ax.bar(["Normal","Fraud"], cnts.values, color=[GREEN,RED], width=0.5, edgecolor="none")
ax.set_yscale("log"); ax.set_ylabel("Count (log)", color=GRAY)
for bar,val in zip(bars,cnts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.5,
            f"{val:,}", ha="center", color="white", fontsize=9, fontweight="bold")

# 2) Amount distribution
ax = fig.add_subplot(gs[0,1])
style(ax, "Transaction Amount Distribution")
ax.hist(df[df.Class==0]["Amount"].clip(0,500), bins=60,
        alpha=0.7, color=GREEN, label="Normal", density=True)
ax.hist(df[df.Class==1]["Amount"].clip(0,500), bins=30,
        alpha=0.9, color=RED, label="Fraud", density=True)
ax.set_xlabel("Amount ($)", color=GRAY); ax.set_ylabel("Density", color=GRAY)
ax.legend(facecolor="#2d3147", labelcolor="white", fontsize=9)
ax.text(0.97,0.95,f"Fraud avg: ${df[df.Class==1].Amount.mean():.0f}",
        transform=ax.transAxes, ha="right", color=RED, fontsize=8)

# 3) Time/hour distribution
ax = fig.add_subplot(gs[0,2])
style(ax, "Fraud Hotspots by Hour of Day")
fh = (df[df.Class==1]["Time"]/3600 % 24).astype(int)
nh = (df[df.Class==0]["Time"].sample(5000,random_state=1)/3600 % 24).astype(int)
ax.hist(nh, bins=24, alpha=0.6, color=BLUE, label="Normal (sample)", density=True)
ax.hist(fh, bins=24, alpha=0.9, color=RED,  label="Fraud", density=True)
ax.set_xlabel("Hour of Day", color=GRAY); ax.set_ylabel("Density", color=GRAY)
ax.legend(facecolor="#2d3147", labelcolor="white", fontsize=9)

# 4) Feature correlations
ax = fig.add_subplot(gs[0,3])
style(ax, "Feature Correlation with Fraud")
top10 = pd.concat([corr.head(5), corr.tail(5)])
bar_c = [RED if v<0 else GREEN for v in top10.values]
ax.barh(top10.index, top10.values, color=bar_c, edgecolor="none")
ax.axvline(0, color=GRAY, lw=0.8)
ax.set_xlabel("Correlation", color=GRAY)
for i,(val,name) in enumerate(zip(top10.values, top10.index)):
    ax.text(val+(0.003 if val>=0 else -0.003), i,
            f"{val:+.3f}", va="center", color="white", fontsize=7,
            ha="left" if val>=0 else "right")

# 5) V1 best separator
ax = fig.add_subplot(gs[1,0])
style(ax, "V1 — Strongest Fraud Separator")
ax.hist(df[df.Class==0]["V1"].clip(-10,6), bins=80,
        alpha=0.65, color=GREEN, label="Normal", density=True)
ax.hist(df[df.Class==1]["V1"].clip(-10,6), bins=40,
        alpha=0.90, color=RED, label="Fraud", density=True)
ax.set_xlabel("V1 (PCA feature)", color=GRAY); ax.set_ylabel("Density", color=GRAY)
ax.legend(facecolor="#2d3147", labelcolor="white", fontsize=9)

# 6) V14 separator
ax = fig.add_subplot(gs[1,1])
style(ax, "V14 — Strong Fraud Separator")
ax.hist(df[df.Class==0]["V14"].clip(-15,8), bins=80,
        alpha=0.65, color=GREEN, label="Normal", density=True)
ax.hist(df[df.Class==1]["V14"].clip(-15,8), bins=40,
        alpha=0.90, color=RED, label="Fraud", density=True)
ax.set_xlabel("V14 (PCA feature)", color=GRAY); ax.set_ylabel("Density", color=GRAY)
ax.legend(facecolor="#2d3147", labelcolor="white", fontsize=9)

# 7) Amount violin
ax = fig.add_subplot(gs[1,2])
style(ax, "Amount by Class (Violin)")
parts = ax.violinplot(
    [df[df.Class==0]["Amount"].clip(0,400).values,
     df[df.Class==1]["Amount"].clip(0,400).values],
    positions=[0,1], showmedians=True)
for i,pc in enumerate(parts["bodies"]):
    pc.set_facecolor(GREEN if i==0 else RED); pc.set_alpha(0.7)
for k in ["cmedians","cmaxes","cmins","cbars"]:
    parts[k].set_colors("white" if k=="cmedians" else GRAY)
ax.set_xticks([0,1]); ax.set_xticklabels(["Normal","Fraud"], color="white")
ax.set_ylabel("Amount ($)", color=GRAY)

# 8) Fraud scatter: time vs amount
ax = fig.add_subplot(gs[1,3])
style(ax, "Fraud: Time vs Amount")
fd = df[df.Class==1].copy()
fd["hour"] = (fd["Time"]/3600 % 24).astype(int)
sc = ax.scatter(fd["hour"], fd["Amount"].clip(0,2000),
                c=fd["Amount"], cmap="Reds", s=25, alpha=0.7, edgecolors="none")
trend = fd.groupby("hour")["Amount"].mean()
ax.plot(trend.index, trend.values, color=GOLD, lw=2, label="Avg by hour")
ax.set_xlabel("Hour of Day", color=GRAY); ax.set_ylabel("Fraud Amount ($)", color=GRAY)
ax.legend(facecolor="#2d3147", labelcolor="white", fontsize=9)

plt.savefig("outputs/01_eda_report.png", dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n  Saved : outputs/01_eda_report.png")
print("  STEP 2 COMPLETE")
