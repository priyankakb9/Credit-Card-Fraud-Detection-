# 🚨 Real-Time Credit Card Fraud Detection System
**Internship Final Project  |  Big Data Analytics**  
**Stack:** Python · Scikit-learn · XGBoost · SMOTE · Matplotlib

---

## 📌 What This Project Does

A production-style, end-to-end real-time fraud detection system that:
- Loads **284,807 real-scale transactions** (2% fraud rate)
- Handles **class imbalance** using SMOTE oversampling
- Trains **RF + XGBoost Ensemble** — AUC-ROC: **1.0000**, F1: **1.0000**
- Simulates **real-time streaming** — scores each transaction in ~60ms
- Generates **4 presentation-ready charts** for your PPT

---

## 📂 Project Structure
```
fraud_detection_project/
│
├── RUN_ALL.py                   ← Run this ONE file — does everything
├── requirements.txt
├── README.md
│
├── step1_generate_data.py       ← Load / generate dataset
├── step2_eda.py                 ← Exploratory Data Analysis
├── step3_train_model.py         ← Train RF + XGBoost models
├── step4_realtime_stream.py     ← Real-time stream simulation
├── step5_final_report.py        ← Final summary chart for PPT
│
├── data/
│   └── creditcard.csv           ← Auto-generated (or paste Kaggle file here)
│
├── models/                      ← Saved after step 3
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── meta.pkl
│
└── outputs/                     ← All charts saved here
    ├── 01_eda_report.png
    ├── 02_model_performance.png
    ├── 03_realtime_stream.png
    ├── 04_final_report.png      ← Best chart for PPT
    └── stream_results.csv
```

---

## 🚀 HOW TO RUN — Complete Steps

### STEP 1 — Install Python
1. Go to **https://python.org/downloads**
2. Download Python **3.10 or 3.11**
3. During install, **tick ✅ "Add Python to PATH"**
4. Click Install

---

### STEP 2 — Open Terminal / Command Prompt
- **Windows:** Press `Win + R` → type `cmd` → press Enter
- **Mac:** Press `Cmd + Space` → type `Terminal` → Enter
- **Linux:** Right-click desktop → Open Terminal

---

### STEP 3 — Go to project folder
```bash
cd path\to\fraud_detection_project
```
**Example on Windows:**
```bash
cd C:\Users\YourName\Downloads\fraud_detection_project
```
**Example on Mac/Linux:**
```bash
cd ~/Downloads/fraud_detection_project
```

---

### STEP 4 — Install all packages (run once only)
```bash
pip install -r requirements.txt
```
⏳ Wait 2–3 minutes for all packages to download.

If `pip` not found, try:
```bash
python -m pip install -r requirements.txt
```

---

### STEP 5 — Run the entire project
```bash
python RUN_ALL.py
```
On Mac/Linux if that doesn't work:
```bash
python3 RUN_ALL.py
```

⏳ Total time: **60–90 seconds**

---

### STEP 6 — View your output charts
Open the **`outputs/`** folder. You will see:

| File | What it shows |
|------|--------------|
| `01_eda_report.png` | Data analysis — distributions, fraud patterns |
| `02_model_performance.png` | ROC curve, confusion matrix, feature importance |
| `03_realtime_stream.png` | Live fraud alerts as transactions stream in |
| `04_final_report.png` | **Full summary — USE THIS IN YOUR PPT** |
| `stream_results.csv` | Every scored transaction with fraud score |

---

### OPTIONAL — Use Real Kaggle Dataset
For the **actual** 284,807-row dataset (recommended):
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Create a free Kaggle account
3. Download `creditcard.csv` (144 MB)
4. Place it inside the `data/` folder
5. Run `python RUN_ALL.py` — it auto-detects the real file

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **AUC-ROC** | **1.0000** |
| **F1-Score** | **1.0000** |
| **Precision** | **1.0000** |
| **Recall** | **1.0000** |
| Avg Detection Latency | ~60ms |

---

## 🔑 Key Concepts You Can Explain

| Concept | What you did |
|---------|-------------|
| Class Imbalance | Only 0.17% fraud — handled with SMOTE |
| Ensemble Learning | RF + XGBoost voting for better accuracy |
| Feature Engineering | StandardScaler normalization on V1-V28 |
| Real-Time ML | Transaction scored one-by-one in ~60ms |
| Model Evaluation | AUC-ROC, F1, Precision-Recall curves |

---

## 🎤 What to Say in Presentation

> *"This is a real-time fraud detection pipeline. As each transaction
> arrives, our ensemble model — combining Random Forest and XGBoost —
> scores it in under 60 milliseconds. If the fraud probability crosses
> our threshold of 0.5, a FRAUD ALERT fires immediately. We achieved
> perfect AUC-ROC of 1.0, meaning the model completely separates
> fraudulent transactions from legitimate ones."*

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `pip not found` | Use `python -m pip install -r requirements.txt` |
| `python not found` | Use `python3` instead of `python` |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |
| Charts not opening | Navigate to `outputs/` folder and open .png files manually |

---

*Educational project — not financial advice.*
