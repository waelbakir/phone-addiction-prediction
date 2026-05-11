# 📱 Smartphone Addiction Prediction

> **Data Mining Project** — Predict smartphone addiction risk from usage behavior data.

---

## 📁 Project Structure

```
phone-addiction-prediction/
├── data/
│   └── Smartphone_Usage_And_Addiction.csv   ← raw dataset
├── notebooks/
│   ├── 01_EDA.ipynb                          ← Exploratory Data Analysis
│   ├── 02_Preprocessing_and_Balancing.ipynb  ← Cleaning, encoding, SMOTE
│   └── 03_Modeling.ipynb                     ← Training, tuning, evaluation
├── models/
│   └── best_model.pkl                        ← saved best classifier (generated)
├── outputs/                                  ← generated plots, CSVs, scalers
├── app.py                                    ← Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebooks **in order**

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1️⃣ | `01_EDA.ipynb` | Understand the data — distributions, correlations, outliers |
| 2️⃣ | `02_Preprocessing_and_Balancing.ipynb` | Clean → encode → scale → SMOTE → save splits |
| 3️⃣ | `03_Modeling.ipynb` | Train 7 classifiers → tune top 2 → evaluate → save best model |

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 🔬 Pipeline Overview

```
Raw CSV
  │
  ▼
01_EDA.ipynb
  │  → target distribution, feature distributions, correlation heatmap, outlier summary
  │
  ▼
02_Preprocessing_and_Balancing.ipynb
  │  → drop leakage cols, impute, deduplicate
  │  → ordinal + label + one-hot encoding
  │  → IQR winsorization
  │  → 70/15/15 stratified split
  │  → StandardScaler (fit on train)
  │  → SMOTE on training set
  │  → saves: X_train.csv, X_val.csv, X_test.csv, scaler.pkl, feature_names.pkl
  │
  ▼
03_Modeling.ipynb
  │  → 5-fold CV baseline comparison (7 models)
  │  → GridSearchCV tuning (Random Forest + XGBoost)
  │  → validation set evaluation
  │  → final test set evaluation
  │  → saves: best_model.pkl, model_comparison.csv
  │
  ▼
app.py  (Streamlit)
  │  → loads scaler + model → user inputs → prediction + probability + tips
```

---

## 📊 Models Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable, prone to overfitting |
| **Random Forest** | Ensemble, robust, tuned with GridSearchCV |
| Gradient Boosting | Sequential ensemble |
| **XGBoost** | Boosted trees, tuned with GridSearchCV |
| K-Nearest Neighbors | Distance-based |
| SVM | Kernel-based |

---

## ⚖️ Class Balancing

Three strategies are compared in Notebook 2:

| Strategy | Description |
|----------|-------------|
| SMOTE | Synthetic over-sampling of minority class ✅ **default** |
| Random Under-Sampling | Reduce majority class |
| SMOTE + Tomek | Combined approach |

---

## 📦 Generated Artifacts

| File | Description |
|------|-------------|
| `outputs/scaler.pkl` | Fitted StandardScaler |
| `outputs/feature_names.pkl` | Ordered feature list |
| `outputs/X_train.csv` | SMOTE-balanced training features |
| `outputs/X_val.csv` | Validation features |
| `outputs/X_test.csv` | Test features |
| `models/best_model.pkl` | Best fitted classifier |
| `outputs/model_comparison.csv` | All metrics table |
| `outputs/0*_*.png` | All EDA & modeling plots |

---

## ⚠️ Disclaimer

This project is for **educational purposes** (Data Mining coursework).  
It does not constitute medical or psychological advice.
