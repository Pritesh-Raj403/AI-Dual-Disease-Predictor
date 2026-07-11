# AI-Powered Dual Disease Prediction System
### Diabetes and 10-Year CHD Risk Using Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-green.svg)](https://xgboost.readthedocs.io)

---

## About This Project

A machine learning-based clinical decision support system that simultaneously
predicts the risk of **Diabetes** and **10-Year Coronary Heart Disease (CHD)**
from clinical parameters — deployed as an interactive Streamlit web application.

**Developer:** Pritesh Raj | Roll No: 230103035  
**Department:** CSE (Artificial Intelligence & Data Science)  
**Institution:** IIIT Senapati, Manipur  
**Supervisor:** Dr. Nongmeikapam Kishorjit Singh, HoD CSE  
**Course:** Project-I (CS3201) | 6th Semester | Batch 2023–2027  

---

## Datasets

| Dataset | Records | Features | Target |
|---------|---------|----------|--------|
| PIMA Indians Diabetes (UCI/Kaggle) | 768 | 8 | Outcome (0/1) |
| Framingham Heart Study (Kaggle) | 4,240 | 15 | TenYearCHD (0/1) |

---

## ML Models Compared

| Model | Diabetes Accuracy | Heart Disease Accuracy |
|-------|------------------|----------------------|
| Logistic Regression | 71.43% | 66.98% |
| Random Forest | 74.68% | 73.58% |
| SVM | 72.73% | 68.75% |
| KNN | 68.83% | 61.32% |
| **XGBoost** | **76.62%** | **81.37%** |
| Gradient Boosting | 74.68% | 80.31% |
| Voting Ensemble | 75.97% | 75.24% |

---

## Unique Features

- **Unified Multi-Disease Prediction** — Both diseases in one system
- **0–100 Risk Score** — With Low / Moderate / High severity levels
- **Feature Engineering** — 4 new clinical features (pulse_pressure, age_risk, heavy_smoker, bmi_category)
- **SMOTE** — Class imbalance correction
- **Personalised Recommendations** — Based on risk level
- **PDF Report Download** — Full prediction report
- **History Tracking** — All predictions saved with trend chart

---

## Project Structure

Diabetes_project/
├── datasets/
│   ├── diabetes.csv          # PIMA Indians Diabetes Dataset
│   └── framingham.csv        # Framingham Heart Study Dataset
├── notebooks/
│   ├── diabetes_model.ipynb  # Diabetes ML pipeline
│   └── heart_model_framingham.ipynb  # Heart Disease ML pipeline
├── models/
│   ├── diabetes_model.pkl    # Trained XGBoost (diabetes)
│   ├── diabetes_scaler.pkl   # StandardScaler (diabetes)
│   ├── heart_model.pkl       # Trained XGBoost (heart disease)
│   └── heart_scaler.pkl      # StandardScaler (heart disease)
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

---

## How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/Diabetes_project.git
cd Diabetes_project
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the web application:**
```bash
python -m streamlit run app.py
```

**4. Open in browser:**

---

## Key Results

- **Best Model:** XGBoost for both diseases
- **Diabetes Accuracy:** 76.62% (consistent with literature: 72–82%)
- **Heart Disease Accuracy:** 81.37% (CV Score: 87.26%)
- **Key Finding:** Engineered feature `age_risk` scored importance **0.218** — highest of all 19 features

---

## Tech Stack

Python | Scikit-learn | XGBoost | SMOTE | Pandas | NumPy | Streamlit | Plotly | Pickle

---

## Disclaimer

This system is for **educational and research purposes only**.  
Always consult a qualified medical professional for clinical diagnosis.