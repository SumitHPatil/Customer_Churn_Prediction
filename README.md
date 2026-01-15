# 📊 Customer Churn Prediction

## Problem Statement
Customer churn is a critical business problem where companies lose customers over time.  
This project aims to **predict whether a customer will churn** based on demographic, service usage, and account-related features using machine learning.

The goal is **binary classification**:  
`Churn = Yes / No`

---
## Dataset

- Source: Telecom customer churn dataset  
- Rows: ~7,000+  
- Target variable: `Churn`  
- Data types:
  - Categorical (gender, contract type, payment method, etc.)
  - Numerical (tenure, monthly charges, total charges)
---

## Project Workflow

### 1. Data Ingestion
<!-- H3 : Sub-sections -->
- Raw data loaded from CSV

### 2. Exploratory Data Analysis (EDA)
- Churn distribution
- Feature relationships with churn
- Missing value analysis

### 3. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Train-test split

### 4. Model Training
- Algorithm used: **XGBoost**
- Reason: handles non-linearity, performs well on tabular data, robust to feature interactions

### 5. Model Evaluation
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

### 6. Model Persistence
- Trained model saved as a `.pkl` file for reuse

---

## Repository Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── 01_Raw_Customer_Churn.csv
│   ├── 02_Cleaned_Customer_Churn.csv
│   └── 03_Encoded_Customer_Churn.csv
│
├── notebooks/
│   ├── 01_EDA_customer_churn.ipynb
│   ├── 02_Preprocessing_customer_churn.ipynb
|   ├── 03_Preprocessing1_customer_churn.ipynb     
│   └── 04_Model_Training_Evaluation.ipynb
│
├── src/
│   ├── 01_Preprocessing_customer_churn.py
│   ├── 02_Preprocessing1_customer_churn.py
│   └── 03_Model_Training_Evaluation.py
│
├── artifacts/
│   ├── encoder.pkl
│   └── xgboost_model.pkl
│
└── README.md
```

## Results ##

## Results

The XGBoost classifier was evaluated on the test dataset with a focus on
handling class imbalance in customer churn prediction.

- Accuracy: 0.79  
- Precision (Churn = Yes): 0.60  
- Recall (Churn = Yes): 0.56  
- F1-score (Churn = Yes): 0.58  

Confusion Matrix:
- True Negatives: 899
- False Positives: 137
- False Negatives: 164
- True Positives: 209

The model shows reasonable overall accuracy but moderate recall for churners,
indicating scope for improvement through threshold tuning and class balancing.


---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost







