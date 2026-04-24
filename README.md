# SepsisGuard — Early Sepsis Prediction using Machine Learning

This project presents a machine learning pipeline designed to support early detection of sepsis in ICU patients.

Developed by:
Fatima Rhouma
Loujein Mbark

Faculty of Sciences of Sfax

---

## Objective

Sepsis is a life-threatening condition requiring early detection.

This project builds a predictive model capable of identifying sepsis risk several hours before clinical deterioration using machine learning.

---

## Dataset

Initial dataset:
1.55 million observations

Sepsis cases:
1.8%

After preprocessing:
83k observations
33% sepsis

Dataset balanced using undersampling.

---

## Pipeline

Data preprocessing:

• Missing value imputation  
• Missing indicators creation  
• Outlier clipping (IQR)  
• StandardScaler normalization  

Feature engineering:

• Shock Index = HR / SBP  
• Pulse Pressure = SBP − DBP  
• Temporal variation features  
• Rolling variability (6h window)  

Feature selection:

Correlation filtering + ANOVA

Final dataset:

22 features

---

## Models tested

Logistic Regression  
Random Forest  
HistGradientBoosting  
XGBoost  

Best model:

HistGradientBoosting

Performance:

ROC-AUC ≈ 0.80

Priority given to Recall for clinical safety.

---

## Future work

Multi-site validation  
Threshold optimisation  
Real-time hospital integration  
Temporal models (LSTM / Transformers)
