📊 Credit Risk Prediction using Random Forest
🧾 Project Overview
This project aims to build a machine learning model that predicts whether a person is likely to default on a loan using historical credit data. The model uses the Random Forest Classifier algorithm to classify individuals as potential defaulters (1) or non-defaulters (0).

💡 Problem Statement
Lending institutions face major financial risk due to credit default. Predicting credit risk beforehand helps in making better lending decisions. The goal is to train a model that can accurately predict loan default risk based on various personal, financial, and loan-related features.

🧰 Technologies Used
Python

Pandas

NumPy

Scikit-learn

Matplotlib & Seaborn (for visualization)

📁 Dataset Features
Feature Name	Description
person_age	Age of the person
person_income	Annual income
person_home_ownership	Home ownership status (RENT/OWN/MORTGAGE)
person_emp_length	Years of employment
loan_intent	Purpose of the loan
loan_grade	Loan grade (A–G)
loan_amnt	Loan amount requested
loan_int_rate	Interest rate on the loan
loan_percent_income	Loan amount as % of income
cb_person_default_on_file	Previous default flag (Y/N)
cb_person_cred_hist_length	Credit history length (years)
loan_status	Target: 1 = Default, 0 = No default

🔍 Model Used
RandomForestClassifier

Tuned using GridSearchCV

Key hyperparameters: n_estimators, max_depth, min_samples_split, criterion

📈 Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

Confusion Matrix

📊 Feature Importance
The model also computes the importance of each feature, which helps understand what factors influence credit risk the most.
