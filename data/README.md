# Data

This folder contains dataset documentation and data-related resources.

Confidential or proprietary data will not be uploaded to this public repository.

This project uses the American Express Default Prediction dataset from Kaggle.

The original dataset contains monthly customer-level financial and behavioural information and a binary default target.

Due to the large size of the dataset and dataset redistribution considerations, the original AMEX data is not included in this public repository.

## Dataset Source

American Express Default Prediction:
https://www.kaggle.com/competitions/amex-default-prediction

## Data Used in This Project

A stratified sample of 100,000 customers was selected from the labelled training population for model development.

The sample preserves the original default/non-default distribution.

- Total sampled customers: 100,000
- Non-default: 74,107
- Default: 25,893
- Sampling random state: 42

The monthly observations for these customers were then processed and aggregated to customer-level features.

## Data Processing

The data processing pipeline included:

1. Customer sampling
2. Missing-value analysis
3. Removal of extremely sparse variables
4. Customer-level aggregation
5. Feature-target correlation analysis
6. Feature redundancy analysis
7. Selection of 30 modelling features

The processed customer-level datasets are retained in the project's working environment but are not included in this public repository.

## Independent Project Development(I used the official Kaggle competition information to understand the dataset and problem context. However, I did not use a competitor Kaggle notebook as the source of my implementation.)

The Kaggle competition information was used as the dataset and problem-context source. No competitor Kaggle notebook was used as the implementation source for this project. The preprocessing, customer-level aggregation, feature screening, feature selection, Logistic Regression baseline, XGBoost model, evaluation and explainability workflow were developed as part of this MSc capstone project.

## Independent Development Evidence
1. The project was developed incrementally in Google Colab notebooks.
2. The dataset was sampled to 100,000 customers using stratified sampling with `random_state=42`.
3. Monthly observations were aggregated into customer-level mean and last-value features.
4. Missing-value analysis was performed before modelling, including analysis of missingness by default status.
5. Pearson feature-target and feature-feature correlation analyses were independently performed.
6. A final set of 30 modelling features was selected using target association and redundancy reduction with `|r| >= 0.90`.
7. Median imputation was fitted only on training data and then applied to validation data to prevent data leakage.
8. Logistic Regression was implemented as the baseline model and XGBoost as the improved model.
9. Model performance was evaluated using ROC-AUC, PR-AUC, accuracy, precision, recall, F1-score, confusion matrices, ROC curves, and Precision-Recall curves.
10. SHAP explainability is being developed as a separate project component.
11. Intermediate datasets, analysis tables, figures, model files, and GitHub commits provide an audit trail.

## Kaggle Competition vs. MSc Capstone Project

| **Aspect** | **American Express Kaggle Competition** | **My MSc Capstone Project** |
|---|---|---|
| **Main question** | Predict the probability that a customer will **default in the future** based on their monthly customer profile. | Can **explainable machine learning** accurately predict credit-card default risk and provide interpretable insights into individual predictions? |
| **Main objective** | Build a highly effective predictive model and achieve a strong competition score. | Develop an **explainable ML framework** that balances prediction performance with interpretability. |
| **Primary focus** | **Prediction performance / ranking** | **Prediction + Explainability** |
| **Dataset** | American Express Default Prediction competition dataset.458,913 unique customers, 5.5 million+ monthly customer observations | A **100,000-customer stratified sample** from the same dataset. |
| **Feature engineering** | Participants could use different techniques to maximise predictive performance. | Customer-level **mean and last-value aggregation** was used to represent historical and recent behaviour. |
| **Feature selection** | Participants could select or create features according to their modelling strategy. | Correlation-based feature screening and redundancy reduction were used to select **30 features**. |
| **Baseline model** | No specific baseline model was required by the competition. | **Logistic Regression** was used as the baseline model. |
| **Improved model** | Participants could use any suitable model or ensemble to maximise competition performance. | **XGBoost** was used as the improved model and compared with Logistic Regression. |
| **Model comparison** | Primarily focused on achieving the best competition performance. | Explicit comparison between the **baseline and improved models**. |
| **Evaluation metrics** | Official competition metric based on **Normalized Gini and default-rate capture at 4%**. | **ROC-AUC, PR-AUC, Accuracy, Precision, Recall and F1-score**. |
| **Explainability** | Explainability was **not the primary objective** of the competition. | **SHAP-based explainability** is a key component of the project. |
| **Individual predictions** | Focused on submitting default probabilities for customers. | Also aims to understand **why an individual customer receives a particular prediction**. |
| **Interpretation** | Strong predictive performance was the main goal. | Identify and communicate **factors influencing model predictions**. |
| **Reproducibility** | Participants were free to develop their own competition solutions. | A reproducible pipeline with documented preprocessing, feature selection, models, parameters and evaluation is being developed. |

### Key Distinction

The dataset and underlying prediction problem originate from the American Express Kaggle competition. However, this MSc capstone is not intended to reproduce the competition solution.

The Kaggle competition primarily focused on **predictive performance and competition ranking**, whereas this project focuses on **predictive performance together with explainability, model comparison, interpretation and reproducibility**.

The project independently developed the data preprocessing, customer-level feature engineering, correlation-based feature selection, Logistic Regression baseline, XGBoost model and evaluation pipeline. SHAP-based explainability is being developed as a separate project component.
