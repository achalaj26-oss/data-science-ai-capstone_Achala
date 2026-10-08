# Machine Learning-Based Credit Risk Assessment
with Explainable AI
CS5998 Capstone Project – Machine Learning-Based Credit Risk Assessment
with Explainable AI
## Dataset

The project uses the **American Express Default Prediction** dataset from Kaggle.

The original training data contains:

- **458,913 unique customers**
- **5.5 million+ monthly customer observations**
- Monthly customer profiles
- 188 predictor variables in the original training data
- A binary target indicating default risk

The dataset contains variables representing different aspects of customer financial behaviour, including:

- Delinquency
- Spend
- Payment
- Balance
- Risk-related variables

### Customer Sampling

Due to the large computational requirements of the original dataset, a stratified sample of **100,000 customers** was selected for model development.

The original target distribution was preserved:

| Target | Customers | Percentage |
|---|---:|---:|
| Non-default | 74,107 | 74.107% |
| Default | 25,893 | 25.893% |
| **Total** | **100,000** | **100%** |

A fixed random seed (`random_state=42`) was used to ensure reproducibility.

---

## Data Preparation

The selected 100,000 customers produced approximately **1.2 million monthly observations**.

The following preprocessing steps were performed:

1. Stratified customer sampling
2. Missing-value analysis
3. Removal of extremely sparse raw variables
4. Customer-level feature aggregation
5. Feature-target correlation analysis
6. Feature-feature correlation analysis
7. Feature selection
8. Train/validation splitting
9. Median imputation
10. Feature standardization for Logistic Regression

### Removing Extremely Sparse Variables

The initial dataset contained variables with extremely high levels of missingness.

Variables with **90% or more missing values** were removed before customer-level aggregation.

This reduced the monthly dataset from:

**190 columns → 172 columns**

After excluding identifier, date, and categorical variables, **168 numerical variables** were available for customer-level aggregation.

---

## Customer-Level Feature Engineering

The original data contains multiple monthly observations for each customer.

To transform the monthly data into a customer-level modelling dataset, two aggregation strategies were used:

### Mean

The mean represents the customer's average historical behaviour across available monthly observations.

### Last

The last observation represents the customer's most recent available behaviour.

Using both statistics provides a balance between historical behaviour and recent customer status while keeping the feature set interpretable.

The 168 numerical variables therefore produced:

**168 mean features + 168 last features = 336 candidate customer-level features**

An observation count was also retained.

---

## Feature Selection

Feature selection was performed using a correlation-based screening approach.

### Feature-Target Correlation

Pearson correlation was calculated between each candidate feature and the binary default target.

### Feature-Feature Correlation

Pairwise Pearson correlations were also calculated to identify highly redundant features.

A threshold of:

**|r| ≥ 0.90**

was used to identify highly correlated feature pairs.

Features were considered in descending order of absolute target correlation. A feature was retained when it provided a strong relationship with the target without being highly redundant with previously selected features.

This resulted in a final set of **30 modelling features**.

---

## Final Modelling Features

The selected features are:

P_2_last
D_48_last
B_2_last
D_61_last
B_9_last
B_18_last
D_55_last
D_61_mean
B_2_mean
D_44_last
B_9_mean
R_1_mean
B_3_last
D_75_last
B_7_last
R_1_last
B_4_last
B_20_last
B_19_last
B_38_last
B_1_last
B_23_mean
R_2_last
B_22_mean
R_10_mean
B_17_mean
B_38_mean
B_22_last
B_1_mean
B_30_last
