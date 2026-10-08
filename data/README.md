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
