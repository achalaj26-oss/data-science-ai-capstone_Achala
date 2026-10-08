# Processed Data

Processed datasets were generated from the American Express Default Prediction dataset during the data preparation and modelling stages.

The processing pipeline produced:

- A 100,000-customer development sample
- Cleaned monthly customer data
- Customer-level aggregated features
- A 70,000-customer modelling dataset
- Training, validation and test customer ID splits

For customer-level privacy, dataset redistribution considerations, and repository size limitations, the processed customer-level datasets are not included in this public repository.

## Main Processed Dataset

The main modelling dataset contains:

- 70,000 customers
- 30 selected predictor variables
- 1 binary target variable
- Customer identifier

Feature engineering used two aggregation statistics:

- Mean: average historical customer behaviour
- Last: most recent available customer observation

The final modelling dataset was used to train and evaluate Logistic Regression and XGBoost models.
