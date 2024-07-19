# Fraud Detection Project

This project involves detecting accounting fraud in publicly traded U.S. firms using a machine learning approach with the RUSBoost algorithm.

## Data

- `data_FraudDetection_JAR2020.csv`: Contains the data used for training and validating the model.
- `AAER_firm_year.csv`: Initial accounting fraud sample from the SEC’s AAERs.
- `identifiers.csv`: Contains identifiers used in the project.

## Scripts

- `FraudDetectionRUSBoost.ipynb`: Jupyter Notebook containing the Python code to preprocess the data, train the RUSBoost model, and visualize the results.

## Usage

1. **Data Preprocessing**: Use `SimpleImputer` to handle missing values.
2. **Model Training and Evaluation**: Train the RUSBoost model using `imblearn` and evaluate it using various metrics.
3. **Results Analysis**: Visualize the results using ROC curve, precision-recall curve, feature importance plot, and confusion matrix.

