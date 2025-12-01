# Bank Churn Prediction
This project was completed as part of a group assignment for DS3000: Introduction to Machine Learning.
The goal was to help a bank identify customers who are likely to leave (churn) and estimate how long a customer will remain with the bank before leaving. This allows the bank to implement timely strategies to retain customers.
- **Classification task**: Predict whether a customer will churn.
- **Regression task**: Predict the tenure of customers who are likely to churn.

## Dataset
The dataset contains demographic and account-related information for bank customers, including features such as `Age`, `Gender`,`CreditScore`, `Balance`, `EstimatedSalary`, `Tenure`, etc. The target variables were `Exited` for classification and `Tenure` for regression.

## Methodology
1. Data Exploration
    - Summary statistics, histograms, boxplots, scatterplots, and two-way tables were used to analyze feature distributions and relationships with churn.
    - Correlation heatmaps helped identify any linear relationships between numerical features.
2. Feature Engineering
    - Dropped irrelevant features: `RowNumber`, `CustomerId`, `Surname`.
    - Combined pairs of existing features such as `Balance_to_Salary`, `Age_Tenure`, and `Product_to_Tenure`.
    - Created `EngagementScore` as a weighted combination of `IsActiveMember`, `HasCrCard`, `NumOfProducts`, and `Tenure`.
    - Converted `CreditScore` into categorical buckets.
    - One-hot encoding for categorical variables.
3. Modeling:
    - **Classification Models**: Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost.
    - **Regression Models**: Linear Regression, Ridge, Lasso, Random Forest Regressor, XGBoost Regressor.
    - Standard scaling applied to numerical features where appropriate.
4. Evaluation Metrics:
    - **Classification**: Accuracy, Precision, Recall, F1 Score.
    - **Regression**: MAE, MSE, RMSE, R² Score.

## Results and Observations
### Classification

| Model               | Accuracy | Precision | Recall  | F1 Score |
|--------------------|----------|-----------|---------|----------|
| Logistic Regression | 0.8085   | 0.5952    | 0.1843  | 0.2814   |
| KNN                 | 0.8170   | 0.6062    | 0.2875  | 0.3900   |
| Random Forest       | 0.8655   | 0.7899    | 0.4619  | 0.5829   |
| XGBoost             | 0.8725   | 0.8040    | 0.4939  | 0.6119   |

- Random Forest and XGBoost perform best overall, likely due to their ability to capture non-linear interactions.
- Logistic Regression and KNN struggled with recall, indicating difficulty identifying all churners.

### Regression

| Model                  | MAE      | MSE      | RMSE     | R2 Score |
|------------------------|----------|----------|----------|----------|
| Linear Regression       | 2.5127   | 8.6121   | 2.9346   | -0.0176  |
| Ridge Regression        | 2.5127   | 8.6120   | 2.9346   | -0.0176  |
| Lasso Regression        | 2.5124   | 8.6099   | 2.9343   | -0.0174  |
| Random Forest Regressor | 2.5378   | 9.0931   | 3.0155   | -0.0745  |
| XGBoost Regressor       | 2.6961   | 10.3203   | 3.2125   | -0.2195  |

- Regression models performed poorly, indicating that predicting tenure before a customer exits is challenging with only demographic and basic account data.
- More detailed behavioral or transactional data would likely be needed for meaningful predictions.
