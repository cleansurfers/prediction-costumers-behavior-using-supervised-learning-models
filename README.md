# Customer Churn Prediction Using Machine Learning

## Project Overview

This project uses a structured telecom customer dataset to predict whether a customer will leave a company (**churn**) or stay.

The main goal is to compare different supervised machine learning algorithms and explore how handling class imbalance affects their performance.

## Dataset and Preprocessing

The dataset contains customer information such as tenure, services, and billing details.

The target variable is **Churn**:
- `0`: Customer stays
- `1`: Customer leaves

The original dataset was imbalanced, with approximately **73% non-churn** and **27% churn** cases.

The preprocessing steps included:

- Exploring the dataset and checking categorical values.
- Converting Yes/No values into 1/0.
- Applying One-Hot Encoding to categorical features.
- Separating features (X) and target variable (y).
- Splitting the dataset into 80% training and 20% testing.
- Using StandardScaler for models that require feature scaling.

## Machine Learning Models

Six supervised classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Random Forest
6. Gradient Boosting

GridSearchCV with 5-fold cross-validation was used for hyperparameter tuning, with F1-score as the selection metric.

## Handling Class Imbalance

Since the dataset was imbalanced, two different techniques were applied:

**1. Random Undersampling**

Reduces the number of samples from the majority class to balance the dataset.

**2. SMOTE (Synthetic Minority Over-sampling Technique)**

Generates synthetic samples for the minority class instead of removing samples from the majority class.

Both methods were applied to the training data, while the
