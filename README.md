# Algorithmic-Game-Regression-Prediction

## Overview

This project uses Machine Learning to predict the **score** of a social media post based on engagement metrics such as likes, dislikes, shares, reposts, comments, and follower count.
The model is trained using the provided training dataset and generates predictions for unseen test data.

## Aim

To build a regression model that can accurately predict the score of a social media post using engagement-related features.

## Approach

A Polynomial Regression model was implemented using Scikit-Learn.
The workflow followed is:

1. Load the datasets:
   - train_3.csv (training dataset)
   - test_3.csv (test dataset)
   - samp_subm_3.csv (reference output format)
2. Separate features and target values.
3. Handle missing values using "SimpleImputer".
4. Standardize features using "StandardScaler".
5. Generate polynomial features using "PolynomialFeatures(degree=2)".
6. Train a "LinearRegression" model.
7. Evaluate model performance using RMSE, MAE, and R² Score.
8. Generate predictions for the test dataset.
9. Export the predictions to "submission.csv".

## Files Required
Download and place the following files in the same folder:
```
train_3.csv
test_3.csv
samp_subm_3.csv
solution.py
```
### File Description

* **train_3.csv** : Dataset used for training the model.
* **test_3.csv** : Dataset used for generating predictions.
* **samp_subm_3.csv** : Reference file showing the required output format.
* **solution.py** : Python script containing the complete implementation.

## Installation

Install the required libraries:
python -m pip install numpy pandas scikit-learn

## How to Run

### Step 1
Create a folder and place all required files inside it.
### Step 2
Open the folder in VS Code.
### Step 3
Open a terminal in VS Code.
### Step 4
Run the Python script:
python solution.py
### Step 5
After execution, a file named:
submission.csv
will be created automatically in the same folder.

## Output

The generated "submission.csv" contains the predicted scores in the required format and can be used for further evaluation.

## Project Report

A detailed report explaining the dataset, approach, model selection, implementation, performance metrics, and prediction results is available in:
**project report.pdf**

Refer to the report for a complete overview of the project methodology and findings.

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn

## Conclusion

A Polynomial Regression model was developed to predict post scores from engagement data. The project demonstrates the complete machine learning workflow, including data preprocessing, model training, evaluation, and prediction generation.
