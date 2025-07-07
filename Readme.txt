Mobile Phone Price Range Prediction
Author: Pramod Nagisetty
University: University of Leicester
Email: pn118@student.le.ac.uk

Project Overview
This project presents a complete machine learning pipeline to predict mobile phone price categories (Low, Medium, High, Very High) using only technical specifications. The workflow includes data loading, exploratory data analysis, visualization, preprocessing, model training, hyperparameter tuning, evaluation, and feature importance analysis.

Dataset
The dataset is loaded from /content/dataset.csv and includes various technical features like RAM, battery power, pixel dimensions, etc., with a target variable price_range.

Key Steps
1. Data Loading and Exploration
Loads the dataset using Pandas.

Displays data types and sample records.

Visualizes each feature using histograms or countplots based on their type.

2. Correlation Analysis
A heatmap is created to explore the correlation between features.

Top 4 features most correlated with price_range are identified: ram, battery_power, px_width, px_height.

3. Advanced Feature Visualization
For the top 4 features:

Bar plots of feature mean across price categories

Point plots to analyze trends

Histograms grouped by price_range

ECDF plots for distribution comparison

4. Data Preprocessing
Standard scaling is applied using StandardScaler.

Dataset is split into training, validation, and test sets using stratified sampling.

5. Model Training and Evaluation
Trains and evaluates the following models:

Random Forest

XGBoost

K-Nearest Neighbors

Multi-Layer Perceptron (Neural Network)

Each model's validation and test accuracies are compared using a grouped bar chart.

6. Hyperparameter Tuning
GridSearchCV is used to tune the XGBoost classifier.

Best parameters and training accuracy are displayed.

7. Final Evaluation
Uses the best model to predict on the test set.

Outputs a classification report and confusion matrix.

Maps numeric predictions to actual price range labels.

8. Feature Importance
Extracts feature importance from the best XGBoost model.

Visualizes the top 10 most important features in descending order.

Dependencies
Required Python libraries:

pandas

matplotlib

seaborn

scikit-learn

xgboost

Install using pip:

bash
Copy
Edit
pip install pandas matplotlib seaborn scikit-learn xgboost
Output
Feature distribution plots

Heatmap of correlations

Comparative model accuracy chart

Confusion matrix and classification report

Top feature importance bar chart

Sample price predictions from the best model

Notes
Dataset is assumed to be pre-balanced.

All steps follow reproducible and interpretable machine learning practices.

XGBoost achieved the highest accuracy after tuning.