# ============================
# STUDENT PERFORMANCE MODEL TRAINING
# Run in VS Code (No Jupyter Needed)
# ============================

# Basic Imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Train Test Split
from sklearn.model_selection import train_test_split


# ====================================
# Load Dataset
# ====================================

df = pd.read_csv("notebook/data/stud.csv")

print("\nDataset Loaded Successfully\n")
print(df.head())


# ====================================
# Prepare Features and Target
# ====================================

X = df.drop(columns=["math_score"], axis=1)
y = df["math_score"]

print("\nFeatures Preview:\n")
print(X.head())

print("\nTarget Preview:\n")
print(y.head())


# ====================================
# Category Info
# ====================================

print("\nCategories in 'gender':")
print(df["gender"].unique())

print("\nCategories in 'race_ethnicity':")
print(df["race_ethnicity"].unique())

print("\nCategories in 'parental_level_of_education':")
print(df["parental_level_of_education"].unique())

print("\nCategories in 'lunch':")
print(df["lunch"].unique())

print("\nCategories in 'test_preparation_course':")
print(df["test_preparation_course"].unique())


# ====================================
# Identify Numeric and Categorical
# ====================================

num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

print("\nNumerical Features:")
print(num_features)

print("\nCategorical Features:")
print(cat_features)


# ====================================
# Preprocessing Pipeline
# ====================================

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),
    ]
)

X = preprocessor.fit_transform(X)

print("\nPreprocessing Completed")
print("Shape:", X.shape)


# ====================================
# Train-Test Split
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ====================================
# Evaluation Function
# ====================================

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)

    return mae, rmse, r2


# ====================================
# Models
# ====================================

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoost Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}


model_list = []
r2_list = []


# ====================================
# Train and Evaluate
# ====================================

print("\nTraining Models...\n")

for model_name, model in models.items():

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae, train_rmse, train_r2 = evaluate_model(
        y_train,
        y_train_pred
    )

    test_mae, test_rmse, test_r2 = evaluate_model(
        y_test,
        y_test_pred
    )

    print("=" * 50)
    print(model_name)

    print("\nTraining Performance")
    print("RMSE:", train_rmse)
    print("MAE :", train_mae)
    print("R2  :", train_r2)

    print("\nTesting Performance")
    print("RMSE:", test_rmse)
    print("MAE :", test_mae)
    print("R2  :", test_r2)

    model_list.append(model_name)
    r2_list.append(test_r2)


# ====================================
# Results Table
# ====================================

results = pd.DataFrame({
    "Model": model_list,
    "R2 Score": r2_list
})

results = results.sort_values(
    by="R2 Score",
    ascending=False
)

print("\nModel Comparison:\n")
print(results)


# ====================================
# Best Model (Linear Regression)
# ====================================

lin_model = LinearRegression()

lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)

score = r2_score(y_test, y_pred) * 100

print("\nLinear Regression Accuracy: %.2f%%" % score)


# ====================================
# Scatter Plot
# ====================================

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

plt.show()


# ====================================
# Regression Plot
# ====================================

plt.figure(figsize=(10, 6))

sns.regplot(
    x=y_test,
    y=y_pred,
    ci=None
)

plt.title("Regression Fit")

plt.show()


# ====================================
# Prediction Comparison
# ====================================

pred_df = pd.DataFrame({
    "Actual Value": y_test,
    "Predicted Value": y_pred,
    "Difference": y_test - y_pred
})

print("\nPrediction Comparison:\n")
print(pred_df.head(20))