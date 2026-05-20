import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("data/stud.csv")

print("\nDataset Loaded\n")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isna().sum())

print("\nDuplicates:")
print(df.duplicated().sum())

print("\nInfo:")
print(df.info())

print("\nUnique Values:")
print(df.nunique())

print("\nStatistics:")
print(df.describe())

# Categories
print(df["gender"].unique())
print(df["race_ethnicity"].unique())
print(df["parental_level_of_education"].unique())
print(df["lunch"].unique())
print(df["test_preparation_course"].unique())

# Numerical and categorical features
numeric_features = [
    feature for feature in df.columns if df[feature].dtype != "O"
]

categorical_features = [
    feature for feature in df.columns if df[feature].dtype == "O"
]

print("\nNumerical Features:", numeric_features)
print("\nCategorical Features:", categorical_features)

# Feature Engineering
df["total_score"] = (
    df["math_score"]
    + df["reading_score"]
    + df["writing_score"]
)

df["average"] = df["total_score"] / 3

print(df.head())

# Full marks
print(
    "Math Full Marks:",
    df[df["math_score"] == 100]["average"].count()
)

print(
    "Reading Full Marks:",
    df[df["reading_score"] == 100]["average"].count()
)

print(
    "Writing Full Marks:",
    df[df["writing_score"] == 100]["average"].count()
)

# Below 20
print(
    "Math <20:",
    df[df["math_score"] <= 20]["average"].count()
)

print(
    "Reading <20:",
    df[df["reading_score"] <= 20]["average"].count()
)

print(
    "Writing <20:",
    df[df["writing_score"] <= 20]["average"].count()
)

# Distribution Plot
plt.figure(figsize=(12, 6))
sns.histplot(df["average"], kde=True)
plt.title("Average Score Distribution")
plt.show()

# Gender comparison
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="average",
    hue="gender",
    kde=True
)
plt.title("Average Score by Gender")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
plt.title("Correlation Heatmap")
plt.show()

# Boxplots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=df["math_score"])

plt.subplot(1, 3, 2)
sns.boxplot(y=df["reading_score"])

plt.subplot(1, 3, 3)
sns.boxplot(y=df["writing_score"])

plt.show()

print("\nEDA Completed Successfully")