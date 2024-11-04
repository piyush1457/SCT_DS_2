import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
df = pd.read_csv('data.csv')
print("First few rows of the data:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)
print("\nMissing values after handling:")
print(df.isnull().sum())
print("\nSummary statistics:")
print(df.describe())
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title("Survival Count by Gender")
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived', hue='Pclass')
plt.title("Survival Count by Passenger Class")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Distribution of Age")
plt.show()

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 80], labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='AgeGroup', hue='Survived')
plt.title("Survival Count by Age Group")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Pclass', y='Fare', hue='Survived')
plt.title("Fare Distribution by Passenger Class and Survival")
plt.show()

profile = ProfileReport(df, title="Train Data Profiling Report", explorative=True)
profile.to_file("data.html")
