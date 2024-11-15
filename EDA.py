# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv('cleaned_car_data.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("Column Data Types and Missing Values:")
print(df.info())

# Display the first few rows of the dataset
print("First 5 Rows of the Dataset:")
print(df.head())

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())


df['Price'] = df['Price'].fillna(df['Price'].median())
df['Registration_Year'] = df['Registration_Year'].fillna(df['Model_Year'])


# Price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=30, kde=True, color='blue')
plt.title('Price Distribution')
plt.xlabel('Price (in Lakh)')
plt.ylabel('Frequency')
plt.show()

# Distribution of KM_Driven
plt.figure(figsize=(10, 6))
sns.histplot(df['KM_Driven'], bins=30, kde=True, color='orange')
plt.title('Distribution of Kilometers Driven')
plt.xlabel('KM Driven')
plt.ylabel('Frequency')
plt.show()

# Distribution of Registration Year
plt.figure(figsize=(10, 6))
sns.histplot(df['Registration_Year'], bins=20, kde=True, color='purple')
plt.title('Registration Year Distribution')
plt.xlabel('Registration Year')
plt.ylabel('Frequency')
plt.show()

# Categorical Feature Distribution
plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='Fuel_Type', palette='Set2')
plt.title('Fuel Type Distribution')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='Body_Type', palette='Set3')
plt.title('Body Type Distribution')
plt.xlabel('Body Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()



# Price vs KM Driven
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='KM_Driven', y='Price', hue='Fuel_Type')
plt.title('Price vs KM Driven')
plt.xlabel('Kilometers Driven')
plt.ylabel('Price (in Lakh)')
plt.show()

# Price vs Engine
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Engine', y='Price', hue='Transmission')
plt.title('Price vs Engine Size')
plt.xlabel('Engine Size (CC)')
plt.ylabel('Price (in Lakh)')
plt.show()

# Price vs Model Year
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Model_Year', y='Price', palette='viridis')
plt.title('Price vs Model Year')
plt.xlabel('Model Year')
plt.ylabel('Price (in Lakh)')
plt.xticks(rotation=45)
plt.show()

# Categorical Price Comparison

# Price by Fuel Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Fuel_Type', y='Price', palette='coolwarm')
plt.title('Price by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Price (in Lakh)')
plt.show()

# Price by Body Type
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='Body_Type', y='Price', palette='Spectral')
plt.title('Price by Body Type')
plt.xlabel('Body Type')
plt.ylabel('Price (in Lakh)')
plt.xticks(rotation=45)
plt.show()

# Correlation Analysis for Numerical Features
plt.figure(figsize=(10, 8))
sns.heatmap(df[['KM_Driven', 'Engine', 'Registration_Year', 'Model_Year', 'Price']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Numerical Features')
plt.show()

# Summary statistics for categorical features
print("\nUnique Counts for Categorical Features:")
for col in ['Fuel_Type', 'Body_Type', 'Transmission', 'Brand', 'City']:
    print(f"{col}: {df[col].nunique()} unique values")

# Display unique values for categorical columns (optional)
print("\nUnique Values for `Brand`:")
print(df['Brand'].unique())

print("\nUnique Values for `Body_Type`:")
print(df['Body_Type'].unique())
