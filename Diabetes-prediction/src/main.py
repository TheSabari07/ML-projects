import pandas as pd

# Load the dataset
diabetes_df = pd.read_csv('../data/diabetes.csv')

# Preview first 5 rows
print(diabetes_df.head())

# Check dataset shape
print("Shape of dataset:", diabetes_df.shape)

# Get basic info
print(diabetes_df.info())

# Summary statistics
print(diabetes_df.describe())

# Value counts of target class
print("Class distribution:")
print(diabetes_df['Outcome'].value_counts())
