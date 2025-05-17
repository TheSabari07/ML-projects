# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


diabetes_df = pd.read_csv('diabetes.csv')


print(diabetes_df.head())
print("Shape of dataset:", diabetes_df.shape)
print(diabetes_df.info())
print(diabetes_df.describe())
print("Class distribution:")
print(diabetes_df['Outcome'].value_counts())


X = diabetes_df.drop('Outcome', axis=1)  
y = diabetes_df['Outcome']             


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)       


print("Training features shape:", X_train_scaled.shape)
print("Testing features shape:", X_test_scaled.shape)
