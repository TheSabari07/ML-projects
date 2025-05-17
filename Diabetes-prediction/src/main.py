import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
diabetes_df = pd.read_csv('../data/diabetes.csv')

# Step 2: Separate features (X) and labels (y)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# Step 3: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Step 7: Predict diabetes for a new patient
new_data = pd.DataFrame({
    'Pregnancies': [2],
    'Glucose': [120],
    'BloodPressure': [70],
    'SkinThickness': [20],
    'Insulin': [85],
    'BMI': [32.0],
    'DiabetesPedigreeFunction': [0.5],
    'Age': [25]
})

prediction = model.predict(new_data)
print("Prediction for new patient:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
