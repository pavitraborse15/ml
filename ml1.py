import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Create dataset
df = pd.DataFrame({
    'Age': [22, 25, np.nan, 30, 35, 40, 28, np.nan],
    'Salary': [25000, 30000, 35000, np.nan, 50000, 60000, 45000, 38000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Department': ['IT', 'HR', 'IT', 'Sales', 'HR', 'Sales', 'IT', 'HR'],
    'Purchased': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
})

print("Original Data:\n", df)

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df.drop_duplicates(inplace=True)

# Data Transformation
df['Salary_Log'] = np.log1p(df['Salary'])

# Feature Encoding
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Purchased'] = le.fit_transform(df['Purchased'])
df = pd.get_dummies(df, columns=['Department'], dtype=int)

# Feature Scaling
scaler = StandardScaler()
df[['Age', 'Salary', 'Salary_Log']] = scaler.fit_transform(
    df[['Age', 'Salary', 'Salary_Log']]
)

# Dataset Preparation
X = df.drop('Purchased', axis=1)
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:\n", X_train)
print("\nTesting Data:\n", X_test)
print("\nPreprocessing Completed!")