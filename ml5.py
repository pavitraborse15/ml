
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dataset
data = {
    "Age": [22, 25, 28, 32, 35, 38, 40, 42, 45, 24],
    "Salary": [20000, 25000, 30000, 40000, 45000, 50000, 55000, 60000, 65000, 22000],
    "Purchased": [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
X = df[["Age", "Salary"]]
y = df["Purchased"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))
print("F1 Score:", f1_score(y_test, pred, zero_division=0))

# Predict a new customer
new_customer = pd.DataFrame({
    "Age": [34],
    "Salary": [45000]
})

print("New Customer:", model.predict(new_customer)[0])