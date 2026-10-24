import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("housing.csv")

# Select input and output
X = df[["median_income"]]
y = df["median_house_value"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices
pred = model.predict(X_test)

# Display results
print("Predicted Prices:", pred[:5])
print("Mean Absolute Error:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# Plot graph
plt.scatter(X_test, y_test)
plt.plot(X_test, pred, color="red")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.show()