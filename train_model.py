import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("house_data.csv")

print("Dataset:")
print(data.head())

# Features
X = data[
    [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Age",
        "Distance_Center"
    ]
]

# Target
y = data["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("--------------------")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# Save model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")