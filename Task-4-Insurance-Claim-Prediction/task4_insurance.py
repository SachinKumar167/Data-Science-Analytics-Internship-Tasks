import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Load dataset
insurance = pd.read_csv("insurance.csv")

# Convert categorical columns to numbers
insurance = pd.get_dummies(
    insurance,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

# Features and Target
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)