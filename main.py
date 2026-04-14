import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# -----------------------------
# STEP 1: Load dataset
# -----------------------------
data = pd.read_csv('data/energy.csv', parse_dates=['Datetime'], index_col='Datetime')

# -----------------------------
# STEP 2: Resample hourly & clean
# -----------------------------
data = data.resample('h').mean()
data = data.ffill()

# -----------------------------
# STEP 3: Feature Engineering
# -----------------------------

# Time-based features
data['hour'] = data.index.hour
data['day'] = data.index.dayofweek

# 🔥 IMPORTANT: Lag Features (this fixes your model)
data['lag_1'] = data['Energy'].shift(1)
data['lag_2'] = data['Energy'].shift(2)
data['lag_3'] = data['Energy'].shift(3)

# Remove null values created by lagging
data = data.dropna()

# -----------------------------
# STEP 4: Define Features & Target
# -----------------------------
X = data[['hour', 'day', 'lag_1', 'lag_2', 'lag_3']]
y = data['Energy']

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# STEP 6: Build Model
# -----------------------------
model = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# STEP 7: Predictions
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# STEP 8: Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# STEP 9: Save Model
# -----------------------------
joblib.dump(model, 'models/energy_model.pkl')

# -----------------------------
# STEP 10: Visualization
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("Energy Consumption Forecasting")
plt.xlabel("Time Steps")
plt.ylabel("Energy")

# Save graph
plt.savefig('images/prediction.png')

# Show graph
plt.show()