import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
data = pd.read_csv("timeseries.csv")
series = data["value"]

# Train ARIMA model
model = ARIMA(series, order=(2, 1, 2))
model_fit = model.fit()

# Forecast (aligned properly)
forecast = model_fit.predict(start=1, end=len(series) - 1)

# Calculate residuals
residuals = series.iloc[1:] - forecast

# Threshold calculation
sigma = np.std(residuals)
threshold = 3 * sigma

# Detect anomalies
anomalies = np.abs(residuals) > threshold

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(series, label="Actual Data")
plt.plot(forecast.index, forecast, label="Forecast")

plt.scatter(
    series.index[1:][anomalies],
    series.iloc[1:][anomalies],
    color="red",
    label="Anomalies",
)

plt.legend()
plt.show()
