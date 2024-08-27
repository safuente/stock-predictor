import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Asegurarse de que el rango de fechas cubra suficiente historial
ticker = 'NVDA'
data = yf.download(ticker, start=datetime.now() - timedelta(days=120), end=datetime.now())

# Revisar si los datos se descargaron correctamente
if data.empty:
    raise ValueError("No data was downloaded. Check the ticker or the date range.")

# Calculate moving averages
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()

# Drop NaN values
data = data.dropna()

# Verificar si los datos después de eliminar NaN no están vacíos
if data.empty:
    raise ValueError("No data available after calculating moving averages and dropping NaN values.")

# Define features and target
X = data[['Close', 'MA_10', 'MA_50']]
y = data['Close'].shift(-1).dropna()
X = X[:-1]

# Asegurarse de que hay suficientes datos para entrenar el modelo
if X.empty or y.empty:
    raise ValueError("No data available to train the model after processing.")

# Initialize and train the model with Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


# Backtesting function
def backtest_strategy(data, initial_investment):
    if data.empty:
        raise ValueError("No data available for backtesting.")

    cash = initial_investment
    stock = 0
    for i in range(len(data) - 1):
        current_row = data.iloc[i]
        next_day_pred = model.predict(current_row[['Close', 'MA_10', 'MA_50']].values.reshape(1, -1))[0]
        if stock == 0 and next_day_pred > current_row['Close']:  # Buy signal
            stock = cash / current_row['Close']
            cash = 0
            print(f"Buying on {current_row.name.date()} at ${current_row['Close']:.2f}, stocks acquired: {stock:.2f}")
        elif stock > 0 and next_day_pred < current_row['Close']:  # Sell signal
            cash = stock * current_row['Close']
            stock = 0
            print(f"Selling on {current_row.name.date()} at ${current_row['Close']:.2f}, cash available: ${cash:.2f}")

    # Final value of the portfolio
    final_value = cash + stock * data.iloc[-1]['Close']
    return final_value


# Apply backtesting on the last 30 days
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()

# Subset the data for the last month
data_last_month = data.loc[start_date:end_date]

# Perform backtesting
initial_investment = 1000  # in euros
final_value = backtest_strategy(data_last_month, initial_investment)

# Display the final results
print(f"\nInitial Investment: €{initial_investment:.2f}")
print(f"Final Portfolio Value: €{final_value:.2f}")
print(f"Net Profit: €{final_value - initial_investment:.2f}")
