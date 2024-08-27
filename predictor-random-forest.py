import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Fetch historical data for Nvidia Corporation (NVDA)
ticker = 'NVDA'
data = yf.download(ticker)

# Calculate moving averages
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()

# Drop NaN values
data = data.dropna()

# Define features and target
X = data[['Close', 'MA_10', 'MA_50']]
y = data['Close'].shift(-1).dropna()
X = X[:-1]

# Initialize and train the model with Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to predict future prices and generate trading signals
def predict_swing_trading(days_ahead, investment_amount):
    future_dates = []
    future_prices = []

    # Start with the current date and predict from the next day
    current_date = datetime.now().date()
    start_date = current_date + timedelta(days=1)  # Start predictions from tomorrow

    last_row = X.iloc[-1].copy()

    for i in range(days_ahead):
        predicted_price = model.predict(last_row.values.reshape(1, -1))[0]

        # Save the predicted price
        future_date = start_date + timedelta(days=i)
        future_dates.append(future_date)
        future_prices.append(predicted_price)

        # Update the row for the next prediction
        last_row['Close'] = predicted_price
        last_row['MA_10'] = (last_row['MA_10'] * 9 + predicted_price) / 10  # Update moving average
        last_row['MA_50'] = (last_row['MA_50'] * 49 + predicted_price) / 50  # Update moving average

    # Find the best entry (buy) and exit (sell) points
    min_price = min(future_prices)
    max_price = max(future_prices)
    min_index = future_prices.index(min_price)
    max_index = future_prices.index(max_price)

    # Calculate a recommended stop loss (e.g., 2% below the buy price)
    stop_loss = min_price * 0.98

    # Calculate the number of shares to buy
    num_shares = investment_amount // min_price

    # Determine the best strategy and calculate expected profit
    if min_index < max_index:
        buy_price = min_price
        sell_price = max_price
        buy_date = future_dates[min_index]
        sell_date = future_dates[max_index]
        expected_profit = (sell_price - buy_price) * num_shares
        strategy = (f"Buy on {buy_date} at ${buy_price:.2f} and sell on {sell_date} "
                    f"at ${sell_price:.2f}. Expected Profit: ${expected_profit:.2f}")
        stop_loss_info = f"Recommended Stop Loss: ${stop_loss:.2f} (2% below buy price)"
    else:
        strategy = "The model suggests no optimal swing trading strategy within the given period."
        expected_profit = 0
        stop_loss_info = "No stop loss recommended as no buy signal was generated."

    # Create a DataFrame for the future predictions
    future_predictions = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_prices})
    future_predictions.set_index('Date', inplace=True)

    return future_predictions, strategy, expected_profit, stop_loss_info, num_shares

# Request investment amount from the user
investment_amount = float(input("Enter the amount of money you want to invest (in USD): "))

# Example: Predict swing trading strategy for the next 7 days
days_ahead = 7
future_predictions, strategy, expected_profit, stop_loss_info, num_shares = predict_swing_trading(days_ahead, investment_amount)

# Display the future predictions and trading strategy
print(f"Predicted prices for the next {days_ahead} days starting from tomorrow:")
print(future_predictions)
print("\nSwing Trading Strategy:")
print(strategy)
print(stop_loss_info)
print(f"Number of Shares to Buy: {num_shares}")
print(f"Expected Profit: ${expected_profit:.2f}")
