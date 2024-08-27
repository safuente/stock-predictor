import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
from datetime import datetime, timedelta

# Descargar datos históricos para una acción específica
ticker = 'AAPL'  # Cambiar por la acción que prefieras
data = yf.download(ticker, start=datetime.now() - timedelta(days=365), end=datetime.now())

# Calcular indicadores
data['MACD'], data['MACD_signal'], _ = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
data['Stochastic_K'], data['Stochastic_D'] = ta.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
data['RSI'] = ta.RSI(data['Close'], timeperiod=14)

# Estrategia de trading basada en las condiciones dadas
def trading_strategy(data, investment_amount, stop_loss_percent):
    positions = []
    cash = investment_amount
    stock = 0

    for i in range(1, len(data) - 7):
        # Verificamos señales de entrada
        if data['MACD'][i] > data['MACD_signal'][i] and data['Stochastic_K'][i] < 20 and data['RSI'][i] < 30:
            buy_price = data['Close'][i + 1]
            stop_loss = buy_price * (1 - stop_loss_percent / 100)
            num_shares = cash // buy_price
            cash -= num_shares * buy_price
            stock += num_shares
            positions.append({
                'Date': data.index[i + 1],
                'Action': 'BUY',
                'Price': buy_price,
                'Shares': num_shares,
                'Stop Loss': stop_loss
            })

        # Verificamos señales de salida
        if stock > 0 and (data['MACD'][i] < data['MACD_signal'][i] or data['Stochastic_K'][i] > 80 or data['RSI'][i] > 70):
            sell_price = data['Close'][i + 1]
            cash += stock * sell_price
            positions.append({
                'Date': data.index[i + 1],
                'Action': 'SELL',
                'Price': sell_price,
                'Shares': stock,
            })
            stock = 0

    # Verificar si queda alguna posición sin cerrar
    if stock > 0:
        sell_price = data['Close'][-1]
        cash += stock * sell_price
        positions.append({
            'Date': data.index[-1],
            'Action': 'SELL',
            'Price': sell_price,
            'Shares': stock,
        })
        stock = 0

    final_value = cash
    return positions, final_value

# Ejecución de la estrategia
investment_amount = 1000  # Euros
stop_loss_percent = 2  # 2% stop loss
positions, final_value = trading_strategy(data, investment_amount, stop_loss_percent)

# Mostrar las órdenes generadas
for position in positions:
    print(f"{position['Date'].date()} - {position['Action']} {position['Shares']} shares at ${position['Price']:.2f}. Stop Loss: ${position.get('Stop Loss', 'N/A'):.2f}")

print(f"\nInitial Investment: €{investment_amount:.2f}")
print(f"Final Portfolio Value: €{final_value:.2f}")
print(f"Net Profit: €{final_value - investment_amount:.2f}")
