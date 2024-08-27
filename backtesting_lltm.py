import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta

# 1. Descargar datos históricos de Meta Platforms (META)
ticker = 'META'
data = yf.download(ticker, start=datetime.now() - timedelta(days=365), end=datetime.now())

# 2. Preprocesamiento de los datos
data = data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Crear las secuencias para entrenar el modelo LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

sequence_length = 60  # Usar los últimos 60 días para predecir el siguiente
X, y = create_sequences(scaled_data, sequence_length)

# Remodelar los datos para el LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 3. Construcción del modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# 4. Entrenar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, batch_size=1, epochs=1)

# 5. Backtesting
# Simulación para los últimos 30 días como período de prueba
test_start = len(data) - 90  # Últimos 90 días para backtesting
test_data = scaled_data[test_start:]
X_test = []
y_test = data['Close'].values[test_start + sequence_length:]

for i in range(sequence_length, len(test_data)):
    X_test.append(test_data[i-sequence_length:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# 6. Evaluación del backtesting
# Simulación de las señales de compra y venta
initial_investment = 1000  # en euros
cash = initial_investment
stock = 0

for i in range(len(predictions) - 1):
    if stock == 0 and predictions[i + 1][0] > predictions[i][0]:  # Señal de compra
        stock = cash / predictions[i][0]
        cash = 0
        print(f"Buying on day {i} at ${predictions[i][0]:.2f}, stocks acquired: {stock:.2f}")
    elif stock > 0 and predictions[i + 1][0] < predictions[i][0]:  # Señal de venta
        cash = stock * predictions[i][0]
        stock = 0
        print(f"Selling on day {i} at ${predictions[i][0]:.2f}, cash available: ${cash:.2f}")

# Valor final de la cartera
final_value = cash + (stock * predictions[-1][0])
print(f"\nInitial Investment: €{initial_investment:.2f}")
print(f"Final Portfolio Value: €{final_value:.2f}")
print(f"Net Profit: €{final_value - initial_investment:.2f}")
