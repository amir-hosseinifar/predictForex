import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to get current price of a forex pair
def get_current_price(currency_pair):
    ticker = yf.Ticker(currency_pair)
    data = ticker.history(period="1d", interval="1m")  # Fetching minute data for today
    if data.empty:
        return None
    return data['Close'].iloc[-1].item()  # Convert Series to float

# Function to get historical forex data for training
def get_forex_data(currency_pair, start_date, end_date, interval='5m'):
    data = yf.download(currency_pair, start=start_date, end=end_date, interval=interval)
    return data

# Parameters
currency_pair = "EURUSD=X"  # Specify the currency pair
end_date = pd.to_datetime("today").normalize()  # Current date
start_date = end_date - pd.DateOffset(months=1)  # 3 months before today
interval = '5m'  # Timeframe for short-term trading

# Load historical forex data
data = get_forex_data(currency_pair, start_date, end_date, interval)

# Check if data was retrieved successfully
if data.empty:
    print("No historical data found for the specified currency pair and date range.")
else:
    # Prepare the dataset
    prices = data['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices.reshape(-1, 1))

    # Create sequences for training and testing (next candlestick prediction)
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data) - 1):  # Predict one step ahead
            X.append(data[i-seq_len:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_len = 60  # Reduced sequence length for short-term trading
    X, y = create_sequences(scaled_data, seq_len)

    # Check if X and y have enough data for reshaping
    if X.shape[0] == 0 or X.shape[1] == 0:
        print("Insufficient data for creating sequences.")
    else:
        # Reshape data for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split the data into training and testing sets
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Build the LSTM model with reduced units
        model = Sequential([
            LSTM(units=30, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=20, return_sequences=True),
            Dropout(0.2),
            LSTM(units=20, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model with reduced epochs
        model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

        # Initialize list to store predictions
        rolling_predictions = []
        last_sequence = X_test[-1]  # Start with the last test sequence

        # Generate only one next candlestick prediction
        next_candle = model.predict(last_sequence.reshape(1, seq_len, 1))
        rolling_predictions.append(next_candle[0, 0])
        
        # Inverse transform predictions for comparison
        rolling_predictions = scaler.inverse_transform(np.array(rolling_predictions).reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Get the current price
        current_price = get_current_price(currency_pair)

        # Print current price and predicted price
        if current_price is not None:
            predicted_price = rolling_predictions[-1, 0]  # Last predicted price
            print(f"Current Price of {currency_pair}: {current_price:.5f}")
            print(f"Predicted Price for {currency_pair} (next period): {predicted_price:.5f}")