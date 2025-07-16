
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📈 Stock Price Prediction with LSTM")
st.write("Enter a stock ticker (e.g., AAPL, TCS.NS):")

ticker = st.text_input("Stock Ticker", "AAPL")

if st.button("Predict"):
    st.write("Fetching data...")
    data = yf.download(ticker, start="2018-01-01", end="2023-12-31")
    close_data = data[['Close']].dropna()

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    # Create sequences
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=1, batch_size=32, verbose=0)

    # Predict
    prediction = model.predict(X)
    prediction_prices = scaler.inverse_transform(prediction)

    # Plot
    st.subheader("Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.plot(close_data.index[60:], close_data.values[60:], label="Actual")
    ax.plot(close_data.index[60:], prediction_prices, label="Predicted")
    ax.legend()
    st.pyplot(fig)
