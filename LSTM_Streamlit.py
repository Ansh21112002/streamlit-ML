import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to create the LSTM model
def create_lstm_model(time_step):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Function to create dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Function to preprocess data and train LSTM model
def train_lstm_model(df, time_step=100):
    data = df.reset_index()['Close']
    data.dropna(inplace=True)

    close_prices = data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices)

    train_size = int(len(scaled_close_prices) * 0.8)
    train_data, _ = scaled_close_prices[0:train_size,:], scaled_close_prices[train_size:len(scaled_close_prices),:]

    X_train, Y_train = create_dataset(train_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = create_lstm_model(time_step)
    model.fit(X_train, Y_train, epochs=10, batch_size=64, verbose=0)

    return model, scaler

# Function to make predictions using the trained LSTM model
def make_predictions(model, scaler, df, time_step=100):
    test_data = scaler.transform(df.values.reshape(-1, 1))
    X_test, Y_test = create_dataset(test_data, time_step)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)
    return test_predict

# Streamlit app
def main():
    st.title("Stock Price Prediction")

    # Dropdown for selecting ticker
    ticker_list = ["^NSEI",
        "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", 
        "BAJFINANCE", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", 
        "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", 
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", 
        "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", 
        "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBIN", 
        "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TATAPOWER", 
        "TCS", "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
    ]

    selected_ticker = st.selectbox("Select Ticker", ticker_list, index=ticker_list.index("^NSEI"))
    start = datetime.now() - timedelta(days=3650)
    end = datetime.now()
    # Fetching data for selected ticker
    if selected_ticker == "^NSEI":
        df = yf.Ticker('^NSEI')
        df=df.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
    else:
        df = yf.Ticker(selected_ticker + ".NS")
        df=df.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

    # Training LSTM model and making predictions
  
    data=df.reset_index()['Close']
    data.dropna(inplace=True)

    st.subheader("OHLC for selected ticker")
    st.write(df)

    close_prices = data.values.reshape(-1, 1)
    train_size = int(len(close_prices) * 0.8)
    train_r2, test_r2 = close_prices[0:train_size,:], close_prices[train_size:len(close_prices),:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices)

    train_size = int(len(scaled_close_prices) * 0.8)
    test_size = len(scaled_close_prices) - train_size
    train_data, test_data = scaled_close_prices[0:train_size,:], scaled_close_prices[train_size:len(scaled_close_prices),:]

    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 100
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    X_r2, Y_r2 = create_dataset(test_r2, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, Y_train, epochs=10,batch_size=64, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Plotting
    

    plt.figure(figsize=(10, 6))
    plt.plot(df.index[len(df)-365:len(df)-30],df['Close'][len(df)-365:len(df)-30],label='orignal')
    # plt.plot(df.index[time_step:len(train_predict)+time_step], data[time_step:len(train_predict)+time_step], label='Train Prediction')
    plt.plot(df.index[len(df)-30:], test_predict[len(test_predict)-30:], label='Test Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('LSTM Model Prediction')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot()

    # Calculating R2 score
    Y_test_float = np.array(Y_r2, dtype=np.float32)
    test_predict_float= np.array(test_predict, dtype=np.float32)

    metric = tf.keras.metrics.R2Score()
    metric.update_state(Y_test_float,test_predict_float )
    result = metric.result()
    st.subheader(f"r2 score for the model for {selected_ticker} is: {result}")

if __name__ == "__main__":
    main()
