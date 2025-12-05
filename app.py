import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Stock Market Analysis & Forecasting', page_icon='📊')

st.title('📊 Stock Market Analysis & Forecasting Application')
st.markdown('Welcome! This app provides real-time and historical market data analysis, multi-model time series forecasting, and interactive model comparison.')

# Sidebar inputs
with st.sidebar:
    st.header('Settings')
    ticker_symbol = st.text_input('Enter Ticker Symbol', 'AAPL')
    start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('2022-12-31'))
    model_selection = st.selectbox('Select Model', ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'])

# Fetch historical market data
def fetch_data(ticker_symbol, start_date, end_date):
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data

# Display line charts, candlestick charts, moving averages, and technical indicators
def display_charts(data):
    st.subheader('Line Chart')
    st.line_chart(data['Close'])
    st.subheader('Candlestick Chart')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close')
    ax.fill_between(data.index, data['Low'], data['High'], alpha=0.2)
    ax.set_title('Candlestick Chart')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
    st.subheader('Moving Averages')
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(data.index, data['MA_50'], label='MA_50')
    ax.plot(data.index, data['MA_200'], label='MA_200')
    ax.set_title('Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
    st.subheader('Technical Indicators')
    data['RSI'] = data['Close'].pct_change().rolling(window=14).apply(lambda x: x.ewm(alpha=1/14-1, adjust=False).std())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['RSI'], label='RSI')
    ax.axhline(y=30, color='red', linestyle='--')
    ax.axhline(y=70, color='green', linestyle='--')
    ax.set_title('Relative Strength Index (RSI)')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.legend()
    st.pyplot(fig)

# Perform multi-model time series forecasting
def arima_forecast(data):
    model = ARIMA(data['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

def sarima_forecast(data):
    model = SARIMAX(data['Close'], order=(5,1,0), seasonal_order=(1,1,1,12))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

def prophet_forecast(data):
    model = Prophet()
    model.fit(data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}))
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast['yhat']

def lstm_forecast(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    X = []
    y = []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])
    X = np.array(X).reshape(-1, 60, 1)
    y = np.array(y)
    model = Sequential()
    model.add(LSTM(50, input_shape=(60, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    forecast = []
    for i in range(30):
        inputs = data_scaled[-60:, 0]
        inputs = inputs.reshape((1, 60, 1))
        output = model.predict(inputs)
        forecast.append(output[0, 0])
        data_scaled = np.append(data_scaled, output[0, 0])
    return np.array(forecast)

# Display forecast plots and evaluation metrics
def display_forecast(data, forecast):
    st.subheader('Forecast Plot')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Actual')
    ax.plot(pd.date_range(start=data.index[-1]+pd.DateOffset(days=1), periods=30), forecast, label='Forecast')
    ax.set_title('Forecast Plot')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
    st.subheader('Evaluation Metrics')
    forecast_eval = pd.DataFrame({'Actual': data['Close'].iloc[-30:], 'Forecast': forecast})
    rmse = np.sqrt(mean_squared_error(forecast_eval['Actual'], forecast_eval['Forecast']))
    mae = mean_absolute_error(forecast_eval['Actual'], forecast_eval['Forecast'])
    st.write('RMSE: ', rmse)
    st.write('MAE: ', mae)

# Main content
if st.button('Analyze', type='primary'):
    data = fetch_data(ticker_symbol, start_date, end_date)
    display_charts(data)
    if model_selection == 'ARIMA':
        forecast = arima_forecast(data)
    elif model_selection == 'SARIMA':
        forecast = sarima_forecast(data)
    elif model_selection == 'Prophet':
        forecast = prophet_forecast(data)
    elif model_selection == 'LSTM':
        forecast = lstm_forecast(data)
    display_forecast(data, forecast)

# Show example
with st.expander('See example'):
    st.write('Example data here...')