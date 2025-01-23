import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import requests
import plotly.graph_objects as go
from textblob import TextBlob
from statsmodels.tsa.arima.model import ARIMA

# Load API key from environment variables
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Function to fetch stock symbols based on user input
def fetch_stock_symbols(api_key, keyword):
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={keyword}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data.get('bestMatches', [])

# Function to fetch stock data
def fetch_stock_data(api_key, symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Function to calculate simple moving averages
def calculate_sma(data, window):
    df = pd.DataFrame(data['Time Series (Daily)']).T.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df[f'SMA_{window}'] = df['4. close'].rolling(window=window).mean()
    return df

# Function for ARIMA prediction
def arima_prediction(data):
    df = pd.DataFrame(data['Time Series (Daily)']).T.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    model = ARIMA(df['4. close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    return forecast

# Function for sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Streamlit app
st.title("Simplified Stock Market Analysis")

# Sidebar for stock search
st.sidebar.header("Stock Search")
keyword = st.sidebar.text_input("Search for a stock:")
selected_stock = None
if keyword and API_KEY:
    matches = fetch_stock_symbols(API_KEY, keyword)
    symbols = [match['1. symbol'] for match in matches]
    selected_stock = st.sidebar.selectbox("Select Stock Symbol:", symbols)

if st.sidebar.button("Fetch Data"):
    if API_KEY and selected_stock:
        with st.spinner("Loading data..."):
            data = fetch_stock_data(API_KEY, selected_stock)

        if 'Time Series (Daily)' in data:
            st.success("Data fetched successfully!")

            # Calculate SMA
            sma_df = calculate_sma(data, 20)

            # Plotting
            st.subheader("Stock Price and SMA")
            st.line_chart(sma_df[['4. close', 'SMA_20']])

            # Candlestick chart
            st.subheader("Candlestick Chart")
            fig = go.Figure(data=[go.Candlestick(x=sma_df.index,
                                                  open=sma_df['1. open'],
                                                  high=sma_df['2. high'],
                                                  low=sma_df['3. low'],
                                                  close=sma_df['4. close'])])
            fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig)

            # ARIMA prediction
            st.subheader("ARIMA Price Prediction for Next 5 Days")
            predicted_prices = arima_prediction(data)
            st.write(predicted_prices)

            # Sentiment analysis example
            st.subheader("Market News Sentiment Analysis")
            news_article = "The market is showing positive growth today."  # Example news
            sentiment_score = analyze_sentiment(news_article)
            st.write(f"Sentiment Score: {sentiment_score:.2f}")

        else:
            st.error("Error fetching data. Please check the stock symbol or API key.")
    else:
        st.warning("Please enter a valid stock keyword.")
