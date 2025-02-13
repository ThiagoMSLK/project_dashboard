import datetime as dt
import pytz

import yfinance as yf
import streamlit as st
import	pandas as pd

ticker = "PETR4.SA"

df = yf.Ticker(ticker).history(period="max", auto_adjust=False)

df.index = pd.to_datetime(df.index.date)

primeiro_dia = df.index[0].date()

start_date, end_date = st.slider("",min_value=None, max_value=dt.date.today(), value=(primeiro_dia, dt.date.today()), format="YYYY-MM-DD")

start_date = pd.to_datetime(start_date).date()
end_date = pd.to_datetime(end_date).date()

df = df.loc[start_date:end_date]

st.area_chart(df['Adj Close'].astype(float))

st.write(df)