# 1.1 Importação das bibliotecas internas
import datetime as dt
from time import sleep

# 1.2 Importação das bibliotecas externas
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


stocks = pd.read_csv('nasdaq_screener_10-01-2025.csv')['Symbol'].values

# 1.3 Difinição do começo e fim da busca
# start_date = dt.datetime(2022, 1, 1)
# end_date = dt.datetime.today()

start_date, end_date = st.slider(
    "Escolha o intervalo de datas:",
    min_value=dt.date(2020, 1, 1),
    max_value=dt.date.today(),
    value=(dt.date(2023, 1, 1), dt.date.today()),
    format="YYYY-MM-DD"
)

# 2. Retornar as informações da API
selecao_ticker = st.selectbox("Escolha a criptomoeda", options= stocks)


df = yf.download(selecao_ticker, start=start_date, end=end_date)

sleep(1)

ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']

selecao_ohlcv = st.selectbox('Escolha os dados que deseja visualizar', options=ohlcv)


st.title(f'Análise de Dados do {selecao_ticker}')

st.line_chart(df[selecao_ohlcv])

st.write(f'## Dados do {selecao_ticker}')
st.write(df)
