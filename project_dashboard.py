# 1.1 Importação das bibliotecas internas
import datetime as dt
from time import sleep

# 1.2 Importação das bibliotecas externas
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# TESTES
# stocks = ('BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'SOL1-USD', 'DOT1-USD', 'DOGE-USD', 'LUNA1-USD', 'AVAX-USD')

stocks = pd.read_csv('nasdaq_screener_10-01-2025.csv')['Symbol'].values

# 1.3 Difinição do começo e fim da busca
start_date = dt.datetime(2022, 1, 1)
end_date = dt.datetime.today()

# 2. Retornar as informações da API
selecao = st.selectbox("Escolha a criptomoeda", options= stocks)

sleep(1)

df = yf.download(selecao, start=start_date, end=end_date)


st.title(f'Análise de Dados do {selecao}')

st.line_chart(df['Volume'])

st.write(f'## Dados do {selecao}')
st.write(df)