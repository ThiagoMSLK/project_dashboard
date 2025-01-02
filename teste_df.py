# 1.1 Importação das bibliotecas internas
import datetime as dt

# 1.2 Importação das bibliotecas externas
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 1.3 Difinição do começo e fim da busca
start_date = dt.datetime(2022, 1, 1)
end_date = dt.datetime.today()

# 2. Retornar as informações da API
df = yf.download('BTC-USD', start=start_date, end=end_date)


print(df.head(20))