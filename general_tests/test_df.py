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

# 1.4 Difinição do Ticker
# ticker = "AAPL"
ticker = "PETR4.SA"
# ticker = ["AAPL", "PETR4.SA"]
# 1.5 Teste historico
# ativo = yf.Ticker(ticker)
# historico = ativo.history(period="max")
# primeiro_dia = historico.index[0].date()

# print(primeiro_dia)

# 2. Retornar as informações da API
df = yf.download(ticker, start=None, end=end_date)

# df.fillna(method="ffill", inplace=True)

# df.index = df.index.tz_localize('UTC').tz_convert('America/Sao_Paulo')
# df.dropna(inplace=True)
df.index = pd.to_datetime(df.index)

# print(df.head(20))
print(df.info())

# print(df['Close'].head(1))

# print(df.index[0].date())

# print(df.columns)


# df.history(period='1d')

# sns.lineplot(data=df["Close"].astype(float))

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x=df.index, y=df["Close"])
plt.title("Gráfico de Área")
plt.xlabel("Data")
plt.ylabel("Close")

plt.show()