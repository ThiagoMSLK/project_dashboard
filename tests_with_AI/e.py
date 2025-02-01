from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Baixar dados do ativo
ticker = "AAPL"
ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)

# Definir o índice de datas com frequência diária (dias úteis)
ativo.index = pd.to_datetime(ativo.index.date)
ativo = ativo.asfreq('B')  # 'B' para dias úteis (business days)

# Preencher valores ausentes, se houver
ativo.ffill(inplace=True)

# Normalizar os dados (Min-Max)
scaler = MinMaxScaler()
ativo_normalizado = scaler.fit_transform(ativo)
ativo_normalizado = pd.DataFrame(ativo_normalizado, columns=ativo.columns, index=ativo.index)


# Teste de estacionariedade para cada coluna
for coluna in ativo_normalizado.columns:
    resultado = adfuller(ativo_normalizado[coluna].dropna())
    print(f"Teste ADF para {coluna}:")
    print(f"Estatística ADF: {resultado[0]}")
    print(f"Valor-p: {resultado[1]}")
    if resultado[1] > 0.05:
        print("Os dados NÃO são estacionários. Considere diferenciar.")
    else:
        print("Os dados são estacionários.")
    print("\n")