import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import BDay

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ticker = "AAPL"

ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)
ativo.index = pd.to_datetime(ativo.index.date)


#####TESTES DE NORMALIZAÇÃO E PADRONIZAÇÃO#####
ativo_colunas = ativo.columns
ativo_index = np.array(ativo.index).reshape(-1, 1)

#####TESTES MODELO LINEAR#####
xtreino, xteste, ytreino, yteste = train_test_split(ativo_index, ativo, test_size=0.33, random_state=42)

modelo = LinearRegression()
modelo.fit(xtreino, ytreino)

#####TESTES NOVOS DADOS#####
ultimo_dia = ativo.index[-1]  
proximo_dia_util = ultimo_dia + BDay(1)  
novos_dias = pd.DataFrame()

for _ in range(10):
    proximo_dia_util += BDay(1)
    nova_linha = {
        "Data": proximo_dia_util.date(),
        "Open": 0.0,       # Valor inicial padrão
        "High": 0.0,       # Valor inicial padrão
        "Low": 0.0,        # Valor inicial padrão
        "Close": 0.0,      # Valor inicial padrão
        "Adj Close": 0.0,  # Valor inicial padrão
        "Volume": 0.0,     # Valor inicial padrão
        "Dividends": 0.0,  # Valor inicial padrão
        "Stock Splits": 0.0,  # Valor inicial padrão
    }

    novos_dias = pd.concat([novos_dias, pd.DataFrame(nova_linha, index=[0])]) 

novos_dias.set_index("Data", inplace=True)

previsao = modelo.predict(novos_dias)

previsao = pd.DataFrame(previsao, columns=ativo_colunas, index=novos_dias.index)

print(previsao["Adj Close"])
#print(ativo.index)
#print(novos_dias.index)


#####TESTES GRAFICOS#####

#graf = sns.lineplot(data=previsao[["Adj Close"]])
#plt.show()
#plt.savefig("grafico.png")
#plt.close()
