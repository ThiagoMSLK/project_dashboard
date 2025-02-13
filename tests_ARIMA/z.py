# 1.1 Importação das bibliotecas internas
import datetime as dt
# from time import sleep

# 1.2 Importação das bibliotecas externas
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# 1.3 Importação das bibliotecas de Machine Learning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

ativo = "AAPL"

df = yf.Ticker(ativo).history(period="max", auto_adjust=False)

def encontrar_d(serie, max_d=3):
    """Determina o número de diferenciações (d) necessárias para tornar a série estacionária."""
    d = 0
    adf_teste = adfuller(serie.dropna())

    while adf_teste[1] > 0.05 and d < max_d:  # Diferencia até ficar estacionária
        d += 1
        serie = serie.diff().dropna()
        adf_teste = adfuller(serie)
        
    d = d
    return d

d = encontrar_d(df["Close"])

def encontrar_p_q(serie):
    """Determina os valores de p e q com base nos gráficos PACF e ACF."""

    if d is None:
       encontrar_d()
        
    serie_d = serie.copy()
    for _ in range(d):
        serie_d = serie_d.diff().dropna()
    
    # Escolhendo o número de lags
    # Regra prática para escolher nlags
    df_len = len(serie_d)
    nlags = int(min(10 * np.log10(df_len), df_len - 1))
    # Garante pelo menos 10 lags
    nlags = max(nlags, 10)

    pacf_vals, confint = pacf(serie_d, nlags=10, alpha=0.05)
    p = np.argmax(np.abs(pacf_vals) < np.abs(confint[:, 1] - pacf_vals).mean()) or 1

    acf_vals, confint = acf(serie_d, nlags=10, alpha=0.05)
    q = np.argmax(np.abs(acf_vals) < np.abs(confint[:, 1] - acf_vals).mean()) or 1    

    return p, q

p, q = encontrar_p_q(df["Close"])


print(f"Parametro P: {p}")
print(f"Parametro Q: {q}")
print(f"Parametro D: {d}")
