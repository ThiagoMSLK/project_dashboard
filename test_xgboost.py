import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import BDay

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#####TESTES DE IMPORTAÇÃO DE DADOS#####

ticker = "AAPL"

ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)
ativo.index = pd.to_datetime(ativo.index)

#####TESTES DE NORMALIZAÇÃO E PADRONIZAÇÃO#####
ativo_colunas = ativo.columns
ativo_index = ativo.index

ativo_normalizado = pd.DataFrame(MinMaxScaler().fit_transform(ativo), columns=ativo_colunas)

ativo_normalizado.index = ativo_index.date
ativo = ativo_normalizado

# Separando preditores e alvo
# colunas_drop = ["Dividends", "Stock Splits"]
# ativo.drop(columns=colunas_drop, inplace=True)

# Separando preditores e alvo
preditor = ativo
alvo = ativo

# Dividindo dados
X_treino, X_teste, Y_treino, Y_teste = train_test_split(preditor, alvo, test_size=0.33, random_state=2025)


#####TESTES XGBOOST#####
# Modelo XGBoost + Treinamento
xgb_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=6, random_state=2025))
xgb_model.fit(X_treino, Y_treino)


y_pred = xgb_model.predict(X_teste)

ultimo_dia = ativo.index[-1]  # Última data no índice
proximo_dia_util = ultimo_dia + BDay(1)  # Adiciona 1 dia útil

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
nova_linha = pd.DataFrame(nova_linha, index=[0])
nova_linha.set_index("Data", inplace=True)


# test_pred_array = xgb_model.predict(nova_linha)
# test_pred = pd.DataFrame(test_pred_array, columns=ativo.columns, index=[proximo_dia_util.date()])

test_pred = pd.DataFrame(xgb_model.predict(nova_linha), columns=ativo.columns, index=[proximo_dia_util.date()])


# Previsões
previsao_df = pd.DataFrame(y_pred, columns=ativo.columns, index=Y_teste.index)

# # Métricas
RMSE = np.sqrt(mean_squared_error(Y_teste, y_pred))
# print("XGBoost RMSE:", RMSE)

MAE = mean_absolute_error(Y_teste, y_pred)
# print("XGBoost MAE:", MAE)

R2 = r2_score(Y_teste, y_pred)
# print("XGBoost R2:", R2)

# print(ativo.columns)
# print(y_pred.shape)

# print(previsão_df.shape)

# Gráfico
st.area_chart(ativo["Adj Close"], color="#ffaa00")
st.area_chart(previsao_df["Adj Close"], color="#ff0000")

st.write("## Teste Previsões")

st.write(ativo_index.shape, Y_teste.shape, y_pred.shape)
st.write(proximo_dia_util.date())

st.write("## Dados para previssão")
st.write(nova_linha)
st.write("## Previsão")
st.write(test_pred)

st.write("## Métricas do Modelo")
st.write(f"**RMSE:** {RMSE}")
st.write(f"**MAE:** {MAE}")
st.write(f"**R2:** {R2}")
# :.2f
st.write("## Dados Reais")
st.write(ativo)