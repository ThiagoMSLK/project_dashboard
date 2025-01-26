import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#####TESTES DE IMPORTAÇÃO DE DADOS#####

ticker = "AAPL"

ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)
ativo.index = pd.to_datetime(ativo.index.date)

#####TESTES DE NORMALIZAÇÃO E PADRONIZAÇÃO#####
ativo_colunas = ativo.columns

ativo_normalizado = pd.DataFrame(MinMaxScaler().fit_transform(ativo), columns=ativo_colunas)
ativo_padronizado = pd.DataFrame(StandardScaler().fit_transform(ativo), columns=ativo_colunas)

ativo = ativo_normalizado
# ativo = ativo_padronizado

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


# y_pred = xgb_model.predict(X_teste)

# Previsões
n_dias_previsao = 10
ultima_data = ativo.index[-1]
novas_datas = pd.date_range(start=ultima_data, periods=n_dias_previsao + 1, freq="B")[1:]

ultimo_valor = preditor.iloc[-1].values.reshape(1, -1)
y_pred = []

for _ in range(n_dias_previsao):
    proxima_previsao = xgb_model.predict(ultimo_valor)
    y_pred.append(proxima_previsao[0])
    ultimo_valor = np.roll(ultimo_valor, -1)
    ultimo_valor = proxima_previsao[0, -1]

previsao_df = pd.DataFrame({"Adj Close": y_pred.flatten()}, index=novas_datas)


# previsão_df = pd.DataFrame(y_pred, columns=ativo.columns, index=Y_teste.index)

# # Métricas
# RMSE = np.sqrt(mean_squared_error(Y_teste, y_pred))
# print("XGBoost RMSE:", RMSE)

# MAE = mean_absolute_error(Y_teste, y_pred)
# print("XGBoost MAE:", MAE)

# R2 = r2_score(Y_teste, y_pred)
# print("XGBoost R2:", R2)

# print(ativo.columns)
# print(y_pred.shape)

# print(previsão_df.shape)

# Gráfico
# sns.lineplot(ativo.index, ativo["Adj Close"]+previsão_df.index, previsão_df["Adj Close"])

fig, ax = plt.subplots()
ax = plt.plot(ativo.index, ativo["Adj Close"], label="Dados Reais", color="blue")
ax = plt.plot(previsao_df.index, previsao_df["Adj Close"], label="Previsões", linestyle="--", color="red")
plt.legend()
# plt.show()
st.pyplot(fig)


st.write("## Dados Reais")
st.write(ativo)