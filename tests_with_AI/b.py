import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from datetime import timedelta
from pandas.tseries.offsets import BDay

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Baixar dados do ativo
ticker = "AAPL"
ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)

# Normalizar os dados
scaler = MinMaxScaler()
ativo_normalizado = pd.DataFrame(scaler.fit_transform(ativo), columns=ativo.columns, index=ativo.index)


# Separar preditores e alvo
X = ativo_normalizado.drop("Adj Close", axis=1)
y = ativo_normalizado["Adj Close"]

# Dividir em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.33, random_state=2025)

# Treinar modelo XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=10, random_state=2025)
xgb_model.fit(X_treino, y_treino)

# Gerar previsões para os próximos 7 dias úteis
n_dias_previsao = 7
ultima_data = ativo.index[-1]
novas_datas = [ultima_data + BDay(i) for i in range(1, n_dias_previsao + 1)]

# Criar base de preditores para simulação
X_ultimo = X.iloc[-1:].copy()
y_pred = []

for _ in range(n_dias_previsao):
    predicao = xgb_model.predict(X_ultimo)[0]
    y_pred.append(predicao)
    
    # Simular o próximo dia atualizando os preditores
    X_ultimo.iloc[0, 0] = predicao  # Exemplo: Atualizar primeira coluna como proxy
    # Outros ajustes podem ser feitos aqui dependendo dos preditores disponíveis.

previsao_df = pd.DataFrame({"Adj Close": y_pred}, index=novas_datas)

# **Transformar previsões de volta para a escala original**
# Criar um array vazio com a mesma estrutura do dataset original
zeros_array = np.zeros((len(previsao_df), ativo.shape[1]))
zeros_df = pd.DataFrame(zeros_array, columns=ativo.columns)

# Inserir as previsões na coluna "Adj Close"
zeros_df["Adj Close"] = previsao_df["Adj Close"].values

# Reverter a normalização
valores_originais = scaler.inverse_transform(zeros_df)
previsao_df["Adj Close"] = valores_originais[:, ativo.columns.get_loc("Adj Close")]

# Restabelecer os valores originais dos dados históricos
ativo["Adj Close"] = scaler.inverse_transform(ativo)[:, ativo.columns.get_loc("Adj Close")]

# Título do aplicativo
st.title(f"Previsão de {ticker} para os próximos {n_dias_previsao} dias úteis")

# Gráfico 1: Dados históricos
st.subheader("Dados Históricos")
dados_historicos = ativo[["Adj Close"]]
st.line_chart(dados_historicos, y=dados_historicos.columns)

# Gráfico 2: Previsões
st.subheader("Previsões")
previsao_df.index = previsao_df.index.date
st.line_chart(previsao_df, y=previsao_df.columns)

st.write("## Métricas do Modelo")
# Calcular métricas
st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_teste, xgb_model.predict(X_teste))):.2f}")
st.write(f"**MAE:** {mean_absolute_error(y_teste, xgb_model.predict(X_teste)):.2f}")
st.write(f"**R2:** {r2_score(y_teste, xgb_model.predict(X_teste)):.2f}")