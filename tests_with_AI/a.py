import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Dados do ativo
ticker = "AAPL"
ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)

ativo_colunas = ativo.columns
ativo = pd.DataFrame(MinMaxScaler().fit_transform(ativo), columns=ativo_colunas)

# Removendo colunas irrelevantes
ativo.drop(columns=["Dividends", "Stock Splits"], inplace=True)

# Separando preditores e alvo
preditor = ativo.drop("Adj Close", axis=1)
alvo = ativo["Adj Close"]

# Dividindo dados
X_treino, X_teste, Y_treino, Y_teste = train_test_split(preditor, alvo, test_size=0.33, random_state=2025)

# Modelo XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=2025)
xgb_model.fit(X_treino, Y_treino)

# Previsões
y_pred = xgb_model.predict(X_teste)

# Métricas
print("XGBoost RMSE:", np.sqrt(mean_squared_error(Y_teste, y_pred)))
print("XGBoost MAE:", mean_absolute_error(Y_teste, y_pred))
print("XGBoost R2:", r2_score(Y_teste, y_pred))