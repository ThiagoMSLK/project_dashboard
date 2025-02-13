import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#####TESTES DE IMPORTAÇÃO DE DADOS#####
ticker = "AAPL"

ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)

#####TESTES DE NORMALIZAÇÃO E PADRONIZAÇÃO#####
ativo_colunas = ativo.columns

ativo_normalizado = pd.DataFrame(MinMaxScaler().fit_transform(ativo), columns=ativo_colunas)

ativo_padronizado = pd.DataFrame(StandardScaler().fit_transform(ativo), columns=ativo_colunas)

ativo = ativo
#####TESTES DE COLUNAS#####
# colunas_drop = ['Open', 'High', 'Low', 'Close', 'Adj Close','Volume', 'Dividends', 'Stock Splits']
colunas_drop = ["Dividends", "Stock Splits"]

ativo.drop(columns=colunas_drop, inplace=True)

#####TESTES DE MODELOS#####
preditor_original = ativo.drop("Adj Close", axis=1)
alvo_original = ativo["Adj Close"]


X_treinoO, X_testeO, Y_treinoO, Y_testeO = train_test_split(preditor_original, alvo_original, test_size=0.33, random_state=2025)

modelo_linear = LinearRegression()
modelo_forest = RandomForestRegressor()

modelo_linear.fit(X_treinoO, Y_treinoO)
modelo_forest.fit(X_treinoO, Y_treinoO)

predicao_linear = modelo_linear.predict(X_testeO)
predicao_forest = modelo_forest.predict(X_testeO)

print("Linear RMSE:", np.sqrt(mean_squared_error(Y_testeO, predicao_linear)))
print("Linear MAE:", mean_absolute_error(Y_testeO, predicao_forest))
print("Linear R2:", r2_score(Y_testeO, predicao_linear))
print("")
print("Forest RMSE:", np.sqrt(mean_squared_error(Y_testeO, predicao_forest)))
print("Forest MAE:", mean_absolute_error(Y_testeO, predicao_forest))
print("Forest R2:", r2_score(Y_testeO, predicao_forest))



#####AREA DE TESTES#####

# print(ativo_colunas)
# print(preditor.head(5),alvo.head(5))

# print(ativo.shape)

# print(ativo.head(5))

# print(ativo.info())

#####AREA DE GRÁFICOS#####

# sns.pairplot(ativo)

# plt.show()