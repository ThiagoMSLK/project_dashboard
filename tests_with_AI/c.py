import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Baixar dados do ativo
ticker = "AAPL"
ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)
ativo.index = pd.to_datetime(ativo.index.date)

# Função para ajustar e prever com ARIMA
def prever_arima(serie, ordem=(5, 1, 0), dias_previsao=10):
    modelo = ARIMA(serie, order=ordem)
    modelo_ajustado = modelo.fit()
    previsao = modelo_ajustado.forecast(steps=dias_previsao)
    return previsao

# Prever cada coluna
dias_previsao = 10
previsoes = {}

for coluna in ativo.columns:
    previsoes[coluna] = prever_arima(ativo[coluna], dias_previsao=dias_previsao)

# Criar DataFrame com as previsões
previsoes_df = pd.DataFrame(previsoes)
print(previsoes_df)