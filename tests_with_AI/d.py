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

# Função para ajustar e prever com ARIMA
def prever_arima(serie, ordem=(5, 1, 0), dias_previsao=10):
    modelo = ARIMA(serie, order=ordem)
    modelo_ajustado = modelo.fit()
    previsao = modelo_ajustado.forecast(steps=dias_previsao)
    return previsao


# Prever cada coluna
dias_previsao = 20
previsoes = {}

for coluna in ativo_normalizado.columns:
    try:
        previsoes[coluna] = prever_arima(ativo_normalizado[coluna], dias_previsao=dias_previsao)
    except Exception as e:
        print(f"Erro ao prever {coluna}: {e}")

# Criar DataFrame com as previsões
if previsoes:
    previsoes_df = pd.DataFrame(previsoes)
else:
    print("Nenhuma previsão foi gerada devido a erros.")

# Desnormalizar as previsões
previsoes_desnormalizadas = scaler.inverse_transform(previsoes_df)

# Converter de volta para DataFrame
previsoes_desnormalizadas = pd.DataFrame(previsoes_desnormalizadas, columns=ativo.columns, index=previsoes_df.index)

print(previsoes_desnormalizadas["Adj Close"])

# Plotar previsões
plt.figure(figsize=(14, 7))
#sns.lineplot(data=ativo["Adj Close"], dashes=False, color="blue")
sns.lineplot(data=previsoes_desnormalizadas["Adj Close"], dashes=True, color="red")
plt.title(f"Previsões ARIMA para {ticker}")
plt.show()

