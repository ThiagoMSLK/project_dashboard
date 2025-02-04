import yfinance as yf
import pandas as pd
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler

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
ativo_normalizado = scaler.fit_transform(ativo[['Close']])
ativo_normalizado = pd.Series(ativo_normalizado.flatten(), index=ativo.index)

# Usar auto_arima para encontrar a melhor ordem do ARIMA
modelo_auto = auto_arima(
    ativo_normalizado,
    seasonal=False,  # Não usar componente sazonal (SARIMA)
    trace=True,      # Mostrar progresso
    error_action='ignore',  # Ignorar erros durante a busca
    suppress_warnings=True,  # Suprimir avisos
    stepwise=True    # Usar busca stepwise para maior eficiência
)

# Resumo do modelo
print(modelo_auto.summary())

# Fazer previsões
dias_previsao = 10
previsoes = modelo_auto.predict(n_periods=dias_previsao)

# Desnormalizar as previsões
previsoes = scaler.inverse_transform(previsoes.reshape(-1, 1))

# Criar DataFrame com as previsões
previsoes_df = pd.DataFrame(previsoes, columns=['Close'], index=pd.date_range(start=ativo.index[-1] + pd.Timedelta(days=1), periods=dias_previsao, freq='B'))

print("Previsões para os próximos dias:")
print(previsoes_df)