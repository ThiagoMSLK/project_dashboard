import datetime as dt

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class SklearnML:
    def __init__(self, serie, coluna, janela=5, test_size=0.1):
        """
        Classe para modelagem de séries temporais com Scikit-Learn.

        Parâmetros:
        - serie: DataFrame contendo a série temporal.
        - coluna: Nome da variável alvo (string).
        - janela: Número de lags (valores passados usados como entrada).
        - test_size: Percentual de dados para teste.
        """
        self.serie = serie
        self.coluna = coluna
        self.colunas_nome = None
        self.janela = janela
        self.test_size = test_size
        self.modelo = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_pred, self.r2, self.rmse, self.mae = None, None, None, None

    def preparar_dados(self):
        """Cria as features de atraso (lags) e divide os dados em treino e teste."""
        df = self.serie[[self.coluna]].copy()  # Seleciona apenas a coluna de interesse
        
        # Criar colunas de lag (valores passados)
        for i in range(1, self.janela + 1):
            df[f"lag_{i}"] = df[self.coluna].shift(i)

        df.dropna(inplace=True)  # Remove valores nulos gerados pelo shift()
        
        # Separar X (features) e y (variável alvo)
        X = df.drop(columns=[self.coluna])
        y = df[self.coluna]

        # Dividir dados em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=50, shuffle=False
        )
    
    def treinar_modelo(self):
        """Treina o modelo de Machine Learning."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Os dados não foram preparados. Chame preparar_dados() antes de treinar o modelo.")

        self.colunas_nome = self.X_train.columns
        self.modelo = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=1000, random_state=42)
        self.modelo.fit(self.X_train, self.y_train)
        
    
    def avaliar_modelo(self):
        """Avalia o modelo treinado."""
        if self.modelo is None:
            raise ValueError("O modelo ainda não foi treinado. Chame treinar_modelo() antes de avaliar.")

        self.y_pred = self.modelo.predict(self.X_test)
        self.r2 = r2_score(self.y_test, self.y_pred)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        self.mae = mean_absolute_error(self.y_test, self.y_pred)

        return self.r2, self.rmse, self.mae, self.y_pred
    
    def prever(self, dias=10):
        """Faz previsões para os próximos dias usando os últimos valores conhecidos."""
        if self.modelo is None:
            raise ValueError("O modelo ainda não foi treinado. Chame treinar_modelo() antes de prever.")

        ultimos_dados = self.serie[[self.coluna]].copy()
        
        for _ in range(dias):
            # Criar features (lags) para previsão
            ultima_janela = ultimos_dados[self.coluna].iloc[-self.janela:].values.reshape(1, -1)
            ultima_janela_df = pd.DataFrame(ultima_janela, columns=self.colunas_nome)
            previsao = self.modelo.predict(pd.DataFrame(data=ultima_janela_df))[0]
            
            # Adicionar previsão ao dataframe
            nova_data = ultimos_dados.index[-1] + dt.timedelta(days=1)
            ultimos_dados.loc[nova_data] = previsao
        
        return ultimos_dados.iloc[-dias:]



OHLCV = ("Adj Close", "Open", "High", "Low", "Close", "Volume")

ticker = "PETR4.SA"

try:
    ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    ativo.index = pd.to_datetime(ativo.index.date)
    ativo.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    ativo = ativo.asfreq("B")
    ativo.ffill(inplace=True)

except Exception as e:
    print(e)

ml = SklearnML(ativo, OHLCV[0], janela=5, test_size=0.1)
ml.preparar_dados()
ml.treinar_modelo()
r2, rmse, mae, y_pred = ml.avaliar_modelo()
previsao = ml.prever(10)
print(previsao)
print(f"R2: {r2*100:.2f}%")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

sns.lineplot(data=previsao, x=previsao.index, y=OHLCV[0], palette="pastel", markers="1", label="Previsão")

#sns.lineplot(data=y_pred, x=y_pred.index,  y=OHLCV[0], palette="pastel", label="Predito")

#sns.lineplot(data=ativo, x=ativo.index, y=OHLCV[0], palette="pastel", markers="1", label="Real")

plt.xticks(rotation=45)
plt.show()

#print(ativo.tail(1))

