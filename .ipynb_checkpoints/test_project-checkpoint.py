# 1.1 Importação das bibliotecas internas
import datetime as dt
# from time import sleep

# 1.2 Importação das bibliotecas externas
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# 1.3 Importação das bibliotecas de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

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
        self.modelo = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1500, activation="identity",learning_rate="constant",learning_rate_init=0.001, random_state=42)
        self.modelo.fit(self.X_train, self.y_train)
        
    
    def avaliar_modelo(self):
        """Avalia o modelo treinado."""
        if self.modelo is None:
            raise ValueError("O modelo ainda não foi treinado. Chame treinar_modelo() antes de avaliar.")

        self.y_pred = self.modelo.predict(self.X_test)
        self.r2 = r2_score(self.y_test, self.y_pred)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        self.mae = mean_absolute_error(self.y_test, self.y_pred)

        return self.r2, self.rmse, self.mae
    
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

# 1.4 Difinição da Tuples OHLCV e do DataFrame Ticker
OHLCV = ("Adj Close", "Open", "High", "Low", "Close", "Volume")

ticker = pd.read_csv("nasdaq_screener_10-01-2025.csv")["Symbol"].values

# 2.1 Configuração da página
st.set_page_config(page_title="Análise de Dados Financeiros", layout="wide")

st.title("Análise de Dados Financeiros")


# 2.2 Difinição dos dados que o usuário deseja visualizar
with st.container():

    st.sidebar.header("Escolha os dados que deseja visualizar")
    box1, box2, box3 = st.columns([1,1,2])
    
    # 2.3 Difinição do Ticker
    with box1:
        # 2.3.1 Colocando o o titulo
        with st.sidebar.container():
            st.write("Escolha o ativo que deseja visualizar")
            
            # 2.3.2 Tamanho das box
            col1, col2 =st.columns([1,1])
            
            # 2.3.3 Seleção do ativo
            with col1:
                ticker_selecionado = st.selectbox("Selecione ou >", options= ticker)
            
            # 2.3.4 Digitar o ativo
            with col2:
                ticker_digitado = st.text_input("Digite")

            # 2.3.5 Garantir que o ativo digitado tenha preferência sobre o selecionado
            if ticker_digitado:
                selecao_ticker = ticker_digitado.upper()
            else:
                selecao_ticker = ticker_selecionado

    # 2.4 Escolha do OHLCV
    with box2:
        selecao_ohlcv = st.sidebar.selectbox("Escolha do OHLCV ", options=OHLCV)
        
    # 2.5 Difinição do começo e fim da busca
    with box3:
        # 2.5.1 Colocando o titulo
        st.sidebar.write("Escolha o intervalo de datas:")

        # 2.5.2 Retornar as informações da API e filtrar e modelando os dados
        df = yf.Ticker(selecao_ticker).history(period="max", auto_adjust=False)
        df.index = pd.to_datetime(df.index.date)
        df.drop(columns=["Dividends", "Stock Splits"], inplace=True)
        df = df.asfreq("B")
        df.ffill(inplace=True)

        # 2.5.3 Procurando a moeda do ativo
        moeda = yf.Ticker(selecao_ticker).info['currency']

        # 2.5.3 Procurando o primeiro dia do ativo
        primeiro_dia = df.index[0].date()

        # 2.5.5 Escolha do intervalo de datas e deixando o label vazio para alinhar com as box
        start_date, end_date = st.sidebar.slider(
            "",min_value=None,
            max_value=dt.date.today(),
            value=(primeiro_dia, dt.date.today()),format="YYYY-MM-DD"
            )
        
        # 2.5.6 Filtrando o DataFrame para ser exibido no gráfico com a data escolhida
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df.loc[start_date:end_date]

st.write(f"## {yf.Ticker(selecao_ticker).info['shortName']}")


# 3.1 Metricas
ult_atualizacao = df.index.max().date() # Data da última atualização

prim_cotacao = round(df.loc[df.index.min(), "Close"], 2) # Primeira cotação

ult_cotacao = round(df.loc[df.index.max(), "Close"], 2) # Última cotação

menor_cotacao = round(df["Close"].min(), 2) # Menor cotação

maior_cotacao = round(df["Close"].max(), 2) # Maior cotação

delta = round(((ult_cotacao - prim_cotacao)/prim_cotacao)*100, 2) # Variação percentual


# 3.2 Visualização dos dados das metricas
with st.container(border=True):
    st.write("## Métricas do Ativo")
    box1, box2, box3 = st.columns([1,1,1])
    with box1:
        st.metric(f"Última cotação: {ult_atualizacao}",f"{moeda} {ult_cotacao}",f"{delta}%")

    with box2:
        st.metric("Maior cotação:",f"{moeda} {maior_cotacao}")

    with box3:
        st.metric("Menor cotação:",f"{moeda} {menor_cotacao}")

# 3.3 Visualização dos dados
st.write(f"## Gráfico de Desenvolvimento do Ativo {selecao_ticker}")

# 3.4 Visualização do gráfico
with st.container(border=True):
    st.area_chart(df[selecao_ohlcv].astype(float))




# 4.2 Prever cada coluna
st.sidebar.header("Previsão")

dias_previsao = st.sidebar.number_input("Dias de Previsão", min_value=1, max_value=100, value=10)

previsoes = {}

try:
    MLPR = SklearnML(df, selecao_ohlcv, janela=10, test_size=0.1)
    MLPR.preparar_dados()
    MLPR.treinar_modelo()
    r2, rmse, mae = MLPR.avaliar_modelo()
    previsoes_df = MLPR.prever(dias=dias_previsao)

except Exception as e:
    st.error(f"Erro ao prever {selecao_ohlcv}: {e}")




# 4.6 Visualização das previsões
ultima_previsao = previsoes_df[selecao_ohlcv].tail(1).reset_index(drop=True)
ultimo_real = df[selecao_ohlcv].tail(1).reset_index(drop=True)

if ultima_previsao.iloc[0] > ultimo_real.iloc[0]:
    st.write("## O ativo tende a subir")
else:
    st.write("## O ativo tende a descer")
with st.container(border=True):
    
    box1, box2 = st.columns([2,1])
    
    with box1:
        st.line_chart(previsoes_df[selecao_ohlcv].astype(float))
    
    with box2:
        st.write(f"# Modelo de Previsão")
        st.write("## Presição do Modelo:")
        st.write(f"## R2: {r2*100:.2f}%")
        st.write("## Erros do Modelo:")
        st.write(f"## RMSE: {rmse:.2f}")
        st.write(f"## MAE: {mae:.2f}")

# 5.1 Informações do Ativo
st.sidebar.header("Informações do Ativo")
sim_nao = st.sidebar.radio("Deseja ver mais informações sobre o ativo?", ("Não", "Sim"))
    
with st.container(border=True):
        
    if sim_nao == "Sim":
        st.write(f"## Informações do Ativo {selecao_ticker}")
        with st.container(border=True):
            box1, box2, box3 = st.columns([2,2,1])
            # DataFrame
            with box1:
                st.write(f"## Dados do {selecao_ticker}\n")
                st.write(df)
            # Informações do DataFrame
            with box2:
                st.write(f"## Informaões dos Dados do {selecao_ticker}")
                st.write(df.describe())
            # Informações do Ticker
            with box3:
                st.write(f"## Informações do Ticker {selecao_ticker}")
                st.write(yf.Ticker(selecao_ticker).info)
                
    if sim_nao == "Não":
        st.write("Obrigado por usar o nosso Dashboard")

# 6.1 Informações do Projeto