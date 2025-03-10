# 1.1 Importação das bibliotecas internas
import datetime as dt
# from time import sleep

# 1.2 Importação das bibliotecas externas
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# 1.3 Importação das bibliotecas de Machine Learning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

class ARIMAForecaster:
    def __init__(self, serie, max_d=3):
        self.serie = serie
        self.max_d = max_d
        self.p = None
        self.d = None
        self.q = None
    
    def encontrar_d(self):
        """Determina o número de diferenciações (d) necessárias para tornar a série estacionária."""
        d = 0
        adf_teste = adfuller(self.serie.dropna())
        
        while adf_teste[1] > 0.05 and d < self.max_d:  # Diferencia até ficar estacionária
            d += 1
            self.serie = self.serie.diff().dropna()
            adf_teste = adfuller(self.serie)
        
        self.d = d
        return self.d
    
    def encontrar_p_q(self):
        """Determina os valores de p e q com base nos gráficos PACF e ACF."""
        if self.d is None:
            self.encontrar_d()
        
        serie_d = self.serie.copy()
        for _ in range(self.d):
            serie_d = serie_d.diff().dropna()
        
        pacf_vals, confint = pacf(serie_d, nlags=10, alpha=0.05)
        self.p = np.argmax(np.abs(pacf_vals) < np.abs(confint[:, 1] - pacf_vals).mean()) or 1

        acf_vals, confint = acf(serie_d, nlags=10, alpha=0.05)
        self.q = np.argmax(np.abs(acf_vals) < np.abs(confint[:, 1] - acf_vals).mean()) or 1   
        
        return self.p, self.q
    
    def prever(self, dias_previsao=10):
        """Treina o modelo ARIMA e faz previsões."""
        if self.p is None or self.q is None or self.d is None:
            self.encontrar_p_q()
        
        print(f"Usando ordem: ({self.p}, {self.d}, {self.q})")
        
        modelo = ARIMA(self.serie, order=(self.p, self.d, self.q))
        modelo_ajustado = modelo.fit()
        previsao = modelo_ajustado.forecast(steps=dias_previsao)
        
        return previsao

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

        # 2.5.2 Retornar as informações da API
        df = yf.Ticker(selecao_ticker).history(period="max", auto_adjust=False)
        df.index = pd.to_datetime(df.index.date)
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


# 4.1 Normalizar os dados (Min-Max)
scaler = MinMaxScaler()
df_normalizado = scaler.fit_transform(df[[selecao_ohlcv]])
df_normalizado = pd.DataFrame(df_normalizado, columns=[selecao_ohlcv], index=df.index)

# 4.2 Prever cada coluna
st.sidebar.header("Previsão")

dias_previsao = st.sidebar.number_input("Dias de Previsão", min_value=1, max_value=100, value=10)

previsoes = {}

try:
    forecaster = ARIMAForecaster(df_normalizado[selecao_ohlcv])
    previsoes[selecao_ohlcv] = forecaster.prever(dias_previsao=dias_previsao)
except Exception as e:
    st.error(f"Erro ao prever {selecao_ohlcv}: {e}")

# 4.3 Criar DataFrame com as previsões
if previsoes:
    previsoes_df = pd.DataFrame(previsoes, index=pd.date_range(start=df.index[-1], periods=dias_previsao+1, freq="B")[1:])
else:
    previsoes_df = None
    st.warning("Nenhuma previsão foi gerada devido a erros.")

# 4.4 Desnormalizar as previsões
if previsoes_df is not None:
    previsoes_desnormalizadas = scaler.inverse_transform(previsoes_df)
    previsoes_desnormalizadas = pd.DataFrame(previsoes_desnormalizadas, columns=[selecao_ohlcv], index=previsoes_df.index)
else:
    previsoes_desnormalizadas = None


# 4.6 Visualização das previsões
ultima_previsao = previsoes_desnormalizadas[selecao_ohlcv].tail(1).reset_index(drop=True)
ultimo_real = df[selecao_ohlcv].tail(1).reset_index(drop=True)

if ultima_previsao.iloc[0] > ultimo_real.iloc[0]:
    st.write("## O ativo tende a subir")
else:
    st.write("## O ativo tende a descer")

st.line_chart(previsoes_desnormalizadas[selecao_ohlcv].astype(float))


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