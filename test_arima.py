# 1.1 Importação das bibliotecas internas
import datetime as dt
# from time import sleep

# 1.2 Importação das bibliotecas externas
import pandas as pd
import yfinance as yf
import streamlit as st
#import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly.express as px

# 1.3 Importação das bibliotecas de Machine Learning
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1.4 Difinição da Tuples OHLCV e do DataFrame Ticker
OHLCV = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

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
        moeda = yf.Ticker(selecao_ticker).info["currency"]

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

# 3.1 Metricas
ult_atualizacao = df.index.max().date() # Data da última atualização

prim_cotacao = round(df.loc[df.index.min(), "Close"], 2).item() # Primeira cotação

ult_cotacao = round(df.loc[df.index.max(), "Close"], 2).item() # Última cotação

menor_cotacao = round(df["Close"].min(), 2).item() # Menor cotação

maior_cotacao = round(df["Close"].max(), 2).item() # Maior cotação

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
st.write(f"## Dados do Ativo {selecao_ticker}")

# 3.4 Visualização do gráfico
with st.container(border=True):
    st.area_chart(df[selecao_ohlcv].astype(float))


# 4.1 Normalizar os dados (Min-Max)
scaler = MinMaxScaler()
df_normalizado = scaler.fit_transform(df)
df_normalizado = pd.DataFrame(df_normalizado, columns=df.columns, index=df.index)

# 4.2 Previsão do ARIMA
def prever_arima(serie, ordem=(6, 6, 1), dias_previsao=10):
    modelo = ARIMA(serie, order=ordem)
    modelo_ajustado = modelo.fit()
    previsao = modelo_ajustado.forecast(steps=dias_previsao)
    return previsao

# Prever cada coluna
st.sidebar.header("Previsão")

dias_previsao = st.sidebar.number_input("Dias de Previsão", min_value=1, max_value=10, value=10)

previsoes = {}


for coluna in df_normalizado.columns:
    try:
        previsoes[coluna] = prever_arima(df_normalizado[coluna], dias_previsao=dias_previsao)
    except Exception as e:
        print(f"Erro ao prever {coluna}: {e}")

# Criar DataFrame com as previsões
if previsoes:
    previsoes_df = pd.DataFrame(previsoes)
    
else:
    st.write("Nenhuma previsão foi gerada devido a erros.")

# Desnormalizar as previsões
previsoes_desnormalizadas = scaler.inverse_transform(previsoes_df)

# Converter de volta para DataFrame
previsoes_desnormalizadas = pd.DataFrame(previsoes_desnormalizadas, columns=df.columns, index=previsoes_df.index)

st.write(previsoes_desnormalizadas[selecao_ohlcv])

st.line_chart(previsoes_desnormalizadas[selecao_ohlcv].astype(float))














#######TESTE######

# 5.1 Visualização do DataFrame
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


#st.write(df.columns)

#tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Todos Dados","Open", "High", "Low","Close","Adj Close","Volume"])