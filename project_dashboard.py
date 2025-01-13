# 1.1 Importação das bibliotecas internas
import datetime as dt
# from time import sleep

# 1.2 Importação das bibliotecas externas
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# 1.3 Difinição da Tuples OHLCV e do DataFrame Ticker
OHLCV = ("Open", "High", "Low", "Close", "Volume")

ticker = pd.read_csv("nasdaq_screener_10-01-2025.csv")["Symbol"].values

# 2.1 Configuração da página
st.set_page_config(page_title="Análise de Dados Financeiros", layout="wide")

st.title("Análise de Dados Financeiros")

# 1.3 Difinição dos dados que o usuário deseja visualizar

with st.container():
    st.header("Escolha os dados que deseja visualizar")

    box1, box2, box3 = st.columns([1,1,2])
    
    # 1.4 Difinição do Ticker
    with box1:
        # 1.4.1 Colocando o o titulo
        with st.container():
            st.write("Escolha o ativo que deseja visualizar")
            
            # 1.4.2 Tamanho das box
            col1, col2 =st.columns([1,1])
            
            # 1.4.3 Seleção do ativo
            with col1:
                ticker_selecionado = st.selectbox("Selecione ou >", options= ticker)
            
            # 1.4.4 Digitar o ativo
            with col2:
                ticker_digitado = st.text_input("Digite")

            # 1.4.5 Garantir que o ativo digitado tenha preferência sobre o selecionado
            if ticker_digitado:
                selecao_ticker = ticker_digitado.upper()
            else:
                selecao_ticker = ticker_selecionado

    # 1.5 Escolha do OHLCV
    with box2:
        with st.container():
            st.write("Escolha os dados que deseja visualizar:")
        selecao_ohlcv = st.selectbox(f"O tcker {selecao_ticker} foi escolhido", options=OHLCV)
        
    # 1.6 Difinição do começo e fim da busca
    with box3:
        # 1.6.1 Colocando o o titulo, mas o princial objetivo é alinhar com as box
        with st.container():
            st.write("Escolha o intervalo de datas:")

        # 1.6.2 Procurando o primeiro dia do ativo
            # Mais explicativo
        ativo = yf.Ticker(selecao_ticker)
        historico = ativo.history(period="max")
        primeiro_dia = historico.index[0].date()
            # Mais simples e direto
        # primeiro_dia = yf.Ticker(selecao_ticker).history(period="max").index[0].date()

        # 1.6.3 Escolha do intervalo de datas e deixando o label vazio para alinhar com as box
        start_date, end_date = st.slider(
    "",
    min_value=None,
    max_value=dt.date.today(),
    value=(primeiro_dia, dt.date.today()),
    format="YYYY-MM-DD"
)

# 1.7 Retornar as informações da API
df = yf.download(selecao_ticker, start=start_date, 
        end=dt.date.today())

# 2.1 Metricas
ult_atualizacao = df.index.max().date()
ult_cotacao = round(df.loc[df.index.max(), "Close"], 2).item()
menor_cotacao = round(df["Close"].min(), 2).item()
maior_cotacao = round(df["Close"].max(), 2).item()
prim_cotacao = round(df.loc[df.index.min(), "Close"], 2).item()
delta = round(((ult_cotacao - prim_cotacao)/prim_cotacao)*100, 2)

# 2.2 Visualização dos dados das metricas
with st.container():
    box1, box2, box3 = st.columns([1,1,1])
    with box1:
        st.metric(f"Última cotação: {ult_atualizacao}",f"US$ {ult_cotacao}",f"{delta}%")

    with box2:
        st.metric("Maior cotação:",f"US$ {maior_cotacao}")

    with box3:
        st.metric("Menor cotação:",f"US$ {menor_cotacao}")

# 2.3 Visualização dos dados
st.write(f"## Dados do Ativo {selecao_ticker}")

# 2.4 Visualização do gráfico
st.area_chart(df[selecao_ohlcv].astype(float))

# 2.5 Visualização do DataFrame
with st.container():
    box1, box2 = st.columns([1,1])
    with box1:
        st.write(f"## Dados do {selecao_ticker}\n")
        st.write(df)
    with box2:
        st.write(f"## Informaões dos Dados do {selecao_ticker}")
        st.write(df.describe())
        st.write(df.info())