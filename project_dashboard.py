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
OHLCV = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

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
        selecao_ohlcv = st.selectbox(f"O Ticker {selecao_ticker} foi escolhido", options=OHLCV)
        
    # 1.6 Difinição do começo e fim da busca
    with box3:

        # 1.6.1 Colocando o o titulo, mas o princial objetivo é alinhar com as box
        with st.container():
            st.write("Escolha o intervalo de datas:")

        # 1.6.2 Retornar as informações da API
        
        df = yf.Ticker(selecao_ticker).history(period="max", auto_adjust=False)
        df.index = pd.to_datetime(df.index.date)
        # 1.6.3 Procurando a moeda do ativo
        moeda = yf.Ticker(selecao_ticker).info["currency"]

        # 1.6.3 Procurando o primeiro dia do ativo
        primeiro_dia = df.index[0].date()

        # 1.6.5 Escolha do intervalo de datas e deixando o label vazio para alinhar com as box
        start_date, end_date = st.slider(
            "",min_value=None,
            max_value=dt.date.today(),
            value=(primeiro_dia, dt.date.today()),format="YYYY-MM-DD"
            )
        
        # 1.6.6 Filtrando o DataFrame para ser exibido no gráfico com a data escolhida
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df.loc[start_date:end_date]

# 2.1 Metricas
ult_atualizacao = df.index.max().date() # Data da última atualização

prim_cotacao = round(df.loc[df.index.min(), "Close"], 2).item() # Primeira cotação

ult_cotacao = round(df.loc[df.index.max(), "Close"], 2).item() # Última cotação

menor_cotacao = round(df["Close"].min(), 2).item() # Menor cotação

maior_cotacao = round(df["Close"].max(), 2).item() # Maior cotação

delta = round(((ult_cotacao - prim_cotacao)/prim_cotacao)*100, 2) # Variação percentual


# 2.2 Visualização dos dados das metricas
with st.container(border=True):
    st.write("## Métricas do Ativo")
    box1, box2, box3 = st.columns([1,1,1])
    with box1:
        st.metric(f"Última cotação: {ult_atualizacao}",f"{moeda} {ult_cotacao}",f"{delta}%")

    with box2:
        st.metric("Maior cotação:",f"{moeda} {maior_cotacao}")

    with box3:
        st.metric("Menor cotação:",f"{moeda} {menor_cotacao}")

# 2.3 Visualização dos dados
st.write(f"## Dados do Ativo {selecao_ticker}")

# 2.4 Visualização do gráfico
st.area_chart(df[selecao_ohlcv].astype(float))

# 2.5 Visualização do DataFrame
with st.container(border=True):
    box1, box2 = st.columns([1,1])

    # DataFrame
    with box1:
        st.write(f"## Dados do {selecao_ticker}\n")
        st.write(df)
    # Informações do DataFrame
    with box2:
        st.write(f"## Informaões dos Dados do {selecao_ticker}")
        st.write(df.describe())
    

    
    with st.container(border=True):
        st.write("Deseja ver mais informações sobre o ativo?")
        
        botao1, botao2 = st.columns(2, border=True)
        with botao1:
            if st.button("Sim"):
                st.write(yf.Ticker(selecao_ticker).info)
                
        with botao2:
            if st.button("Não"):
                st.write("Obrigado por usar o nosso Dashboard")