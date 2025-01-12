# 1.1 Importação das bibliotecas internas
import datetime as dt
# from time import sleep

# 1.2 Importação das bibliotecas externas
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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

        with st.container():
            st.write("Escolha o ativo que deseja visualizar")
            
            col1, col2 =st.columns([1,1])
            
            with col1:
                ticker_selecionado = st.selectbox("Selecione ou >", options= ticker)
            with col2:
                ticker_digitado = st.text_input("Digite")

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

        # 1.6.1 Procurando o primeiro dia do ativo
        ativo = yf.Ticker(selecao_ticker)
        historico = ativo.history(period="max")
        primeiro_dia = historico.index[0].date()

        # 1.6.2 Escolha do intervalo de datas
        start_date, end_date = st.slider(
    "Escolha o intervalo de datas:",
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
st.area_chart(df[selecao_ohlcv])

# st.write_stream(sns.pairplot(df))

# 2.5 Visualização do DataFrame
st.write(f"## Dados do {selecao_ticker}")
st.write(df)
