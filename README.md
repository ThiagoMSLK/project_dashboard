# 📈 Dashboard de Previsão de Ativos Financeiros
## Com ativos do YFinance e visualização com Streamlit 

> 🚧 **Status do Projeto:** Em desenvolvimento

📈 Financial Dashboard com Previsão de 10 Dias  

### 📊 Descrição do Projeto  

Este projeto apresenta uma dashboard interativa desenvolvida em Python e Streamlit, que consome dados diretamente da API do Yahoo Finance (YFinance) para analisar e prever preços de ativos financeiros.

O sistema permite selecionar qualquer ativo disponível na bolsa (ex: PETR4.SA, AAPL, BTC-USD), visualizar seus dados históricos OHLCV (Open, High, Low, Close, Volume) e gerar previsões para os próximos 10 dias utilizando Regressão Linear.

O objetivo é demonstrar o uso integrado de ciência de dados, machine learning e visualização interativa no contexto de finanças.  
  
### 🧠 Principais Recursos

- Coleta automática de dados de ativos via API YFinance

- Visualização de gráficos interativos com Streamlit

- Modelo de Regressão Linear para prever preços futuros

- Opção de escolha do tipo de dado a ser previsto (Open, High, Low, Close, Volume)

- Pipeline de análise totalmente automatizado

### ⚙️ Tecnologias Utilizadas

Python 3.10+

- Streamlit → interface interativa e visual

- Pandas / NumPy → manipulação e estruturação dos dados

- YFinance → coleta de dados financeiros em tempo real

- Scikit-learn → implementação da Regressão Linear

- Matplotlib / Plotly → geração dos gráficos

### 🚀 Como Executar o Projeto

Clone o repositório:

- `git clone https://github.com/seuusuario/project_dashboard.git`  
- `cd project_dashboard`  


Instale as dependências:

- `pip install pandas numpy yfinance streamlit scikit-learn`


Execute o projeto no Streamlit:

- `streamlit run project_dashboard.py`

### 📉 Resultados Esperados

- Visualização clara das variações de preço histórico.

- Projeções automáticas de preços para os próximos 10 dias.

- Ferramenta útil para explorar padrões e tendências de mercado.

### 💡 Próximos Passos

- Implementar outros modelos preditivos (ARIMA, XGBoost, LSTM).

- Criar comparativos entre previsões e valores reais.

- Publicar versão online com Streamlit Cloud.

### 💼 Autor: Thiago Martins LK
- <a href="https://www.linkedin.com/in/thiagomartinslk" target="_blank">Meu LinkedIn</a>
- <a href="https://www.kaggle.com/thiagomartinslk" target="_blank">Meu Kaggle</a>
- <a href="https://github.com/ThiagoMSLK" target="_blank">Meu GitHub</a>
