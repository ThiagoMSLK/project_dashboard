# project_dashboard

📈 Financial Dashboard com Previsão de 10 Dias
🇧🇷 Versão em Português
📊 Descrição do Projeto

Este projeto apresenta uma dashboard interativa desenvolvida em Python e Streamlit, que consome dados diretamente da API do Yahoo Finance (YFinance) para analisar e prever preços de ativos financeiros.

O sistema permite selecionar qualquer ativo disponível na bolsa (ex: PETR4.SA, AAPL, BTC-USD), visualizar seus dados históricos OHLCV (Open, High, Low, Close, Volume) e gerar previsões para os próximos 10 dias utilizando Regressão Linear.

O objetivo é demonstrar o uso integrado de ciência de dados, machine learning e visualização interativa no contexto de finanças.

🧠 Principais Recursos

Coleta automática de dados de ativos via API YFinance

Visualização de gráficos interativos com Streamlit

Modelo de Regressão Linear para prever preços futuros

Opção de escolha do tipo de dado a ser previsto (Open, High, Low, Close, Volume)

Pipeline de análise totalmente automatizado

⚙️ Tecnologias Utilizadas

Python 3.10+

Streamlit → interface interativa e visual

Pandas / NumPy → manipulação e estruturação dos dados

YFinance → coleta de dados financeiros em tempo real

Scikit-learn → implementação da Regressão Linear

Matplotlib / Plotly → geração dos gráficos

🚀 Como Executar o Projeto

Clone o repositório:

git clone https://github.com/seuusuario/project_dashboard.git
cd project_dashboard


Instale as dependências:

pip install -r requirements.txt


Execute o projeto no Streamlit:

streamlit run project_dashboard.py

📉 Resultados Esperados

Visualização clara das variações de preço histórico.

Projeções automáticas de preços para os próximos 10 dias.

Ferramenta útil para explorar padrões e tendências de mercado.

💡 Próximos Passos

Implementar outros modelos preditivos (ARIMA, XGBoost, LSTM).

Adicionar métricas de avaliação (RMSE, MAE, R²).

Criar comparativos entre previsões e valores reais.

Publicar versão online com Streamlit Cloud.

💼 Autor

Thiago Martins LK
🔗 [LinkedIn](www.linkedin.com/in/thiagomartinslk)
🔗 [Kaggle](https://www.kaggle.com/thiagomartinslk)
🔗 [GitHub](https://github.com/ThiagoMSLK/ThiagoMSLK/blob/main/README.md)

🌍 English Version
📈 Financial Dashboard with 10-Day Forecast
📊 Project Description

This project features an interactive financial dashboard built with Python and Streamlit, using the Yahoo Finance API (YFinance) to analyze and forecast stock prices.

Users can select any stock or asset (e.g., PETR4.SA, AAPL, BTC-USD), view historical OHLCV data (Open, High, Low, Close, Volume), and generate 10-day price predictions using Linear Regression.

The main goal is to demonstrate the integration of data science, machine learning, and financial visualization in an intuitive and interactive way.

🧠 Key Features

Automated data fetching from YFinance API

Interactive charts built with Streamlit

Linear Regression model for future price prediction

User selection of which metric to forecast (Open, High, Low, Close, Volume)

End-to-end automated data analysis pipeline

⚙️ Technologies Used

Python 3.10+

Streamlit – interactive dashboard framework

Pandas / NumPy – data wrangling and structuring

YFinance – real-time financial data collection

Scikit-learn – Linear Regression model

Matplotlib / Plotly – visualizations

🚀 How to Run
git clone https://github.com/seuusuario/project_dashboard.git
cd project_dashboard
pip install -r requirements.txt
streamlit run project_dashboard.py

📉 Expected Results

Clear visualization of historical stock performance.

Automatic 10-day price forecast.

Practical tool to explore financial trends and patterns.

💡 Next Steps

Add new forecasting models (ARIMA, XGBoost, LSTM).

Implement evaluation metrics (RMSE, MAE, R²).

Compare predictions with actual results.

Deploy using Streamlit Cloud or Hugging Face Spaces.

💼 Author

Thiago Martins LK
🔗 [LinkedIn](www.linkedin.com/in/thiagomartinslk)
🔗 [Kaggle](https://www.kaggle.com/thiagomartinslk)
🔗 [GitHub](https://github.com/ThiagoMSLK/ThiagoMSLK/blob/main/README.md)
