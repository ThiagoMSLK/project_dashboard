import yfinance as yf

ticker = "PETR4.SA"

ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)