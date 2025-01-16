import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

ticker = "PETR4.SA"

ativo = yf.Ticker(ticker).history(period="max", auto_adjust=False)

moeda = yf.Ticker(ticker).info["currency"]


# sns.lineplot(data=ativo[["Open", "Close", "High", "Low"]].astype(float), palette="tab10")
# plt.show()

print(moeda)

# print(ativo.info())