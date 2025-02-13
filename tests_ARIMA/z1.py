import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
import yfinance as yf
import numpy as np

ativo = "PETR4.SA"

df = yf.Ticker(ativo).history(period="max", auto_adjust=False)

# Regra prática para escolher nlags
n = len(df["Close"])
nlags = int(min(10 * np.log10(n), n - 1))  
# Garante pelo menos 10 lags
nlags = max(nlags, 10)  

pacf_vals = pacf(df["Close"], nlags=nlags, method="ols")
acf_vals = acf(df["Close"], nlags=nlags)

# Plot da PACF
plt.figure(figsize=(12, 6))
plt.stem(pacf_vals)
plt.title("Função de Autocorrelação Parcial (PACF)")
plt.xlabel("Lag")
plt.ylabel("Valor da PACF")
plt.axhline(y=0, color='black', linestyle='--')
plt.savefig("z_pacf.png")
plt.close()

# Plot da ACF
plt.figure(figsize=(12, 6))
plt.stem(acf_vals)
plt.title("Função de Autocorrelação (ACF)")
plt.xlabel("Lag")
plt.ylabel("Valor da ACF")
plt.axhline(y=0, color='black', linestyle='--')
plt.savefig("z_acf.png")
plt.close()