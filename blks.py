import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import datetime as dt

# Defining the cumulative distribution function for the normal distribution
N = norm.cdf

# Black-Scholes Call option pricing function
def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r * T) * N(d2)

# Black-Scholes Put option pricing function
def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * N(-d2) - S * N(-d1)

# Parameters
K = 100
r = 0.1
T = 1
sigma = 0.3

# Stock prices
S = np.arange(60, 140, 0.1)

# Calculating call and put option prices for a range of stock prices
calls = [BS_CALL(s, K, T, r, sigma) for s in S]
puts = [BS_PUT(s, K, T, r, sigma) for s in S]

plt.figure(figsize=(10, 6))
plt.plot(S, calls, label='Call Value')
plt.plot(S, puts, label='Put Value')
plt.xlabel('Stock Price $S_0$')
plt.ylabel('Option Value')
plt.title('Black-Scholes Option Pricing')
plt.legend()
plt.grid(True)
plt.show()

# Calculating call and put option prices for a range of times to maturity
T_range = np.arange(0.1, 2.1, 0.1)
calls_time = [BS_CALL(S[-1], K, t, r, sigma) for t in T_range]
puts_time = [BS_PUT(S[-1], K, t, r, sigma) for t in T_range]

plt.figure(figsize=(10, 6))
plt.plot(T_range, calls_time, label='Call Value')
plt.plot(T_range, puts_time, label='Put Value')
plt.xlabel('Time to Maturity $T$ in years')
plt.ylabel('Option Value')
plt.title('Black-Scholes Option Pricing with Time to Maturity')
plt.legend()
plt.grid(True)
plt.show()

start = dt.datetime(2010, 1, 1)
end = dt.datetime(2020, 10, 1)
symbol = 'TATAMOTORS.NS'  

data = yf.download(symbol, start=start, end=end)
data['change'] = data['Adj Close'].pct_change()
data['rolling_sigma'] = data['change'].rolling(20).std() * np.sqrt(252)

data['rolling_sigma'].plot(figsize=(10, 6))
plt.ylabel('Volatility $\sigma$')
plt.title('Tata Motors Rolling Volatility')
plt.grid(True)
plt.show()
