import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt

def combos(n, i):
    return math.factorial(n) / (math.factorial(n - i) * math.factorial(i))

def binom_EU1(S0, K, T, r, sigma, N, type_='call'):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    value = 0
    for i in range(N + 1):
        node_prob = combos(N, i) * p**i * (1 - p)**(N - i)
        ST = S0 * (u)**i * (d)**(N - i)
        if type_ == 'call':
            value += max(ST - K, 0) * node_prob
        elif type_ == 'put':
            value += max(K - ST, 0) * node_prob
        else:
            raise ValueError("type_ must be 'call' or 'put'")
    return value * np.exp(-r * T)

def get_data(symbol):
    ticker = yf.Ticker(symbol)
    expiries = ticker.options
    options = []
    for expiry in expiries:
        opt = ticker.option_chain(expiry)
        calls = opt.calls
        puts = opt.puts
        calls['Type'] = 'call'
        puts['Type'] = 'put'
        calls['expiry'] = expiry
        puts['expiry'] = expiry
        options.append(calls)
        options.append(puts)
    df = pd.concat(options)
    df['mid_price'] = (df['ask'] + df['bid']) / 2
    df['expiry'] = pd.to_datetime(df['expiry'])
    df['Time'] = (df['expiry'] - pd.Timestamp.now()).dt.days / 365.0
    return df[(df['bid'] > 0) & (df['ask'] > 0)]

df = get_data('TSLA')

prices = []
for row in df.itertuples():
    price = binom_EU1(row.lastPrice, row.strike, row.Time, 0.01, 0.5, 20, row.Type)
    prices.append(price)

df['Price'] = prices
df['error'] = df['mid_price'] - df['Price']

exp1 = df[(df['expiry'] == df['expiry'].unique()[2]) & (df['Type'] == 'call')]

plt.figure(figsize=(10, 6))
plt.plot(exp1['strike'], exp1['mid_price'], label='Mid Price')
plt.plot(exp1['strike'], exp1['Price'], label='Calculated Price')
plt.xlabel('Strike')
plt.ylabel('Call Value')
plt.title('Option Prices vs Strike Prices')
plt.legend()
plt.grid(True)
plt.show()

# Simulate different number of steps
S0 = df.iloc[0]['lastPrice']
K = df.iloc[0]['strike']
T = df.iloc[0]['Time']
r = 0.01
sigma = 0.5
type_ = df.iloc[0]['Type']

num_steps = range(1, 101)
prices_steps = [binom_EU1(S0, K, T, r, sigma, N, type_) for N in num_steps]

plt.figure(figsize=(10, 6))
plt.plot(num_steps, prices_steps, label='Estimated Option Price')
plt.xlabel('Number of Simulations')
plt.ylabel('Option Price')
plt.title('Estimated Option Price vs Number of Simulations')
plt.legend()
plt.grid(True)
plt.show()
