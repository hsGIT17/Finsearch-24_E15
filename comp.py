import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# Black-Scholes Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        option_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option_type == "put":
        option_price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
    return option_price

# Binomial Model
def binomial_model(S, K, T, r, sigma, N, option_type="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    ST = np.zeros(N + 1)
    for i in range(N + 1):
        ST[i] = S * (u ** (N - i)) * (d ** i)
    
    # Initialize option values at maturity
    if option_type == "call":
        C = np.maximum(0, ST - K)
    elif option_type == "put":
        C = np.maximum(0, K - ST)
    
    # Step back through the tree
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            C[i] = np.exp(-r * dt) * (p * C[i] + (1 - p) * C[i + 1])
    
    return C[0]

# Parameters
S = 2500  # Current stock price
K = 2550  # Strike price
T = 1     # Time to maturity in years
r = 0.05  # Risk-free interest rate
sigma = 0.25  # Volatility
N = 100   # Number of binomial steps

# Calculate option prices
bs_call_price = black_scholes(S, K, T, r, sigma, option_type="call")
bs_put_price = black_scholes(S, K, T, r, sigma, option_type="put")
binom_call_price = binomial_model(S, K, T, r, sigma, N, option_type="call")
binom_put_price = binomial_model(S, K, T, r, sigma, N, option_type="put")

# Print results
print(f"Black-Scholes Call Price: {bs_call_price:.2f}")
print(f"Black-Scholes Put Price: {bs_put_price:.2f}")
print(f"Binomial Call Price: {binom_call_price:.2f}")
print(f"Binomial Put Price: {binom_put_price:.2f}")

# Graphs
steps = range(1, N+1)
binom_call_prices = [binomial_model(S, K, T, r, sigma, n, option_type="call") for n in steps]
binom_put_prices = [binomial_model(S, K, T, r, sigma, n, option_type="put") for n in steps]

plt.figure(figsize=(12, 6))

# Call Option Prices
plt.subplot(1, 2, 1)
plt.plot(steps, binom_call_prices, label='Binomial Model Call Price')
plt.axhline(y=bs_call_price, color='r', linestyle='-', label='Black-Scholes Call Price')
plt.xlabel('Number of Steps (N)')
plt.ylabel('Call Option Price')
plt.title('Call Option Price Comparison')
plt.legend()

# Put Option Prices
plt.subplot(1, 2, 2)
plt.plot(steps, binom_put_prices, label='Binomial Model Put Price')
plt.axhline(y=bs_put_price, color='r', linestyle='-', label='Black-Scholes Put Price')
plt.xlabel('Number of Steps (N)')
plt.ylabel('Put Option Price')
plt.title('Put Option Price Comparison')
plt.legend()

plt.tight_layout()
plt.show()
