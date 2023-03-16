import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class OptionPricer:
    def __init__(self, S0, K, r, T, sigma, option_type='call', num_steps=100):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.option_type = option_type
        self.num_steps = num_steps

    def black_scholes(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_type == 'call':
            price = self.S0 * stats.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S0 * stats.norm.cdf(-d1)
        return price

    def monte_carlo(self, num_simulations=100000):
        Z = np.random.standard_normal(num_simulations)
        ST = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        if self.option_type == 'call':
            payoff = np.maximum(ST - self.K, 0)
        else:
            payoff = np.maximum(self.K - ST, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price

    def leisen_reimer_binomial_tree(self):
        dt = self.T / self.num_steps
        u = np.exp(self.sigma * np.sqrt(3 * dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        q = 1 - p

        stock_price = np.zeros((self.num_steps + 1, self.num_steps + 1))
        stock_price[0, 0] = self.S0

        for i in range(1, self.num_steps + 1):
            stock_price[i, 0] = stock_price[i - 1, 0] * u
            for j in range(1, i + 1):
                stock_price[i, j] = stock_price[i - 1, j - 1] * d

        if self.option_type == 'call':
            option_value = np.maximum(stock_price[:, -1] - self.K, 0)
        else:
            option_value = np.maximum(self.K - stock_price[:, -1], 0)

        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                option_value[j] = np.exp(-self.r * dt) * (p * option_value[j] + q * option_value[j + 1])

        return option_value[0]
    
    @staticmethod
    def implied_volatility(price, S0, K, r, T, option_type='call', tol=1e-6, max_iter=100):
        if price <= 0:
            return np.nan

        sigma = 0.2  # initial guess for implied volatility
        for _ in range(max_iter):
            pricer = OptionPricer(S0, K, r, T, sigma, option_type)
            option_price = pricer.black_scholes()
            vega = pricer.vega()

            diff = option_price - price
            if abs(diff) < tol:
                return sigma

            sigma -= diff / vega

        return np.nan
    
    def vega(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        return self.S0 * stats.norm.pdf(d1) * np.sqrt(self.T)

    def plot_payoff(self, option_prices, spot_prices):
        if self.option_type == 'call':
            payoff = np.maximum(spot_prices - self.K, 0)
        else:
            payoff = np.maximum(self.K - spot_prices, 0)

        plt.plot(spot_prices, payoff, label='Payoff')
        plt.plot(spot_prices, option_prices, label='Option Price')
        plt.xlabel('Spot Price')
        plt.ylabel('Payoff / Option Price')
        plt.title(f'{self.option_type.capitalize()} Option Payoff and Price')
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_volatility_surface(self, option_prices, spot_prices, strike_prices, expirations):
        implied_volatilities = np.zeros((len(spot_prices), len(strike_prices), len(expirations)))

        for i, S0 in enumerate(spot_prices):
            for j, K in enumerate(strike_prices):
                for k, T in enumerate(expirations):
                    option_price = option_prices[i, j, k]
                    implied_volatilities[i, j, k] = self.implied_volatility(option_price, S0, K, self.r, T, self.option_type)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X, Y = np.meshgrid(strike_prices, expirations)
        Z = implied_volatilities.mean(axis=0)

        ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Time to Expiration')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'{self.option_type.capitalize()} Option Implied Volatility Surface')
        plt.show()
