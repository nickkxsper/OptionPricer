# OptionPricer

OptionPricer is a Python library for pricing options using various methods, including Black-Scholes, Monte Carlo, and Leisen-Reimer Binomial Tree. It also provides functionality to plot the payoff of a set of options and visualize the implied volatility surface.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```bash
pip install numpy scipy matplotlib


# OptionPricer

OptionPricer is a Python library for pricing options using various methods, including Black-Scholes, Monte Carlo, and Leisen-Reimer Binomial Tree. It also provides functionality to plot the payoff of a set of options and visualize the implied volatility surface.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```bash
pip install numpy scipy matplotlib
Usage
python
Copy code
from option_pricer import OptionPricer
import numpy as np

S0 = 100
K = 110
r = 0.05
T = 1
sigma = 0.2
option_type = 'call'
num_steps = 100

option_pricer = OptionPricer(S0, K, r, T, sigma, option_type, num_steps)

# Price the option using Black-Scholes
black_scholes_price = option_pricer.black_scholes()

# Price the option using Monte Carlo
monte_carlo_price = option_pricer.monte_carlo()

# Price the option using Leisen-Reimer Binomial Tree
lr_binomial_tree_price = option_pricer.leisen_reimer_binomial_tree()

print(f'Black-Scholes: {black_scholes_price}')
print(f'Monte Carlo: {monte_carlo_price}')
print(f'Leisen-Reimer Binomial Tree: {lr_binomial_tree_price}')

# Plot the payoff of a set of options
spot_prices = np.linspace(50, 150, 100)
option_prices = [OptionPricer(S, K, r, T, sigma, option_type, num_steps).black_scholes() for S in spot_prices]
option_pricer.plot_payoff(option_prices, spot_prices)

# Plot the implied volatility surface
spot_prices = np.linspace(50, 150, 10)
strike_prices = np.linspace(80, 120, 10)
expirations = np.linspace(0.1, 2, 10)
option_prices = ...  # Replace with real option price data
option_pricer.plot_volatility_surface(option_prices, spot_prices, strike_prices, expirations)
Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

License
MIT
