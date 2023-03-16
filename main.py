from utils import *


## Example to price an option and plot its payoff

def price_and_plot(
        S0 = 100, #initial stock price
        K = 110, # strike price
        r = 0.05, # risk free rate
        T = 1, # days to expiration
        sigma = 0.2, #implied volatility
        option_type = 'put', #call or put
        num_steps = 100 # number of steps for leisen reimer binomial tree
        ):
    
    option_pricer = OptionPricer(S0, K, r, T, sigma, option_type, num_steps)
    black_scholes_price = option_pricer.black_scholes()
    monte_carlo_price = option_pricer.monte_carlo()
    lr_binomial_tree_price = option_pricer.leisen_reimer_binomial_tree()

    print(f'Black-Scholes: {black_scholes_price}')
    print(f'Monte Carlo: {monte_carlo_price}')
    print(f'Leisen-Reimer Binomial Tree: {lr_binomial_tree_price}')

    spot_prices = np.linspace(50, 150, 100)
    option_prices = [OptionPricer(S, K, r, T, sigma, option_type, num_steps).black_scholes() for S in spot_prices]

    option_pricer.plot_payoff(option_prices, spot_prices)
    
def price_and_plot_vol_surface(
        S0 = 100, #initial stock price
        K = 110, # strike price
        r = 0.05, # risk free rate
        T = 1, # days to expiration
        sigma = 0.2, #implied volatility
        option_type = 'put', #call or put
        num_steps = 100 # number of steps for leisen reimer binomial tree
        ):
        option_pricer = OptionPricer(S0, K, r, T, sigma, option_type, num_steps)
        black_scholes_price = option_pricer.black_scholes()
        monte_carlo_price = option_pricer.monte_carlo()
        lr_binomial_tree_price = option_pricer.leisen_reimer_binomial_tree()

        print(f'Black-Scholes: {black_scholes_price}')
        print(f'Monte Carlo: {monte_carlo_price}')
        print(f'Leisen-Reimer Binomial Tree: {lr_binomial_tree_price}')

        spot_prices = np.linspace(50, 150, 10)
        strike_prices = np.linspace(80, 120, 10)
        expirations = np.linspace(0.1, 2, 10)

        option_prices = np.random.rand(len(spot_prices), len(strike_prices), len(expirations)) * 20 + 5

        option_pricer.plot_volatility_surface(option_prices, spot_prices, strike_prices, expirations)

if __name__ == '__main__':
    price_and_plot()
    price_and_plot_vol_surface()