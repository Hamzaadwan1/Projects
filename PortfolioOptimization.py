# importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize
import yfinance as yf
import seaborn as sns

# Stage 1: Pull the data of different asset classes and assess their correlation

# Choosing assets: S&P, US-5Y, Gold
tickers = ["^GSPC", "^FVX", "GLD", "EEA", "LULU", "TSLA", "CL=F" ]
data = yf.download(tickers, start='2018-01-01', end='2024-01-01')['Adj Close']
print(data.head())


# Parameters for your portfolio
start_date = '2018-01-01'
end_date = '2024-01-01'


# Calculating daily returns
daily_returns = data.pct_change().dropna()
print("\nThe daily percent changes are:")
print(daily_returns.head())

# Computing correlation matrix
corr_matrix = daily_returns.corr()

# Plotting the correlation between the assets
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Stage 2: Annualize mean returns and plot a covariance matrix

mean_returns = daily_returns.mean() * 252  # annualizing returns and covariance 
cov_matrix = daily_returns.cov() * 252

print('\nAnnualized Returns:')
print(mean_returns)
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Covariance Matrix')
plt.show()

# Stage 3: Calculate the Sharpe Ratio of the portfolio

# Assuming a risk free rate of 0.045 based on average yield on 5yr treasury bills
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.045): 
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Total weight of the portfolio is 1
bounds = tuple((0, 1) for asset in range(len(tickers)))
initial_guess = np.array([1 / len(tickers)] * len(tickers))

result = minimize(negative_sharpe, initial_guess, args=(mean_returns, cov_matrix, 0.045), method='SLSQP', bounds=bounds, constraints=constraints)
optimized_weights = result.x
optimized_return = np.dot(optimized_weights, mean_returns)
optimized_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
optimized_sharpe_ratio = (optimized_return - 0.045) / optimized_volatility
print("\nOptimized Portfolio Weights:", optimized_weights)
print("Optimized Portfolio Return:", optimized_return)
print("Optimized Portfolio Volatility:", optimized_volatility)
print('Optimized Sharpe Ratio', optimized_sharpe_ratio)

# Stage 4: Plot the efficient frontier
num_portfolios = 25000
risk_free_rate = 0.045

# Initializing empty lists
portfolio_returns = []
portfolio_volatility = []
sharpe_ratios = []
stock_weights = []

# Assigning random weights to the assets of 25,000 different portfolios and then finding the sharpe, volatility and return of each, outputting the one with the highest sharpe 

np.random.seed(42)  # ensures the same output of random numbers each time
for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)  # normalizing weights to 1
    portfolio_return = np.dot(weights, mean_returns)  # expected portfolio return
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # expected volatility
    sharpe = (portfolio_return - risk_free_rate) / volatility  # sharpe ratios

    # appending results
    portfolio_returns.append(portfolio_return)
    portfolio_volatility.append(volatility)
    sharpe_ratios.append(sharpe)
    stock_weights.append(weights)

# Turning the lists into arrays so it's easier to handle
portfolio_returns = np.array(portfolio_returns)
portfolio_volatility = np.array(portfolio_volatility)
sharpe_ratios = np.array(sharpe_ratios)

# Finding the portfolio with the maximum Sharpe ratio
max_sharpe_idx = np.argmax(sharpe_ratios)
max_sharpe_return = portfolio_returns[max_sharpe_idx]
max_sharpe_volatility = portfolio_volatility[max_sharpe_idx]

# Plotting the efficient frontier
plt.figure(figsize=(12, 8))
scatter = plt.scatter(portfolio_volatility, portfolio_returns, c=sharpe_ratios, cmap='viridis')
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Volatility (STD)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')

# Adding the optimized method to the chart to see the difference in numbers
plt.scatter(optimized_volatility, optimized_return, c='blue', s=50, marker='*', label='Optimized Sharpe Ratio')
plt.scatter(max_sharpe_volatility, max_sharpe_return, c='red', s=50, marker='*', label='Maximum Sharpe Ratio')
plt.legend(labelspacing=0.8)
plt.show()

# Stage 5: Compare between best portfolios found through optimization and the efficient frontier

comparison = pd.DataFrame({
    "\nMetric": ["Return", "Volatility", "Sharpe Ratio"],
    "Optimized Portfolio": [optimized_return, optimized_volatility, optimized_sharpe_ratio],
    "Efficient Frontier Max": [max_sharpe_return, max_sharpe_volatility, sharpe_ratios[max_sharpe_idx]]
})

print(comparison)

# Stage 6: Conduct Monte Carlo simulation using geometric brownian motion model to simulate stock prices, this model accounts for both drift (average daily return) and volatility

# Function to calculate historical returns and volatilities
def calculate_mu_sigma(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    daily_returns = data.pct_change().dropna()
    mean_returns = daily_returns.mean().values * 252  # Annualized mean returns
    volatilities = daily_returns.std().values * np.sqrt(252)  # Annualized volatilities
    return mean_returns, volatilities, daily_returns.corr().values

# Function to simulate GBM for multiple assets and aggregate portfolio
def gbm_multi_sim(S0, mu, sigma, corr_matrix, T, dt, num_simulations, weights):
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N)
    num_assets = len(S0)
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(corr_matrix)
    
    # Initialize an array to store all simulations
    portfolio_simulations = np.zeros((num_simulations, N))
    
    for sim in range(num_simulations):
        # Simulate independent Wiener processes
        W = np.random.standard_normal((N, num_assets))
        
        # Introduce correlations using the Cholesky matrix
        W = np.dot(W, L.T)
        
        # Cumulative sum for the Wiener process over time
        W = np.cumsum(W, axis=0) * np.sqrt(dt)
        
        # Initialize the array for simulated prices
        S = np.zeros_like(W)
        
        for i in range(num_assets):
            X = (mu[i] - 0.5 * sigma[i] ** 2) * t + sigma[i] * W[:, i]
            S[:, i] = S0[i] * np.exp(X)
        
        # Normalize to initial value of 1
        portfolio_value = np.dot(S, weights)
        portfolio_simulations[sim, :] = portfolio_value / portfolio_value[0]
    
    return t, portfolio_simulations

# Fetch latest prices
latest_data = yf.download(tickers, period='1d')['Adj Close']
initial_prices = latest_data.values[-1]

# Calculate mu and sigma
mu, sigma, corr_matrix = calculate_mu_sigma(tickers, start_date, end_date)

# Simulation parameters
T = 6.0  # Time period in years
dt = 0.01  # Time step
num_simulations = 500  # Number of simulations

# Simulate GBM
t, portfolio_simulations = gbm_multi_sim(initial_prices, mu, sigma, corr_matrix, T, dt, num_simulations, optimized_weights)


# Plot the simulations for the portfolio
plt.figure(figsize=(12, 8))
for sim in range(num_simulations):
    plt.plot(t, portfolio_simulations[sim, :], alpha=0.1, color='blue')
plt.plot(t, np.mean(portfolio_simulations, axis=0), color='red', label='Mean Portfolio Value')
plt.xlabel('Time (years)')
plt.ylabel('Portfolio Value')
plt.title('Geometric Brownian Motion Simulation for Portfolio')
plt.legend()
plt.show()

# Stage 7: Normalizing the data gathered from the Monte-Carlo simulations, finding mean, STD and variance

# Calculate the final normalized portfolio values
final_values = portfolio_simulations[:, -1]

# Calculate mean, standard deviation, skewness, and kurtosis of the final portfolio values
mean_final_value = np.mean(final_values)
std_final_value = np.std(final_values)
skewness = skew(final_values)
kurt = kurtosis(final_values)

print(f"Mean: {mean_final_value}, Std: {std_final_value}, Skewness: {skewness}, Kurtosis: {kurt}")

# Plot the histogram of the final portfolio values
plt.figure(figsize=(12, 8))
sns.histplot(final_values, bins=100, kde=True, color='g', stat='density')

# Plot the normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_final_value, std_final_value)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f, skewness = %.2f, kurtosis = %.2f" % (mean_final_value, std_final_value, skewness, kurt)
plt.title(title)
plt.xlabel('Final Portfolio Value (Normalized)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Stage 8: Calculating portfolio returns and plotting it against the S&P 500

# Calculate portfolio returns
p_returns = daily_returns.dot(optimized_weights)
p_cum_returns = (1 + p_returns).cumprod()

# Fetch S&P 500 data and calculate cumulative returns
sp500 = yf.download("^GSPC", start='2018-01-01', end='2024-01-01')['Adj Close']
sp500_returns = sp500.pct_change().dropna()
sp500_cum_returns = (1 + sp500_returns).cumprod()

# Plot the cumulative returns of the portfolio and the S&P 500
plt.figure(figsize=(12, 8))
plt.plot(p_cum_returns, label='Optimized Portfolio')
plt.plot(sp500_cum_returns, label='S&P 500')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Portfolio Performance vs. S&P 500')
plt.legend()
plt.show()
