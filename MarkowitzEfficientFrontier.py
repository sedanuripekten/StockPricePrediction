import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, plotting
from pypfopt import expected_returns
from scipy.optimize import minimize
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

#Reading the final_df we created in FBProphetForecast.py file
df = pd.read_csv("//stock_forecast.csv")
#Converting df to pivot table style
def prepare_pivot_data(dataframe):
    dataframe["DATE"] = pd.to_datetime(dataframe["DATE"])
    dataframe = dataframe.pivot(index="DATE", columns="STOCK_NAME", values="STOCK_PRICE")
    return dataframe
df = prepare_pivot_data(df)
df.head()
df.shape


#Portfolio optimization with Efficient Frontier strategy
#TICKERS
tickers = df.columns.tolist()

#RETURNS
log_returns = np.log(df/df.shift(1)).dropna()
log_returns.isnull().sum()

#COVARIANCE MATRIX
cov_matrix = log_returns.cov()*252

#RISK FREE RATE
risk_free_rate = 0.02

#VOLATILITY=ST DEVIATION
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)
#Expected Return
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252
#Sharpe Ratio
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

#CONSTRAINTS, BOUNDS, INITIAL_WEIGHTS
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.4) for _ in range(len(tickers))]
initial_weights = np.array([1/len(tickers)]*len(tickers))

#Portfolios for 3 Types of Investors
# RISK AVERSE -> Min Volatility
def risk_averse_investor():
    def min_volatility(weights, cov_matrix):
        return standard_deviation(weights, cov_matrix)

    min_vol_result = minimize(min_volatility, initial_weights, args=(cov_matrix),
                          method='SLSQP', constraints=constraints, bounds=bounds)
    min_vol_weights = min_vol_result.x
    min_vol_return = expected_return(min_vol_weights, log_returns)
    min_vol_volatility = standard_deviation(min_vol_weights, cov_matrix)
    min_vol_sharpe = sharpe_ratio(min_vol_weights, log_returns, cov_matrix, risk_free_rate)
    # Saving min_vol filtered weights
    min_vol_filtered_weights = {ticker: round(weight, 4) for ticker, weight in zip(tickers, min_vol_weights) if
                                weight > 0.001}
    return min_vol_filtered_weights, min_vol_return, min_vol_volatility, min_vol_sharpe
min_vol_filtered_weights, min_vol_return, min_vol_volatility, min_vol_sharpe = risk_averse_investor()

# RISK TAKING -> Max Return
def risk_taking_investor():
    def neg_expected_return(weights, log_returns):
        return -expected_return(weights, log_returns)
    max_ret_result = minimize(neg_expected_return, initial_weights, args=(log_returns),
                              method='SLSQP', constraints=constraints, bounds=bounds)
    max_ret_weights = max_ret_result.x
    max_ret_return = expected_return(max_ret_weights, log_returns)
    max_ret_volatility = standard_deviation(max_ret_weights, cov_matrix)
    max_ret_sharpe = sharpe_ratio(max_ret_weights, log_returns, cov_matrix, risk_free_rate)
    # Saving max_ret filtered weights
    max_ret_filtered_weights = {ticker: round(weight, 4) for ticker, weight in zip(tickers, max_ret_weights) if
                                weight > 0.001}
    return max_ret_filtered_weights, max_ret_return, max_ret_volatility, max_ret_sharpe
max_ret_filtered_weights, max_ret_return , max_ret_volatility, max_ret_sharpe = risk_taking_investor()

# RISK NEUTRAL → OPTIMAL PORTFOLIO=Max Sharpe Ratio
def risk_neutral_investor():
    def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
    optimal_result = minimize(neg_sharpe_ratio, initial_weights,
                              args=(log_returns, cov_matrix, risk_free_rate),
                              method='SLSQP', constraints=constraints, bounds=bounds)
    optimal_weights = optimal_result.x
    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
    # Saving optimal filtered weights
    optimal_filtered_weights = {ticker: round(weight, 4) for ticker, weight in zip(tickers, optimal_weights) if
                                weight > 0.001}
    return optimal_filtered_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio
optimal_filtered_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio = risk_neutral_investor()

#Plots for visualizing investors and their stock weight distributions
def plots(optimal_pie=False, investors_bar=False, investors_pie=False):
    ticker_color_map = {
        'AAPL': '#FF6F61',  # Coral
        'ABBV': '#6B5B95',  # Purple
        'AMZN': '#FFB6B9',  # Soft Pink
        'ATRO': '#D4E157',  # Lime Green
        'BAC': '#FFEB3B',  # Yellow
        'BK': '#F44336',  # Red
        'C': '#607D8B',  # Blue Grey
        'F': '#9C27B0',  # Purple
        'GE': '#2196F3',  # Bright Blue
        'GM': '#795548',  # Brown
        'HMC': '#FF9800',  # Orange
        'JNJ': '#009688',  # Teal
        'JPM': '#3F51B5',  # Indigo
        'LLY': '#8BC34A',  # Light Green
        'META': '#9E9E9E',  # Grey
        'MSFT': '#4CAF50',  # Green
        'NOC': '#E91E63',  # Pink
        'NVDA': '#FF5722',  # Deep Orange
        'NVO': '#8E24AA',  # Deep Purple
        'PFE': '#00BCD4',  # Cyan
        'RTX': '#FF4081',  # Pink
        'SPR': '#3F4C6B',  # Charcoal
        'TM': '#0288D1',  # Bright Blue
        'TSLA': '#CDDC39',  # Lime
        'WFC': '#FF9800',  # Amber
    }

    if optimal_pie:
        # Optimal Portfolio (Risk Neutral) Pie Chart
        # Extract labels and sizes
        labels = list(optimal_filtered_weights.keys())
        sizes = list(optimal_filtered_weights.values())
        # Use ticker_color_map to get colors for the pie chart
        colors = [ticker_color_map.get(ticker, '#000000') for ticker in labels]  # Default to black if not found
        # Plot
        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            textprops={'fontsize': 12}
        )
        # Customize text
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(12)
        # Title adjustment
        plt.title("Optimal Portfolio Allocation", fontsize=16, y=1.05)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    if investors_bar:
        # INVESTOR TYPES BAR CHART
        fig, ax = plt.subplots(figsize=(14, 6))
        bar_width = 0.25
        x = np.arange(len(tickers))

        # Convert weight dictionaries to lists in ticker order
        # Union of all tickers involved in any of the portfolios
        all_tickers = sorted(set(min_vol_filtered_weights) |
                             set(optimal_filtered_weights) |
                             set(max_ret_filtered_weights))
        fig, ax = plt.subplots(figsize=(14, 6))
        bar_width = 0.25
        x = np.arange(len(all_tickers))

        # Fill missing weights with 0
        min_vol_weights = [min_vol_filtered_weights.get(t, 0) for t in all_tickers]
        opt_weights = [optimal_filtered_weights.get(t, 0) for t in all_tickers]
        max_ret_weights = [max_ret_filtered_weights.get(t, 0) for t in all_tickers]

        # Plot bars
        ax.bar(x - bar_width, min_vol_weights, width=bar_width, label='Risk Averse', edgecolor="black")
        ax.bar(x, opt_weights, width=bar_width, label='Risk Neutral', edgecolor="black")
        ax.bar(x + bar_width, max_ret_weights, width=bar_width, label='Risk Taking', edgecolor="black")

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels(all_tickers, rotation=45)
        ax.set_ylabel('Portfolio Weights')
        ax.set_title('Portfolio Weights for Different Investor Types')
        ax.legend()
        ax.grid(axis='y')
        plt.tight_layout()
        plt.show()

    if investors_pie:
        # INVESTOR TYPES PIE CHART
        # Define your 10 custom hex color codes
        hex_colors = ['#E63946', '#F1FAEE', '#A8DADC', '#457B9D', '#1D3557',
                      '#FFB703', '#FB8500', '#06D6A0', '#118AB2', '#073B4C']

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        titles = ['Risk Averse', 'Risk Neutral', 'Risk Taking']
        weights_list = [min_vol_filtered_weights, optimal_filtered_weights, max_ret_filtered_weights]
        returns = [min_vol_return, optimal_portfolio_return, max_ret_return]
        volatilities = [min_vol_volatility, optimal_portfolio_volatility, max_ret_volatility]
        sharpes = [min_vol_sharpe, optimal_sharpe_ratio, max_ret_sharpe]

        for i, ax in enumerate(axes):
            weights = weights_list[i]
            filtered = {ticker: weight for ticker, weight in weights.items() if weight > 0}

            # Use only as many hex colors as needed (loop if more than 10 tickers)
            colors = [ticker_color_map.get(ticker, '#000000') for ticker in filtered.keys()]  # Default to black if not found

            wedges, texts, autotexts = ax.pie(
                filtered.values(),
                labels=filtered.keys(),
                autopct='%1.1f%%',
                startangle=140,
                colors=colors,
                textprops={'fontsize': 11}
            )

            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(10)

            ax.set_title(titles[i], fontsize=14)
            ax.axis('equal')

            ax.text(0, -1.3,
                    f"Expected Return: {returns[i]*100:.2f}%\nExpected Volatility: {volatilities[i]*100:.2f}%\nSharpe: {sharpes[i]:.2f}",
                    ha='center', fontsize=10)

        plt.tight_layout()
        plt.show()
plots(optimal_pie=True, investors_bar=True, investors_pie=True)

def efficient_frontier_plot(dataframe):
    price_df = dataframe

    # Calculate mean returns (mu) and covariance (S)
    mu = expected_returns.mean_historical_return(price_df)
    S = risk_models.sample_cov(price_df)

    # Create Efficient Frontier object for max Sharpe (Risk Neutral)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()  # Max Sharpe portfolio (Risk Neutral)
    ret, vol, sharpe = ef.portfolio_performance()

    # Capital Allocation Line (CAL)
    x_vals = np.linspace(0, 0.5, 100)
    y_vals = risk_free_rate + (ret - risk_free_rate) / vol * x_vals

    ef_plot = EfficientFrontier(mu, S)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Efficient Frontier curve
    plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)

    # CAL
    ax.plot(x_vals, y_vals, label="Capital Allocation Line (CAL)", linestyle="--", color="red")

    # Risk Neutral (Optimal Portfolio)
    ax.scatter(vol, ret, marker="*", color="gold", s=200, label="Optimal Portfolio (Max Sharpe)")

    # Risk Averse (Min Volatility)
    ax.scatter(min_vol_volatility, min_vol_return, color="skyblue", s=100, label="Risk Averse (Min Volatility)",
               marker='o')

    # Risk Taking (Max Return Portfolio)
    # We generate a range of portfolios on the efficient frontier
    frontier_volatilities = np.linspace(0, 0.5, 100)
    frontier_returns = []

    # Loop over frontier volatilities and compute corresponding returns
    for volatility in frontier_volatilities:
        ef_temp = EfficientFrontier(mu, S)  # Create a new instance
        ef_temp.efficient_return(target_return=volatility)
        ret, vol, _ = ef_temp.portfolio_performance()
        frontier_returns.append(ret)

    # Find the max return portfolio along the efficient frontier
    max_return_index = np.argmax(frontier_returns)
    max_return_volatility = frontier_volatilities[max_return_index]
    max_return = frontier_returns[max_return_index]

    # Plot the Risk Taking portfolio (Max Return)
    ax.scatter(max_return_volatility, max_return, color="purple", s=100, label="Risk Taking (Max Return)", marker='s')

    # Risk-Free
    ax.scatter(0, risk_free_rate, color="blue", s=100, label="Risk-Free Rate", marker='x')

    ax.set_title("Efficient Frontier with CAL & Investor Types")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()
efficient_frontier_plot(df)

#Simulation of the investments for 1 year hold
def simulation(risk_neutral=False, risk_averse=False, risk_taking=False):
    initial_balance = 100000  # Example initial balance of $100,000
    def simulate_investment(initial_balance, filtered_weights, start_date='2024-01-02', end_date='2025-01-02'):
        # Retrieve stock data for the first market day of 2024 and 2025
        stock_prices = yf.download(list(filtered_weights.keys()), start=start_date, end=end_date)["Close"]

        # Get prices for the first market day of 2024 (start_date) and 2025 (end_date)
        start_prices = stock_prices.iloc[0]  # First market day of 2024
        end_prices = stock_prices.iloc[-1]  # First market day of 2025

        # Calculate the number of shares bought for each stock on the first market day of 2024
        shares_bought = {}
        for ticker, weight in filtered_weights.items():
            price_at_start = start_prices[ticker]
            shares_bought[ticker] = (initial_balance * weight) / price_at_start

        # Calculate the portfolio value on the first market day of 2025 (after selling all the stocks)
        portfolio_value_end = sum(shares_bought[ticker] * end_prices[ticker] for ticker in filtered_weights)

        # Calculate gain/loss
        total_invested = initial_balance  # This is the total amount invested initially
        gain_loss = portfolio_value_end - total_invested

        # Calculate the percentage change (return)
        return_percentage = (gain_loss / total_invested) * 100

        # Display the results
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Portfolio Value on 2025-01-02: ${portfolio_value_end:.2f}")
        print(f"Gain/Loss: ${gain_loss:.2f}")
        print(f"Return: {return_percentage:.2f}%")

        # Return the results
        return {
            "initial_balance": total_invested,
            "portfolio_value_end": portfolio_value_end,
            "gain_loss": gain_loss,
            "return_percentage": return_percentage
        }

    if risk_neutral:
        print("RISK NEUTRAL (OPTIMAL) PORTFOLIO")
        result = simulate_investment(initial_balance, optimal_filtered_weights)

    if risk_averse:
        print("RISK AVERSE PORTFOLIO")
        result = simulate_investment(initial_balance, min_vol_filtered_weights)

    if risk_taking:
        print("RISK TAKING PORTFOLIO")
        result = simulate_investment(initial_balance, max_ret_filtered_weights)
simulation(risk_neutral=False, risk_averse=True, risk_taking=False)

