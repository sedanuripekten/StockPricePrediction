import streamlit as st
#pip install streamlit
from matplotlib import cm
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import expected_returns
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#from MarkowitzEfficientFrontier import max_ret_filtered_weights, max_ret_volatility, max_ret_sharpe

pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df = pd.read_csv("//stock_forecast.csv")

def prepare_pivot_data(dataframe):
    dataframe["DATE"] = pd.to_datetime(dataframe["DATE"])
    dataframe = dataframe.pivot(index="DATE", columns="STOCK_NAME", values="STOCK_PRICE")
    return dataframe
df = prepare_pivot_data(df)

tickers = df.columns.tolist()

#RETURNS
log_returns = np.log(df/df.shift(1)).dropna()
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

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.4) for _ in range(len(tickers))]
initial_weights = np.array([1/len(tickers)]*len(tickers))

#Portfolios for 3 Types of Investors + the weights

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

# previous code here for portfolio weights, calculations, and optimization...

# Simulate investment function
def simulate_investment(initial_balance, filtered_weights, start_date='2024-01-02', end_date='2025-01-02'):
    stock_prices = yf.download(list(filtered_weights.keys()), start=start_date, end=end_date)["Close"]
    start_prices = stock_prices.iloc[0]
    end_prices = stock_prices.iloc[-1]

    shares_bought = {}
    for ticker, weight in filtered_weights.items():
        price_at_start = start_prices[ticker]
        shares_bought[ticker] = (initial_balance * weight) / price_at_start

    portfolio_value_end = sum(shares_bought[ticker] * end_prices[ticker] for ticker in filtered_weights)

    gain_loss = portfolio_value_end - initial_balance
    return_percentage = (gain_loss / initial_balance) * 100

    return {
        "initial_balance": initial_balance,
        "portfolio_value_end": portfolio_value_end,
        "gain_loss": gain_loss,
        "return_percentage": return_percentage
    }

# Streamlit UI
def main():
    st.title("Stock Portfolio Simulator")

    # Option for user to choose investor type
    investor_type = st.selectbox(
        "Select Investor Type",
        ["Risk Averse", "Risk Neutral (Optimal)", "Risk Taking"]
    )

    # Set the initial balance for investment (user input option)
    initial_balance = st.number_input("Enter your initial investment balance:", min_value=0, value=100000)

    # Dictionary mapping the investor types to the filtered portfolio weights
    investor_weights = {
        "Risk Averse": min_vol_filtered_weights,
        "Risk Neutral (Optimal)": optimal_filtered_weights,
        "Risk Taking": max_ret_filtered_weights
    }

    # Simulate investment based on selected investor type
    if st.button(f"Simulate {investor_type} Portfolio"):
        filtered_weights = investor_weights[investor_type]
        result = simulate_investment(initial_balance, filtered_weights)

        # Display the results
        st.write(f"Initial Balance: ${result['initial_balance']:,.2f}")
        st.write(f"Portfolio Value on 2025-01-02: ${result['portfolio_value_end']:,.2f}")
        st.write(f"Gain/Loss: ${result['gain_loss']:,.2f}")
        st.write(f"Return: {result['return_percentage']:.2f}%")

        # Plot pie chart of portfolio weights
        st.subheader(f"Portfolio Weight Distribution for {investor_type}")
        labels = list(filtered_weights.keys())
        sizes = list(filtered_weights.values())
        colors = ['#FF6F61', "#6B5B95", "#88B04B", "#F7B7A3", "#F1D00A", "#4A90E2", "#D85C5C", "#9B5F5F", "#00A99D",
                  "#F1A7B3"]

        fig, ax = plt.subplots()

        # Plot the pie chart with labels directly on wedges
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1},
            labeldistance=1.1,  # Pushes the labels slightly away from the center
            pctdistance=0.75  # Moves percentage labels closer to center
        )

        # Add a border to the pie chart area
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)

        # Add a title
        plt.title(f"Portfolio Distribution for {investor_type} Investor")

        # Display in Streamlit
        st.pyplot(fig)

if __name__ == "__main__":
    main()



