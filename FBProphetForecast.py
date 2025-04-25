import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import streamlit as st

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Define stock tickers and their respective sectors
stocks = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "META": "Technology",
    "AMZN": "Technology",
    "JPM": "Finance",
    "BAC": "Finance",
    "C": "Finance",
    "BK": "Finance",
    "WFC": "Finance",
    "GE": "Aerospace_Defense",
    "NOC": "Aerospace_Defense",
    "RTX": "Aerospace_Defense",
    "SPR": "Aerospace_Defense",
    "ATRO": "Aerospace_Defense",
    "GM": "Automobile_AutoParts",
    "F": "Automobile_AutoParts",
    "TM": "Automobile_AutoParts",
    "HMC": "Automobile_AutoParts",
    "TSLA": "Automobile_AutoParts",
    "NVO": "Health",
    "JNJ": "Health",
    "PFE": "Health",
    "ABBV": "Health",
    "LLY": "Health"
}

def forecast_regressor(df, column_name, days_ahead):
    model = Prophet()
    df_reg = df[['ds', column_name]].rename(columns={column_name: 'y'})
    model.fit(df_reg)

    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)


    return forecast[['ds', 'yhat']].rename(columns={'yhat': column_name})

# Forecast function
def forecast_stock(stocks, start_date="2013-01-01", end_date="2023-01-01", days_ahead=365, plot=False):
    all_forecasts = []  # List to store forecasted data for each ticker

    rmse_results = {}  # Dictionary to store RMSE for each ticker

    for ticker, sector in stocks.items():
        try:
            # Fetch stock data
            df = yf.download(ticker, start=start_date, end=end_date)

            # Feature Engineering
            df["VOLATILITY"] = df["Close"].rolling(window=10).std()
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
            df["SMA_200"] = df["Close"].rolling(window=200).mean()

            # RSI Calculation
            delta = df["Close"].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # Fill NaN values
            df.fillna(method="bfill", inplace=True)

            # Prepare Data for Prophet
            df = df.reset_index()
            df = df[["Date", "Close", "VOLATILITY", "SMA_50", "SMA_200", "RSI"]]
            df.columns = ["ds", "y", "VOLATILITY", "SMA_50", "SMA_200", "RSI"]

            # Forecast each regressor separately
            volatility_forecast = forecast_regressor(df, "VOLATILITY", days_ahead)
            sma50_forecast = forecast_regressor(df, "SMA_50", days_ahead)
            sma200_forecast = forecast_regressor(df, "SMA_200", days_ahead)
            rsi_forecast = forecast_regressor(df, "RSI", days_ahead)

            # Merge regressors forecasts
            future_regressors = volatility_forecast.merge(sma50_forecast, on="ds") \
                .merge(sma200_forecast, on="ds") \
                .merge(rsi_forecast, on="ds")

            # Define Prophet Model
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                daily_seasonality=False,
                weekly_seasonality=True
            )
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

            # Add Regressors
            model.add_regressor("VOLATILITY")
            model.add_regressor("SMA_50")
            model.add_regressor("SMA_200")
            model.add_regressor("RSI")

            # Fit Model
            model.fit(df)

            # Create future dataframe for forecasting
            future = model.make_future_dataframe(periods=days_ahead)

            # Merge with forecasted regressor values
            future = future.merge(future_regressors, on="ds", how="left")

            # Make Prediction
            forecast = model.predict(future)

            # Extract Future Predictions
            hist_forecast = forecast
            hist_forecast["STOCK_NAME"] = ticker
            hist_forecast["SECTOR"] = sector
            hist_forecast = hist_forecast[["ds", "yhat", "VOLATILITY", "STOCK_NAME", "SECTOR"]]
            hist_forecast.rename(columns={"ds": "DATE", "yhat": "STOCK_PRICE"}, inplace=True)

            # Add to the list of all forecasts
            all_forecasts.append(hist_forecast)

            if plot:
                # Plot results
                model.plot(forecast)
                plt.title(f"Stock Price Forecast for {ticker}")
                plt.show()

            # Define cutoff period for cross-validation
            df_cv = cross_validation(model, horizon="365 days", period="90 days", initial="2190 days")

            # Calculate performance metrics
            df_p = performance_metrics(df_cv, metrics=["rmse", "mape", "mae"])

            # Store RMSE for this ticker
            rmse_results[ticker] = df_p["rmse"].mean()

        except Exception as e:
            print(f"Error forecasting {ticker}: {e}")

    # Combine all forecasted data into a single DataFrame
    forecasted_df = pd.concat(all_forecasts, ignore_index=True)

    # Print all RMSE values
    print("\nRMSE Results:")
    total_rmse = 0
    for stock, rmse in rmse_results.items():
        print(f"{stock}: {rmse:.4f}")
        total_rmse += rmse
    print(f"Total RMSE: {total_rmse:.4f}")

    return forecasted_df

forecasted_df = forecast_stock(stocks)
#RMSE for only historical predicted (not the forecasted stock data!)
#RMSE Error:
    # RMSE Results:
    # AAPL: 12.4271
    # MSFT: 14.0565
    # NVDA: 2.6829
    # META: 37.8447
    # AMZN: 15.8900
    # JPM: 9.7897
    # BAC: 2.6635
    # C: 4.8849
    # BK: 3.3068
    # WFC: 4.1431
    # GE: 7.4332
    # NOC: 22.4789
    # RTX: 5.5295
    # SPR: 9.7211
    # ATRO: 3.8613
    # GM: 4.8791
    # F: 1.4050
    # TM: 7.2025
    # HMC: 1.2842
    # TSLA: 57.8517
    # NVO: 2.2089
    # JNJ: 4.4157
    # PFE: 3.4377
    # ABBV: 5.1696
    # LLY: 14.3555
    # Total RMSE: 258.9232
#Dropping weekends out of the forecasted dataset

def only_future(dataframe, date_threshold = "2023-01-01"):
    dataframe["DOW"] = dataframe["DATE"].dt.day_name()
    dataframe = dataframe[((dataframe["DOW"] != "Saturday") & (dataframe["DOW"] != "Sunday"))]
    # Keeping only the future forecasted stock prices
    only_future_df = dataframe[dataframe["DATE"] > date_threshold]
    print(only_future_df.head())
    return only_future_df
only_future_df = only_future(forecasted_df)

def keep_only_marketdays(dataframe):
    # List of public holidays (excluding weekends) for 2023 when the market is closed
    market_holidays = [
        "2023-01-02",  # New Year's Day (observed)
        "2023-01-16",  # Martin Luther King Jr. Day
        "2023-02-20",  # Presidents' Day
        "2023-04-07",  # Good Friday
        "2023-05-29",  # Memorial Day
        "2023-06-19",  # Juneteenth National Independence Day
        "2023-07-04",  # Independence Day
        "2023-09-04",  # Labor Day
        "2023-11-23",  # Thanksgiving Day
        "2023-12-25",  # Christmas Day
        "2023-11-24",  # Black Friday (early close)
        "2023-12-24",  # Day before Christmas (Sunday, markets closed)
        "2023-07-03",  # Day before Independence Day (early close)
    ]
    # Exclude these holidays from your dataset
    dataframe = dataframe[~dataframe['DATE'].isin(market_holidays)]
    dataframe.shape
    # (6200, 6)
    dataframe = dataframe.sort_values(by="DATE")
    return dataframe
only_future_df = keep_only_marketdays(only_future_df)


#Pulling only historical data to a dataframe
def get_historical_stocks(stocks, start_date = "2013-01-01", end_date = "2023-01-01", plot=False):
    all_data = [] #List to store each stock

    for ticker, sector in stocks.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date)

            df.columns = df.columns.droplevel(1)  # Drop the first level of column names

            #Feature Engineering
            df["VOLATILITY"] = df["Close"].rolling(window=10).std()

            #Fill missing values
            df.fillna(method="bfill", inplace=True)

            #Add stock info
            df = df.reset_index()
            df["STOCK_NAME"] = ticker
            df["SECTOR"] = sector

            #Keep only relevant columns
            df = df[["Date", "Close", "Volume", "VOLATILITY", "STOCK_NAME", "SECTOR"]]

            #Rename columns
            df.rename(columns={"Date": "DATE", "Close": "STOCK_PRICE"}, inplace=True)

            if plot:
                plt.figure(figsize=(10, 5))
                sns.lineplot(x=df["DATE"], y=df["STOCK_PRICE"], label=ticker)
                plt.title(f"Stock Price Trend: {ticker}")
                plt.xlabel("Date")
                plt.ylabel("Close Price (USD)")
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.show()

            all_data.append(df)

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    #Combine all stock data into a single dataframe
    historical_df = pd.concat(all_data, ignore_index=True)
    historical_df["DOW"] = historical_df["DATE"].dt.day_name()
    historical_df = historical_df.sort_values(by="DATE")
    return historical_df
historical_df = get_historical_stocks(stocks)
historical_df.shape

#Combining historical data (2013-2023) + forecasted data (2023-2024)
final_df = pd.concat([historical_df, only_future_df], ignore_index=True)
final_df.shape #total row count
final_df.info()
final_df.head()


#KNN Imputation for null Volume values in final_df
def knn_imputation(dataframe):
    #Scaling
    features_to_scale = ["STOCK_PRICE", "Volume", "VOLATILITY"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dataframe[features_to_scale])
    dataframe[features_to_scale] = scaled_features

    # Apply KNNImputer to fill missing values in the entire dataframe
    knn_imputer = KNNImputer(n_neighbors=5)
    dataframe[['STOCK_PRICE', 'Volume', 'VOLATILITY']] = knn_imputer.fit_transform(dataframe[['STOCK_PRICE', 'Volume', 'VOLATILITY']])

    # Inverse transform the scaled columns back to the original range
    dataframe[features_to_scale] = scaler.inverse_transform(dataframe[features_to_scale])

    #Uppercase the column name
    dataframe = dataframe.rename(columns={"Volume": "VOLUME"})
    return dataframe
final_df = knn_imputation(final_df)
final_df.isnull().sum()
final_df.tail(20)
final_df.shape

# Save to CSV (optional)
final_df.to_csv("stock_forecast.csv", index=False)

#Pull historical full dataframe from 2013 to 2024 to compare the performance of portfolio simulations later on with historical + forecasted dataframe (final_df)
def full_historical():
    historical_full_df = get_historical_stocks(stocks, start_date="2013-01-01", end_date="2024-01-01")
    historical_full_df["DOW"] = historical_full_df["DATE"].dt.day_name()
    return historical_full_df
historical_full_df = full_historical()

historical_full_df.to_csv("stock_full_historical.csv", index=False)


