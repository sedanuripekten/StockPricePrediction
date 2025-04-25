import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = pd.read_csv("//stock_forecast.csv")

def percentage_changes(dataframe):
    dataframe["DATE"] = pd.to_datetime(dataframe["DATE"])

    # Find the earliest and latest dates for each stock
    first_close = dataframe.loc[dataframe.groupby("STOCK_NAME")["DATE"].idxmin(), ["STOCK_NAME", "STOCK_PRICE"]]
    last_close = dataframe.loc[df.groupby("STOCK_NAME")["DATE"].idxmax(), ["STOCK_NAME", "STOCK_PRICE"]]

    first_close.set_index(first_close["STOCK_NAME"], inplace = True)
    first_close = first_close.drop("STOCK_NAME", axis = 1)

    last_close.set_index(last_close["STOCK_NAME"], inplace = True)
    last_close = last_close.drop("STOCK_NAME", axis = 1)

    stock_changes_pct = ((last_close["STOCK_PRICE"] - first_close["STOCK_PRICE"]) / first_close["STOCK_PRICE"]) * 100
    stock_changes_pct = stock_changes_pct.to_frame(name="PERCENTAGE_CHANGE")
    stock_changes_pct.reset_index(inplace=True)

    technology = ["AAPL","MSFT","NVDA","META","AMZN"]
    finance = ["JPM","BAC","C","BK","WFC"]
    health = ["NVO","JNJ","PFE","ABBV","LLY"]
    aero_defense = ["GE","NOC","RTX","SPR","ATRO"]
    auto = ["GM","F","TM","HMC","TSLA"]

    # Define conditions
    conditions = [
        stock_changes_pct["STOCK_NAME"].isin(technology),
        stock_changes_pct["STOCK_NAME"].isin(finance),
        stock_changes_pct["STOCK_NAME"].isin(health),
        stock_changes_pct["STOCK_NAME"].isin(auto),
        stock_changes_pct["STOCK_NAME"].isin(aero_defense)
    ]

    # Define corresponding values
    choices = ["Technology", "Finance", "Health", "Automobile_Autoparts", "Aerospace_Defense"]

    # Assign sector using np.select()
    stock_changes_pct["SECTOR"] = np.select(conditions, choices, default="Other") # Default for unknown stocks

    return stock_changes_pct

stock_changes_pct = percentage_changes(df)

stock_changes_pct.to_csv("stock_changes.csv", index=False)
