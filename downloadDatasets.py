import os
import numpy as np
import pandas as pd
import datetime
import time
from datetime import date
import pandas_ta as ta
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as si
import datetime
import yfinance as yf
from random import randint
from time import sleep

#%%
def load_historic_data(symbol, start_date_str, today_date_str, period, interval, prepost):
    try:
        df = yf.download(symbol, start=start_date_str, end=today_date_str, period=period, interval=interval, prepost=prepost)
        #  Add symbol
        df["Symbol"] = symbol
        return df
    except:
        print('Error loading stock data for ' + symbols)
        return None

#%%
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# fetch data by interval (including intraday if period < 60 days)
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
period = 'max'
interval = '1d'
prepost = True
today = datetime.date.today()
today_date_str = today.strftime("%Y-%m-%d")
#  NOTE: 7 days is the max allowed
days = datetime.timedelta(20000)
start_date = today - days
start_date_str = datetime.datetime.strftime(start_date, "%Y-%m-%d")
#  Coins to download
symbols = ['BTC-USD']
#  Fetch data for coin symbols
for symbol in symbols:
    print(f"Loading data for {symbol}")
    df = load_historic_data(symbol, start_date_str, today_date_str, period, interval, prepost)
    #  Save df
    file_name = f"{today_date_str}_{symbol}_{period}_{interval}.csv"
    df.to_csv(f"D:/Files/Datasets/cryptocurrency/main/{file_name}")
    #  Avoid DOS issues
    sleep(randint(0,3))