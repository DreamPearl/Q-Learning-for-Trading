import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(col='Adj_Close'):
  """ Returns a 3 x n_step array """
  msft = pd.read_csv('data/MSFT_Stock.csv', usecols=[col])
 
  #ibm = pd.read_csv('data/daily_IBM.csv', usecols=[col])
  #qcom = pd.read_csv('data/daily_QCOM.csv', usecols=[col])
  # recent price are at top; reverse it
  #return np.array([msft[col].values[::-1],ibm[col].values[::-1],
               #   qcom[col].values[::-1]])
  return np.array([msft[col].values[::]])

def get_data2(col='RSI'):
     msft2 = pd.read_csv('data/MSFT_Stock.csv', usecols=[col])
     return np.array([msft2[col].values[::]])

def get_data_date(col='Trade_Date'):
     msft2 = pd.read_csv('data/MSFT_Stock.csv', usecols=[col])
     return np.array([msft2[col].values[::]])

def get_scaler(env):
  """ Takes a env and returns a scaler for its observation space """
  low = [0] * (env.n_stock * 2 + 1)
  # low = [0] * (env.n_stock)

  high = []
  max_price = env.stock_price_history.max(axis=1)
  min_price = env.stock_price_history.min(axis=1)
  max_rsi = env.stock_rsi_history.max(axis=1)
  max_cash = env.init_invest * 3 # 3 is a magic number...
  # rsi=env.stock_rsi
  # print(max_price)
  # print(min_price)
  # print(max_cash)
  max_stock_owned = max_cash // min_price
  # print(max_stock_owned)
  # exit()
  # need to check again for multiple companies later
  for i in max_stock_owned:
    high.append(i)
  for i in max_rsi:
    high.append(i)
  # high.append(max_rsi[0])           #not for 2 company
  # for i in max_price:
  #   high.append(i)
  high.append(max_cash)
  

  scaler = StandardScaler()
  scaler.fit([low, high])
  # print(scaler, low, high)
  # exit()
  
  # print([low, high])
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)