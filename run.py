from __future__ import print_function
import pickle
import time
import numpy as np
import argparse
import re
import math
import pandas as pd
import talib as tb
from backtester.dataSource.yahoo_data_source import YahooStockDataSource
import matplotlib.pyplot as plt
import datetime
from datetime import date



from envs import TradingEnv
from agent import DQNAgent
# from GetExternalData.py import weeklyTrade
from utils import get_data,get_data2,get_data_date,get_scaler, maybe_make_dir


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode', type=int, default=2000,
                      help='number of episode to run')
  parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='batch size for experience replay')
  parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                      help='initial investment amount')
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
  parser.add_argument('-c' , '--company', type=str,required=True,help='company name(Like MSFT for Microsoft etc..)')
  args = parser.parse_args()

  maybe_make_dir('weights')
  maybe_make_dir('portfolio_val')


  startDateStr = '2015/12/01'
  endDateStr = date.today().strftime("%Y/%m/%d")
  cachedFolderName = 'yahooData/'
  dataSetId = 'weeklyTradeTest'#can be anything 


# In[12]:
  stkName = args.company
  dir_path = r'yahooData/weeklyTradeTest/' + stkName + '_Stock'+'.csv'

  class weeklyTrade:
      def __init__(self,stkName):
          self.stkName=stkName
          self.instrumentIds = [self.stkName]
          self.stkData =[]
          self.LoadData()
              
      def LoadData(self):
          
          ds = YahooStockDataSource(cachedFolderName=cachedFolderName,
                              dataSetId=dataSetId,
                              instrumentIds=self.instrumentIds,
                              startDateStr=startDateStr,
                              endDateStr=endDateStr,
                              event='history')
         
          self.stkData = [ds.getBookDataByFeature()['open'],
                  ds.getBookDataByFeature()['high'],
                  ds.getBookDataByFeature()['low'],
                  ds.getBookDataByFeature()['close'],
                  ds.getBookDataByFeature()['adjClose'],
                  ds.getBookDataByFeature()['volume']]
          tbRSI = tb.RSI(self.stkData[4][self.instrumentIds[0]],14)
          
          tbChaikin = tb.ADOSC(self.stkData[1][self.instrumentIds[0]],
                              self.stkData[2][self.instrumentIds[0]],
                              self.stkData[3][self.instrumentIds[0]],
                              self.stkData[5][self.instrumentIds[0]])
          
      
          '''
          Task1
          Create new dataframe here with 5 columns 
          column  1 : Date of stock price 
          column  2 : Day of the week (Monday Tuesday etc)
          Columnn 3 :  tbRSI
          Column 4 : tbChaikin
          column 5 :  Stock's adjusted price.
          Index : column1 
          '''
          dfdata = {'Adj_Close':self.stkData[4][self.instrumentIds[0]],
                   'RSI':tbRSI,
                   'Chaikin':tbChaikin}
          df = pd.DataFrame(dfdata)
          df['Trade_Date'] = pd.to_datetime(df.index)
          df['day_of_week'] = df['Trade_Date'].dt.day_name()
          
          dfFinal = df[["Trade_Date", "day_of_week", "RSI", "Chaikin", "Adj_Close"]]
          # dfFinal2 = df[["Trade_Date", "day_of_week", "RSI", "Chaikin", "Adj_Close"]]
          #df_rsi = pd.DataFrame(dfFinal['RSI'])
          #print(df_rsi)
          #df_rsi.to_csv('yahooData/weeklyTradeTest/RSI_Value.csv')
          
          
          
          '''
          Task 2
          filter above dataframe using column  2 value (e.g. 'Wednesday') and store in other dataframe 
          '''
          # dfFinal2 = dfFinal.loc[df['day_of_week'] == 'Wednesday']
          # dfFinal.to_csv(dir_path)    
          # print(dfFinal.head(50)
          # dfFinal = dfFinal.loc[df['day_of_week'] == 'Friday']
          dfFinal.to_csv(dir_path)    
          print(dfFinal.head(50))
     
  w1 = weeklyTrade(stkName)
  timestamp = time.strftime('%Y%m%d%H%M')

  data = np.around(get_data(stkName))
  # print(data)
  data_rsi=np.ceil((get_data2(stkName)/10))*10
  data_date=get_data_date(stkName)
  length=len(data[0])
  # print(length)
  train_data_length=int(round(length*0.8))
  train_data = data[:, 15:train_data_length]
  train_data_rsi=data_rsi[:, 15:train_data_length]
  train_data_dates=data_date[:, 15:train_data_length]
  test_data_dates = data_date[:, train_data_length:]
  test_data = data[:, train_data_length:]
  test_data_rsi=data_rsi[:, train_data_length:]
 

  env = TradingEnv(train_data,train_data_rsi, args.initial_invest)
  state_size = env.observation_space.shape
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size,args.mode)
  scaler = get_scaler(env)
  
  print("state(stock_owned),state(rsi),state(cash_in_hand),action,net_worth,accuracy,loss")
  portfolio_value = []

  if args.mode == 'test':
    
    # remake the env with test data
    env = TradingEnv(test_data,test_data_rsi, args.initial_invest)
    # load trained weights
    agent.load(args.weights)
    # when test, the timestamp is same as time when weights was trained
    timestamp = re.findall(r'\d{12}', args.weights)[0]
    
  for e in range(args.episode):
    state = env.reset()
    state = scaler.transform([state])  
    for time in range(env.n_step):
      before_transform_state = scaler.inverse_transform(state)
      qvalue_current= agent.model.predict(np.array(state))
      
      print('{},{},{},'.format(before_transform_state[0][0],before_transform_state[0][1],before_transform_state[0][2]),end='')
      
      action = agent.act(state)
      print('{}'.format(action),end='')
      if action==0:
       print('(sell),',end='')
      elif action==1:
        print('(hold),',end='')
      else:
        print('(buy),',end='')
  
      next_state, reward, done, info = env.step(action)
     
      next_state = scaler.transform([next_state])    
     
      before_transform_next_state = scaler.inverse_transform(next_state)   
      if args.mode == 'train':
        agent.remember(state, action, reward, next_state, done)
        
      state = next_state
      if done:
        print("episode: {}/{}, episode end value: {}".format(
          e + 1, args.episode, info['cur_val']))
        portfolio_value.append(info['cur_val']) # append episode end portfolio value
        break
      
      
      if args.mode == 'train' and len(agent.memory) > args.batch_size:
        

        history=agent.replay(e,action,args.batch_size)
      print()
    
    if args.mode == 'train': # and (e + 1) % 10 == 0: 
      weight_name='weights/{}-dqn.h5'.format(timestamp)
      print("saving the weights file {}".format(weight_name)) # checkpoint weights
      agent.save(weight_name)

  # save portfolio value history to disk
  with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
    pickle.dump(portfolio_value, fp)




