from __future__ import print_function
import pickle
import time
import numpy as np
import argparse
import re
import math



from envs import TradingEnv
from agent import DQNAgent
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
  args = parser.parse_args()

  maybe_make_dir('weights')
  maybe_make_dir('portfolio_val')

  timestamp = time.strftime('%Y%m%d%H%M')

  data = np.around(get_data())
  data_rsi=np.ceil((get_data2()/10))*10
  data_date=get_data_date()

  train_data = data[:, 3:600]
  train_data_rsi=data_rsi[:, 3:600]
  train_data_dates=data_date[:, 3:600]
  test_data_dates = data_date[:, 600:]
  test_data = data[:, 600:]
  test_data_rsi=data_rsi[:, 600:]
 

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
      print("saving the weights file") # checkpoint weights
      agent.save('weights/{}-dqn.h5'.format(timestamp))

  # save portfolio value history to disk
  with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
    pickle.dump(portfolio_value, fp)




