from __future__ import print_function
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools

TRADE_BENEFIT = 0

class TradingEnv(gym.Env):
  """
  A 3-stock (MSFT, IBM, QCOM) trading environment.

  State: [# of stock owned, current stock prices, cash in hand]
    - array of length n_stock * 2 + 1
    - price is discretized (to integer) to reduce state space
    - use close price for each stock
    - cash in hand is evaluated at each step based on action performed

  Action: sell (0), hold (1), and buy (2)
    - when selling, sell all the shares
    - when buying, buy as many as cash in hand allows
    - if buying multiple stock, equally distribute cash in hand and then utilize the balance
  """
  def __init__(self, train_data,train_data_rsi, init_invest=20000):
    # data
    self.stock_price_history = np.around(train_data) # round up to integer to reduce state space
    self.n_stock, self.n_step = self.stock_price_history.shape
    self.stock_rsi_history=train_data_rsi

    # instance attributes
    self.init_invest = init_invest
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.stock_rsi= None
    self.cash_in_hand = None

    # action space
    self.action_space = spaces.Discrete(3**self.n_stock)

    # observation space: give estimates in order to sample and build scaler
    # print(self.stock_price_history)
    stock_max_price = self.stock_price_history.max(axis=1)
    # print(stock_max_price)
    # exit()
    stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
    price_range = [[0, mx] for mx in stock_max_price]
    cash_in_hand_range = [[0, init_invest * 2]]
    rsi_range=[[0,100]]
    # print(stock_range)
    # print(price_range)
    # print(cash_in_hand_range)
    # print(stock_range + price_range + cash_in_hand_range)
    # exit()
    self.observation_space = spaces.MultiDiscrete(stock_range + rsi_range + cash_in_hand_range)
    # self.observation_space = spaces.MultiDiscrete(rsi_range)


    # seed and start
    self._seed()
    self._reset()


  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def _reset(self):
    self.cur_step = 0
    self.stock_owned = [0] * self.n_stock
    self.stock_price = self.stock_price_history[:, self.cur_step]
    self.stock_rsi = self.stock_rsi_history[:, self.cur_step]
    self.cash_in_hand = self.init_invest
    return self._get_obs()


  def _step(self, action):
    assert self.action_space.contains(action)
    prev_val = self._get_val()
    self.cur_step += 1
    self.stock_price = self.stock_price_history[:, self.cur_step]    # update price
    self.stock_rsi = self.stock_rsi_history[:, self.cur_step] # update rsi
    self._trade(action)
    cur_val = self._get_val()
    reward = cur_val - prev_val
    # if reward < 0:
    #     reward = reward * 2
    # reward = reward + TRADE_BENEFIT
    done = self.cur_step == self.n_step - 1
    info = {'cur_val': cur_val}
    return self._get_obs(), reward, done, info


  def _get_obs(self):
    obs = []
    # obs.extend(self.stock_price)
    obs.extend(self.stock_owned)
    obs.extend(list(self.stock_rsi))
    obs.append(self.cash_in_hand)
    return obs


  def _get_val(self):
    return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand


  def _trade(self, action):
    # all combo to sell(0), hold(1), or buy(2) stocks
    action_combo = map(list, itertools.product([0, 1, 2], repeat=self.n_stock))
    action_vec = action_combo[action]

    # one pass to get sell/buy index
    sell_index = []
    buy_index = []
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)
    # print('{},'.format(self.stock_owned[0]),end='')
    # print('{},'.format(self.cash_in_hand),end='')
    print('{},'.format(self._get_val()),end='')  # net worth


    # TRADE_BENEFIT = -0.05
    # two passes: sell first, then buy; might be naive in real-world settings
    if sell_index:
      for i in sell_index:
        if self.stock_owned[i] > 0:
            self.cash_in_hand += self.stock_price[i]
            self.stock_owned[i] -= 1
            # TRADE_BENEFIT = 0.10
        
            # TRADE_BENEFIT = -200
    if buy_index:
      can_buy = True
      while can_buy:
        # for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1 # buy one share
            self.cash_in_hand -= self.stock_price[i]
            can_buy=False
            # TRADE_BENEFIT = 0.10
          else:
            # TTRADE_BENEFIT = -200
            can_buy = False

