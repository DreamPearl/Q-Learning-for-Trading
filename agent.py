from __future__ import print_function
from collections import deque
import random
import numpy as np
from model import mlp
import argparse

class DQNAgent(object):
  """ A simple Deep Q agent """
  def __init__(self, state_size, action_size,mode):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
    self.mode=mode
    self.gamma = 0.90  # discount rate
    self.epsilon = 1.0  # exploration rate #original one
    # self.epsilon = 0.5  # exploration rate
    self.epsilon_min = 0.10 # asdadasds
    # self.epsilon_decay = 0.2
    self.epsilon_decay = 0.998    #original one
    self.model = mlp(state_size, action_size)


  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
    #print(self.memory[-1])


  def act(self, state):
    # print(state)
    # exit()
    
    if np.random.rand() <= self.epsilon and self.mode=='train':
        print('Taking_random_move,',end='')
        # print('epsilon {},'.format(self.epsilon))
        # print(random.randrange(self.action_size))
        # exit()
        return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    print('Taking_smart__move,',end='')
    # print(np.argmax(act_values[0]))
    # exit()
    return np.argmax(act_values[0])  # returns action
   

  def replay(self,e,action,batch_size=32):
    """ vectorized implementation; 30x speed up compared with for loop """
    minibatch = random.sample(self.memory, batch_size)
    ########## CHANGE
    # minibatch = self.memory[-batch_size:]
    

    # print(minibatch)
    # print('')
    # print('Inside replay function:')

    states = np.array([tup[0][0] for tup in minibatch])
    actions = np.array([tup[1] for tup in minibatch])
    rewards = np.array([tup[2] for tup in minibatch])
    next_states = np.array([tup[3][0] for tup in minibatch])
    done = np.array([tup[4] for tup in minibatch])

    # Q(s', a)
    target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
    # end state target is reward itself (no lookahead)
    target[done] = rewards[done]
    # print(target,end='')
    # print('')
    
    # Q(s, a)
    target_f = self.model.predict(states)
    # print('target_f is {}'.format(target_f))
    # make the agent to approximately map the current state to future discounted reward
    target_f[range(batch_size), actions] = target
    # before_transform_state = scaler.inverse_transform(states)
    # print('replay function state is {},{},{},'.format(before_transform_state[0][0],before_transform_state[0][1],before_transform_state[0][2]))
    # if e==99:
    #   print('{},{},{},'.format(target_f[0][0],target_f[0][1],target_f[0][2]),end='')
    #   print('{}'.format(action))

    # if actions[0]==0:
    #   print('sell')
    # elif actions[0]==1:
    #   print('hold')
    # else:
    #   print('buy')
  
    # print('')
    history=self.model.fit(states, target_f,validation_split=0.2, epochs=1, verbose=0)
    # print(history.history)
    print('{},'.format(history.history['acc'][0]),end='')
    print('{},'.format(history.history['loss'][0]),end='')
    print('{},'.format(history.history['val_acc'][0]),end='')
    print(history.history['val_loss'][0],end='')
   

    # self.model.fit(states, target_f, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    return history


  def load(self, name):
    # i=1
    self.model.load_weights(name)
    # for layer in self.model.layers:
    #   print('')
    #   print('Printing layer {}'.format(i))
    #   i+=1
    #   print('')
    #   print(layer.get_weights())
    # exit()


  def save(self, name):
    self.model.save_weights(name)
