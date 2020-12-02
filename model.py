from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
# from keras.layers import LSTM



def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
  """ A multi-layer perceptron """
  model = Sequential()
  # model.add(LSTM(60, input_dim=n_obs))
  model.add(Dense(60, input_dim=n_obs, activation=activation))
  model.add(BatchNormalization())
  for _ in range(n_hidden_layer):
    model.add(Dense(n_neuron_per_layer, activation=activation))
  model.add(Dense(n_action, activation='softmax'))
  model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])
  print(model.summary())
  return model