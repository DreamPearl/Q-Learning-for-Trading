from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


def mlp(n_obs, n_action, n_hidden_layer=4, n_neuron_per_layer=100,
        activation='relu', loss='mse'):
  """ A multi-layer perceptron """
  model = Sequential()
  model.add(Dense(200, input_dim=n_obs, activation=activation))
  model.add(BatchNormalization())
  model.add(Dropout(0.15))
  for _ in range(n_hidden_layer):
    model.add(Dense(n_neuron_per_layer, activation=activation))
  model.add(Dense(n_action, activation='softmax'))
  model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])
  print(model.summary())
  return model
