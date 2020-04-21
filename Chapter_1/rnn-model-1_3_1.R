# A Simple RNN model with 30 x 12 input and 5-dim one-hot vector
# https://raw.githubusercontent.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/master/chapter1-keras-quick-tour/rnn-model-1.3.1.py
rm(list = ls());

# keras modules
library(tensorflow);
library(keras);

timesteps <- 30
input_dim <- 12
# number of units in RNN cell
units <- 512
# number of classes to be identified
n_activities <- 5
model <- keras::keras_model_sequential();
# RNN with dropout
model %>% 
  keras::layer_simple_rnn(units = units, dropout = 0.2, input_shape = c(timesteps, input_dim)) %>%
  keras::layer_dense(units = n_activities, activation = 'softmax')

model %>% keras::compile(loss = "categorical_crossentropy",
                         optimizer = keras::optimizer_adam(),
                         metrics = c('accuracy'));
summary(model)
