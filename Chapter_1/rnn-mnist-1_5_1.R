# RNN for MNIST digits classification
# 98.3% test accuracy in 20epochs
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/rnn-mnist-1.5.1.py
rm(list = ls());

library(tensorflow);
library(keras);

# load mnist dataset
mnist <- keras::dataset_mnist();
x_train <- mnist$train$x;
y_train <- mnist$train$y;

x_test <- mnist$test$x;
y_test <- mnist$test$y;

# compute the number of labels
num_labels <- length(unique(y_train));

# convert to one-hot vector
y_train <- keras::to_categorical(y_train);
y_test <- keras::to_categorical(y_test);

# resize and normalize
image_size <- dim(x_train)[2];
x_train <- keras::k_reshape(x_train, c(-1, image_size, image_size));
x_test <- keras::k_reshape(x_test, c(-1, image_size, image_size));
x_train <- keras::k_cast_to_floatx(x_train) / 255.0;
x_test <- keras::k_cast_to_floatx(x_test) / 255.0;

# network parameters
input_shape <- c(image_size, image_size)
batch_size  <- 128
units       <- 256
dropout     <- 0.2

# model is RNN with 256 units, input is 28-dim vector 28 timesteps
model <- keras::keras_model_sequential();
model %>% 
  keras::layer_simple_rnn(units=units, dropout=dropout, input_shape=input_shape) %>%
  keras::layer_dense(num_labels) %>%
  keras::layer_activation('softmax')

summary(model);
# plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

# loss function for one-hot vector
# use of sgd optimizer
# accuracy is good metric for classification tasks
model %>% compile(loss = 'categorical_crossentropy',
                  optimizer = 'sgd',
                  metrics = c('accuracy'));

# train the network
model %>% fit(x_train, y_train, epochs = 20, batch_size = batch_size);

hist <- model %>% evaluate(x_test, y_test, batch_size = batch_size, verbose = 0);
cat(sprintf("\nTest accuracy: %.1f%%", 100.0 * hist$accuracy));
