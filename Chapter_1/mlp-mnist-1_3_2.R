rm(list = ls());
# '''
# A MLP network for MNIST digits classification
# 
# 98.3% test accuracy in 20epochs
# 
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
# '''

library(keras);
library(tensorflow);

set.seed(1234);

# load mnist dataset
mnist <- keras::dataset_mnist()
x_train <- mnist$train$x;
y_train <- mnist$train$y;
x_test <- mnist$test$x;
y_test <- mnist$test$y;
# compute the number of labels

num_labels <- length(unique(y_train))

# convert to one-hot vector
y_train = keras::to_categorical(y_train)
y_test = keras::to_categorical(y_test)

# image dimensions (assumed square)
image_size <- dim(x_train)[2];
input_size <- image_size * image_size;

# resize and normalize
x_train <- keras::k_reshape(x_train, c(-1, input_size))
x_train <- keras::k_cast_to_floatx(x_train) / 255.0;
x_test <- keras::k_reshape(x_test, c(-1, input_size))
x_test <- keras::k_cast_to_floatx(x_test) / 255.0;

# network parameters
batch_size <- 128
hidden_units <- 256
dropout <- 0.45

# model is a 3-layer MLP with ReLU and dropout after each layer
model <- keras::keras_model_sequential();
model %>%
  keras::layer_dense(units = hidden_units, input_shape = input_size) %>%
  keras::layer_activation(activation = "relu") %>% 
  keras::layer_dropout(dropout) %>%
  keras::layer_dense(units = hidden_units) %>%
  keras::layer_activation(activation = "relu") %>% 
  keras::layer_dropout(dropout) %>%
  keras::layer_dense(units = num_labels) %>%
  keras::layer_activation(activation = "softmax");
  
# this is the output for one-hot vector
summary(model)
# plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model %>% keras::compile(loss = 'categorical_crossentropy',
                         optimizer = 'adam',
                         metrics = c('accuracy'));
# train the network
model %>% keras::fit(x_train, y_train, epochs = 20, batch_size = batch_size);

# validate the model on test dataset to determine generalization
eval <- model %>% keras::evaluate(x_test, y_test, batch_size = batch_size, verbose = 0);
cat(sprintf("\nTest accuracy: %.1f%%", (100.0 * eval$accuracy)));
cat(sprintf("\nTest Loss: %.1f%%", (100.0 * eval$loss)));
