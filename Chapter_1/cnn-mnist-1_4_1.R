# CNN MNIST digits classification
# 
# 3-layer CNN for MNIST digits classification 
# First 2 layers - Conv2D-ReLU-MaxPool
# 3rd layer - Conv2D-ReLU-Dropout
# 4th layer - Dense(10)
# Output Activation - softmax
# Optimizer - Adam
# 
# 99.4% test accuracy in 10epochs
# 
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/cnn-mnist-1.4.1.py
rm(list = ls());

library(tensorflow);
library(keras);

set.seed(1234);

# load mnist dataset
mnist <- keras::dataset_mnist();
x_train <- mnist$train$x;
y_train <- mnist$train$y;

x_test <- mnist$test$x;
y_test <- mnist$test$y;

# compute the number of labels
num_labels <- length(unique(y_train));

# convert to one-hot vector
y_train <- keras::to_categorical(y_train)
y_test <- keras::to_categorical(y_test)

# input image dimensions
image_size = dim(x_train)[2];

# resize and normalize
x_train <- keras::k_reshape(x_train, c(-1, image_size, image_size, 1));
x_test  <- keras::k_reshape(x_test, c(-1, image_size, image_size, 1));
x_train <- keras::k_cast(x_train, 'float32') / 255.0;
x_test  <- keras::k_cast(x_test, 'float32') / 255.0;

# network parameters
# image is processed as is (square grayscale)
input_shape <- c(image_size, image_size, 1);
batch_size  <- 128;
kernel_size <- 3;
pool_size   <- 2;
filters     <- 64;
dropout     <- 0.2;

# model is a stack of CNN-ReLU-MaxPooling
model <- keras::keras_model_sequential();
model %>%
  keras::layer_conv_2d(filters = filters, kernel_size = c(kernel_size, kernel_size),
                       activation = 'relu', input_shape = input_shape) %>%
  keras::layer_max_pooling_2d(pool_size) %>%
  keras::layer_conv_2d(filters = filters, kernel_size = kernel_size, activation = 'relu') %>%
  keras::layer_max_pooling_2d(pool_size) %>%
  keras::layer_conv_2d(filters = filters, kernel_size = kernel_size, activation = 'relu') %>%
  keras::layer_flatten() %>%
  # dropout added as regularizer
  keras::layer_dropout(dropout) %>%
  # output layer is 10-dim one-hot vector
  keras::layer_dense(num_labels) %>%
  keras::layer_activation('softmax');

summary(model);

# plot_model(model, to_file='cnn-mnist.png', show_shapes=True)

# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model %>% keras::compile(loss = 'categorical_crossentropy',
                         optimizer = 'adam',
                         metrics = c('accuracy'));
# train the network
model %>% keras::fit(x_train, y_train, epochs = 10, batch_size = batch_size);

hist <- model %>% keras::evaluate(x_test, y_test, batch_size = batch_size, verbose = 0)
cat(sprintf("\nTest accuracy: %.1f%%", 100.0 * hist$accuracy));
