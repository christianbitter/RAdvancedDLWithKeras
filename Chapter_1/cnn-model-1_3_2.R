# 
# A sample CNN network for classification
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/cnn-model-1.3.2.py


rm(list = ls());

library(tensorflow);
library(keras)

n_digits <- 10;
model <- keras::keras_model_sequential();

model %>%
  keras::layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu'
                       strides = 2, input_shape = c(28, 28, 1), padding='same') %>%
  keras::layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation='relu',
                       strides=2) %>%
  keras::layer_flatten() %>%
  keras::layer_dense(n_digits, activation='softmax');

model %>% 
  keras::compile(loss='categorical_crossentropy',
                 optimizer = keras::optimizer_rmsprop(),
                 metrics = list('accuracy'));

summary(model);
