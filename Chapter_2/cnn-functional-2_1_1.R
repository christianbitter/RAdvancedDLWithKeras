# Using Functional API to build CNN
# ~99.3% test accuracy
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/cnn-functional-2.1.1.py
# https://tensorflow.rstudio.com/guide/keras/functional_api/
rm(list = ls());

library(keras);

set.seed(1234);

# load MNIST dataset
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
filters     <- 64;
dropout     <- 0.3

# use functional API to build cnn layers
inputs <- keras::layer_input(shape = input_shape);
y <- inputs %>% keras::layer_conv_2d(filters = filters, kernel_size = kernel_size, activation = 'relu');
y <- y %>% keras::layer_max_pooling_2d();
y <- y %>% keras::layer_conv_2d(filters = filters, kernel_size = kernel_size, activation = 'relu');
y <- y %>% keras::layer_max_pooling_2d();
y <- y %>% keras::layer_conv_2d(filters = filters, kernel_size = kernel_size, activation = 'relu');

# image to vector before connecting to dense layer
y <- y %>% keras::layer_flatten();

# dropout regularization
y <- y %>% keras::layer_dropout(dropout);
outputs <- y %>% keras::layer_dense(num_labels, activation = 'softmax');

# build the model by supplying inputs/outputs
model <- keras::keras_model(inputs = inputs, outputs = outputs);
# network model in text
summary(model);

# classifier loss, Adam optimizer, classifier accuracy
model %>% compile(loss = 'categorical_crossentropy', 
                  optimizer = keras::optimizer_adam(),
                  metrics = c('accuracy'));

# train the model with input images and labels
model %>% keras::fit(x_train, y_train, 
                     validation_data = list(x_test, y_test),
                     epochs = 20, batch_size = batch_size);

# model accuracy on test dataset
score <- model %>% keras::evaluate(x_test, y_test, batch_size = batch_size, verbose = 0);

cat(sprintf("\nTest accuracy: %.1f%%", 100.0 * score$accuracy));