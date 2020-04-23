# Implements a Y-Network using Functional API
# 
# ~99.3% test accuracy
# https://sacmehta.github.io/YNet/
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/cnn-y-network-2.1.2.py

rm(list = ls());

library(keras);

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
dropout     <- 0.4
n_filters   <- 32

# left branch of Y network
left_inputs <- keras::layer_input(shape = input_shape);
x           <- left_inputs;
filters     <- n_filters;
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for (i in seq(1, 3)) {
  x <- x %>% keras::layer_conv_2d(filters = filters, kernel_size = kernel_size,
                                  padding = 'same', activation = 'relu');
  x <- x %>% keras::layer_dropout(dropout);
  x <- x %>% keras::layer_max_pooling_2d();
  filters <- filters * 2;
}

# right branch of Y network
right_inputs <- keras::layer_input(shape = input_shape);
y            <- right_inputs;
filters      <- n_filters;
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for (i in seq(1, 3)) {
  y <- y %>% keras::layer_conv_2d(filters = filters, kernel_size = kernel_size, padding = 'same', 
                            activation = 'relu', dilation_rate = 2);
  y <- y %>% keras::layer_dropout(dropout);
  y <- y %>% keras::layer_max_pooling_2d();
  filters <- filters * 2;
}

# merge left and right branches outputs
y <- keras::layer_concatenate(list(x, y));
# feature maps to vector before connecting to Dense 
y <- y %>% keras::layer_flatten();
y <- y %>% keras::layer_dropout(dropout);

outputs <- y %>% keras::layer_dense(num_labels, activation='softmax');

# build the model in functional API
model <- keras::keras_model(list(left_inputs, right_inputs), outputs);

# verify the model using graph
# plot_model(model, to_file='cnn-y-network.png', show_shapes=True)
# verify the model using layer text description
summary(model);

# classifier loss, Adam optimizer, classifier accuracy
model %>% keras::compile(loss = 'categorical_crossentropy',
                         optimizer = 'adam',
                         metrics = list('accuracy'));

# train the model with input images and labels
model %>% keras::fit(list(x_train, x_train), y_train,
                     #TODO: validation_data=([x_test, x_test], y_test),
                     epochs = 20, batch_size = batch_size);

# model accuracy on test dataset
score <- model %>% keras::evaluate(list(x_test, x_test), y_test,
                                   batch_size = batch_size, verbose = 0)
cat(sprintf("\nTest accuracy: %.1f%%", 100.0 * score$accuracy));
