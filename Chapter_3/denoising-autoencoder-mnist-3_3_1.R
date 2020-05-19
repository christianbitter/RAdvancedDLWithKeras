# Trains a denoising autoencoder on MNIST dataset.
# 
# Denoising is one of the classic applications of autoencoders.
# The denoising process removes unwanted noise that corrupted the
# true data.
# 
# Noise + Data ---> Denoising Autoencoder ---> Data
# 
# Given a training dataset of corrupted data as input and
# true data as output, a denoising autoencoder can recover the
# hidden structure to generate clean data.
# 
# This example has modular design. The encoder, decoder and autoencoder
# are 3 models that share weights. For example, after training the
# autoencoder, the encoder can be used to  generate latent vectors
# of input data for low-dim visualization like PCA or TSNE.
# .rs.restartR();

rm(list = ls());

library(keras);
library(deepviz);

set.seed(1337)

try(keras::k_cast(1, "float32"));

# load MNIST dataset
mnist <- keras::dataset_mnist()
x_train <- mnist$train$x;
x_test  <- mnist$test$x;

# reshape to (28, 28, 1) and normalize input images
image_size = dim(x_train)[2];

x_train <- keras::k_reshape(x_train, c(-1, image_size, image_size, 1));
x_test  <- keras::k_reshape(x_test, c(-1, image_size, image_size, 1));
x_train <- keras::k_cast(x_train, 'float32') / 255.;
x_test  <- keras::k_cast(x_test, 'float32') / 255.;

# generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
noise         <- keras::k_random_normal(mean = 0.5, stddev = 0.5, dim(x_train));
x_train_noisy <- x_train + noise;
noise        <- keras::k_random_normal(mean = 0.5, stddev = 0.5, dim(x_test));
x_test_noisy <- x_test + noise;

# adding noise may exceed normalized pixel values>1.0 or <0.0
# clip pixel values >1.0 to 1.0 and <0.0 to 0.0
x_train_noisy <- keras::k_clip(x_train_noisy, 0., 1.);
x_test_noisy  <- keras::k_clip(x_test_noisy, 0., 1.);

# network parameters
input_shape <- c(image_size, image_size, 1);
batch_size <- 32;
kernel_size <- 3;
latent_dim <- 16;

# encoder/decoder number of CNN layers and filters per layer
layer_filters <- c(32, 64);

# build the autoencoder model
# first build the encoder model
inputs <- keras::layer_input(shape = input_shape, name = 'encoder_input');
x      <- inputs;

# stack of Conv2D(32)-Conv2D(64)
for (filters in layer_filters) {
  x <- x %>% keras::layer_conv_2d(filters = filters, kernel_size = kernel_size,
                                  strides = 2, activation = 'relu',
                                  padding = 'same');
}
# shape info needed to build decoder model so we don't do hand computation
# the input to the decoder's first Conv2DTranspose will have this shape
# shape is (7, 7, 64) which can be processed by the decoder back to (28, 28, 1)
shape <- keras::k_int_shape(x);

# generate the latent vector
x      <- x %>% keras::layer_flatten();
latent <- x %>% keras::layer_dense(latent_dim, name = 'latent_vector');

# instantiate encoder model
encoder <- keras::keras_model(inputs, latent);
summary(encoder);

# build the decoder model
latent_inputs = keras::layer_input(shape = list(latent_dim), name = 'decoder_input');
# use the shape (7, 7, 64) that was earlier saved
x <- latent_inputs %>% keras::layer_dense(shape[[2]] * shape[[3]] * shape[[4]]);
# from vector to suitable shape for transposed conv
x <- x %>% keras::layer_reshape(list(shape[[2]], shape[[3]], shape[[4]]));

# stack of Conv2DTranspose(64)-Conv2DTranspose(32)
for (filters in rev(layer_filters)) {
  # upsampling
  x <- x %>% keras::layer_conv_2d_transpose(filters = filters, kernel_size = kernel_size,
                                            activation = 'relu', strides = 2, padding = 'same');
}

# reconstruct the input
outputs <- x %>% keras::layer_conv_2d_transpose(filters = 1, kernel_size = kernel_size,
                                                activation = 'sigmoid', padding = 'same',
                                                name = 'decoder_output');

# instantiate decoder model
decoder <- keras::keras_model(latent_inputs, outputs);
summary(decoder);

# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder <- keras::keras_model(inputs, decoder(encoder(inputs)));
summary(autoencoder);

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder %>% compile(loss = 'mse', optimizer = 'adam');

# train the autoencoder
autoencoder %>% fit(x_train_noisy, x_train, validation_data = list(x_test_noisy, x_test),
                    epochs = 10, batch_size = batch_size);

# predict the autoencoder output from corrupted test images
x_decoded <- autoencoder %>% predict(x_test_noisy);

# instead of the 9 images we do like last time, we look at 8 images
# where the left side shows the corruped image and the right hand side shows the corrected image
png(filename = 'Chapter_3/corrupted_and_denoised.png');
par(mar = c(0, 0, 0, 0));
par(mfrow = c(8, 2));
for (i in 1:8) {
  img_i_in  <- x_test_noisy[i, , , 1]$numpy()
  img_i_out <- x_decoded[i, , , ];
  image(img_i_in,  useRaster = TRUE, axes = FALSE, col = grey(seq(0, 1, length = 256)));
  image(img_i_out, useRaster = TRUE, axes = FALSE, col = grey(seq(0, 1, length = 256)));
}
dev.off();
