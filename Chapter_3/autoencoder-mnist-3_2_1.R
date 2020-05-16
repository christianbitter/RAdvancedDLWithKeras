# Example of autoencoder model on MNIST dataset
# This autoencoder has modular design. The encoder, decoder and autoencoder
# are 3 models that share weights. For example, after training the
# autoencoder, the encoder can be used to  generate latent vectors
# of input data for low-dim visualization like PCA or TSNE.

# .rs.restartR();

rm(list = ls());

# library("devtools")
# devtools::install_github("andrie/deepviz")

library(keras);
library(deepviz);

try(keras::k_cast(1, "float32"));

# load MNIST dataset
mnist   <- keras::dataset_mnist();
x_train <- mnist$train$x;
x_test  <- mnist$test$x;

# reshape to (28, 28, 1) and normalize input images
dim_img    <- dim(x_train);
image_size <- dim_img[2];
x_train <- keras::k_reshape(x_train, c(-1, dim_img[2], dim_img[3], 1));
x_test  <- keras::k_reshape(x_test, c(-1, dim_img[2], dim_img[3], 1));
x_train <- keras::k_cast(x_train, 'float32') / 255.
x_test  <- keras::k_cast(x_test, 'float32') / 255.

# network parameters
input_shape <- c(image_size, image_size, 1);
batch_size <- 32
kernel_size <- 3
latent_dim <- 16

# encoder/decoder number of CNN layers and filters per layer
layer_filters <- c(32, 64);

# build the autoencoder model
# first build the encoder model
inputs <- keras::layer_input(shape = input_shape, name = 'encoder_input');
x      <- inputs;
# stack of Conv2D(32)-Conv2D(64)
for (filters in layer_filters) {
  x <- x %>% keras::layer_conv_2d(filters = filters, kernel_size = kernel_size,
                                  activation = 'relu', strides = 2, padding = 'same');
}
# shape info needed to build decoder model
# so we don't do hand computation
# the input to the decoder's first
# Conv2DTranspose will have this shape
# shape is (7, 7, 64) which is processed by
# the decoder back to (28, 28, 1)
shape <- keras::k_int_shape(x);

# generate latent vector
x <- x %>% keras::layer_flatten();
latent <- x %>% keras::layer_dense(latent_dim, name='latent_vector'); # this is the embedding

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
plot_model(decoder, to_file='decoder.png', show_shapes=True)

# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder <- keras::keras_model(inputs, decoder(encoder(inputs)));
summary(autoencoder);

encoder %>% deepviz::plot_model();
decoder %>% deepviz::plot_model();
autoencoder %>% deepviz::plot_model();

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder %>% keras::compile(loss = 'mse', optimizer = 'adam');

# train the autoencoder
autoencoder %>% fit(x_train, x_train, validation_data = list(x_test, x_test), 
                    epochs = 1, batch_size = batch_size);

# predict the autoencoder output from test data
x_decoded <- autoencoder %>% predict(x_test);

png(filename = 'input_and_decoded.png');
par(mar = c(0, 0, 0, 0));
par(mfrow = c(8, 2));
for (i in 1:8) {
  img_i_in  <- x_test[i, , , 1]$numpy()
  img_i_out <- x_decoded[i, , , ];
  image(img_i_in,  useRaster = TRUE, axes = FALSE, col = grey(seq(0, 1, length = 256)));
  image(img_i_out, useRaster = TRUE, axes = FALSE, col = grey(seq(0, 1, length = 256)));
}
dev.off();