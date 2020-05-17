# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/autoencoder-2dim-mnist-3.2.2.py
# 
# Example of autoencoder model on MNIST dataset using 2dim latent
# 
# The autoencoder forces the encoder to discover 2-dim latent vector
# that the decoder can recover the original input. The 2-dim latent
# vector is projected on 2D space to analyze the distribution of code
# in the latent space. The latent space can be navigated by varying the
# values of latent vector to produce new MNIST digits.
# 
# This autoencoder has modular design. The encoder, decoder and autoencoder
# are 3 models that share weights. For example, after training the
# autoencoder, the encoder can be used to  generate latent vectors
# of input data for low-dim visualization like PCA or TSNE.
# .rs.restartR();

rm(list = ls());

library(keras);
library(deepviz);
library(ggplot2);
library(dplyr);

try(keras::k_cast(1, "float32"));

set.seed(1234);

#'@name plot_results
#'@description Plots 2-dim latent values as scatter plot of digits
#'then, plot MNIST digits as function of 2-dim latent vector
#'@param models (list): encoder and decoder models
#'@param data (list): test data and label
#'@param batch_size (int): prediction batch size
#'@param model_name (string): which model is using this function
plot_results <- function(models, data, batch_size=32, model_name="autoencoder_2dim") {
  encoder <- models[[1]];
  decoder <- models[[2]];
  x_test <- data[[1]];
  y_test <- data[[2]]
  xmin <- ymin <- -4;
  xmax <- ymax <- +4;
  
  if (!dir.exists(model_name)) {
    dir.create(path = model_name, showWarnings = F);
  }
  
  filename <- sprintf("%s/%s", model_name, "latent_2dim.png");
  # display a 2D plot of the digit classes in the latent space
  z <- encoder %>% predict(x_test, batch_size = batch_size);
  
  # subsample to reduce density of points on the plot
  no_samples <- 1e3;
  sample_ix <- sample(x = 1:nrow(z), size = no_samples, replace = F);
  z_sample <- z[sample_ix, ];
  y_sample <- y_test[sample_ix];
  
  p <- tibble("X1" = z_sample[, 1], "X2" = z_sample[, 2], "N" = y_sample) %>% 
    ggplot(aes(x = X1, y = X2)) + 
    geom_text(aes(label = N)) + 
    xlim(xmin, xmax) + ylim(ymin, ymax) + 
    labs(x = "x", y = "y") + 
    theme_light();
  
  ggsave(filename = filename, plot = p);
  
  filename <- sprintf("%s/%s", model_name, "digits_over_latent.png");
  # display a 30x30 2D manifold of the digits
  n <- 30
  digit_size <- 28;
  figure <- matrix(NA_real_, nrow = digit_size * n, ncol = digit_size * n);
  # linearly spaced coordinates corresponding to the 2D plot
  # of digit classes in the latent space
  grid_x <- seq(from = xmin, to = xmax, length.out = n);
  grid_y <- rev(seq(from = ymin, to = ymax, length.out = n));
  
  for (i in 1:length(grid_y)) {
    yi <- grid_y[i];
    for (j in 1:length(grid_x)) {
      xi <- grid_x[j];
      
      z <- matrix(c(xi, yi));
      z <- keras::k_reshape(z, c(1, 2));
      x_decoded <- decoder %>% predict(z)
      digit     <- x_decoded %>% keras::k_reshape(shape = c(digit_size, digit_size));
      digit     <- digit$numpy();
      
      # pay attention to the zero indexing of python
      i_digit_size <- (i - 1) * digit_size + 1;
      j_digit_size <- (j - 1) * digit_size + 1;
      figure[i_digit_size:(i_digit_size + digit_size - 1),
             j_digit_size:(j_digit_size + digit_size - 1)] <- digit;
    }
  }
  
  # now plot
  png(filename = filename);
  par(mar = c(0, 0, 0, 0));
  image(figure, xlab = "z0", ylab = "z1",
        useRaster = TRUE, axes = FALSE, col = grey(seq(0, 1, length = 256)));
  dev.off();  
}

# load MNIST dataset
mnist   <- keras::dataset_mnist();
x_train <- mnist$train$x;
y_train <- mnist$train$y;
x_test  <- mnist$test$x;
y_test  <- mnist$test$y;

# reshape to (28, 28, 1) and normalize input images
dim_img    <- dim(x_train);
image_size <- dim_img[2];
x_train <- keras::k_reshape(x_train, c(-1, dim_img[2], dim_img[3], 1));
x_test  <- keras::k_reshape(x_test, c(-1, dim_img[2], dim_img[3], 1));
x_train <- keras::k_cast(x_train, 'float32') / 255.
x_test  <- keras::k_cast(x_test, 'float32') / 255.

# network parameters
input_shape <- c(image_size, image_size, 1);
batch_size  <- 32;
kernel_size <- 3;
latent_dim  <- 2;
# encoder/decoder number of CNN layers and filters per layer
layer_filters <- c(32, 64);

# build the autoencoder model
# first build the encoder model
inputs <- keras::layer_input(shape = input_shape, name = 'encoder_input')
x      <- inputs;
# stack of Conv2D(32)-Conv2D(64)
for (filters in layer_filters) {
  x <- x %>% keras::layer_conv_2d(filters = filters, kernel_size = kernel_size,
                                  activation = 'relu', strides = 2, padding = 'same');
}
# shape info needed to build decoder model so we don't do hand computation
# the input to the decoder's first Conv2DTranspose will have this shape
# shape is (7, 7, 64) which is processed by the decoder back to (28, 28, 1)
shape <- keras::k_int_shape(x);

# generate latent vector
x      <- x %>% keras::layer_flatten();
latent <- x %>% keras::layer_dense(latent_dim, name = 'latent_vector');

# instantiate encoder model
encoder <- keras::keras_model(inputs, latent);
summary(encoder);

deepviz::plot_model(encoder);

# build the decoder model
latent_inputs <- keras::layer_input(shape = list(latent_dim), name = 'decoder_input');

# use the shape (7, 7, 64) that was earlier saved
x <- latent_inputs %>% keras::layer_dense(shape[[2]] * shape[[3]] * shape[[4]]);
# from vector to suitable shape for transposed conv
x <- x %>% keras::layer_reshape(list(shape[[2]], shape[[3]], shape[[4]]));

# stack of Conv2DTranspose(64)-Conv2DTranspose(32)
for (filters in rev(layer_filters)) {
  x <- x %>% keras::layer_conv_2d_transpose(filters = filters, kernel_size = kernel_size,
                                            activation = 'relu', strides = 2, padding = 'same');
}

# reconstruct the input
outputs = x %>% keras::layer_conv_2d_transpose(filters = 1, kernel_size = kernel_size,
                                               activation = 'sigmoid', padding = 'same', 
                                               name = 'decoder_output');

# instantiate decoder model
decoder <- keras::keras_model(latent_inputs, outputs);
summary(decoder)
deepviz::plot_model(decoder);

# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder <- keras::keras_model(inputs, decoder(encoder(inputs)));
summary(autoencoder);
deepviz::plot_model(autoencoder);

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder %>% compile(loss = 'mse', optimizer = 'adam');

# train the autoencoder
autoencoder %>% fit(x_train, x_train, validation_data = list(x_test, x_test),
                    epochs = 20, batch_size = batch_size);

# predict the autoencoder output from test data
x_decoded <- autoencoder %>% predict(x_test);

# display the 1st 8 test input and decoded images
png(filename = 'Chapter_3/input_and_decoded.png');
par(mar = c(0, 0, 0, 0));
par(mfrow = c(8, 2));
for (i in 1:8) {
  img_i_in  <- x_test[i, , , 1]$numpy()
  img_i_out <- x_decoded[i, , , ];
  image(img_i_in,  useRaster = TRUE, axes = FALSE, col = grey(seq(0, 1, length = 256)));
  image(img_i_out, useRaster = TRUE, axes = FALSE, col = grey(seq(0, 1, length = 256)));
}
dev.off();

# project the 2-dim latent on 2D space
models <- list(encoder, decoder);
data <- list(x_test, y_test);
model_name <- "autoencoder-2dim";
plot_results(models, data, batch_size = batch_size, model_name = "autoencoder-2dim");
