# Colorization autoencoder
# 
# The autoencoder is trained with grayscale images as input
# and colored images as output.
# Colorization autoencoder can be treated like the opposite
# of denoising autoencoder. Instead of removing noise, colorization
# adds noise (color) to the grayscale image.
# 
# Grayscale Images --> Colorization --> Color Images
# .rs.restartR();
library(keras);

try(keras::k_cast(1, "float32"))

set.seed(1234);

#'@name
#'@description Convert from color image (RGB) to grayscale.
#'Source: opencv.org
#'grayscale = 0.299*red + 0.587*green + 0.114*blue
#'@param rgb (tensor): rgb image
#'@return (tensor): grayscale image
rgb2gray <-function(rgb) {
  m <- keras::k_reshape(matrix(c(0.299, 0.587, 0.114), ncol = 3), list(3, 1));
  return(keras::k_dot(rgb, m));
}

# load the CIFAR10 data
# load MNIST dataset
cifar10   <- keras::dataset_cifar10();
x_train <- cifar10$train$x;
x_test  <- cifar10$test$x;

# input image dimensions
# we assume data format "channels_last"
img_rows <- dim(x_train)[2]
img_cols <- dim(x_train)[3]
channels <- dim(x_train)[4]

# create saved_images folder
imgs_dir <- 'saved_images'
save_dir <- sprintf("%s/%s", getwd(), imgs_dir)
if (!dir.exists(save_dir)) dir.create(save_dir);

# display the 1st n input images (color and gray)
n    <- 100;
imgs <- x_test[1:n, , , ];
png(filename = sprintf('%s/test_color.png', imgs_dir));
par(mar = c(0, 0, 0, 0));
par(mfrow = c(10, 10));
for (i in 1:100) {
  img_j <- raster::as.raster(imgs[i, , , ] / 255.);
  raster::plot(img_j, interpolate=FALSE)
}
dev.off();

# convert color train and test images to gray
# note that the dot function requires tensors, and values scaled to 0..1
x_train_gray <- rgb2gray(keras::k_reshape(x_train / 255., dim(x_train)));
x_test_gray  <- rgb2gray(keras::k_reshape(x_test / 255., dim(x_test)));
# dim(x_train_gray)

# display grayscale version of test images
n    <- 100;
imgs <- x_test_gray$numpy()[1:n, , , ];
png(filename = sprintf('%s/test_gray.png', imgs_dir));
par(mar = c(0, 0, 0, 0));
par(mfrow = c(10, 10));
for (i in 1:100) {
  img_j <- raster::as.raster(imgs[i, , ]);
  raster::plot(img_j, interpolate=FALSE)
}
dev.off();

# note: we should move this actually further up, since the col2rgb benefits from it
# but for the sake of code-symetry, we will not do this here
# normalize output train and test color images
x_train <- keras::k_cast(x_train, 'float32') / 255.;
x_test  <- keras::k_cast(x_test, 'float32') / 255.;

# note: there is no need to do this, since the train gray is already float
# for for code symetry we leave this in
# normalize input train and test grayscale images
x_train_gray <- keras::k_cast(x_train_gray, 'float32') / 255.;
x_test_gray <- keras::k_cast(x_test_gray, 'float32') / 255.;

# note: there is no need to do this
# reshape images to row x col x channel for CNN output/validation
x_train <- keras::k_reshape(x_train, c(dim(x_train)[1], img_rows, img_cols, channels));
x_test <- keras::k_reshape(x_test, c(dim(x_test)[1], img_rows, img_cols, channels));

# note: there is no need to do this
# reshape images to row x col x channel for CNN input
x_train_gray <- keras::k_reshape(x_train_gray, c(dim(x_train_gray)[1], img_rows, img_cols, 1));
x_test_gray <- keras::k_reshape(x_test_gray, c(dim(x_test_gray)[1], img_rows, img_cols, 1));

# network parameters
input_shape <- c(img_rows, img_cols, 1);
batch_size  <- 32;
kernel_size <- 3;
latent_dim  <- 256;
# encoder/decoder number of CNN layers and filters per layer
layer_filters <- c(64, 128, 256);

# build the autoencoder model
# first build the encoder model
inputs <- keras::layer_input(shape = input_shape, name='encoder_input')
x <- inputs
# stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
for (filters in layer_filters) {
  x <- x %>% keras::layer_conv_2d(filters = filters, kernel_size = kernel_size,
                                  strides = 2, activation = 'relu', padding = 'same');
}
# shape info needed to build decoder model so we don't do hand computation
# the input to the decoder's first Conv2DTranspose will have this shape
# shape is (4, 4, 256) which is processed by the decoder back to (32, 32, 3)
shape <- keras::k_int_shape(x);

# generate a latent vector
x <- x %>% keras::layer_flatten();
latent <- x %>% keras::layer_dense(latent_dim, name = 'latent_vector');

# instantiate encoder model
encoder <- keras::keras_model(inputs, latent);
summary(encoder);

# build the decoder model
latent_inputs <- keras::layer_input(shape = c(latent_dim), name = 'decoder_input');
x <- latent_inputs %>% keras::layer_dense((shape[[2]]*shape[[3]]*shape[[4]]));
x <- x %>% keras::layer_reshape(list(shape[[2]], shape[[3]], shape[[4]]));

# stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
for (filters in rev(layer_filters)) {
  x <- x %>% keras::layer_conv_2d_transpose(filters = filters, kernel_size = kernel_size,
                                            strides = 2, activation = 'relu', padding = 'same');
}

outputs <- x %>% keras::layer_conv_2d_transpose(filters = channels, kernel_size = kernel_size,
                                                activation = 'sigmoid', padding = 'same',
                                                name = 'decoder_output');

# instantiate decoder model
decoder <- keras::keras_model(latent_inputs, outputs);
summary(decoder);

# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder <- keras::keras_model(inputs, decoder(encoder(inputs)));
summary(autoencoder);

# prepare model saving directory.
save_dir <- sprintf("%s/%s", getwd(), 'saved_models');
model_name <- 'colorized_ae_model.{epoch:03d}.h5'
if (!dir.exists(save_dir)) dir.create(save_dir);

filepath <- sprintf("%s/%s", save_dir, model_name);

# reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs
lr_reducer <- keras::callback_reduce_lr_on_plateau(factor = sqrt(0.1), cooldown = 0,
                                                   patience = 5, verbose = 1, min_lr = 0.5e-6)

# save weights for future use (e.g. reload parameters w/o training)
checkpoint <- keras::callback_model_checkpoint(filepath = filepath, monitor = 'val_loss', 
                                               verbose = 1, save_best_only = T);

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder %>% compile(loss = 'mse', optimizer = 'adam');

# called every epoch
callbacks <- list(lr_reducer, checkpoint);

# train the autoencoder
autoencoder %>% fit(x_train_gray, x_train,
                    validation_data = list(x_test_gray, x_test),
                    epochs = 30, batch_size = batch_size, 
                    callbacks = callbacks);

# predict the autoencoder output from test data
x_decoded <- autoencoder %>% predict(x_test_gray);

# display the 1st 100 colorized images
n    <- 100;
imgs <- x_decoded[1:n, , , ];
png(filename = sprintf('%s/colorized.png', imgs_dir));
par(mar = c(0, 0, 0, 0));
par(mfrow = c(10, 10));
for (i in 1:100) {
  img_j <- raster::as.raster(imgs[i, , , ]);
  raster::plot(img_j, interpolate=FALSE)
}
dev.off();
