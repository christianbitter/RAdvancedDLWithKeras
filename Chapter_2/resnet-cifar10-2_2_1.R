# Trains a ResNet on the CIFAR10 dataset.
# 
# ResNet v1
# [a] Deep Residual Learning for Image Recognition
# https://arxiv.org/pdf/1512.03385.pdf
# 
# ResNet v2
# [b] Identity Mappings in Deep Residual Networks
# https://arxiv.org/pdf/1603.05027.pdf
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/resnet-cifar10-2.2.1.py

rm(list = ls());

library(keras);

set.seed(1234);

# this may be needed if you receive an error such as symbol not found when invoking a keras function.
# try(keras::k_cast(1, "float32"));

# training parameters
batch_size <- 32; # orig paper trained all networks with batch_size=128
epochs     <- 200;
data_augmentation <- T;
num_classes <- 10;

# subtracting pixel mean improves accuracy
subtract_pixel_mean <- T;

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n <- 3;

# model version
# orig paper: version = 1 (ResNet v1), 
# improved ResNet: version = 2 (ResNet v2)
version <- 2;

# computed depth from supplied model parameter n
if (version == 1) {
  depth <- n * 6 + 2;
} else if (version == 2) {
  depth <- n * 9 + 2;
} else {
  stop("Version has to be either 1 or 2");
}

# model name, depth and version
model_type <- sprintf('ResNet%dv%d', depth, version);

# load the CIFAR10 data.
cifar10 <- keras::dataset_cifar10();
x_train <- cifar10$train$x;
y_train <- cifar10$train$y;
x_test  <- cifar10$test$x;
y_test  <- cifar10$test$y;

# input image dimensions.
input_shape <- dim(x_train)[-1];

# normalize data.
x_train <- keras::k_cast(x_train, 'float32') / 255.;
x_test  <- keras::k_cast(x_test, 'float32') / 255.;

# if subtract pixel mean is enabled
if (subtract_pixel_mean) {
  x_train_mean <- keras::k_mean(x_train, axis = list(1));
}

x_train <- x_train - x_train_mean;
x_test  <- x_test - x_train_mean;

cat('x_train shape:', dim(x_train), "\r\n");
cat(dim(x_train)[1], 'train samples\r\n');
cat(dim(x_test)[1], 'test samples\r\n');
cat('y_train shape:', dim(y_train), "\r\n");

# convert class vectors to binary class matrices.
y_train <- keras::to_categorical(y_train, num_classes);
y_test  <- keras::to_categorical(y_test, num_classes);

#'@name lr_schedule
#'@description
#'Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
#'Called automatically every epoch as part of callbacks during training.
#'@param epoch (int): The number of epochs
#'@return lr (float32): learning rate
lr_schedule <- function(epoch, current_lr) {
  lr <- 1e-3
  if (epoch > 180) {
    lr <- lr * 0.5e-3;
  } else if (epoch > 160) {
    lr <- lr * 1e-3;
  } else if (epoch > 120) {
    lr <- lr * 1e-2;
  } else if (epoch > 80) {
    lr <- lr * 1e-1;
  } else {
    lr <- 1e-3;
  }
  cat('Learning rate: ', lr, "\r\n");
  return(lr);
}

#'@name resnet_layer
#'@description 2D Convolution-Batch Normalization-Activation stack builder
#'@param inputs (tensor): input tensor from input image or previous layer
#'@param num_filters (int): Conv2D number of filters
#'@param kernel_size (int): Conv2D square kernel dimensions
#'@param strides (int): Conv2D square stride dimensions
#'@param activation (string): activation name
#'@param batch_normalization (bool): whether to include batch normalization
#'@param conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)
#'@eturns x (tensor): tensor as input to the next layer
resnet_layer <- function(inputs, num_filters = 16, kernel_size = 3,
                         strides = 1, activation = 'relu', 
                         batch_normalization = T, conv_first = T) {

  conv <- keras::layer_conv_2d(filters = num_filters, kernel_size = kernel_size, strides = strides,
                               padding = 'same', kernel_initializer = 'he_normal',
                               kernel_regularizer = keras::regularizer_l2(1e-4));

  x <- inputs;
  if (conv_first) {
    x <- x %>% conv();
  
    if (batch_normalization) x <- x %>% keras::layer_batch_normalization();
    if (!is.na(activation)) x <- x %>% keras::layer_activation(activation);
  } else {
    if (batch_normalization) x <- x %>% keras::layer_batch_normalization();
    if (!is.na(activation)) x <- x %>% keras::layer_activation(activation);
    
    x <- x %>% conv();
  }
  
  return(x);
}

#'@name resnet_v1
#'@description
#'ResNet Version 1 Model builder [a]
#'Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
#'Last ReLU is after the shortcut connection.
#'At the beginning of each stage, the feature map size is halved
#'(downsampled) by a convolutional layer with strides=2, while 
#'the number of filters is doubled. Within each stage, 
#'the layers have the same number filters and the
#'same number of filters.
#'Features maps sizes:
#'stage 0: 32x32, 16
#'stage 1: 16x16, 32
#'stage 2:  8x8,  64
#'The Number of parameters is approx the same as Table 6 of [a]:
#'ResNet20 0.27M
#'ResNet32 0.46M
#'ResNet44 0.66M
#'ResNet56 0.85M
#'ResNet110 1.7M
#'@param input_shape (tensor): shape of input image tensor
#'@param depth (int): number of core convolutional layers
#'@param num_classes (int): number of classes (CIFAR10 has 10)
#'@return model (Model): Keras model instance
resnet_v1 <- function(input_shape, depth, num_classes = 10) {
  if ((depth - 2) %% 6 != 0) {
    stop('depth should be 6n+2 (eg 20, 32, in [a])');
  }
  
  # start model definition.
  num_filters    <- 16;
  num_res_blocks <- (depth - 2) / 6;
  
  inputs <- keras::layer_input(shape = input_shape);
  x      <- resnet_layer(inputs = inputs);
  
  # instantiate the stack of residual units
  for (stack in seq(0, 2)) {
    for (res_block in seq(0, num_res_blocks - 1)) {
      strides <- 1;
      # first layer but not first stack
      if (stack > 0 && res_block == 0) {  
        strides <- 2  # downsample
      }
      
      y <- resnet_layer(inputs = x, num_filters = num_filters, strides = strides);
      y <- resnet_layer(inputs = y, num_filters = num_filters, activation = NA);
      
      # first layer but not first stack
      if (stack > 0 && res_block == 0) {
        # linear projection residual shortcut
        # connection to match changed dims
        x <- resnet_layer(inputs = x, num_filters = num_filters, kernel_size = 1,
                          strides = strides, activation = NA, batch_normalization = F);
      }
      
      x <- keras::layer_add(list(x, y));
      x <- x %>% keras::layer_activation('relu');
    }
    num_filters <- num_filters * 2;
  }
  # add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x <- x %>% keras::layer_average_pooling_2d(pool_size = 8);
  y <- x %>% keras::layer_flatten();
  
  outputs <- y %>% keras::layer_dense(units = num_classes, activation = 'softmax', 
                                      kernel_initializer = 'he_normal');
  
  # instantiate model.
  model <- keras::keras_model(inputs = inputs, outputs = outputs);
  return(model);
}

#'@name resnet_v2
#'@param input_shape (tensor): shape of input image tensor
#'@param depth (int): number of core convolutional layers
#'@param num_classes (int): number of classes (CIFAR10 has 10)
#'@Return model (Model): Keras model instance
#'@description
#'ResNet Version 2 Model builder [b]
#'
#'Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or 
#'also known as bottleneck layer.
#'First shortcut connection per layer is 1 x 1 Conv2D.
#'Second and onwards shortcut connection is identity.
#'At the beginning of each stage, 
#'the feature map size is halved (downsampled)
#'by a convolutional layer with strides=2, 
#'while the number of filter maps is
#'doubled. Within each stage, the layers have 
#'the same number filters and the same filter map sizes.
#'Features maps sizes:
#'conv1  : 32x32,  16
#'stage 0: 32x32,  64
#'stage 1: 16x16, 128
#'stage 2:  8x8,  256
resnet_v2 <- function(input_shape, depth, num_classes = 10) {
  if ((depth - 2) %% 9 != 0) stop('depth should be 9n+2 (eg 110 in [b])');
  
  # start model definition.
  num_filters_in <- 16;
  num_res_blocks <- (depth - 2) / 9;
  
  inputs = keras::layer_input(shape = input_shape);
  
  # v2 performs Conv2D with BN-ReLU
  # on input before splitting into 2 paths
  x <- resnet_layer(inputs = inputs, num_filters = num_filters_in, conv_first = T);
  
  # instantiate the stack of residual units
  for (stage in seq(0, 2)) {
    for (res_block in seq(0, num_res_blocks - 1)) {
      activation          <- 'relu';
      batch_normalization <- T;
      strides             <- 1;
      if (stage == 0) {
        num_filters_out <- num_filters_in * 4;
        # first layer and first stage
        if (res_block == 0) activation <- NA;
        batch_normalization <- F;
      } else {
        num_filters_out <- num_filters_in * 2;
        # first layer but not first stage
        if (res_block == 0) strides <- 2; # downsample
      }
  
      # bottleneck residual unit
      y <- resnet_layer(inputs = x, num_filters = num_filters_in, kernel_size = 1, 
                        strides = strides, activation = activation, 
                        batch_normalization = batch_normalization, conv_first = F);
      y <- resnet_layer(inputs = y, num_filters = num_filters_in, conv_first = F)
      y <- resnet_layer(inputs = y, num_filters = num_filters_out, kernel_size = 1, conv_first = F);
      
      if (res_block == 0) {
        # linear projection residual shortcut connection to match changed dims
        x <- resnet_layer(inputs = x, num_filters = num_filters_out, kernel_size = 1,
                          strides = strides, activation = NA, batch_normalization = F);
      }
      
      x <- keras::layer_add(list(x, y));
    }
    
    num_filters_in <- num_filters_out;
  }
  
  # add classifier on top -  v2 has BN-ReLU before Pooling
  x <- x %>% keras::layer_batch_normalization();
  x <- x %>% keras::layer_activation('relu');
  x <- x %>% keras::layer_average_pooling_2d(pool_size = 8);
  y <- x %>% keras::layer_flatten();
  
  outputs <- y %>% keras::layer_dense(units = num_classes, activation = 'softmax',
                                      kernel_initializer = 'he_normal');
  
  # instantiate model.
  model <- keras::keras_model(inputs = inputs, outputs = outputs);
  return(model);
}

model <- NULL;
if (version == 2) {
  model <- resnet_v2(input_shape = input_shape, depth = depth);
} else {
  model <- resnet_v1(input_shape = input_shape, depth = depth);
}

model %>% keras::compile(loss = 'categorical_crossentropy',
                         optimizer = keras::optimizer_adam(lr = lr_schedule(0)),
                         metrics = list('acc'));
summary(model);

# plot_model(model, to_file="%s.png" % model_type, show_shapes=True)
cat("Model Type:", model_type, "\r\n");

# prepare model model saving directory.
save_dir   <- sprintf("%s/saved_models", getwd());
model_name <- sprintf('cifar10_%s_model.{epoch:03d}.h5', model_type);

if (!dir.exists(save_dir)) dir.create(save_dir);

filepath <- sprintf("%s/%s", save_dir, model_name);

# prepare callbacks for model saving and for learning rate adjustment.
checkpoint <- keras::callback_model_checkpoint(filepath = filepath,
                                               monitor = 'val_acc',
                                               verbose = 1,
                                               save_best_only = T);

lr_scheduler <- keras::callback_learning_rate_scheduler(lr_schedule);
lr_reducer <- keras::callback_reduce_lr_on_plateau(factor = sqrt(0.1),
                                                   cooldown = 0,
                                                   patience = 5,
                                                   min_lr = 0.5e-6);

callbacks <- list(checkpoint, lr_reducer, lr_scheduler);
callbacks <- list();
# run training, with or without data augmentation.
if (!data_augmentation) {
  cat('Not using data augmentation.\r\n');
  model %>% keras::fit(x_train, y_train,
                       batch_size = batch_size, epochs = epochs,
                       validation_data = list(x_test, y_test),
                       shuffle = T,
                       callbacks = callbacks);
} else {
  cat('Using real-time data augmentation.')
  # this will do preprocessing and realtime data augmentation:
  datagen <- keras::image_data_generator(
    featurewise_center = F, # set input mean to 0 over the dataset
    samplewise_center = F, # set each sample mean to 0
    featurewise_std_normalization = F, # divide inputs by std of dataset
    samplewise_std_normalization = F, # divide each input by its std
    zca_whitening = F, # apply ZCA whitening
    rotation_range = 0, # randomly rotate images in the range (deg 0 to 180)
    width_shift_range = 0.1, # randomly shift images horizontally
    height_shift_range = 0.1, # randomly shift images vertically
    horizontal_flip = T, # randomly flip images
    vertical_flip = F # randomly flip images
  );
  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied).
  datagen         <- datagen %>% keras::fit_image_data_generator(x_train);
  steps_per_epoch <- ceiling(nrow(x_train) / batch_size);
  datagen         <- keras::flow_images_from_data(x = x_train, y = y_train, batch_size = batch_size)
  # fit the model on the batches generated by datagen.flow().
  model %>% fit_generator(datagen, verbose = 1,
                          epochs = epochs,
                          validation_data = list(x_test, y_test),
                          steps_per_epoch = steps_per_epoch,
                          callbacks = callbacks);
}

# score trained model
scores <- model %>% keras::evaluate(x_test, y_test, batch_size = batch_size, verbose = 0);
cat('Test loss:', scores$loss, "\r\n");
cat('Test accuracy:', scores$acc, "\r\n");