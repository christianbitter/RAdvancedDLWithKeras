# Trains a 100-Layer DenseNet on the CIFAR10 dataset.
# 
# With data augmentation:
# Greater than 93.55% test accuracy in 200 epochs
# 225sec per epoch on GTX 1080Ti
# 
# Densely Connected Convolutional Networks
# https://arxiv.org/pdf/1608.06993.pdf
# http://openaccess.thecvf.com/content_cvpr_2017/papers/
#     Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
# Network below is similar to 100-Layer DenseNet-BC (k=12)
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/densenet-cifar10-2.4.1.py
# https://towardsdatascience.com/densenet-2810936aeebb

library(keras);

set.seed(1234);

try(keras::k_cast(1, "float32"));


# training parameters
batch_size <- 32;
epochs     <- 200;
data_augmentation <- T;

# network parameters
num_classes <- 10;
num_dense_blocks <- 3;
use_max_pool <- F;

# DenseNet-BC with dataset augmentation
# Growth rate   | Depth |  Accuracy (paper)| Accuracy (this)      |
# 12            | 100   |  95.49%          | 93.74%               |
# 24            | 250   |  96.38%          | requires big mem GPU |
# 40            | 190   |  96.54%          | requires big mem GPU |
growth_rate <- 12;
depth       <- 100;
num_bottleneck_layers <- (depth - 4) / (2 * num_dense_blocks);

num_filters_bef_dense_block <- 2 * growth_rate;
compression_factor <- 0.5;

# load the CIFAR10 data
cifar10 <- keras::dataset_cifar10();
x_train <- cifar10$train$x;
y_train <- cifar10$train$y;
x_test  <- cifar10$test$x;
y_test  <- cifar10$test$y;

# input image dimensions
input_shape <- dim(x_train)[-1];

# mormalize data
x_train <- keras::k_cast(x_train, 'float32') / 255.
x_test  <- keras::k_cast(x_test, 'float32') / 255.

cat('x_train shape:', dim(x_train), "\r\n");
cat(dim(x_train)[1], 'train samples\r\n');
cat(dim(x_test)[1], 'test samples\r\n');
cat('y_train shape:', dim(y_train), "\r\n");

# convert class vectors to binary class matrices.
y_train <- keras::to_categorical(y_train, num_classes);
y_test  <- keras::to_categorical(y_test, num_classes);

#'@title Learning Rate Schedule
#'@name lr_schedule
#'@description Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
#'Called automatically every epoch as part of callbacks during training.
#'@param epoch (int): The number of epochs
#'@return lr (float32): learning rate
lr_schedule <- function(epoch, current_lr) {
  lr <- 1e-3
  if (epoch > 180) {
    lr <- lr * 0.5e-3
  } else if (epoch > 160) {
    lr <- lr * 1e-3;
  } else if (epoch > 120) {
    lr <- lr * 1e-2;
  } else if (epoch > 80) {
    lr <- lr * 1e-1;
  }
  cat('Learning rate: ', lr, "\r\n");
  return(lr);
}

# start model definition
# densenet CNNs (composite function) are made of BN-ReLU-Conv2D
inputs <- keras::layer_input(shape = input_shape);
x <- inputs %>% keras::layer_batch_normalization();
x <- x %>% keras::layer_activation('relu');
x <- x %>% keras::layer_conv_2d(num_filters_bef_dense_block,
                                kernel_size = 3,
                                padding = 'same',
                                kernel_initializer = 'he_normal');
# instead of resnet addition, this is concatenation
x <- keras::layer_concatenate(list(inputs, x));

# stack of dense blocks bridged by transition layers
for (i in seq(0, num_dense_blocks - 1)) {

  # a dense block is a stack of bottleneck layers
  for (j in seq(0, num_bottleneck_layers - 1)) {
    y <- x %>% keras::layer_batch_normalization();
    y <- y %>% keras::layer_activation('relu');
    y <- y %>% keras::layer_conv_2d(4 * growth_rate,
                                    kernel_size = 1,
                                    padding = 'same',
                                    kernel_initializer = 'he_normal');
    
    if (!data_augmentation) y <- y %>% keras::layer_dropout(0.2);
    y <- y %>% keras::layer_batch_normalization();
    y <- y %>% keras::layer_activation('relu');
    y <- y %>% keras::layer_conv_2d(growth_rate,
                                    kernel_size = 3,
                                    padding = 'same',
                                    kernel_initializer = 'he_normal');
    
    if (!data_augmentation) y <- y %>% keras::layer_dropout(0.2);
    x <- keras::layer_concatenate(list(x, y));
  }
  # no transition layer after the last dense block
  if (i == num_dense_blocks - 1) next;

  # transition layer compresses num of feature maps and reduces the size by 2
  num_filters_bef_dense_block <- num_filters_bef_dense_block + num_bottleneck_layers * growth_rate;
  num_filters_bef_dense_block <- round(num_filters_bef_dense_block * compression_factor);
  
  y <- y %>% keras::layer_batch_normalization();
  y <- y %>% keras::layer_conv_2d(num_filters_bef_dense_block,
                                  kernel_size = 1,
                                  padding = 'same',
                                  kernel_initializer = 'he_normal');
  
  if (!data_augmentation) y <- y %>% keras::layer_dropout(0.2);
  
  x <- x %>% keras::layer_average_pooling_2d();
}

# add classifier on top
# after average pooling, size of feature map is 1 x 1
x <- x %>% keras::layer_average_pooling_2d(pool_size = 8);
y <- x %>% keras::layer_flatten();

outputs <- y %>% keras::layer_dense(units = num_classes,
                                    kernel_initializer = 'he_normal',
                                    activation = 'softmax');

# instantiate and compile model
# orig paper uses SGD but RMSprop works better for DenseNet
model <- keras::keras_model(inputs = inputs, outputs = outputs);

model %>% compile(loss = 'categorical_crossentropy',
                  optimizer = keras::optimizer_rmsprop(1e-3),
                  metrics = list('acc'));

summary(model);

# prepare model model saving directory
save_dir   <- sprintf("%s/%s", getwd(), 'saved_models');
model_name <- 'cifar10_densenet_model.{epoch:02d}.h5'
if (!dir.exists(save_dir)) dir.create(save_dir);
filepath <- sprintf("%s/%s", save_dir, model_name);

# prepare callbacks for model saving and for learning rate reducer
checkpoint <- keras::callback_model_checkpoint(filepath = filepath,
                                               monitor = 'val_acc',
                                               verbose = 1,
                                               save_best_only = T);

lr_scheduler <- keras::callback_learning_rate_scheduler(lr_schedule);

lr_reducer <- keras::callback_reduce_lr_on_plateau(factor = sqrt(0.1),
                                                   cooldown = 0,
                                                   patience = 5,
                                                   min_lr = 0.5e-6);

callbacks = list(checkpoint, lr_reducer, lr_scheduler);

# run training, with or without data augmentation
if (!data_augmentation) {
  cat('Not using data augmentation.\r\n');
  model %>% fit(x_train, y_train,
                batch_size = batch_size,
                epochs = epochs,
                validation_data = list(x_test, y_test),
                shuffle = T,
                callbacks = callbacks);
} else {
  cat('Using real-time data augmentation.\r\n');
  # preprocessing  and realtime data augmentation
  datagen <- keras::image_data_generator(
    featurewise_center = F,  # set input mean to 0 over the dataset
    samplewise_center = F,  # set each sample mean to 0
    featurewise_std_normalization = F,  # divide inputs by std of dataset
    samplewise_std_normalization = F,  # divide each input by its std
    zca_whitening = F,  # apply ZCA whitening
    rotation_range = 0,  # randomly rotate images in the range (deg 0 to 180)
    width_shift_range = 0.1,  # randomly shift images horizontally
    height_shift_range = 0.1,  # randomly shift images vertically
    horizontal_flip = T,  # randomly flip images
    vertical_flip = F);  # randomly flip images

  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied)
  datagen %>% keras::fit_image_data_generator(x_train);

  steps_per_epoch = ceiling(nrow(x_train) / batch_size);
  # fit the model on the batches generated by datagen.flow().
  datagen <- keras::flow_images_from_data(x = x_train, y = y_train, batch_size = batch_size)
  model %>% fit_generator(datagen,
                          verbose = 1,
                          epochs = epochs,
                          validation_data = list(x_test, y_test),
                          steps_per_epoch = steps_per_epoch,
                          callbacks = callbacks);
}

# fit the model on the batches generated by datagen.flow()
#model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
##                    steps_per_epoch=x_train.shape[0] // batch_size,
#                    validation_data=(x_test, y_test),
#                    epochs=epochs, verbose=1,
#                    callbacks=callbacks)

# score trained model
scores <- model %>% evaluate(x_test, y_test, verbose = 0);
print('Test loss:', scores$loss)
print('Test accuracy:', scores$acc)