# A simple MLP in Keras implementing linear regression.
# we could also do the same using the tidy models approach

rm(list = ls());

library(tensorflow);
library(keras);
library(ggplot2);

set.seed(1234);

# 
# # generate x data
x <- seq(from = -1, to = 1, length.out = 10);
dim(x) <- c(length(x), 1);
# 
# generate y data
y <- 2 * x + 3;
# 
# True if noise is added to y
is_noisy <- F;
# 
# add noise if enabled
if (is_noisy) {
    noise <- runif(n = length(x), min = -0.1, max = 0.1);
    x <- x + noise
}

plot(x, y)
# deep learning method
# build 2-layer MLP network
model <- keras::keras_model_sequential();
model %>%
  layer_dense(units = 1, input_shape = 1, name = "Input") %>%
  layer_dense(units = 1, activation = "linear", name = "Output");

summary(model);

model %>% keras::compile(
  loss = 'mse',
  optimizer = keras::optimizer_sgd()
);
# # feed the network with complete dataset (1 epoch) 100 times
# # batch size of sgd is 4
history <- model %>% 
  keras::fit(x, y, epochs = 10, batch_size = 4);

history$params
model %>% evaluate(x, y)

# simple validation by predicting the output based on x
ypred <- model %>% predict(x);

data.frame("y" = y, "yhat" = ypred) %>%
  ggplot(aes(x = y, y = yhat)) + 
  geom_line() + geom_point() +
  theme_light()

