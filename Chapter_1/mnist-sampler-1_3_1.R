# Demonstrates how to sample and plot MNIST digits
# using Keras API
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras

rm(list = ls());


library(tensorflow);
library(keras);
library(ggplot2);
library(reshape2);
library(gridExtra);

set.seed(1234);

mnist <- keras::dataset_mnist()

plot_title   <- "MNIST - Sampler"
plot_caption <- "(c) 2020 Christian Bitter"

# load dataset
x_train <- mnist$train$x;
x_test  <- mnist$test$x;
y_train <- mnist$train$y;
y_test  <- mnist$test$y;

# count the number of unique train labels
cat("Train labels: ");
table(y_train)

# count the number of unique test labels
table("Test Labels:")
table(y_train);

# sample 25 mnist digits from train dataset
N       <- 5;
indexes <- sample(x = 0:length(y_train), size = N, replace = F);
images  <- x_train[indexes, ,];
labels  <- y_train[indexes];

# plot the 25 mnist digits
plots <- list();
label_text <- character(length = N);
for (i in 1:N) {
  x_i <- images[i,,];
  y_i <- labels[i];
  label_text[i] <- sprintf("Number: %d", y_i);
  df.long <- reshape2::melt(x_i);
  p_i <- df.long %>%
    ggplot(aes(x = Var1, y = Var2)) +
    geom_raster(aes(fill = value)) +
    labs(x = "x", y = "y", subtitle = label_text[i]) + 
    scale_colour_discrete() + 
    guides(fill = F) + 
    theme_light();
  
  plots[[i]] <- p_i;
}

# https://wilkelab.org/cowplot/articles/plot_grid.html
plot_row <- cowplot::plot_grid(plotlist = plots);

# now add the title
title <- ggdraw() + 
  draw_label(plot_title, fontface = 'bold', x = 0, hjust = 0) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = ggplot2::margin(0, 0, 0, 30)
  )

plot_grid(title, plot_row, rel_heights = c(0.1, 1), nrow = 2);
