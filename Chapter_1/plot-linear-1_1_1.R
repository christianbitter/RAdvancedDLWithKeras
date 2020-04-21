# Utility for plotting a linear function with and without noise

rm(list = ls());

library(ggplot2);

want_noise <- T;

# generate data bet -1,1 interval of 0.2
x <- seq(-1,1,0.2)
y <- 2*x + 3
df <- data.frame(x, y);
df %>%
  ggplot() + 
  geom_point(aes(x, y)) + 
  labs(x = "x", y = "f(x)") + 
  theme_light();

if (want_noise) {
  # generate data with uniform distribution
  noise <- runif(min = -0.2, max = 0.2, n = nrow(df));
  xn    <- x + noise;
}

data.frame("x" = xn, "y" = y) %>%
  ggplot() + 
  geom_point(aes(x, y)) + 
  labs(x = "x", y = "f(x)",
       subtitle = "y with noised x") + 
  theme_light();
