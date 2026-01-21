library(tidyverse)
library(mdsr)
presidential



select(
  filter(presidential, lubridate::year(start) > 1973 & party == "Democratic"),
  name
)

presidential %>% 
  filter(lubridate::year(start)>1973 & party == "Republican") %>% 
  select(name)
filter(presidential, end > 1961)
select(presidential, name)

presidential %>% 
  filter(lubridate::year(start) > 2008) %>% 
  select(party)

word <- filter(presidential, party == "Republican")
liners <- length(word)
print(liners)


install.packages("tidyverse")
library("tidyverse")
library()
library(mdsr)

xxx <- c(4, 1, 3, 8, 6, 7, 5, 3, 0, 9)
xxx
xxx[3]
xxx[3:7]


factoringvector <- factor(c(69, 69, 69, 70, 70, 71))
factoringvector
#can do with characters too

#paste function
pastingVectors <- paste(1:7, "xy", sep = "")
pastingVectors

matr <- matrix(1:11, ncol = 4, nrow = 4, byrow = FALSE)
matr
#dont really need nrow specification for matrices
#byrow specifying whether to fill matrix horizontally, or false = vertically
matr[1,4]
matr[3,]

#list practice
vector3 <- c(5, 6, 7)
valuesList <- list(666, "Franky", rep(vector3, 2), matrix(c(1:10), ncol = 5, nrow = 2, byrow = TRUE))
valuesList
valuesList[2]

#DF creation
df1 <- data.frame(Sect1 = 1:5, Sect2 = rep(5,5), Sect3 = seq(2, 10, 2))
df1
df1[,3]

summary(df1)
table(df1)

library(tidyverse)
library(mdsr)
presidential

select(presidential, name, party)
filter(presidential, end > "2000-01-01")

presidential %>% 
  filter(start > 1973-01-01 & party == "Republican") %>% 
  select(name)

x<-1:1000
y<-head(x)
plot(y)


install.packages("readxl")


library(tidyverse)
view(mpg)
?mpg
glimpse(mpg)
?filter

filter(mpg, cty >= 20)

mpg_efficiency <- filter(mpg, cty >= 20)
view(mpg_efficiency)
plot(mpg_efficiency)
plot(mpg)


mpg_fordname <- filter(mpg, manufacturer == "ford")
view(mpg_fordname)

view(mpg)


view(mpg)




mpg_kilo <- mpg %>% 
  mutate(city_kilo = cty * 0.425144) %>% 
  filter(city_kilo >= 8)
view(mpg_kilo)

view(mpg)
mpg %>% 
  group_by(class) %>% 
  summarise(mean(cty))
            #median(cty))
# data comment for example, about to do some data vis ~ ggplot2


ggplot(mpg, aes(x = cty)) +
  geom_histogram() +
  geom_freqpoly() +
  labs(x = "City Mileage")

#using regression lines
ggplot(mpg, aes(x = cty,
                y = hwy)) +
  geom_point() + 
  geom_smooth(method = "lm")
  labs(x = "City Mileage", y = "Highway Mileage")

ggplot(mpg, aes(x = cty,
                y = hwy,
                color = class)) +
  geom_point() +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "City Mileage", y = "Highway Mileage")
 
install.packages("pacman")
library(pacman)
?pacman
??pacman

