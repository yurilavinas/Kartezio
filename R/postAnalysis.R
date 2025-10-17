#!/usr/bin/env Rscript
rm(list = ls(all = TRUE))

# Libraries
library(ggstatsplot)
library(viridis)
library(ggplot2)
library(plyr)
library(dplyr)
# library(ggh4x)

options(scipen = 10000)

source("~/Documents/MCF/Kartezio/R/funcs.R")

maximum = 1000000
maxis = seq(from = 0,
            to = maximum,
            length.out = 20)
maxis[1] = 5100
maxis[2] = 100000
maxis[3] = 150000
maxis[4] = 200000
maxis[5] = 250000
maxis[6] = 300000
maxis[7] = 350000
maxis[8] = 400000
maxis[9] = 450000
maxis[10] = 500000
maxis[11] = 550000
maxis[12] = 600000
maxis[13] = 650000
maxis[14] = 700000
maxis[15] = 750000
maxis[16] = 800000
maxis[17] = 850000
maxis[18] = 900000
maxis[19] = 950000
maxis[19] = maximum

data = c()

for (maxi in maxis) {
  cluster = load_al(filename = "Documents/MCF/results/cluster/_oneplus_nMut_1_nDiv_10/raw_test_data.txt",
                    name="cluster",
                    maxi = maxi)
  cluster = cluster[cluster$Images_used ==
                      max(cluster$Images_used), ]
  
  
  ppsnlike = load_al(filename = "Documents/MCF/results/ppsn_like/_oneplus_nMut_1_nDiv_99999/raw_test_data.txt",
                     name="ppsnlike",
                     maxi = maxi)
  ppsnlike = ppsnlike[ppsnlike$Images_used ==
                        max(ppsnlike$Images_used), ]
  
  typical = load_al(filename = "Documents/MCF/results/typical/_oneplus_nMut_1_nDiv_10/raw_test_data.txt",
                    name="typical",
                    maxi = maxi)
  typical = typical[typical$Images_used ==
                      max(typical$Images_used), ]
  
  
  
  rnd = load_al(filename = "Documents/MCF/results/rnd/_oneplus_nMut_1_nDiv_20/raw_test_data.txt",
                    name="random",
                    maxi = maxi)
  rnd = rnd[rnd$Images_used ==
                      max(rnd$Images_used), ]
  
  
  train_data = rbind(cluster,ppsnlike,typical,rnd)
  train_data$Images_used = as.numeric(train_data$Images_used)
  train_data$test = as.numeric(train_data$test)
  
  
  
  data = rbind(data, train_data)
}

tmp = data[which(data$Images_used==max(data$Images_used)),]
for (algo in unique(tmp$algorithm)){
  cat(algo,":",mean(tmp[tmp$algorithm==algo,]$test),sd(tmp[tmp$algorithm==algo,]$test),'\n')
}

data$Images_used = as.factor(data$Images_used)

p <- ggplot(data = data, aes(x = Images_used, y = test, fill = Images_used)) +
  geom_boxplot() +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  facet_wrap(
    ~ algorithm,
    strip.position = "bottom",
    scales = "free_x",
    ncol = 4  # Let ggplot decide
  ) +
  theme_minimal(base_size = 12) +
  labs(
    x = "",
    y = "test IOU - higher is better",
    fill = "Images used",
    caption = "Images Used"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 0.1
    ),
    legend.position = "none",
    plot.caption = element_text(hjust = 0.5, size = rel(1.2))
  )

ggsave("~/Desktop/test_iou.pdf",plot = p, dpi = 150,width = 1920, height = 700,units = 'px')


p <- ggplot(data = data, aes(x = Images_used, y = train, fill = Images_used)) +
  geom_boxplot() +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  facet_wrap(
    ~ algorithm,
    strip.position = "bottom",
    scales = "free_x",
    ncol = 4  # Let ggplot decide
  ) +
  theme_minimal(base_size = 12) +
  labs(
    x = "",
    y = "train IOU - higher is better",
    fill = "Images used",
    caption = "Images Used"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 0.1
    ),
    legend.position = "none",
    plot.caption = element_text(hjust = 0.5, size = rel(1.2))
  )

ggsave("~/Desktop/train_iou.pdf",plot = p, dpi = 150,width = 1920, height = 700,units = 'px')


p <- ggplot(data = data, aes(x = Images_used, y = time, fill = Images_used)) +
  geom_boxplot() +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 0.9)) +
  facet_wrap(
    ~ algorithm,
    strip.position = "bottom",
    scales = "free_x",
    ncol = 4  # Let ggplot decide
  ) +
  theme_minimal(base_size = 12) +
  labs(
    x = "",
    y = "Time",
    fill = "Images used",
    caption = "Images Used"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 0.1
    ),
    legend.position = "none",
    plot.caption = element_text(hjust = 0.5, size = rel(1.2))
  )

ggsave("~/Desktop/time.pdf",plot = p, dpi = 150,width = 1920, height = 700,units = 'px')



p <- ggplot(data = data, aes(x = Images_used, y = size, fill = Images_used)) +
  geom_boxplot() +
  # scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 0.9)) +
  facet_wrap(
    ~ algorithm,
    strip.position = "bottom",
    scales = "free_x",
    ncol = 4  # Let ggplot decide
  ) +
  theme_minimal(base_size = 12) +
  labs(
    x = "",
    y = "Size",
    fill = "Images used",
    caption = "Images Used"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 0.1
    ),
    legend.position = "none",
    plot.caption = element_text(hjust = 0.5, size = rel(1.2))
  )

ggsave("~/Desktop/size.pdf",plot = p, dpi = 150,width = 1920, height = 700,units = 'px')


p <- ggplot(data = data, aes(x = Images_used, y = test-train, fill = Images_used)) +
  geom_boxplot() +
  # scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 0.9)) +
  facet_wrap(
    ~ algorithm,
    strip.position = "bottom",
    scales = "free_x",
    ncol = 4  # Let ggplot decide
  ) +
  theme_minimal(base_size = 12) +
  labs(
    x = "",
    y = "Generalization",
    fill = "Images used",
    caption = "Images Used"
  ) +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 0.1
    ),
    legend.position = "none",
    plot.caption = element_text(hjust = 0.5, size = rel(1.2))
  )

ggsave("~/Desktop/Generalization.pdf",plot = p, dpi = 150,width = 1920, height = 700,units = 'px')

