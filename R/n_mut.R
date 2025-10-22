#!/usr/bin/env Rscript
rm(list = ls(all = TRUE))

# Libraries
# library(ggstatsplot)
library(viridis)
library(ggplot2)
library(plyr)
library(dplyr)
# library(ggh4x)

options(scipen = 10000)

source("~/Documents/MCF/Kartezio/R/funcs.R")

maximum = 1000200
maxis = seq(from = 0,
            to = maximum,
            length.out = 1)
maxis[1] = 10000
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
maxis[20] = maximum

data = c()

for (maxi in maxis) {
  nMut_1 = load_al(filename = "Documents/MCF/results/cluster/_oneplus_nMut_1_nDiv_20_n_future_99/raw_test_data.txt",
                       name="nMut_1",
                       maxi = maxi)
  nMut_1 = nMut_1[nMut_1$Images_used ==
                            max(nMut_1$Images_used), ]
  
  
  nMut_2 = load_al(filename = "Documents/MCF/results/cluster/_oneplus_nMut_2_nDiv_20_n_future_99/raw_test_data.txt",
                        name="nMut_2",
                        maxi = maxi)
  nMut_2 = nMut_2[nMut_2$Images_used ==
                              max(nMut_2$Images_used), ]
  
  nMut_5 = load_al(filename = "Documents/MCF/results/cluster/_oneplus_nMut_5_nDiv_20_n_future_99/raw_test_data.txt",
                        name="nMut_5",
                        maxi = maxi)
  nMut_5 = nMut_5[nMut_5$Images_used ==
                              max(nMut_5$Images_used), ]
  
  
  nMut_10 = load_al(filename = "Documents/MCF/results/cluster/_oneplus_nMut_10_nDiv_20_n_future_99/raw_test_data.txt",
                        name="nMut_10",
                        maxi = maxi)
  nMut_10 = nMut_10[nMut_10$Images_used ==
                              max(nMut_10$Images_used), ]
  
  
  
  
  train_data = rbind(nMut_1,nMut_2,nMut_5,nMut_10)
  train_data$Images_used = as.numeric(train_data$Images_used)
  train_data$test = as.numeric(train_data$test)
  
  
  
  data = rbind(data, train_data)
}

maxImg_data = data[which(data$Images_used==max(data$Images_used)),]


data$Images_used = as.factor(data$Images_used)

p <- ggplot(data = data, aes(x = Images_used, y = test, fill = Images_used)) +
  geom_boxplot() +
  scale_y_continuous(breaks = seq(0.1, 1, 0.2), limits = c(0, 0.95)) +
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

ggsave("~/Desktop/test_iou_n_mut.pdf",plot = p, dpi = 150,width = 1920, height = 700,units = 'px')


for (algo in unique(maxImg_data$algorithm)){
  cat(algo,":",mean(maxImg_data[maxImg_data$algorithm==algo,]$test),sd(maxImg_data[maxImg_data$algorithm==algo,]$test),'\n')
}
b=boxplot(test ~ algorithm, data = maxImg_data)
print(b$n)
pairwise.wilcox.test(maxImg_data$test, maxImg_data$algorithm, p.adjust.method = "bonf", paired = F)

