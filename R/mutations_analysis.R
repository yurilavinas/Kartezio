library(tidyverse)
options(scipen = 999)
library(viridis)



closest_rows <- function(df, target, error = 200) {
  df = df[which(df$eval <= target),]
  diffs <- abs(df$eval - target)
  min_diff <- min(diffs, na.rm = TRUE)
  df = df[abs(diffs - min_diff) <= error, ]
  df$eval= target
  return (df)
}
changeCloset = function(df,targetList){
  tmp = data.frame(col1 = character(),
                   col2 = numeric(),
                   col3 = numeric(),
                   col4 = numeric(),
                   col5 = numeric(),
                   col6 = numeric(),
                   col7 = numeric(),
                   col8 = numeric(),
                   stringsAsFactors = FALSE)
  colnames(tmp)=colnames(df)
  for (i in 1:length(targetList)){
    tmp=rbind(tmp,closest_rows(df, targetList[i]))
  }
  return (tmp)
}


nmut1  = read.csv('Documents/MCF/results/ppsn_like/_oneplus/raw_test_data.txt', sep = '\t')
nmut1$init_idx="mut_1"
nmut5 = read.csv('Documents/MCF/results/ppsn_like/_oneplus_nMut_5_nDiv_99999/raw_test_data.txt', sep = '\t')
nmut5$init_idx="mut_5"
nmut10 = read.csv('Documents/MCF/results/ppsn_like/_oneplus_nMut_10_nDiv_99999/raw_test_data.txt', sep = '\t')
nmut10$init_idx="mut_10"
nmut50 = read.csv('Documents/MCF/results/ppsn_like/_oneplus_nMut_50_nDiv_99999/raw_test_data.txt', sep = '\t')
nmut50$init_idx="mut_50"


minVal=5100
maxVal=1000000
targetList = seq(0,maxVal,by=as.integer(maxVal/20))
targetList[1]=minVal
targetList[length(targetList)]=maxVal

tmp=nmut1$size
nmut1$size = nmut1$time
nmut1$time=tmp

tmp=nmut5$size
nmut5$size = nmut5$time
nmut5$time=tmp

tmp=nmut10$size
nmut10$size = nmut10$time
nmut10$time=tmp



idx=c(1,4,6,7,8,11,12,13)
nmut1 = nmut1[,idx]
nmut5 = nmut5[,idx]
nmut10 = nmut10[,idx]

nmut1 = changeCloset(nmut1, targetList)
nmut5 = changeCloset(nmut5, targetList)
nmut10 = changeCloset(nmut10, targetList)



data = rbind(
  nmut1,
  nmut5,
  nmut10
)
data = data.frame(data)
data$test = 1 - data$test
data$train = 1 - data$train


ggplot(data, aes(x = factor(eval), y = size, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  scale_y_continuous(
    limits = c(0, 25),           # y-axis range
    breaks = seq(0, 25, by = 1)  # y-axis ticks every 0.1
  ) +
  labs(
    title = "Active nodes over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  )+ 
  geom_hline(yintercept = 8, color = "red", linewidth = 1)


ggplot(data, aes(x = factor(eval), y = sharpness, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  scale_y_continuous(
    limits = c(0, 1),           # y-axis range
    breaks = seq(0, 1, by = 0.05)  # y-axis ticks every 0.1
  ) +
  labs(
    title = "Sharpness over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  )+ 
  geom_hline(yintercept = 0.1, color = "red", linewidth = 1)


ggplot(data, aes(x = factor(eval), y = time, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  scale_y_continuous(
    limits = c(0, 0.2),           # y-axis range
    breaks = seq(0, 0.2, by = 0.05)  # y-axis ticks every 0.1
  ) +
  labs(
    title = "Time over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  )+ 
  geom_hline(yintercept = 0.05, color = "red", linewidth = 1)




ggplot(data, aes(x = factor(eval), y = updatedElite, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  # scale_y_continuous(
  #   limits = c(0, 1),           # y-axis range
  #   breaks = seq(0, 1, by = 0.05)  # y-axis ticks every 0.1
  # ) +
  labs(
    title = "Updated Elites over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  )

ggplot(data, aes(x = factor(eval), y = test, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  scale_y_continuous(
    limits = c(0, 1),           # y-axis range
    breaks = seq(0, 1, by = 0.2)  # y-axis ticks every 0.1
  ) +
  labs(
    title = "IOU (test) over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  ) + 
  geom_hline(yintercept = 0.84, color = "red", linewidth = 1)

ggplot(data, aes(x = factor(eval), y = train, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  scale_y_continuous(
    limits = c(0, 1),           # y-axis range
    breaks = seq(0, 1, by = 0.05)  # y-axis ticks every 0.1
  ) +
  labs(
    title = "IOU (train) over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  ) + 
  geom_hline(yintercept = 0.95, color = "red", linewidth = 1)

data$generalization=data$test-data$train
ggplot(data, aes(x = factor(eval), y = generalization, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  # scale_y_continuous(
  #   limits = c(0, 1),           # y-axis range
  #   breaks = seq(0, 1, by = 0.05)  # y-axis ticks every 0.1
  # ) +
  labs(
    title = "Generalization over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  ) + 
  geom_hline(yintercept = 0., color = "red", linewidth = 1)


tmp = data[which(data$eval==maxVal),]
aggregate(tmp$test, FUN=mean, by=list( tmp$init_idx))
aggregate(tmp$test, FUN=sd, by=list( tmp$init_idx))




