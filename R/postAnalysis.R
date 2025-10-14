library(tidyverse)
options(scipen = 999)
library(viridis)



closest_rows <- function(df, target, error = 3200) {
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


cluster = read.csv('Documents/MCF/results/cluster/_oneplus/raw_test_data.txt', sep = '\t')
typical = read.csv('Documents/MCF/results/typical/_oneplus/raw_test_data.txt', sep = '\t')
rnd  = read.csv('Documents/MCF/results/rnd/_oneplus/raw_test_data.txt', sep = '\t')
ppsnlike  = read.csv('Documents/MCF/results/ppsn_like/_oneplus/raw_test_data.txt', sep = '\t')

minVal=5100
maxVal=1000000
targetList = seq(0,maxVal,by=as.integer(maxVal/20))
targetList[1]=minVal
targetList[length(targetList)]=maxVal

tmp=cluster$size
cluster$size = cluster$time
cluster$time=tmp

tmp=typical$size
typical$size = typical$time
typical$time=tmp
typical$init_idx="Typical"

tmp=rnd$size
rnd$size = rnd$time
rnd$time=tmp

tmp=ppsnlike$size
ppsnlike$size = ppsnlike$time
ppsnlike$time=tmp


idx=c(1,4,6,7,8,11,12,13)
cluster = cluster[,idx]
rnd = rnd[,idx]
typical = typical[,idx]
ppsnlike = ppsnlike[,idx]
ppsnlike$init_idx="ppsnlike"

cluster = changeCloset(cluster, targetList)
rnd = changeCloset(rnd, targetList)
typical = changeCloset(typical, targetList)
ppsnlike = changeCloset(ppsnlike, targetList)



data = rbind(
  cluster,
  rnd,
  ppsnlike,
  typical
)
data = data.frame(data)
data$test = 1 - data$test
data$train = 1 - data$train


v=ggplot(data, aes(x = factor(eval), y = size, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  scale_y_continuous(
    limits = c(0, 25),           # y-axis range
    breaks = seq(0, 25, by = 1)  # y-axis ticks every 0.1
  ) +
  labs(
    # title = "Active nodes over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  )
# + geom_hline(yintercept = 8, color = "red", linewidth = 1)
ggsave('~/Desktop/Active_nodes.pdf',plot = v, units='px', dpi = 150,width = 960, height = 480)


# v=ggplot(data, aes(x = factor(eval), y = sharpness, fill = eval)) +
#   geom_boxplot(position = position_dodge(width = 0.8)) +
#   facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
#   scale_y_continuous(
#     limits = c(0, 1),           # y-axis range
#     breaks = seq(0, 1, by = 0.05)  # y-axis ticks every 0.1
#   ) +
#   labs(
#     title = "Sharpness over images",
#     x = "Images",
#     y = "Value"
#   ) +
#   theme_minimal() +
#   theme(
#     axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
#   )
# # +   geom_hline(yintercept = 0.1, color = "red", linewidth = 1)
# ggsave('~/Desktop/train.pdf',plot = v, dpi = 150,width = 960, height = 480)


v=ggplot(data, aes(x = factor(eval), y = time, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  scale_y_continuous(
    limits = c(0, 0.2),           # y-axis range
    breaks = seq(0, 0.2, by = 0.05)  # y-axis ticks every 0.1
  ) +
  labs(
    # title = "Time over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  )
# + geom_hline(yintercept = 0.05, color = "red", linewidth = 1)
ggsave('~/Desktop/exec_time.pdf',plot = v, units='px',dpi = 150,width = 960, height = 480)




v=ggplot(data, aes(x = factor(eval), y = updatedElite, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  # scale_y_continuous(
  #   limits = c(0, 1),           # y-axis range
  #   breaks = seq(0, 1, by = 0.05)  # y-axis ticks every 0.1
  # ) +
  labs(
    # title = "Updated Elites over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  )
ggsave('~/Desktop/eliteUpdated_after_AL.pdf',units='px',plot = v, dpi = 150,width = 960, height = 480)


v=ggplot(data, aes(x = factor(eval), y = test, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  scale_y_continuous(
    limits = c(0, 1),           # y-axis range
    breaks = seq(0, 1, by = 0.2)  # y-axis ticks every 0.1
  ) +
  labs(
    # title = "IOU (test) over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  ) 
# +  geom_hline(yintercept = 0.84, color = "red", linewidth = 1)
ggsave('~/Desktop/test.pdf',plot = v, units='px',dpi = 150,width = 960, height = 480)

v=ggplot(data, aes(x = factor(eval), y = train, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  scale_y_continuous(
    limits = c(0, 1),           # y-axis range
    breaks = seq(0, 1, by = 0.05)  # y-axis ticks every 0.1
  ) +
  labs(
    # title = "IOU (train) over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  ) 
# + geom_hline(yintercept = 0.95, color = "red", linewidth = 1)
ggsave('~/Desktop/train.pdf',plot = v, units='px',dpi = 150,width = 960, height = 480)

data$generalization=data$test-data$train
v = ggplot(data, aes(x = factor(eval), y = generalization, fill = eval)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ init_idx, scales = "free_x", nrow = 1) +
  # scale_y_continuous(
  #   limits = c(0, 1),           # y-axis range
  #   breaks = seq(0, 1, by = 0.05)  # y-axis ticks every 0.1
  # ) +
  labs(
    # title = "Generalization over images",
    x = "Images",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  ) 
# + geom_hline(yintercept = 0., color = "red", linewidth = 1)

ggsave('~/Desktop/generalization.pdf',units='px',plot = v, dpi = 150,width = 960, height = 480)

tmp = data[which(data$eval==maxVal),]
aggregate(tmp$test, FUN=mean, by=list( tmp$init_idx))
aggregate(tmp$test, FUN=sd, by=list( tmp$init_idx))


for (i in unique(tmp$init_idx)){
  print(i)
  print(sum(tmp$init_idx==i))
}
