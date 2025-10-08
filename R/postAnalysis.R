library(tidyverse)
options(scipen = 999)
library(viridis)



getCloser = function(df,target){
  which.min(abs(df$V4 - target))
}
closest_rows <- function(df, target, error = 500) {
  diffs <- abs(df$V4 - target)
  min_diff <- min(diffs, na.rm = TRUE)
  df = df[abs(diffs - min_diff) <= error, ]
  df$V4= target
  return (df)
}
changeCloset = function(df,targetList){
  tmp = data.frame(col1 = character(),
                   col2 = numeric(),
                   col3 = numeric(),
                   stringsAsFactors = FALSE)
  colnames(tmp)=colnames(df)
  for (i in 1:length(targetList)){
    # id = closest_rows(df, targetList[i])
    # df[id]$V4=targetList[i]
    tmp=rbind(tmp,closest_rows(df, targetList[i]))
  }
  return (tmp)
}



targetList = c(4100,50000,100000, 500000,1000000)
targetList = seq(0,1000000,by=25000)

cluster = read.csv('Documents/MCF/results/cluster/_oneplus/raw_test_data.txt', sep = '\t', header = F,skip = 1)
two = read.csv('Documents/MCF/results/typical/_oneplus/raw_test_data.txt', sep = '\t', header = F,skip = 1)
rnd = two[which(two$V1=='rnd'),]
typical = two[which(two$V1=='52'),]



cluster = cluster[,c(1,4,6)]
rnd = rnd[,c(1,4,6)]
typical = typical[,c(1,4,6)]

cluster = changeCloset(cluster, targetList)
rnd = changeCloset(rnd, targetList)
typical = changeCloset(typical, targetList)



data = rbind(
  cluster,
  rnd,
  typical
)
data = data.frame(data)
data$V6 = 1 - data$V6



ggplot(data, aes(x = factor(V4), y = V6, fill = V1)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ V1, scales = "free_x") +
  scale_y_continuous(
    limits = c(0, 1),           # y-axis range
    breaks = seq(0, 1, by = 0.05)  # y-axis ticks every 0.1
  ) +
  labs(
    title = "Boxplot of Values Over Time per Group",
    x = "Timestamp",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)  # vertical labels
  )

tmp = data[which(data$V4==max(data$V4)),]
aggregate(data$V6, FUN=mean, by=list( data$V1))
