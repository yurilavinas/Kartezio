load_al = function(filename, name, maxi) {
  # filename = paste0("../data/",folder,"/", name, "/raw_test_data.txt")
  # print(filename)
  data = read.csv(filename,
                  sep = "\t",
                  header = T,
                  skip = 0)
  idx=c(1,3,4,6,7,8,12,13)
  data = data[, idx]
  
  colnames(data) = c(
    "algorithm",
    "Gen",
    "Images_used",
    "train",
    "test",
    "size",
    "updatedElite",
    "time")
  data$algorithm=name
  data = data[data$Images_used <= maxi,]
  val = max(data$Images_used)
  data[which(data$Images_used==val),]$Images_used = maxi
  data$train = 1 - data$train
  data$test = 1 - data$test
  data
}


stats_fitness = function(data){
  cdata <- ddply(data, c("algorithm", "Images_used"), summarise,
                 N    = sum(!is.na(Fitness)),
                 mean = mean(Fitness, na.rm=TRUE),
                 median = median(Fitness, na.rm=TRUE),
                 sd   = sd(Fitness, na.rm=TRUE),
                 se   = sd / sqrt(N)
  )
  cdata
}

stats_active = function(data){
  cdata <- ddply(data, c("algorithm", "Images_used"), summarise,
                 N    = sum(!is.na(active)),
                 mean = mean(active, na.rm=TRUE),
                 sd   = sd(active, na.rm=TRUE),
                 se   = sd / sqrt(N)
  )
  cdata
}

stats_fitness_best = function(data){
  cdata <- ddply(data, c("algorithm", "Images_used"), summarise,
                 N    = sum(!is.na(Fitness)),
                 mean = mean(Fitness, na.rm=TRUE),
                 median = median(Fitness, na.rm=TRUE),
                 sd   = sd(Fitness, na.rm=TRUE),
                 se   = sd / sqrt(N)
  )
  cdata
}


create_cgp_data = function(base_cgp){
  data_files <-
    list.files(base_cgp, pattern = "\\.txt$")
  cgp_data = do.call(rbind, lapply(paste0(base_cgp, data_files), function(x)
    read.csv(
      x,
      sep = "\t",
      header = F,
      skip = 0
    )))
  colnames(cgp_data) = c("Run", "Gen", "Fitness", "Images_used")
  cgp_data$Fitness = 1 - cgp_data$Fitness
  cgp_data$algorithm = "CGP"
  cgp_data
}

scale = function(data){
  data / max(data, na.rm = TRUE)
}

