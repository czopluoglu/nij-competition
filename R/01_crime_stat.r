require(dplyr)
require(here)
library(readxl)
################################################################################


crime13 <- read_excel("data/crime stat.xlsx", sheet = "2013")
crime14 <- read_excel("data/crime stat.xlsx", sheet = "2014")
crime15 <- read_excel("data/crime stat.xlsx", sheet = "2015")
crime16 <- read_excel("data/crime stat.xlsx", sheet = "2016")
crime17 <- read_excel("data/crime stat.xlsx", sheet = "2017")


sum(crime13$county==crime14$county)
sum(crime13$county==crime15$county)
sum(crime13$county==crime16$county)
sum(crime13$county==crime17$county)

################################################################################

crime_ave <- crime13
crime_ave[,2:11] <- (crime13[,2:11]+crime14[,2:11]+crime15[,2:11]+crime16[,2:11]+crime17[,2:11])/5
  
for(i in 1:nrow(crime_ave)){
  crime_ave[i,3:11] <- (crime_ave[i,3:11]/as.numeric(crime_ave[i,2]))*100000
}

################################################################################


pumas          <- read_excel("data/puma.xlsx")[1:25,]

pumas$murder   <- NA
pumas$rape     <- NA
pumas$robbery  <- NA
pumas$assault  <- NA
pumas$burglary <- NA
pumas$larceny  <- NA
pumas$theft    <- NA
pumas$arson    <- NA
pumas$total    <- NA

pumas <- as.data.frame(pumas)

for(i in 1:25){
  
  tmp <- strsplit(as.character(pumas[i,3]),",")[[1]]
  tmp <- trimws(tmp)
  
  pumas[i,4:12] <- as.numeric(colMeans(crime_ave[crime_ave$county%in%tmp,3:11]))
  
}



write.csv(pumas, 
          here('data','crime_summary.csv'),
          row.names = FALSE)

















