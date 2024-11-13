getwd()
setwd("/Users/skeppler/Desktop/")

Crime<-read.csv("v1crimedata.csv")

install.packages("ggplot2")
library(ggplot2)

##Type of Offense Percentage Chart
ggplot(Crime, aes(x = CIBRS.Offense.Description)) + 
  geom_bar(aes(y = ..prop.., group = 1), fill = 'blue', color = 'black') +
  geom_text(aes(y = ..prop.., label = sprintf("%.2f%%", ..prop.. * 100), group = 1), 
            stat = "count", vjust = -0.5, size = 3) +
  labs(title = 'Type of Offense', x = 'Type', y = 'Percentage') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0, 0.75)

## Victim Race
ggplot(Crime, aes(x = Victim.Race)) + 
  geom_bar(aes(y = ..prop.., group = 1), fill = 'orange', color = 'black') +
  geom_text(aes(y = ..prop.., label = sprintf("%.2f%%", ..prop.. * 100), group = 1), 
            stat = "count", vjust = -0.5, size = 3) +
  labs(title = 'Victim Race', x = 'Race', y = 'Percentage') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0, 0.50)

##Victim Age
Crime$Age.Group <- cut(Crime$Victim.Age,
                       breaks = seq(0, 100, by = 5),
                       right = FALSE,
                       labels = paste(seq(0, 95, by = 5), seq(5, 100, by = 5) - 1, sep = "-"))

ggplot(Crime, aes(x = Age.Group)) + 
  geom_bar(aes(y = ..prop.., group = 1), fill = 'purple', color = 'black') +
  geom_text(aes(y = ..prop.., label = sprintf("%.2f%%", ..prop.. * 100), group = 1), 
            stat = "count", vjust = -0.5, size = 2) + 
  labs(title = 'Frequency of Victim Age Groups', x = 'Age Group', y = 'Percentage') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0, 0.15)

##City
ggplot(Crime, aes(x = City)) + 
  geom_bar(aes(y = ..prop.., group = 1), fill = 'green', color = 'black') +
  geom_text(aes(y = ..prop.., label = sprintf("%.1f%%", ..prop.. * 100), group = 1), 
            stat = "count", vjust = -0.5, size = 3) +
  labs(title = 'Frequency of Cities', x = 'City', y = 'Percentage') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0, 0.45)

##Domestic Violence
ggplot(Crime, aes(x = Domestic.Violence.Incident)) + 
  geom_bar(aes(y = ..prop.., group = 1), fill = 'red', color = 'black') +
  geom_text(aes(y = ..prop.., label = sprintf("%.2f%%", ..prop.. * 100), group = 1), 
            stat = "count", vjust = -0.5, size = 4) + 
  labs(title = 'Frequency of Domestic Violence Cases', x = 'Domestic Violence', y = 'Percentage') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0, 1)

##Race Breakdown
##Using v2 of the data where subcategories in Race have been combined
Crime2<-read.csv("v2crimedata.csv")

ggplot(Crime2, aes(x = Overall.Race)) + 
  geom_bar(aes(y = ..prop.., group = 1), fill = 'orange', color = 'black') +
  geom_text(aes(y = ..prop.., label = sprintf("%.2f%%", ..prop.. * 100), group = 1), 
            stat = "count", vjust = -0.5, size = 3) + 
  labs(title = 'Frequency of Race', x = 'Race Category', y = 'Percentage') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0, 0.6)



