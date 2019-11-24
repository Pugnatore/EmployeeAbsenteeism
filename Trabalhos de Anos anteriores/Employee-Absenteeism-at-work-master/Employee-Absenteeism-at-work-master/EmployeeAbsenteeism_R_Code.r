rm(list = ls())
setwd("C:/Users/Click/Desktop/EmployeeAbsenteeism_project")
getwd()
# #loading Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "e1071",
      "DataCombine", "pROC", "doSNOW", "class", "readxl","ROSE","dplyr", "plyr", "reshape","xlsx","pbapply", "unbalanced", "dummies", "MASS" , "gbm" ,"Information", "rpart")

# #install.packages if not
#lapply(x, install.packages)

# #load libraries
lapply(x, require, character.only = TRUE)
rm(x)


#Input Data Source
df = data.frame(read_xls('Absenteeism_at_work_Project.xls', sheet = 1))

#Creating backup of orginal data 
data_Original  = df


###########################################################################
#                  EXPLORING DATA										  #
###########################################################################

#viewing the data
head(df,4)
dim(df)

#structure of data or data types
str(df)  

#Summary of data 
summary(df)

#unique value of each count
apply(df, 2,function(x) length(table(x)))

#Replacing the dot b/w collumn name to underscore for easy to use
names(df) <- gsub('\\.','_',names(df))

# From the above EDA and problem statement categorising data in 2 category "continuous" and "catagorical"
cont_vars = c('Distance_from_Residence_to_Work', 'Service_time', 'Age',
              'Work_load_Average_day', 'Transportation_expense',
              'Hit_target', 'Weight', 'Height', 
              'Body_mass_index', 'Absenteeism_time_in_hours')

cata_vars = c('ID','Reason_for_absence','Month_of_absence','Day_of_the_week',
              'Seasons','Disciplinary_failure', 'Education', 'Social_drinker',
              'Social_smoker', 'Son', 'Pet')


#########################################################################
#          Checking Missing data										#
#########################################################################
apply(df, 2, function(x) {sum(is.na(x))}) # in R, 1 = Row & 2 = Col 

#Creating dataframe with missing values present in each variable
null_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
null_val$Columns = row.names(null_val)
names(null_val)[1] =  "null_percentage"

#Calculating percentage missing value
null_val$null_percentage = (null_val$null_percentage/nrow(df)) * 100

# Sorting null_val in Descending order
null_val = null_val[order(-null_val$null_percentage),]
row.names(null_val) = NULL

# Reordering columns
null_val = null_val[,c(2,1)]

# Saving output result into csv file
write.csv(null_val, "MissingVal_perc_R.csv", row.names = F)


#########################################################################
#                     Visualizing the data                              #
#########################################################################

#library(ggplot2)
#Missing data percentage 

#library(ggplot2)
#Missing data percentage 
ggplot(data = null_val[1:18,], aes(x=reorder(Columns, -null_percentage),y = null_percentage))+
  geom_bar(stat = "identity",fill = "grey")+xlab("Variables")+
  ggtitle("Missing data percentage") + theme(axis.text.x = element_text( color="#993333", size=6, angle=90))

################################################################
#               Data Imputation 					      	   #
################################################################
#Here we are analysing the best method for imputation by trying to generate a value for existing data in our data set.
# Actual Value = 23
# Mean = 26.68
# Median = 25
# KNN = 23

#Mean Method
# df$Body_mass_index[is.na(df$Body_mass_index)] = mean(df$Body_mass_index, na.rm = T)

#Median Method
# df$Body_mass_index[is.na(df$Body_mass_index)] = median(df$Body_mass_index, na.rm = T)

# kNN Imputation
df = knnImputation(df, k = 3)

# Checking for missing value
sum(is.na(df))


################################################################
#               Outlier Analysis					                	   #
################################################################

## BoxPlots - Distribution and Outlier Check

# Boxplot for continuous variables
for (i in 1:length(cont_vars))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cont_vars[i]), x = "Absenteeism_time_in_hours"), data = subset(df))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cont_vars[i],x="Absenteeism_time_in_hours")+
           ggtitle(paste("Box plot of absenteeism for",cont_vars[i])))
}

# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)
gridExtra::grid.arrange(gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,ncol=2)

# #Remove outliers using boxplot method
# #loop to remove from all variables
for(i in cont_vars)
{
  print(i)
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  #print(length(val))
  df = df[which(!df[,i] %in% val),]
}

#Replace all outliers with NA and impute
for(i in cont_vars)
{
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  #print(length(val))
  df[,i][df[,i] %in% val] = NA
}

# Imputing missing values
df = knnImputation(df,k=3)


################################################################
#               Feature Selection                              #
################################################################

#Here we will use corrgram to find corelation

##Correlation plot
#library('corrgram')

corrgram(df,
         order = F,  #we don't want to reorder
         upper.panel=panel.pie,
         lower.panel=panel.shade,
         text.panel=panel.txt,
         main = 'CORRELATION PLOT')


#We can see that the highly corr related vars in plot are marked in dark blue. 
#Dark blue color means highly positive correlation

##------------------ANOVA validation_set--------------------------##

## ANOVA validation_set for Categprical variable
summary(aov(formula = Absenteeism_time_in_hours~ID,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Reason_for_absence,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Month_of_absence,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Day_of_the_week,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Seasons,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Disciplinary_failure,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Education,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Social_drinker,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Social_smoker,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Son,data = df))
summary(aov(formula = Absenteeism_time_in_hours~Pet,data = df))

## Dimension Reduction
df = subset(df, select = -c(Weight))

################################################################
#               Feature Scaling		                     	   #
################################################################

# Updating the continuous and catagorical variables		 
cont_vars = c('Distance_from_Residence_to_Work', 'Service_time', 'Age',
              'Work_load_Average_day', 'Transportation_expense',
              'Hit_target', 'Height', 
              'Body_mass_index')

cata_vars = c('ID','Reason_for_absence','Month_of_absence','Day_of_the_week',
              'Seasons','Disciplinary_failure', 'Education', 'Social_drinker',
              'Social_smoker', 'Son', 'Pet')


#Normality check
#Checking Data for Continuous Variables

################  Histogram   ##################
hist(df$Absenteeism_time_in_hours)
hist(df$Distance_from_Residence_to_Work)
hist(df$Transportation_expense)
hist(df$Work_load_Average_day)
hist(df$Body_mass_index)
hist(df$Service_time)

#We have seen that our data is not normally distributed. Hence, we will go for Normalization.	

#Normalization
for(i in cont_vars)
{
  print(i)
  df[,i] = (df[,i] - min(df[,i]))/(max(df[,i])-min(df[,i]))
}


#Creating dummy variables for categorical variables
library(mlr)
df1 = dummy.data.frame(df, cata_vars)

################################################################
#          		        Sampling of Data        			           #
################################################################

# #Divide data into trainset and validation_set using stratified sampling method

#install.packages('caret')
#library(caret)
set.seed(101)
split_index = createDataPartition(df$Absenteeism_time_in_hours, p = 0.66, list = FALSE)
trainset = df[split_index,]
validation_set  = df[-split_index,]

#Checking df Set Target Class
table(trainset$Absenteeism_time_in_hours)

############################################################################################################################################################
##                                                   Basic approach for ML - Models																		  ##
##               We will first get a basic idea of how different models perform on our preprocesed data and then select the best model and make it        ##
##                                                    more efficient for our Dataset																	  ##
############################################################################################################################################################

#------------------------------------------Decision tree-------------------------------------------#
#Develop Model on training data -> https://www.guru99.com/r-decision-trees.html

fit_DT = rpart(Absenteeism_time_in_hours ~., data = trainset, method = "anova")

#Summary of DT model
summary(fit_DT)

#write rules into disk
write(capture.output(summary(fit_DT)), "Rules.txt")

#Lets predict for training data
pred_DT_train = predict(fit_DT, trainset[,names(trainset) != "Absenteeism_time_in_hours"])

#rpart.plot(fit_DT, extra = 106)

# For training data 
print(postResample(pred = pred_DT_train, obs = trainset[,10]))

#RMSE   Rsquared        MAE 
#4.44522583 0.01624301 3.84331513 

#------------------------------------------Linear Regression-------------------------------------------#

#Develop Model on training data
fit_LR = lm(Absenteeism_time_in_hours ~ ., data = trainset)

#Lets predict for training data
pred_LR_train = predict(fit_LR, trainset[,names(validation_set) != "Absenteeism_time_in_hours"])

# For training data 
print(postResample(pred = pred_LR_train, obs = trainset[,10]))

#RMSE   Rsquared        MAE 
#4.17968459 0.03094267 3.87604411 

#-----------------------------------------Random Forest----------------------------------------------#

#Develop Model on training data
fit_RF = randomForest(Absenteeism_time_in_hours~., data = trainset)

#Lets predict for training data
pred_RF_train = predict(fit_RF, trainset[,names(validation_set) != "Absenteeism_time_in_hours"])

# For training data 
print(postResample(pred = pred_RF_train, obs = trainset[,10]))

#     RMSE   Rsquared        MAE 
# 4.52959371 0.01351095 3.87059587 

#--------------------------------------------XGBoost-------------------------------------------#

#Develop Model on training data
fit_XGB = gbm(Absenteeism_time_in_hours~., data = trainset, n.trees = 500, interaction.depth = 2)

#Lets predict for training data
pred_XGB_train = predict(fit_XGB, trainset[,names(validation_set) != "Absenteeism_time_in_hours"], n.trees = 500)

# For training data 
print(postResample(pred = pred_XGB_train, obs = trainset[,10]))

#      RMSE   Rsquared        MAE 
#4.53665834 0.01470712 3.86070501 


#-------------------------------------------Decision tree for classification-------------------------------------------------#

#Develop Model on training data
fit_DT = rpart(Absenteeism_time_in_hours ~., data = trainset, method = "anova")

#Lets predict for training data
pred_DT_train = predict(fit_DT, trainset)

# For training data 
print(postResample(pred = pred_DT_train, obs = trainset$Absenteeism_time_in_hours))

#     RMSE  Rsquared       MAE 
# 2.3747264 0.4801571 1.6626688

#------------------------------------------Linear Regression-------------------------------------------#

#Develop Model on training data
fit_LR = lm(Absenteeism_time_in_hours ~ ., data = trainset)

#Lets predict for training data
pred_LR_train = predict(fit_LR, trainset)

# For training data 
print(postResample(pred = pred_LR_train, obs = trainset$Absenteeism_time_in_hours))

#RMSE Rsquared      MAE 
#2.8151420 0.2694573 2.0619282 

#-----------------------------------------Random Forest----------------------------------------------#

#Develop Model on training data
fit_RF = randomForest(Absenteeism_time_in_hours~., data = trainset)

#Lets predict for training data
pred_RF_train = predict(fit_RF, trainset)

# For training data 
print(postResample(pred = pred_RF_train, obs = trainset$Absenteeism_time_in_hours))

#   RMSE Rsquared      MAE 
#1.4310323 0.8522856 1.0158687


#--------------------------------------------XGBoost-------------------------------------------#

#Develop Model on training data
fit_XGB = gbm(Absenteeism_time_in_hours~., data = trainset, n.trees = 500, interaction.depth = 2)

#Lets predict for training data
pred_XGB_train = predict(fit_XGB, trainset, n.trees = 500)

# For training data 
print(postResample(pred = pred_XGB_train, obs = trainset$Absenteeism_time_in_hours))

#    RMSE Rsquared      MAE 
#1.804408 0.706962 1.312741 


########################################################################################
#             			 Saving output to file										   #
########################################################################################

#swrite.csv(submit,file = 'C:/Users/Click/Desktop/EmployeeAbsenteeism_project/FinalAbsenteeism_R.csv',row.names = F)
rm(list = ls())
