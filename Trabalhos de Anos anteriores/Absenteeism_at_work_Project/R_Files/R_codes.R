# Clearing the environment
rm(list=ls(all=T))

# getting working directory
getwd()

# Loading libraries
library(xlsx)
library(ggplot2)
library(corrgram)
library(caret)
library(MASS)
library(rpart)
library(DataCombine)
library(corrgram)
library(DMwR)
library(randomForest)
library(unbalanced)
library(dummies)
library(e1071)
library(Information)
library(rpart)
library(gbm)
library(ROSE)


## Reading the data
df = read.xlsx('Absenteeism_at_work_Project.xls', sheetIndex = 1)

############# Exploratory Data Analysis #################

# Shape of the data
dim(df)

# Viewing data
# View(df)

# Structure of the data
str(df)

# Variable namesof the data
colnames(df)

# From the problem statement categorising data in 2 category "continuous" and "catagorical"
continuous_vars = c('Distance.from.Residence.to.Work', 'Service.time', 'Age',
                    'Work.load.Average.day.', 'Transportation.expense',
                    'Hit.target', 'Weight', 'Height', 
                    'Body.mass.index', 'Absenteeism.time.in.hours')

catagorical_vars = c('ID','Reason.for.absence','Month.of.absence','Day.of.the.week',
                     'Seasons','Disciplinary.failure', 'Education', 'Social.drinker',
                     'Social.smoker', 'Son', 'Pet')



############ Missing Values Analysis ###############

#Creating a dataframe with missing values present in each variable
missing_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"

#Calculating percentage missing value
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(df)) * 100

# Sorting missing_val in Descending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL

# Reordering columns
missing_val = missing_val[,c(2,1)]

# Saving output result into csv file
write.csv(missing_val, "Missing_perc_R.csv", row.names = F)

## Plot
ggplot(data = missing_val[1:18,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
geom_bar(stat = "identity",fill = "grey")+xlab("Variables")+
ggtitle("Missing data percentage") + theme_bw()

# Actual Value = 23
# Mean = 26.68
# Median = 25
# KNN = 23


#Mean Method
# df$Body.mass.index[is.na(df$Body.mass.index)] = mean(df$Body.mass.index, na.rm = T)

#Median Method
# df$Body.mass.index[is.na(df$Body.mass.index)] = median(df$Body.mass.index, na.rm = T)

# kNN Imputation
df = knnImputation(df, k = 3)

# Checking for missing value
sum(is.na(df))


############# Outlier Analysis ################

# BoxPlots - Distribution and Outlier Check

# Boxplot for continuous variables
for (i in 1:length(continuous_vars))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (continuous_vars[i]), x = "Absenteeism.time.in.hours"), data = subset(df))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=continuous_vars[i],x="Absenteeism.time.in.hours")+
           ggtitle(paste("Box plot of absenteeism for",continuous_vars[i])))
}

### Plotting plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)
gridExtra::grid.arrange(gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,ncol=2)


##Remove outliers using boxplot method

##loop to remove from all variables
for(i in continuous_vars)
{
  print(i)
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  #print(length(val))
  df = df[which(!df[,i] %in% val),]
}

#Replace all outliers with NA and impute

for(i in continuous_vars)
{
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  #print(length(val))
  df[,i][df[,i] %in% val] = NA
}

# Imputing missing values
df = knnImputation(df,k=3)


############### Feature Selection ###############

## Correlation Plot 
corrgram(df[,continuous_vars], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

## ANOVA test for Categprical variable
summary(aov(formula = Absenteeism.time.in.hours~ID,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Reason.for.absence,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Month.of.absence,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Day.of.the.week,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Seasons,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Disciplinary.failure,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Education,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Social.drinker,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Social.smoker,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Son,data = df))
summary(aov(formula = Absenteeism.time.in.hours~Pet,data = df))


## Dimension Reduction
df = subset(df, select = -c(Weight))


############# Feature Scaling ##############
#Normality check
hist(df$Absenteeism.time.in.hours)

# Updating the continuous and catagorical variable
continuous_vars = c('Distance.from.Residence.to.Work', 'Service.time', 'Age',
                    'Work.load.Average.day.', 'Transportation.expense',
                    'Hit.target', 'Height', 
                    'Body.mass.index')

catagorical_vars = c('ID','Reason.for.absence','Disciplinary.failure', 
                     'Social.drinker', 'Son', 'Pet', 'Month.of.absence', 'Day.of.the.week', 'Seasons',
                     'Education', 'Social.smoker')


# Normalization
for(i in continuous_vars)
{
  print(i)
  df[,i] = (df[,i] - min(df[,i]))/(max(df[,i])-min(df[,i]))
}

# Creating dummy variables for categorical variables
library(mlr)
df = dummy.data.frame(df, catagorical_vars)


############ Model Development ################

#Cleaning the environment
rmExcept("df")

#Divide data into train and test using stratified sampling method
set.seed(123)
train.index = sample(1:nrow(df), 0.8 * nrow(df))
train = df[ train.index,]
test  = df[-train.index,]

##Decision tree for classification
#Develop Model on training data
fit_DT = rpart(Absenteeism.time.in.hours ~., data = train, method = "anova")

#Summary of DT model
summary(fit_DT)

#write rules into disk
write(capture.output(summary(fit_DT)), "Rules.txt")

#Lets predict for training data
pred_DT_train = predict(fit_DT, train[,names(test) != "Absenteeism.time.in.hours"])

#Lets predict for training data
pred_DT_test = predict(fit_DT,test[,names(test) != "Absenteeism.time.in.hours"])


# For training data 
print(postResample(pred = pred_DT_train, obs = train[,107]))

# For testing data 
print(postResample(pred = pred_DT_test, obs = test[,107]))


############## Linear Regression ############

set.seed(123)

#Develop Model on training data
fit_LR = lm(Absenteeism.time.in.hours ~ ., data = train)

#Lets predict for training data
pred_LR_train = predict(fit_LR, train[,names(test) != "Absenteeism.time.in.hours"])

#Lets predict for testing data
pred_LR_test = predict(fit_LR,test[,names(test) != "Absenteeism.time.in.hours"])

# For training data 
print(postResample(pred = pred_LR_train, obs = train[,107]))

# For testing data 
print(postResample(pred = pred_LR_test, obs = test[,107]))


######### Random Forest ############

set.seed(123)

#Develop Model on training data
fit_RF = randomForest(Absenteeism.time.in.hours~., data = train)

#Lets predict for training data
pred_RF_train = predict(fit_RF, train[,names(test) != "Absenteeism.time.in.hours"])

#Lets predict for testing data
pred_RF_test = predict(fit_RF,test[,names(test) != "Absenteeism.time.in.hours"])

# For training data 
print(postResample(pred = pred_RF_train, obs = train[,107]))

# For testing data 
print(postResample(pred = pred_RF_test, obs = test[,107]))




