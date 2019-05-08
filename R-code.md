---
title: "ML Pump it Up"
author: "Camille Chauliac, Chia-Yu Lin, Hsiu-Chi Liu, Julius Oldorf"
date: "February 28, 2019"
output: html_document
---

# Executive summary

## Models we are using
Five distinct types of classification models were constructed for purposes of predicting the current status of a given water pump:

### Logistic Regression
*Convenient probability scores*

*Efficient implementations available*

*Wide spread industry comfort for*

*Logistic regression solutions*

Although it is easy to understand and really simple to execute, there are some down sides about it:
*Doesn't perform well when feature space is too large*
*Doesn't handle large number of categorical features/variables well*
*Relies on transformations for nonlinear features*

As a result we are just using this as a baseline model to compare how good do the rest perform.

### Naive Bayes
*Very fast, low storage requirements*
*Robust to irrelevant features*
  *irrelevant features cancel each other without affecting results*
*Very good in domains with many equally important features.*
*Optimal if the independence assumptionshold:*
  *If assumed independence of events is correct, then Bayes is the optimal classifier for problem*
  *If not, remember: "All models are wrong but some are useful".*

### Decision Tree
*Intuitive Decision Rules*
*Can handle non-linear features*
*Take into account variable interactions*
*Recursive Partitioning is relatively fast, even with large data sets and many attributes*
*Recursive Partitioning does feature selection*
*Small-medium size trees usually intelligible*
*Can be converted to rules*
But we have to be careful about that it is highly biased to training set, as a result, we decided to try Random Forest to check out the performance as well.

### Random Forest
Has all the pros from the decision tree, and further solve the problem of highly biased to training set.
*Independent Classifiers*
*Reduce Variance*
*Handles Overfitting*

### XGBoost
Instead of fitting a large decision tree to the data, boosting tries to learn slowly from the errors
*Extremely fast (parallel computation) and highly efficient.*
*Versatile (Can be used for classification, regression or ranking).*
*Can be used to extract variable importance.*
*Do not require feature transformation (missing values imputation, scaling and normalization)*
However, we still have to tune hyperparameters properly, otherwise it might be overfitted.

## Data Preparation
More than 53% of the records in the data set contain either missing or invalid data values. Our data exploration efforts allowed us to uncover relationships between various independent variables that could serve as the basis of statistically valid imputation algorithms for the missing values of the amount_tsh, gps_height, construction_year, latitude and longitude variables.

Eventually we decided to drop columns based 3 reasons:
*Meaningless variable: recorded_by, num_private*
*Duplicate columns: waterpoint_type_group, source, source_class, quantity_group, payment_type, extraction_type, extraction_type_group, scheme_management, water_quality, region_code, management_group, region, district_code*
*High level variables: wpt_name, subvillage, scheme_name, ward*

Impute NAs and 0s by unknown and median

## Feature Engineering
We created new features based on date and month to gain more insightful information from it, unfortunately our best score, which is 0.8270 from XGBoost performs better without these new features.

# Result
Best Score came from the XGBoost model we tried: 0.8270

-----------------------------------------------------------------------------------------------------------


# Import libraries
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(ggplot2)
library(plyr)
library(dplyr)     # To compute the `union` of the levels.
library(png)       # To include images in this document.
library(knitr)     # To include images inline in this doc.
library(moments)   # Skewness
library(e1071)     # Alternative for Skewness
library(glmnet)    # Lasso
library(caret)     # To enable Lasso training with CV.
library(corrplot)  # For plotting correlation plot
library(xgboost)   # For the XGBoost Model
library(Matrix)
library(MatrixModels)
library(data.table)
library(lubridate)
library(leaflet)
```

# Data Reading and preparation
The dataset is offered in two separated fields, one for the training and another one for the test set. Another dataset containing the labels is given and has been integrated as well. 

```{r Load Data and set blank into NAs}
# For the original_training_data I precombined the lable to the train dataset in excel
original_training_data = read.csv("TrainPump_0.csv", na.strings = "")
original_test_data = read.csv("TestPump.csv", na.strings = "")
label = read.csv("TrainPump_Labels.csv")
```

Joining them together in order to do datacleaning and feature engineering on both sets:
```{r Joinning datasets}
# Combine the training sets: Labels and values
original_training_data <- cbind(original_training_data, label)
original_training_data <- original_training_data[,-41]
dim(original_training_data)

# Create an empty coloumn status_group in order to merge the test and train sets
original_test_data$status_group <- 0
dim(original_test_data)

dataset <- rbind(original_training_data, original_test_data)
```

## Summarize and visualize
Look at the data we are dealing with: 
```{r Dataset Visualization}
# summary(dataset)
dim(dataset)
```

Let's now summarize the dataset to see where to begin
```{r Dataset summarization}
str(dataset)
```

Distribution of the Water pumps:
```{r distribution of water pumps}
# absolute numbers
table(original_training_data$status_group) 

# proportions
prop.table(table(original_training_data$status_group))
```
The tables above provides us with a benchmark. The plot tells us that 54.3% of all pumps are functional, 38.4% are non-functional, and 7.3% are functional but in need of repair.

```{r check coordinates of the waterpumps}
# mapping the coordinates
map.original_training_data <- original_training_data %>% select(latitude, longitude, status_group) %>% sample_frac(size = 1)

leaflet() %>% addTiles() %>% addMarkers(lat = map.original_training_data$latitude, lng = map.original_training_data$longitude, clusterOptions = markerClusterOptions())
```
Due to this interactive map, we can see that some of the coordinates for latitude and longitude in this dataset do not make any sense as it is not possible to have waterpumps in the middle of the ocean.

# Dropping columns and cleaning the dataset

## Remove meaningless features 
We found recorded_by that just had one category in all. Since it would be in no way useful for predicting the final class label, we decided to remove it from our model.

Id is also not of a high predictive value as it is unique for every well, but as this has to be kept as a column for the test data this variable is not to be removed.

```{r drop recorded_by}
dataset <- dataset[,-which(names(dataset) == "recorded_by")]
#dataset <- dataset[,-which(names(dataset) == "id")]
#id can be droped because it is unique for each instance.
```

Exclude amount_tsh because about 70% of the values are missing, but what we found over here is that once we drop this column, it actually lower downs the result.
```{r drop amount_tsh}
#dataset <- dataset[,-which(names(dataset) == "amount_tsh")]
```

No clear explanation is given about the num_private variable. Therefore we dive deeper into this variable to try and make more sense out of it. 

Num_private is mostly (~99%) zeros, only a small amount of this variable has values different than zero. Since we are not sure of what this variable actually means, it is a shot in the dark but we decide to drop this variable as it will most likely not give us any predictive value.
```{r Feature Selection num_private}
summary(dataset$num_private)
boxplot(as.numeric(dataset$num_private), col = "yellow")
hist(as.numeric(dataset$num_private), col = 'yellow')
length(dataset$num_private[which(dataset$num_private > 0)])

dataset$num_private<-NULL
```

## Remove redundant features as well
We remove redundant features as well. As all those features technically portray the same variable. (duplicates/similar to other variables). They contain similar representation of the data, the only difference is the factor levels. If all features would be included during the training of our data we would risk overfitting. Therefore, we decided to drop the following variables: 

-> region_code = region
-> water_quality = quality_group
-> quantity = quantity_group
-> payment = payment_type
-> waterpoint_type = waterpoint_type_group
-> scheme_name = scheme_management
-> extraction_type_class = extraction_type & extraction_type_group
-> management = management_group
-> source_type = source and source_class
```{r remove redundant features}
Redundant <- c("waterpoint_type_group","source", "source_class", "quantity_group","payment_type","extraction_type", "extraction_type_group","management_group","scheme_management","water_quality","region_code")
for (x in Redundant)    dataset <- dataset[,-which(names(dataset) == x)]
```

### Left over location variables
We have 2 more location variables (district_code, region) left in the dataset that portray the same thing - Geographical location (just with different factor levels):
```{r district_code}
# count distinct district_code
length(unique(dataset$district_code))

# calc number of wells per district_code
c_subv <- arrange(summarise(group_by(dataset,district_code), 
                     TotalWells = length(unique(id)) ), desc(TotalWells) )
head(c_subv, n = 20)
```

```{r region}
# count distinct region
length(unique(dataset$region))

# calc number of wells per region
c_subv <- arrange(summarise(group_by(dataset,region), 
                     TotalWells = length(unique(id)) ), desc(TotalWells) )
head(c_subv, n = 20)
```

Technically, region and district_code could be representing the same thing. As we believe this might cause overfitting, we will only include longitude and latitude as location variables. We think this should be sufficient.
```{r drop region and district_code}
dataset$region<-NULL
dataset$district_code<-NULL
```

### Installer and funder
The installer variable represents the name of the organization that installed the given pumps. This variable contains a lot of unique values and missing values (NA's). The NA is even the second biggest group when you calculate the number of wells per installer.
```{r exploration of installer}
# count distinct installer
length(unique(dataset$installer))

# calc number of wells per installer
c_installers <- arrange(summarise(group_by(dataset, installer), TotalWells = length(unique(id)) ), desc(TotalWells) )
head(c_installers, n = 20)
```
Same goes for funder. This variable also contains a lot of different unique values and a numerous amount of missing values. 

```{r exploration of funder}
# count distinct funder
length(unique(dataset$funder))

# calc number of wells per funder
c_funders <- arrange(summarise(group_by(dataset, funder), 
                     TotalWells = length(unique(id)) ), desc(TotalWells) )
head(c_funders, n = 20)
```

But the most valuable insight we get is when we check for coincidences between the missing values of these two variables. 
```{r cursory review of installer and funder}
f_i <- subset(dataset, is.na(dataset$funder) & is.na(dataset$installer))
nrow(f_i)
```
The fact that nearly all of the missing funder and installer values are coincident makes it difficult to find a decent and correct imputation for these missing values. Therefore we decide to remove installer, keep funder and focus on a way to impute funder. 

```{r drop installer}
dataset$installer<-NULL
```

## Discover NAs and unique values
```{r NAs discovery}
na.cols <- which(colSums(is.na(dataset)) > 0)
paste('There are', length(na.cols), 'columns with missing values')
sort(colSums(sapply(dataset[na.cols], is.na)), decreasing = TRUE)
```

```{r unique values discovery}
sort(apply(dataset, 2, function(x) length(unique(x))), decreasing = T)
```

## Remove high level features
We decided to remove columns with levels higher than 20,000 and column that have too many missing values and given the large number of possible values as a proportion of the total records, we believe these variables offer limited value for our prediction. Therefore, we decide to drop the subvillage variable as well as wpt_name and scheme_name.
```{r remove highlevels features}
highLevels <- c("wpt_name","subvillage")
for (x in highLevels)    dataset <- dataset[,-which(names(dataset) == x)]
```

```{r drop ward}
length(unique(dataset$ward))

c_ward <- arrange(summarise(group_by(dataset, ward), 
                     TotalWells = length(unique(id)) ), desc(TotalWells) )
head(c_ward, n = 20)

#Removed because there were too many unique values
dataset$ward<- NULL
```

## Lower down the level for funder
We explored the funder column and found out that there are 2141 unique values in this column even after cleaning and replacing meaningless values with other. But we realize that by doing this actually lowers down the final score so we exclude this step at last.
```{r cleaning and exploring the funder column}
#dataset$funder <- tolower(dataset$funder)
#dataset$funder[dataset$funder %in% c(" ", "", "0", "_", "-", "unknown")] <- "other"
```

After exploring, we decide to narrow down the unique value of column funder to 50 unique values, since RandomForest can take no more than 55 categorical features per column. But we realize that by doing this actually lowers down the final score so we exclude this step at last.
```{r lower down unique values in funder}
#funderTop <- names(summary(as.factor(dataset$funder)))[1:50]
#dataset$funder[!(dataset$funder %in% funderTop)] <- "other"
```

# Replace NAs with Unknown

After taking a close look to the remaining columns, we decided to replace NAs by unknown since the missing records might mean something they don't know
```{r replace NAs with unknown for the rest}
theRest <- c("permit","public_meeting")
for (x in theRest )
{
dataset[[x]] <- factor( dataset[[x]], levels= c(levels(dataset[[x]]),c('unknown')))
dataset[[x]][is.na(dataset[[x]])] <- "unknown"
}

# Preparing dataset for Random Forest
dataset2 <- dataset
```

# Feature Engineering

With this step, we hashtagged out everything, due to the reason that XGBoost perfrom better without all the settings down there. We have tried them with all the other models too, but they have never given us a better accuracy score than with XGBoost.

## Impute 0s

```{r impute 0s for gps_height and construction_year}
#dataset$gps_height[dataset$gps_height==0] <- #median(dataset$gps_height[dataset$gps_height>0])

#dataset$construction_year[dataset$construction_year==0] <- median(dataset$construction_year[dataset$construction_year>0])
```


## Change Variables types

```{r change date_recorded}
#Changed date_recorded to the Date variable type
#dataset$date_recorded <- as.Date(as.character(dataset$date_recorded))
#class(dataset$date_recorded)
```

## Create new features

### Add 2 derived features : of days since Jan 1 2014, month recorded as factor
```{r create features based on date_recorded}
#dataset[["date_recorded_offset_days"]] <- as.numeric(as.Date("2014-01-01") - dataset$date_recorded)

#dataset[["date_recorded_month"]] <- factor(format(as.Date(dataset$date_recorded), "%b"))
```

### Categorize months into seasonal description
From Expert Africa: Tanzania has two rainy seasons: The short rains from late-October to late-December, a.k.a. the Mango Rains, and the long rains from March to May.

Seasons in Tanzania:
Short dry season: January, February
Long rains: March, April, May
Long dry season: June, July, August, September, October
Short rains: November, December
```{r}
#dataset[["rain_seasons"]] <- ifelse(dataset$date_recorded_month == c("Jan", "Feb"), "dry short", ifelse(dataset$date_recorded_month == c("Mar", "Apr", "May"), "wet long", ifelse(dataset$date_recorded_month == c("Jun", "Jul", "Aug", "Sep", "Oct"), "dry long", "wet short")))
```

### Set up operation_years based on date_recorded and construction_year
There is some interesting time information as well: `date_recorded` and `construction_year`. Unfortunately, the year of construction is missing for about 35% of the data points. I convert it to `operation_years` by subtracting the year in which the status was recorded.
```{r}
#dataset[["operation_years"]] <- lubridate::year(dataset[["date_recorded"]]) - dataset[["construction_year"]]
```

# Train, Validation Spliting 

To facilitate the data cleaning and feature engineering we merged train and test datasets. We now split them again to create our final model.
```{r Train test split}
training_data <- dataset[1:59400,]
test <- dataset[59401:74250,]
```

We are going to split the annotated dataset in training and validation for the later evaluation of our models
```{r Train Validation split}
# I found this function, that is worth to save for future ocasions.
splitdf <- function(dataframe, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
 	index <- 1:nrow(dataframe)
 	trainindex <- sample(index, trunc(length(index)/1.5))
 	trainset <- dataframe[trainindex, ]
 	testset <- dataframe[-trainindex, ]
 	list(trainset=trainset,testset=testset)
}
splits <- splitdf(training_data, seed=1)
training <- splits$trainset
validation <- splits$testset
```

# Logistic Regression

```{r libraries for logistic regression}
# for multinom logistic regression
library(nnet)
library(FSelector)
```

```{r chisquared}
chisquared <- data.frame(chi.squared(status_group~., training))
chisquared$features <- rownames(chisquared)

# Plot the result, and remove those below the 1st IQR
(inter-quartile-range) --aggressive
par(mfrow=c(1,2))
boxplot(chisquared$attr_importance)
bp.stats <- boxplot.stats(chisquared$attr_importance)$stats   

# Get the statistics from the boxplot
chisquared.threshold = bp.stats[2]  # This element represent the 1st quartile.
text(y = bp.stats, labels = bp.stats, x = 1.3, cex=0.7)
barplot(sort(chisquared$attr_importance), names.arg =
chisquared$features, cex.names = 0.6, las=2, horiz = T)
abline(v=chisquared.threshold, col='red')  # Draw a red line over the 1st IQR
```

```{r logistic regression}
mlr1 <- multinom(status_group ~ quantity + waterpoint_type + latitude + longitude + management + payment + basin + population, data = training, maxit = 250)
probs <- predict(mlr1, newdata=validation, type = "prob")
classes <- predict(mlr1, newdata=validation, type = "class")

probs <- predict(mlr1, newdata=test, type = "prob")
classes <- predict(mlr1, newdata=test, type = "class")

submission <- data.frame(id=original_test_data$id, status_group=classes)

write.csv(submission, file="Water_solution_LogReg.csv", row.names = FALSE)

```

# Naive Bayes

```{r Naive Bayes model}
nb_model <- naiveBayes(training, training$status_group, laplace = 1)
probs <- predict(nb_model, newdata=validation, type = "raw")
classes <- predict(nb_model, newdata=validation, type = "class")

probs <- predict(nb_model, newdata=test, type = "raw")
classes <- predict(nb_model, newdata=test, type = "class")

submission <- data.frame(id=original_test_data$id, status_group=classes)

write.csv(submission, file = "Water_solution_naivebayes.csv", row.names = FALSE)
```

# XGBoost model

```{r preparation for XGBoost}

#Remove the id and status_group columns from the train and test dataset.I don't want these columns to affect the the model

data_train <- subset(training_data, select = c(-id, -status_group))
data_test_1 <- subset(test, select = c(-id, -status_group))

#Convert data frames to numeric matrices. Xgboost requires user to enter data as a numeric matrix
data_test_1 <- as.matrix(as.data.frame(lapply(data_test_1, as.numeric)))
data_train <- as.matrix(as.data.frame(lapply(data_train, as.numeric)))
label <- as.numeric(label$status_group)
```

```{r select the parameters by cross validation}
#Create a xgb.DMatrix which is the best format to use to create an xgboost model
train.DMatrix <- xgb.DMatrix(data = data_train,label = label, missing = NA)

#For loop to run model 11 time with different random seeds. Using an ensemble technique such as this improved the model performance

#Set i=2 because the first column is for the id variable
i=2

#Create data frame to hold the 11 solutions developed by the model
solution.table<-data.frame(id=test[,"id"])
for (i in 2:12){

  #Set seed so that the results are reproducible
  set.seed(i)

#Cross validation to determine the number of iterations to run the model.
#I tested this model with a variety of parameters to find the most accurate model
xgb.tab = xgb.cv(data = train.DMatrix, objective = "multi:softmax", booster = "gbtree", nrounds = 500, nfold = 4, early_stopping_rounds = 10, num_class = 4, maximize = FALSE, evaluation = "merror", eta = .2, max_depth = 12, colsample_bytree = .4)

#Create variable that identifies the optimal number of iterations for the model
min.error.idx = which.min(xgb.tab[["evaluation_log"]][["test_merror_mean"]])

#Create model using the same parameters used in xgb.cv
model <- xgboost(data = train.DMatrix, objective = "multi:softmax", booster = "gbtree", eval_metric = "merror", nrounds = min.error.idx, num_class = 4,eta = .2, max_depth = 14, colsample_bytree = .4)

#Predict. Used the data_test because it contained the same number of columns as the train.DMatrix used to build the model.
predict <- predict(model,data_test_1)

#Modify prediction labels to match submission format
predict[predict==1]<-"functional"
predict[predict==2]<-"functional needs repair"
predict[predict==3]<-"non functional"

#View prediction
table(predict)

#Add the solution to column i of the solutions data frame. This creates a data frame with a column for each prediction set. Each prediction is a vote for that prediction. Next I will count the number of votes for each prediction as use the element with the most votes as my final solution.
solution.table[,i]<-predict
}
```

```{r prediction}
#Count the number of votes for each solution for each row
solution.table.count<-apply(solution.table,MARGIN=1,table)

#Create a vector to hold the final solution
predict.combined <- vector()
x=1

#Finds the element that has the most votes for each prediction row
for (x in 1:nrow(data_test_1)){
  predict.combined[x]<-names(which.max(solution.table.count[[x]]))}

#View the number of predictions for each classification
table(predict.combined)
```

```{r create solution}
#Create solution data frame
solution<- data.frame(id=test[,"id"], status_group=predict.combined)
```

```{r write the solution into csv file}
#Create csv submission file
write.csv(solution, file = "Water_solution_xgboost.csv", row.names = FALSE)
```

# RandomForest

To facilitate the data cleaning and feature engineering we merged train and test datasets. We now split them again to create our final model.
```{r Train, Validation Spliting for Random Forest}
# Excluding variables that contain more than 53 categories as RandomForest cannot deal with them
drops <- c("funder","installer",
           "lga", "ward", "date_recorded")
dataset2 <- dataset2[ , !(names(dataset2) %in% drops)]

training_data1 <- dataset2[1:59400,]
test1 <- dataset2[59401:74250,]
```

We are going to split the annotated dataset in training and validation for the later evaluation of our models.
```{r Train Validation split}
splitdf <- function(dataframe, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
 	index <- 1:nrow(dataframe)
 	trainindex <- sample(index, trunc(length(index)/1.5))
 	trainset <- dataframe[trainindex, ]
 	testset <- dataframe[-trainindex, ]
 	list(trainset=trainset,testset=testset)
}
splits <- splitdf(training_data, seed=1)
training1 <- splits$trainset
validation1 <- splits$testset
```

Now we are training a first baseline model to test what score we achieve without optimization. 
```{r setup the first baseline model}
library(randomForest)
# Baseline model
model1 <- randomForest(status_group ~ ., data = training_data1, importance = TRUE, na.action = na.roughfix)
model1
```

With this loop we are aiming to optimize the mtry value. This operation indicates the highest accuracy for mtry = 8.
```{r}
# Using For loop to identify the right mtry for the model
# (THIS OPERATION TAKES SOME TIME)

# a=c()
# i=5
# 
# for (i in 3:8) {
#   model2 <- randomForest(status_group ~ ., data = training_data1, ntree = 500, mtry = i, importance = TRUE)
#   predValid <- predict(model2, training_data1, type = "class")
#   a[i-2] = mean(predValid ==  training_data1$status_group)
# }
# 
# a
# plot(3:8,a)

# Results: 
#0.8738047 0.9053030 0.9281650 0.9457912 0.9582323 0.9671380
```

Model 2 utilizes the mtry optimization result. We will now train the new model respectively to compare it with Model 1. 
```{r}
model2 <- randomForest(formula = status_group ~ ., data = training_data1, importance = TRUE, mtry = 8)
model2
```


##Model1
We are now predicting on the train set and have closer look on the actual predictions.
```{r}
# Predicting on train set 
predTrain <- predict(model1, training1, type = "class")

# Checking classification accuracy
table(predTrain, training1$status_group)
```

Thie code runs Model 1 on the validation set. The accuraccy is 94.82%. 
```{r}
# Predicting on Validation set
predValid <- predict(model1, validation1, type = "class")

# Checking classification accuracy
mean(predValid == validation1$status_group)                    
table(predValid,validation1$status_group)
```

To get a better understanding of the model we are looking into the importance of each variable.
```{r}
# To check important variables
importance(model1)        
varImpPlot(model1)  
```


##Model2
We are now predicting on the train set and have closer look on the actual predictions.
```{r}
# Predicting on train set 
predTrain <- predict(model2, training1, type = "class")
# Checking classification accuracy
table(predTrain, training1$status_group)
```

Thie code runs Model 2 on the validation set. The accuraccy is 98.15% which is much better than model 1. However, we have to validate the success with uploading both datasets on drivendata.org.
```{r}
# Predicting on Validation set
predValid <- predict(model2, validation1, type = "class")

# Checking classification accuracy
mean(predValid == validation1$status_group)                    
table(predValid,validation1$status_group)
```

To get a better understanding of the model we are looking into the importance of each variable.
```{r}
# To check important variables
importance(model2)        
varImpPlot(model2)  
```


Preparing the final document of Model 1 to upload on drivendata.org.
```{r}
predTest1 <- predict(model1, test1, type = "class")
solution_RF1 <- data.frame(id=test1[,"id"], status_group=predTest1)
write.csv(solution_RF1, file = "Water_solution_Random_Forest_1.csv", row.names = FALSE)
```

Preparing the final document of Model 2 to upload on drivendata.org.
```{r}
predTest2 <- predict(model2, test1, type = "class")
solution_RF2 <- data.frame(id=test1[,"id"], status_group=predTest2)
write.csv(solution_RF2, file = "Water_solution_Random_Forest_2.csv", row.names = FALSE)
```

After submitting the results from both models, we received a better score with the baseline model, which is why Model1 is the best Random Forest model.

## Decision Tree
We are using a decision tree model to challenge our Random Forest models. This code is training the model. 
```{r Compare model to Decision Tree}
library(tree)
tree1 <- tree(formula = status_group~., data = training_data1)
```

We visualize the forest built to get a better understanding of the logic behind decision tree. 
```{r}
summary(tree1)
plot(tree1)
text(tree1)
```

Thie code runs the Tree1 model on the validation set. The accuraccy is 71.43% which is lower better than the Random Forest models. We will not proceed with this model. 
```{r}
# Predicting on Validation set
predValid <- predict(tree1, validation1, type = "class")

# Checking classification accuracy
mean(predValid == validation1$status_group)                    
table(predValid,validation1$status_group)
```

Preparing the final document of Model 1 to upload on drivendata.org.
```{r}
Test_Tree <- predict(tree1, test1, type = "class")
solution_DT <- data.frame(id=test1[,"id"], status_group=Test_Tree)
write.csv(solution_DT, file = "Water_solution_Decision_Tree.csv", row.names = FALSE)
```
