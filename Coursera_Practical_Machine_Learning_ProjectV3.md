Coursera Practical Machine Learning Project
================
Chris Blycha
16/01/2018

1: Background
=============

#### Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your oal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting aExercise Dataset).

2: Project Description & Dataset Overview
=========================================

Project Description
===================

#### The goal of the project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

#### our submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to &lt; 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders.

#### You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details.

### Dataset Overview

##### The training data for this project are available here:

##### <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

##### The test data are available here:

##### <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

###### The data for this project come from <http://groupware.les.inf.puc-rio.br/har>.

3: Environment Preparation , Loading Required Packages & Data loading
=====================================================================

``` r
library(knitr)
```

    ## Warning: package 'knitr' was built under R version 3.4.3

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning in as.POSIXlt.POSIXct(Sys.time()): unknown timezone 'zone/tz/2017c.
    ## 1.0/zoneinfo/Australia/Sydney'

``` r
library(rpart)
library(rpart.plot)
library(randomForest)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(mboost)
```

    ## Loading required package: parallel

    ## Loading required package: stabs

    ## This is mboost 2.8-1. See 'package?mboost' and 'news(package  = "mboost")'
    ## for a complete list of changes.

    ## 
    ## Attaching package: 'mboost'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     %+%

``` r
library(Amelia)
```

    ## Loading required package: Rcpp

    ## ## 
    ## ## Amelia II: Multiple Imputation
    ## ## (Version 1.7.4, built: 2015-12-05)
    ## ## Copyright (C) 2005-2018 James Honaker, Gary King and Matthew Blackwell
    ## ## Refer to http://gking.harvard.edu/amelia/ for more information
    ## ##

``` r
library(DT)
library(markdown)
```

#### Creating folder structure to download files:

``` r
if(!file.exists("./data")){dir.create("./data")}
```

#### Downloading files & removing any \#DIV/0 entries

``` r
Training_data <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                          na.strings=c('#DIV/0', '', 'NA') ,stringsAsFactors = F)
Testing_data  <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                          na.strings= c('#DIV/0', '', 'NA'),stringsAsFactors = F)
```

#### Keeping the original downloaded files & creating copies:

``` r
Train_data <- Training_data
Test_data <- Testing_data
```

4: Exploratory Analysis
=======================

#### check missing values

``` r
missmap(Train_data, main = "Missing values Map - Train_data")
```

![](Coursera_Practical_Machine_Learning_ProjectV3_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
missmap(Test_data , main = "Missing values Map - Test_data ")
```

![](Coursera_Practical_Machine_Learning_ProjectV3_files/figure-markdown_github/unnamed-chunk-5-2.png) \#\#\#\#We can see there is allot of missing data with in this data set. \#\#\#\#Checking the total NA values in the training data set.

``` r
print(sum(is.na(Train_data))) # Checking for NA Values. 
```

    ## [1] 1921600

#### There is 1921600 NA values in the data set.

#### -Checking for zero variance predictors

``` r
x = nearZeroVar(Train_data, saveMetrics = TRUE)
str(x, vec.len=1)
```

    ## 'data.frame':    160 obs. of  4 variables:
    ##  $ freqRatio    : num  1 ...
    ##  $ percentUnique: num  100 ...
    ##  $ zeroVar      : logi  FALSE ...
    ##  $ nzv          : logi  FALSE ...

#### For this project we shall remove the zero variance predictors.

5: Split the dataset & Cleaning data.
=====================================

### Split the dataset

``` r
inTrain  <- createDataPartition(Train_data$classe, p=0.7, list=FALSE)
TrainSet <- Training_data[inTrain, ]
TestSet  <- Training_data[-inTrain,]
dim(TrainSet)
```

    ## [1] 13737   160

``` r
dim(TestSet)
```

    ## [1] 5885  160

#### Cleaning data

#### remove variables with Nearly Zero Variance

``` r
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]
dim(TrainSet)
```

    ## [1] 13737   124

``` r
dim(TestSet)
```

    ## [1] 5885  124

#### remove variables that are mostly NA

``` r
RNA <- (colSums(is.na(TrainSet)) == 0)
TrainSet <- TrainSet[, RNA]
TestSet<- TestSet[, RNA]
rm(RNA)
dim(TrainSet)
```

    ## [1] 13737    59

``` r
dim(TestSet)
```

    ## [1] 5885   59

#### remove identification only variables (columns 1 to 5)

``` r
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
dim(TrainSet)
```

    ## [1] 13737    54

``` r
dim(TestSet)
```

    ## [1] 5885   54

6: Model building
=================

### In this section, our plan is to build Decision Tree, Random forest, Generalized Boosted Model(Boosting) and then \#choose with model has the best the out-of-sample accuracy. Then use this model to predict the manner in which they \#did the exercise

### Decision Tree

#### Using ML algorithms for prediction: Decision Tree

``` r
set.seed(12345)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
```

#### Note: to view the decision tree with this command:

``` r
prp(modFitDecTree, extra=6, box.palette="auto")
```

    ## Warning: extra=6 but the response has 5 levels (only the 2nd level is
    ## displayed)

![](Coursera_Practical_Machine_Learning_ProjectV3_files/figure-markdown_github/unnamed-chunk-13-1.png)

### Now, we estimate the performance of the model on the testing data set.

``` r
predictTree <- predict(modFitDecTree, TestSet, type = "class")
Model_1 <- confusionMatrix(TestSet$classe, predictTree)
```

#### We can see the Accuracy : 0.74%. Lets see Accuracy of the other ML models.

Random Forest
=============

#### Now, we run a random forest algorithm with in is the caret package & use cross validation to select the number \#\#\#\#of the predictors. Here we use five fold cross validation in this model due the computational cost.

``` r
modelRF <- train(classe ~ ., data = TrainSet, method = "rf", trControl = trainControl(method = "cv", 5), ntree = 250)
modelRF
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    53 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10990, 10989, 10990, 10989, 10990 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9917739  0.9895934
    ##   27    0.9975250  0.9968694
    ##   53    0.9935211  0.9918042
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

#### Now, we estimate the performance of the model on the testing data set.

``` r
predictRF <- predict(modelRF, TestSet)
ModelRF_2 <-confusionMatrix(TestSet$classe, predictRF)
ModelRF_2
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    2    0    0    0
    ##          B    1 1134    4    0    0
    ##          C    0    1 1024    1    0
    ##          D    0    0    2  962    0
    ##          E    0    0    0    1 1081
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.998           
    ##                  95% CI : (0.9964, 0.9989)
    ##     No Information Rate : 0.2843          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9974          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9974   0.9942   0.9979   1.0000
    ## Specificity            0.9995   0.9989   0.9996   0.9996   0.9998
    ## Pos Pred Value         0.9988   0.9956   0.9981   0.9979   0.9991
    ## Neg Pred Value         0.9998   0.9994   0.9988   0.9996   1.0000
    ## Prevalence             0.2843   0.1932   0.1750   0.1638   0.1837
    ## Detection Rate         0.2841   0.1927   0.1740   0.1635   0.1837
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9995   0.9982   0.9969   0.9988   0.9999

#### We can see the Accuracy is over 0.99%. The Accuracy has increased, the Lets see Accuracy of the other ML \#\#\#\#models.

Method: Generalized Boosted Model(Boosting)
===========================================

### In the boosting tree model, we first use five fold cross-validation

``` r
set.seed(12345)
Mod_3 <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
GModel_3  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = Mod_3 , verbose = FALSE)
```

    ## Loading required package: survival

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loaded gbm 2.1.3

#### Now, we estimate the performance of the model on the testing data set.

#### out-of-sample errors using testing dataset

``` r
PGModel_3 <- predict(GModel_3, newdata=TestSet)
CGModel_3 <- confusionMatrix(PGModel_3, TestSet$classe)
```

``` r
CGModel_3
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1669    9    0    0    0
    ##          B    4 1116   11    2    6
    ##          C    0   14 1010   10    0
    ##          D    1    0    4  950   11
    ##          E    0    0    1    2 1065
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.9873        
    ##                  95% CI : (0.9841, 0.99)
    ##     No Information Rate : 0.2845        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.9839        
    ##  Mcnemar's Test P-Value : NA            
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9970   0.9798   0.9844   0.9855   0.9843
    ## Specificity            0.9979   0.9952   0.9951   0.9967   0.9994
    ## Pos Pred Value         0.9946   0.9798   0.9768   0.9834   0.9972
    ## Neg Pred Value         0.9988   0.9952   0.9967   0.9972   0.9965
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2836   0.1896   0.1716   0.1614   0.1810
    ## Detection Prevalence   0.2851   0.1935   0.1757   0.1641   0.1815
    ## Balanced Accuracy      0.9974   0.9875   0.9897   0.9911   0.9918

Prediction Model Selection
==========================

``` r
AccuracyResults <- data.frame(
  Model = c('Decision Tree', 'Random Forest', 'GBM(Boosting)'),
  Accuracy = rbind(Model_1$overall[1], ModelRF_2$overall[1], CGModel_3$overall[1])
)
AccuracyResults 
```

    ##           Model  Accuracy
    ## 1 Decision Tree 0.7157179
    ## 2 Random Forest 0.9979609
    ## 3 GBM(Boosting) 0.9872557

#### Based on an assessment of these 3 model fits and out-of-sample results, it looks like both random forests and GBM(Boosting) were better fit than the Decision Tree, with random forests being slightly more accurate. Therefore

#### we will use random forests to predict the manner in which they did the exercise

Preduction
==========

#### As a last step in the project, I’ll use the testing data sample to predict a classe for each of the 20 \# observations based on the other information we know about these observations contained in the validation sample.

``` r
predict(modelRF, Test_data)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
