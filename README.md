
# Starbucks Capstone Project


## Introduction:

   
It is estimated that there are 16.8 million users using the Starbucks mobile app. To keep its user engaged, once every few days, Starbucks sent out an offer to its users using the app.   

An offer can be an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). 
The offer also varies in rewards, difficulty levels, and duration.   

In this project, I am going to combine transaction, demographic, and offer data to determine which demographic groups respond best to the promotions. And I am also going to create a machine learning model to predict who is likely to complete an offer.


## Problem Statement:   
   

1. Which demographic groups respond best to the promotions? What offers and channels are converting the best?
2. Which factors are highly correlated with the completion of an offer?
3. Can we predict who is going to complete the offer?

## Data:

   
The data is contained in three files:
portfolio.json - containing offer ids and metadata about each offer (duration, type, etc.)
profile.json - demographic data for each customer
transcript.json - records for transactions, offers received, offers viewed, and offers complete

Here is the schema and explanation of each variable in the files:

#### portfolio.json  
   
id (string) - offer id
offer_type (string) - a type of offer ie BOGO, discount, informational
difficulty (int) - the minimum required to spend to complete an offer
reward (int) - reward is given for completing an offer
duration (int) - time for the offer to be open, in days
channels (list of strings)

#### profile.json
   
age (int) - age of the customer
became_member_on (int) - the date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer-id
income (float) - customer's income

#### transcript.json
   
event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer-id
time (int) - time in hours since the start of the test. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record
   
## Packages:

   
A list of libraries that we used and how to import them:
import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
   
## Data Cleaning:
   

### 1. Examing the data frames
   

*Portfolio data frame:    
   
The channels column in this data frame contains "[]" and a list of strings. The symbol needs to be removed and string values will be used to create new columns.   
   
*Transcript data frame:   
   
The value column in this data frame also includes symbols I don't need. And the list of strings will be turned into individual columns.   
   
*Portfolio data frame:   
   
This data frame contains a lot of missing values. I will need to handle the missing values later on.   
   
### 2. Merge data sets


After initial data cleaning is done. I was able to merge the data frames into one.   

The merged data frame is really messy with multiple duplicated entries of the same customer_id and id (for the offers):   

Each time a customer has made a purchase, the purchase amount is also recorded in the data as a new entry.   

Each time a customer has received or viewed or completed an offer, a new entry was also created.    

Having multiple duplicated entries will decrease the accuracy of our analysis.   

My end goal is to create a data frame with one customer_id per (offer) id.   

Variables such as "offer viewed", "view time", "offer completed", and "completed time" will need to be compressed into a single row corresponding to the correct customer_id and (offer) id.   


**To complete this goal:**

I first created 3 different data frames based on the event types. Such as offer received, offer viewed, and offer completed. 
Every time I created a data frame I made sure to drop duplicates in both "customer id" and "id" subsets.   

I kept customer id and (offer) id for this data frame so I can merge these new data frames.   

A new data frame has been created with one customer per one kind of offer with their corresponding interactions with a given offer.

### 3. Data Cleaning   

#### Handle outliers and null values


From examing the data, I can see that there are 8066 users in the data frame whose ages are 118 years old. I found it very odd.   

I created a function to check for outliers. And my suspicion was confirmed. The age of 118 was indeed an outlier.   

I then replaced the outliers with a value of -0.5. The outliers could be due to an error or the users were reluctant to disclose their ages. I will treat this outlier as an "unknown" age variable in the data engineering process later on.   

Age BoxplotCoincidentally there are 8066 null values in the 'income' data frame.   

I temporarily filled in the null values with a value of -0.5 so I can engineer new features later on.   

There are quite a few null values for the following data frames. It is due to the absence of these interactions from a user. So we will just fill in the null values with 0.   

offer viewed        14153
viewed time         14153
offer completed     34292

Remove users who completed offers without viewing


Since my goal for this data set is to predict who is going to follow through an offer and complete it.
I removed all the entries which indicate that a user has completed an offer without viewing or completed an offer before viewing.
Exploratory Data Analysis


The median age is 51 years old. The 25th percentile is 31years old and the 75th percentile is 64. years old.   

The median income in 57,000. The 25th percentile is 39,000 and the 75th percentile is 75,000.   

There are more male users than female users in the dataset. But as we can see, female users are more likely to complete an offer.

About 49% of female users have completed an offer. And only 37% of male users completed the offer. 
![gender comparison](https://miro.medium.com/max/1400/1*WDq4vU4uvDN7dpSCKdFOOg.png)

The offers also have the highest conversion rate among high-income users. The second in place is middle-income users.   
![income comparison](https://miro.medium.com/max/1400/1*L4hr9QlYJIFfuJw1--U6Kg.png)

Social media has the highest conversion rate - 45%. The web is 43% and the mobile is 39%.   
![channel comparison](https://miro.medium.com/max/1400/1*7e5gNLQu4A9fQsDizhaveg.png)

With a 53% conversion rate, discount offers have the highest conversion rate in the dataset.   
![offer comparison](https://miro.medium.com/max/1400/1*drLOBqXnJDp9nJSYgUUicg.png)

There are no completed offers for informational offers. Because they don't have any reward, difficulty level, or duration, so technically they are not real promotional offers.   

#### Correlation Matrix:

The variables that are most positively correlated with completed offers are age, discount, income, duration and offer viewed.   

age                  0.217355
discount             0.238568
income               0.272171
duration             0.304960
offer viewed         0.357929   

![correlation matrix](https://miro.medium.com/max/1400/1*QA8il4-8EPgItsByn5PR3w.png)

On the other side of the spectrum, informational offers are negatively correlated to offer completed. That makes sense because informational offers just provide information, there is no associated reward, difficulty, and duration.   



# Baseline model   

We establish the baseline model by using a basic logistic regression model.   

Since this is a classification problem, I will be using an accuracy score and F1 score as performance metrics.   

I like F1 because it is taken into consideration both precision and recall scores.  

Intial results:
The accuracy score for this logistic regression model is 0.61
The f1 score for this logistic regression model is 0.39

The baseline model is not ideal. The accuracy score is just a little better than guessing.   
To improve performance, I propose the following:   
The age variable has a large standard deviation. We can standardize the data or alternative we can categories the ages into the following groups: unknown, young, middle-aged, older adults   
The same goes for the income variable as well. We can categorize this variable into the following group: "unknown", "low income", "middle income", and "high income"   
We can use standard scaler inside Scikitlearn to standardize those variables: "reward",  "difficulty", "duration", and "membership length".   
We can use other machine learning models such as random forest classifiers since it is more Robust to Outliers and Non-linear Data.

# Results:


I used the grid search method for the logistic regression model and random forest classifier and able to achieve an accuracy score of 0.8. And f1 score of 0.73   
The accuracy score is: 0.80   
The f1 score is: 0.73  


# Discussion:


Now that we have greatly improved the performance of the machine learning model. We can focus our marketing efforts on people who are likely to take up on our offer.   
To improve performance, we can test another robust machine learning model like Gradient Boosting Classifier and fine-tune the hyperparameters.   
I also recommend that we develop another model to predict who is going to purchase without offers. If someone is going to buy without a promo offer, it helps the company save a lot of money in the long run.   

# Conclusion:


1. From the graphs, we can see those female users are more likely to complete an offer than male users.   
2. Offers are also converted better among middle income and high-income users than low-income users.   
3. Discount offers are the highest converting offers and social media is the most effective channel.   
4. Which factors are highly correlated with the completion of an offer? The variables that are most positively correlated with completed offers are age, discount, income, duration, and offer viewed.   
5. By utilizing the best estimator from Random Forest Classifier, I was able to achieve an accuracy score of 0.80 and an F1 score of 0.73.   