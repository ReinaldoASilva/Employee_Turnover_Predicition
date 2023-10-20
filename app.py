#------------------------------------------------- Libraries ------------------------------------------------- 
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
#------------------------------------------------- Read Dataset ------------------------------------------------- 
data = pd.read_csv("datasets_9768_13874_HR_comma_sep.csv")
print(data)

#------------------------------------------------- Info Dataset ------------------------------------------------- 

data.info()
data.head()
data.isnull().sum()
data.shape
# Our data is pretty clean, with no missing values, so let's move futher and see how many\
#employees work in the organization

#------------------------------------------------- Data Cleaning ------------------------------------------------- 

# rename column name from "sales" to "departament"
data = data.rename(columns= {"sales":"department"})
data["department"].unique()

#------------------------------------------------- Data Analysis ------------------------------------------------- 

#The left column is the outcome variable recoring 1 and o.1 for employees who left the company\
#and 0 for those who are still working here.
# The department column of the dataset has many categories an we need to reduce the categories \
# for better modelling. Let's see all the categories of the department column.
# Letzs add all the 'technical', 'support and 'TI' columns into one column to make our analysis easier.
data["department"] = np.where(data["department"] == "support", "technical", data["department"])
data["department"] = np.where(data["department"] == "IT", "technical", data["department"])

#------------------------------------------------------------------------------------------------------ 

# Creating Variables for Categorical Varibles
# As there are two categorical variables(deparmente, salary)in the dataset and they need to be\
# converted to dummy varyables before they can be used for modelling.
cat_vars = ["department","salary"]
for var in cat_vars:
    cat_list="var"+"_"+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

#------------------------------------------------------------------------------------------------------ 
# Now the actual variables nees to be removed after the dummy variable have been created. Column names\
# after creating dummy variables for categorical variables
data.drop(data.columns[[8,9]],axis=1,inplace=True)
data.columns.values

#------------------------------------------------------------------------------------------------------ 

# The outcome variable is "left", and all the other variables are predictors.
data_vars=data.columns.values.tolist()
y=["left"]
x=[i for i in data_vars if i not in y]


#------------------------------------------------- Model Predict ------------------------------------------------- 
# Features Selection for employeed turnover prediction

# Let's use the feature selection method to decid which variables are the best option that can predict\
# employeed turnover with great accuracy. There are a total of 18 columns in x, and now "left" see how\
# we can select about 10 from them:
model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=10)
rfe = rfe.fit(data[x], data[y])
print(rfe.support_)
print(rfe.ranking_)

#------------------------------------------------------------------------------------------------------ 

#You can see that or feature selection chose the 10 variables for us, which are marked True in the \ 
#support_ array and marked with a choice “1” in the ranking_array. Now lets have a look at these columns:

cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low']

x = data[cols]
y = ['left']

#------------------------------------------------------------------------------------------------------ 

# Logistic Regression Model to Predict Employee Turnover