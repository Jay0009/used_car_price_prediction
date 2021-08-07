# -*- coding: utf-8 -*-

# -- Sheet --

# ### Predicting used car price


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

car_df = pd.read_csv("/data/workspace_files/cars_data.csv")

car_df.head()

car_df.tail()

car_df.columns

car_df.shape

car_df.info()

car_df.describe()

# Check if there are any missing elements
car_df.isnull().sum()

sns.heatmap(car_df.isnull())
plt.show()

car_df[car_df['Cylinders'].isnull()]

car_df.dropna(inplace=True)

car_df.isnull().sum()

# We can see that from the calculation above as well as the heatmap that there are a few(2) missing values in the cylinders column


car_df.info()

# Converting Invoice to int and removing $ and , from values

car_df["Invoice"] = car_df["Invoice"].str.replace("$", "")
car_df["Invoice"] = car_df["Invoice"].str.replace(",", "")
car_df["Invoice"] = car_df["Invoice"].astype(int)

# Converting MSRP to int and removing $ and , from values

car_df["MSRP"] = car_df["MSRP"].str.replace("$", "")
car_df["MSRP"] = car_df["MSRP"].str.replace(",", "")
car_df["MSRP"] = car_df["MSRP"].astype(int)

car_df.head()

car_df.describe()

sns.pairplot(car_df)
plt.show()

# checking correlation
sns.heatmap(car_df.corr(), annot=True)
plt.show()

#  From the above histogram plots, we can see that almost all features depict the skewness to the right.
# Hence, we can say that there are more outliers to the right except for the length of the car which looks that it is normally distributed.
# 
# Secondly we can see strong positive correlation between engine size and horsepower, which seems pretty obvious as the engine size increases the horse power increases. Also the increase in weight can be because of the increase in size of an engine.
# 
# On the other hand the mileage decreases as it seems the higher capacity engines could be of some type of a high performance car
# 
# 
# We can see the skewness to the right for almost every feature which depicts that those outliers could be these high performance cars as the engine size, horse power and cylinders histogram skew towards right.
# 
# 
# While exploring the correlation between MSRP and other features, we can see that there is a strong positive correlation between MSRP and the horsepower.


# Lets see all the companies in car_df

car_df.Make.unique()

import plotly as px

fig = px.hist_series(car_df, x = "Make",
                   labels = {"Make": "Manufacturer"},
                   title = "MAKE OF THE CAR",
                   color_discrete_sequence = ["maroon"])

fig.show()

fig = px.hist_series(car_df, x = "Type",
                   labels = {"Type": "Vehicle Type"},
                   title = "Type of Vehicle",
                   color_discrete_sequence = ["green"])

fig.show()

fig = px.hist_series(car_df, x = "Origin",
                   labels = {"Type": "Location"},
                   title = "Origin of company",
                   color_discrete_sequence = ["blue"])

fig.show()


fig = px.hist_series(car_df, x = "DriveTrain",
                   labels = {"Type": "Drive Train"},
                   title = "Drive Trainy",
                   color_discrete_sequence = ["red"])

fig.show()

fig = px.hist_series(car_df, x = "Make",
                     color = "Origin",
                   labels = {"Make": "Manufacturer"},
                   title = "Origin of company and its name",
                    )

fig.show()

fig = px.hist_series(car_df, x = "Make",
                     color = "Type",
                   labels = {"Make": "Manufacturer"},
                   title = "Make and type of a vehicle",
                    )

fig.show()

# Above chart depict that all the green color segments could be the outliers as the green color in the bar represents the "sports" type and the make Porsche makes highest number of sports car and only Toyota and Honda makes the hybrid cars. 


# Visualisation using wordcloud

from wordcloud import WordCloud, STOPWORDS

text = car_df.Model.values

stopwords = set(STOPWORDS)

wc = WordCloud(background_color='black', max_words= 2000, max_font_size= 100, random_state=3, stopwords=stopwords,
               contour_width=3).generate(str(text))

fig = plt.figure(figsize = (25,15))
plt.imshow(wc)
plt.axis("off")
plt.show()

# ### Cleaning and preparing the data for training


df_dum = pd.get_dummies(car_df, columns = ['Make', 'Model', 'Type', 'Origin', 'DriveTrain'], drop_first=True) # aka one hot 

df_dum

df_data = df_dum.drop(['Invoice'], axis=1)

df_data.shape

# Splitting the data

X = df_data.drop(['MSRP'], axis=1)
y = df_data['MSRP']

# Converting the data into an array

X = np.array(X)
y = np.array(y)

# Training the data

from sklearn.model_selection import  train_test_split

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# ### Now we will predict the continuous variable MSRP through Linear Regression


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score
from math import sqrt

linearRegression_model = LinearRegression()
linearRegression_model.fit(X_train, y_train)

accuracy_linearRegression = linearRegression_model.score(X_test, y_test)
print(accuracy_linearRegression)

# ### Train and Evaluate Decision tree and Random forest Models


from sklearn.tree import DecisionTreeRegressor
DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(X_train, y_train)

accuracy_DecisionTree = DecisionTree_model.score(X_test, y_test)
print(accuracy_DecisionTree)

from sklearn.ensemble import RandomForestRegressor
RandomForest_model = RandomForestRegressor(n_estimators=5, max_depth=5)
RandomForest_model.fit(X_train, y_train)

accuracy_RandomForest = RandomForest_model.score(X_test, y_test)
print(accuracy_RandomForest)

# ### NOw we will use XGBOOST


from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)

accuracy_XGBoost = model.score(X_test, y_test)
print(accuracy_XGBoost)

# ## Compare models and calculate regression KPI's


# Linear

y_predict_linear = linearRegression_model.predict(X_test)

#plotting against y_test

fig = sns.regplot(y_predict_linear, y_test, color='r', marker='^')
fig.set(title = "Linear Regression Model", xlabel= "Predicted MSRP", ylabel= "True MSRP")

r2 = r2_score(y_test, y_predict_linear)
MAE = mean_absolute_error(y_test, y_predict_linear)
MSE = mean_squared_error(y_test, y_predict_linear)
RMSE = np.sqrt(MSE)

print('r2 = ' ,r2, 'MAE = ',MAE, 'MSE = ',MSE, 'RMSE = ',RMSE)

# Random forest

y_predict_RandomForest = RandomForest_model.predict(X_test)

#plotting against y_test

fig = sns.regplot(y_predict_linear, y_test, color='r', marker='o')
fig.set(title = "Random Forest Regression Model", xlabel= "Predicted MSRP", ylabel= "True MSRP")

r2 = r2_score(y_test, y_predict_linear)
MAE = mean_absolute_error(y_test, y_predict_linear)
MSE = mean_squared_error(y_test, y_predict_linear)
RMSE = np.sqrt(MSE)

print('r2 = ' ,r2, 'MAE = ',MAE, 'MSE = ',MSE, 'RMSE = ',RMSE)

# XGBoost
y_predict_XGBoost = model.predict(X_test)

#plotting against y_test

fig = sns.regplot(y_predict_linear, y_test, color='y', marker='*')
fig.set(title = " XGBoost Regression Model", xlabel= "Predicted MSRP", ylabel= "True MSRP")

r2 = r2_score(y_test, y_predict_linear)
MAE = mean_absolute_error(y_test, y_predict_linear)
MSE = mean_squared_error(y_test, y_predict_linear)
RMSE = np.sqrt(MSE)

print('r2 = ' ,r2, 'MAE = ',MAE, 'MSE = ',MSE, 'RMSE = ',RMSE)

y_predict_DecesionTree = DecisionTree_model.predict(X_test)

#plotting against y_test

fig = sns.regplot(y_predict_linear, y_test, color='g', marker='o')
fig.set(title = "Decesion Tree Regression Model", xlabel= "Predicted MSRP", ylabel= "True MSRP")

r2 = r2_score(y_test, y_predict_linear)
MAE = mean_absolute_error(y_test, y_predict_linear)
MSE = mean_squared_error(y_test, y_predict_linear)
RMSE = np.sqrt(MSE)

print('r2 = ' ,r2, 'MAE = ',MAE, 'MSE = ',MSE, 'RMSE = ',RMSE)





