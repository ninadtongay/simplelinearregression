"""
@author: ninadtongay
"""
# Simple Linear Regression
# Dataset taken from : https://www.livechennai.com/Chennai_petrol_Price_History.asp
# Data Preprocessing #

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

# Importing the dataset
dataset = pd.read_csv('November_Chennai.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
n_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Date VS Petrol Price (Training Set)')
plt.xlabel('Date')
plt.ylabel('Petrol Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Date VS Petrol Price (Training Set)')
plt.xlabel('Date')
plt.ylabel('Petrol Price')
plt.show()


