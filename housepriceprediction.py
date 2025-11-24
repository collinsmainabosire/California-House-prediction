#importing packages for the regression model 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Loading dataset
House_df = pd.read_csv('housing.csv')
#Displaying first few rows of the dataset
#print(House_df.head())
#Checking for missing values
#print(House_df.isnull().sum())
#Dropping rows with missing values
House_df = House_df.dropna()
#dropping irrelevant columns
House_df = House_df.drop(['Address'], axis=1)
#print(House_df.info())
#Separating features and target variable
X = House_df.drop('Price', axis=1)
y = House_df['Price']
#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#checking the shape of training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
#buidling the regression model
Housing_Regression_model = LinearRegression()
#training the model
Housing_Regression_model.fit(X_train, y_train)
#making predictions
House_price_predictions = Housing_Regression_model.predict(X_test)
#evaluating the model