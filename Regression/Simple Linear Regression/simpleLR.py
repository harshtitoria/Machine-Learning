import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset=pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
#splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
#fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(x_train,y_train)
#predicting
y_pred = regressor.predict(x_test)
#visualising training data
plt.scatter(x_train,y_train,color = "red")
plt.plot(x_train,regressor.predict(x_train), color="blue")
plt.title("Salary Vs Years of Experiance(Training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.plot()
#visualing test data
plt.scatter(x_test,y_test,color = "red")
plt.plot(x_test,regressor.predict(x_test), color="blue")
plt.title("Salary Vs Years of Experiance(Test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.plot()