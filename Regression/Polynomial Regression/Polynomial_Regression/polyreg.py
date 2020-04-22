import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 6)
x_poly = polyreg.fit_transform(x) #x_poly create x,x^2,x^3,x^4,x^5

regressor2 = LinearRegression()
regressor2.fit(x_poly,y)

y_pred = regressor2.predict(x_poly)
 
plt.scatter(x,y, color = "red")
plt.plot(x,regressor1.predict(x),color = "blue")
plt.title("Truth or Bluff")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x,y, color = "red")
plt.plot(x_grid,regressor2.predict(polyreg.fit_transform(x_grid)),color = "blue")
plt.title("Truth or Bluff")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

