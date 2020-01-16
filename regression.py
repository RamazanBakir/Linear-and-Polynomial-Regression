#! /usr/bin/env python  
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("2019dollar.csv")

x = data["Day"]
y = data["Price"]

x=x.ravel().reshape(len(data),1)
y=y.ravel().reshape(len(data),1)

plt.title("Linear and Polynomial Regression")
plt.xlabel("Day")
plt.ylabel("Price")
plt.grid()
plt.scatter(x,y)  #indicates existing values as dots.
#plt.show()

#Linear Regression
estimate_linear=LinearRegression()
estimate_linear.fit(x,y) #For fitting on x and y axis.
estimate_linear.predict(x) #We are looking for prices by days.Predict for forecasting.
plt.plot(x,estimate_linear.predict(x),c="red") #For plotting
#print(estimate_linear.predict(x)) #to print forecasts ...
#plt.show() #to show linear ...

#Polinom Regresyon
a=[2,3,9,12] #degree tanımlama
b=["black","green","gray","orange"]
i=0
for i in range(len(a)):
    tahminpolinom = PolynomialFeatures(degree=a[i])#we have defined the degree of polynomial.
    Xyeni = tahminpolinom.fit_transform(x)#Create new matrix for x.

    polinommodel = LinearRegression() 
    polinommodel.fit(Xyeni,y) #We put the y value on the axis with the new matrix.
    polinommodel.predict(Xyeni) #According to the new matrix we have predicted.
    plt.plot(x,polinommodel.predict(Xyeni),c=b[i]) 
    i=i+1
plt.show()
#To find out which degree is better.
a=0
hatakaresipolinom=0
for a in range(50):

    tahminpolinom = PolynomialFeatures(degree=a+1)
    Xyeni = tahminpolinom.fit_transform(x)

    polinommodel = LinearRegression()
    polinommodel.fit(Xyeni,y)
    polinommodel.predict(Xyeni)
    for i in range(len(Xyeni)):
        hatakaresipolinom = hatakaresipolinom + (float(y[i])-float(polinommodel.predict(Xyeni)[i]))**2
    print("----------")
    print(a+1,":", hatakaresipolinom)
    print("----------")

    hatakaresipolinom = 0
"""
In order to see the errors, 
we take the square of the data between 
the estimated value and the actual value 
of the data and collect it for all values.
"""
hatakaresilineer=0
hatakaresipolinom=0
#To see the error of polynomial regression ....
for i in range(len(Xyeni)):              #(My real value - my estimated value)**2
    hatakaresipolinom=hatakaresipolinom+(float(y[i]-float(polinommodel.predict(Xyeni)[i])))**2  
for i in range(len(y)):
    hatakaresilineer = hatakaresilineer + (float(y[i])-float(estimate_linear.predict(x)[i]))**2
