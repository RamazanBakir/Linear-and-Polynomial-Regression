# Problem Definations
Changes in exchange rates affect both domestic and international investment and consumption decisions and affect the economy as a whole through the real and financial sectors. Considering the weight of this activity on the economy, the volatilities in exchange rates have the capacity to affect the economy as a whole. Therefore, analyzing the volatility of the exchange rate is extremely important in terms of foreseeing the risks that may arise and taking precautions.

With artificial intelligence, organizations can benefit from growing data pools, better meet formal regulations, increase profits, improve customer experience and more, moreover, diversify retail financial services and make the right budget. In cases where the data is not linear, Polynomial Regression is used. The aim of linear regression is to find the line that passes on the two-dimensional line closest to all points.

In addition, the use of regressions which are the estimation method is discussed.
In this regard, linear and polynomial regression estimation is examined. These two regression models are used for different situations.

# Literature Review
In order to follow the correct way in the evaluation of the project, the last projects, techniques, and technologies were examined. In our literature review, research articles between 2000-2018 the estimation was taken into consideration for accuracy. The more valued and cited papers are only considered for our literature review to get a strong foundation for implementing the project.
Articles related to regression models, their use and applications are taken into consideration.
# METHODOLOGY 
In this section, regression model and estimation are used. In line with the researches, information is given about the project. We used 2 models: polynomial regression and linear regression. We experienced in our project which of these two models is important for us through trial and error. In short, Linear Regression is an analysis method that enables us to examine the statistical relationship between two scalar variables. In predictive (statistical) relationships, looking at some of the data we have, a hypothetical relationship is drawn between them.In order to make a prediction in linear regression, we need to create a model.
This model is as follows;
```sh
y=b0+b1.x1
```
  - b0 is  constant value.
  - b1 represents the slope of  model.
  - The value x1 represents the point to be estimated.
The process we will do in Python is about finding b0 and b1.
The aim of linear regression is to obtain the function of the relationship between parameters. The simplest machine learning model. The main objective is to determine whether there is a linear 
### Advantages of Linear Regression
> -In the linear model, a linear class boundary is created in the classification problem.
>-The theory is very well developed, its properties and behavior are well known.
>-Parameters are easy to estimate and develop according to problems.
>-Very wide and varied relationships can be expressed.

Data may sometimes be non-linear. In these cases, Polynomial Regression is also used.
```sh
Linear Regression
y=α+βx,
yi=α+βxi+£i.
```
```sh
Multiple Linear Regression
y=β0+β1x1+β2x2+β3x3+£.
```
```sh
Polynomial Regression
y=β0+β1x+β2x²+...+βhXh+£.
```
In the above equation h is the polynomial degree.
According to the polynomial and linear regression, we can now go through the sample project step by step together. I will try my best to be descriptive.
### Adding The Necessary Libraries To The Project.

We will use the numpy library as we will do all the operations on the matrix.
```sh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
```
Since the incoming data will probably come as a DataFrame, which is the data retention object of the pandas library, we have imported the pandas to make the necessary coding for that case.
```sh
x = data["Day"]
``` 
We introduce the Pyplot library for the visualization process we will use at the end.
```sh
y = data["Price"]
```
We give the values in the .csv file from the user into the variable named data.
```sh
x=x.ravel().reshape(len(data),1)
y=y.ravel().reshape(len(data),1)
```
In the above function we took the extension of the variable with len (). Because we will apply the number of data reads from csv. In the meantime, the process here is to create a new order to create data matrix.
```sh
plt.title("Linear and Polynomial Regression")
plt.xlabel("Day")
plt.ylabel("Price")
plt.grid()
plt.scatter(x,y)
```
Here we charted the raw data of our data and identified the names of the labels.
The reason for graphing our data was that we can easily see how much error rate regressions are.
In this way we get a graph. Then we go to the linear regression event.
```sh
#Linear Regression
estimate_linear=LinearRegression()
estimate_linear.fit(x,y) #For fitting on x and y axis.
estimate_linear.predict(x) #We are looking for prices by days.Predict for forecasting.
plt.plot(x,estimate_linear.predict(x),c="red") #For plotting
#print(estimate_linear.predict(x)) #to print forecasts ...
#plt.show() #to show linear ...
```
In this section, we first created a variable named estimatelineer. We equalized our constant method called LinearRegression () to this variable. Then, we fit our x, y values to their axes with the fit (x, y) function. We are looking for prices according to our function in the form of predict (x).    
```sh
#Linear Regression
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
```
Here, we have determined the error rate of the degree of the polynomial regression according to which we have seen which would be good.
```sh
hatakaresilineer=0
hatakaresipolinom=0
#Polinom regresyonun hatasını görmek için....
for i in range(len(Xyeni)):              #(Gerçek değerim - tahmini değerim)**2
    hatakaresipolinom=hatakaresipolinom+(float(y[i]-float(polinommodel.predict(Xyeni)[i])))**2  
for i in range(len(y)):
    hatakaresilineer = hatakaresilineer + (float(y[i])-float(tahminlineer.predict(x)[i]))**2
```
And next;

We first defined an array. To understand how it looks like we're wondering. Then we said that if this graphic is the same color, we don't understand anything.
And where ”c” is written, we'd better draw the colors from the array again.

•	2nd degree black ones,
•	3rd degree green,
•	9th degree gray

# Conclusion
This project includes a study on dollar estimation with artificial intelligence.
In our project, linear and polymer regression analyzes were used.We chose the Python programming language because of its library width and compatibility.

As we think that artificial intelligence will provide serious conveniences and benefits in a serious sector such as dollar rate estimation as in all fields, we have implemented such a project.

In conclusion, polynomial regression is more appropriate and 9th degree polynomial is more appropriate in this study.
