# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:28:26 2021

@author: Jahidul Islam
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
df = pd.read_csv('linear reg.csv')
print (df)

#matplotlib inline
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.Area,df.Price,color='red',marker='+')


new_df = df.drop('Price',axis='columns')
new_df

price = df.Price
price
# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)

print (reg.predict([[3300]]),reg.coef_,reg.intercept_)


area_df = pd.read_csv("areas.csv")
area_df.head(3)


p = reg.predict(area_df)

area_df['prices']=p
area_df

area_df.to_csv("prediction.csv",index =False)

