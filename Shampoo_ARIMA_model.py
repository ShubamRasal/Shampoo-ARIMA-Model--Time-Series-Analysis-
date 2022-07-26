# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:45:45 2022

@author: Shubham
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

shampoo = pd.read_csv(r"D:\Data Science and Artificial Inteligence\Semister- ll\Machine Learning Sujit Deokar Sir\shampoo_without_inflation.csv")

shampoo.head()

type(shampoo)

shampoo = pd.read_csv('D:\Data Science and Artificial Inteligence\Semister- ll\Machine Learning Sujit Deokar Sir\shampoo_without_inflation.csv',index_col=[0],parse_dates=True,squeeze=True)

type(shampoo)

#plot:
shampoo.plot()

shampoo.plot(style = 'k.')

shampoo.size

shampoo.describe()

#Smoothing the time series
shampoo_ma = shampoo.rolling(window=10).mean()
shampoo_ma.plot()

#previous value is the best reflector of the next value
#The previous value are related to next value
from pandas import concat
shampoo_base = pd.concat([shampoo,shampoo.shift(1)],axis=1)
shampoo_base 

shampoo_base.columns  
shampoo_base.columns = ['Actual_Sales','Forecast_Sales']

shampoo_base.head()

shampoo_base.dropna(inplace=True)

shampoo_base.head()

from sklearn.metrics import mean_squared_error
import numpy as np

shampoo_error = mean_squared_error(shampoo_base.Actual_Sales,shampoo_base.Forecast_Sales)
shampoo_error 

np.sqrt(shampoo_error)

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot_acf(shampoo)

#Q -3 p- 2 , d - 0-2
plot_pacf(shampoo)

from statsmodels.tsa.arima_model import ARIMA

shampoo_train = shampoo[0:25]
shampoo_test = shampoo[25:36]

shampoo_model = ARIMA(shampoo_train,order=(3,1,2))

shampoo_model_fit = shampoo_model.fit()

shampoo_model_fit.aic

shampoo_forecast = shampoo_model_fit.forecast(steps=11)[0]

np.sqrt(mean_squared_error(shampoo_test,shampoo_forecast))
 
p_value = range(0,5)
d_value = range(0,3)
q_value = range(0,5)

import warnings 
warnings.filterwarnings('ignore')

for p in p_value:
    for d in d_value:
        for q in q_value:
            order = (p,d,q)
            train,test = shampoo[0:25],shampoo[25:36]
            predictions = list()
            for i in range(len(test)):
                try:
                    model = ARIMA(train,order)
                    model_fit = model.fit(disp=0)
                    pred_y = model_fit.forecast()[0]
                    predictions.append(pred_y)
                    error = mean_squared_error(test,predictions)
                    print('ARIMA%s RMSE = %.2f'% (order,error))
                except:
                    continue










